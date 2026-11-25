"""Run a model on one suite and write one JSON line per episode (resumable).

  python -m hdb.run clarify --provider openai --model gpt-5.4-mini --condition plain --reps 3 --out runs/clarify_gpt54mini.jsonl
  python -m hdb.run dispute --provider openai --model gpt-5.4-mini --temperature 1.0 --out runs/dispute_gpt54mini.jsonl
  python -m hdb.run dispute --provider mock --model mock --split lite --out runs/mock.jsonl

Then: python -m hdb.score runs/dispute_gpt54mini.jsonl"""
from __future__ import annotations
import argparse, json, os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

from . import clarify_core as C
from . import dispute_core as D
from .client import make_client

DATA = Path(__file__).resolve().parents[1] / "data"


def load_items(suite, split="full"):
    items = [json.loads(l) for l in open(DATA / f"{suite}.jsonl", encoding="utf-8")]
    if suite == "dispute" and split == "lite":
        items = [i for i in items if i["variant"] in (1, 3, 5, 7, 9)]
    return items


def _env(item):
    e = dict(item["env"])
    return e


def run_clarify(item, client, condition):
    env = _env(item)
    msgs = [{"role": "system", "content": item["system_prompt"]}, {"role": "user", "content": item["initial_prompt"][condition]}]
    r = client.chat(msgs, json_schema=item["initial_schema"])
    init = C.parse_initial(r["text"], env)
    invalid_init = init is None
    if invalid_init:
        init = ("act", C.FALLBACK_ACTION)
    cont, invalid_cont, raw = {}, 0, {"initial": r["text"], "continuation": {}}
    if init[0] == "ask":
        q = init[1]
        for key, prompt in item["continuation_prompts"][condition].items():
            if not key.startswith(q + ":"):
                continue
            s = tuple(key.split(":", 1)[1].split("|"))
            r2 = client.chat([{"role": "system", "content": item["system_prompt"]}, {"role": "user", "content": prompt}],
                             json_schema=item["final_schema"])
            a = C.parse_final(r2["text"], env)
            raw["continuation"][key] = r2["text"]
            if a is None:
                invalid_cont += 1
                a = C.FALLBACK_ACTION
            cont[s] = a
    sc = C.score_policy(env, init, cont)
    return {"choice": "act" if init[0] == "act" else init[1], "initial": f"{init[0]}:{init[1]}",
            "continuation": {"|".join(k): v for k, v in cont.items()},
            "invalid_initial": invalid_init, "invalid_continuations": invalid_cont,
            "value": float(sc["value"]), "regret": float(sc["regret"]),
            "acquisition_regret": float(sc["acq_regret"]), "execution_regret": float(sc["cont_regret"]), "raw": raw}


def run_dispute(item, client):
    msgs = list(item["messages"])
    r1 = client.chat(msgs, tools=item["tools"])
    calls = [c for c in r1["tool_calls"] if c["name"] in D.RESOLUTION_TOOLS]
    nudged, r2 = False, None
    if not calls:  # one reminder, as in the original study
        nudged = True
        r2 = client.chat(msgs + [{"role": "assistant", "content": r1["text"] or "(no text)"},
                                 {"role": "user", "content": item["nudge"]}], tools=item["tools"])
        calls = [c for c in r2["tool_calls"] if c["name"] in D.RESOLUTION_TOOLS]
    return {"outcome": D.code_outcome(item, calls), "calls": calls, "nudged": nudged,
            "final_text": (r2["text"] if r2 and r2["text"] else r1["text"]) or ""}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("suite", choices=["clarify", "dispute"])
    ap.add_argument("--provider", required=True, choices=["openai", "anthropic", "mock"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key-env", default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--thinking-budget", type=int, default=None, help="Anthropic extended thinking budget")
    ap.add_argument("--condition", default="plain", choices=["plain", "computed"], help="clarify only")
    ap.add_argument("--reps", type=int, default=1, help="clarify only: repetitions per item")
    ap.add_argument("--split", default="full", choices=["full", "lite"], help="dispute only: lite = variants 1,3,5,7,9")
    ap.add_argument("--label", default=None, help="name stored with each row (default: model)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    kw = dict(base_url=a.base_url, temperature=a.temperature, max_tokens=a.max_tokens, thinking_budget=a.thinking_budget)
    if a.api_key_env:
        kw["api_key_env"] = a.api_key_env
    if a.provider == "openai":
        kw.pop("thinking_budget")
    client = make_client(a.provider, a.model, **kw)
    label = a.label or a.model

    items = load_items(a.suite, a.split)
    jobs = []
    for it in items:
        if a.suite == "clarify":
            for rep in range(1, a.reps + 1):
                jobs.append((f"{label}|{a.condition}|{it['item_id']}|r{rep}", it, rep))
        else:
            jobs.append((f"{label}|{it['item_id']}", it, 1))
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            try:
                done.add(json.loads(l)["run_key"])
            except Exception:
                pass
    todo = [j for j in jobs if j[0] not in done]
    print(f"{a.suite}: {len(jobs)} episodes, {len(done)} done, {len(todo)} to run", file=sys.stderr)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    def work(job):
        key, it, rep = job
        if a.suite == "clarify":
            res = run_clarify(it, client, a.condition)
            meta = {"condition": a.condition, "rep": rep, "pair_id": it["pair_id"], "portal": it["portal"],
                    "shift_pair": it["shift_pair"], "oracle_choice": it["oracle"]["best_choice"]}
        else:
            res = run_dispute(it, client)
            meta = {k: it[k] for k in ("holder", "record_type", "transition", "order", "last_speaker", "policy_line",
                                        "variant", "request_form", "holder_speaks_last")}
        return {"run_key": key, "suite": a.suite, "model": label, "item_id": it["item_id"], **meta, **res}

    n = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex, open(a.out, "a", encoding="utf-8") as f:
        futs = [ex.submit(work, j) for j in todo]
        for fu in as_completed(futs):
            try:
                row = fu.result()
            except Exception as e:
                print("episode failed:", e, file=sys.stderr)
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush(); n += 1
            if n % 50 == 0:
                print(f"  {n}/{len(todo)}", file=sys.stderr)
    print(f"done: {n} new episodes -> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
