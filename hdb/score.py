"""Summarise run files written by hdb.run.

  python -m hdb.score runs/*.jsonl            # prints a table per suite
  python -m hdb.score runs/*.jsonl --csv out.csv"""
from __future__ import annotations
import argparse, collections, csv, json, sys

from . import dispute_core as D


def clarify_summary(rows):
    out = []
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["model"], r["condition"])].append(r)
    for (m, c), g in sorted(groups.items()):
        n = len(g)
        # selective responsiveness: does the first move change when the portal changes the optimal move?
        by = collections.defaultdict(dict)
        for r in g:
            by[(r["pair_id"], r["rep"])][r["portal"]] = r
        sh, st = [], []
        for (pid, rep), d in by.items():
            if True in d and False in d:
                changed = d[True]["choice"] != d[False]["choice"]
                (sh if d[True]["shift_pair"] else st).append(changed)
        sr = (sum(sh) / len(sh) - sum(st) / len(st)) if sh and st else float("nan")
        out.append({"suite": "clarify", "model": m, "condition": c, "episodes": n,
                    "mean_regret": sum(r["regret"] for r in g) / n,
                    "acquisition_regret": sum(r["acquisition_regret"] for r in g) / n,
                    "execution_regret": sum(r["execution_regret"] for r in g) / n,
                    "oracle_first_move": sum(r["choice"] == r["oracle_choice"] for r in g) / n,
                    "selective_responsiveness": sr,
                    "invalid_initial": sum(bool(r["invalid_initial"]) for r in g) / n})
    return out


def dispute_summary(rows):
    out = []
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["model"], "on" if r["policy_line"] else "off")].append(r)
    for (m, p), g in sorted(groups.items()):
        out.append({"suite": "dispute", "model": m, "policy_line": p, **D.metrics(g),
                    "reminded": sum(bool(r.get("nudged")) for r in g) / len(g)})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args(argv)
    rows = [json.loads(l) for f in a.files for l in open(f, encoding="utf-8") if l.strip()]
    res = clarify_summary([r for r in rows if r["suite"] == "clarify"]) + dispute_summary([r for r in rows if r["suite"] == "dispute"])
    for r in res:
        print(", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    if a.csv and res:
        keys = sorted({k for r in res for k in r}, key=lambda k: list(res[0]).index(k) if k in res[0] else 99)
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(res)


if __name__ == "__main__":
    main()
