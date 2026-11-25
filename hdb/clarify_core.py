"""Clarify suite: when should a household AI agent ask a clarifying question before delivering a personalised offer?

Exact decision model (fractions.Fraction), prompts, strict parsing and regret scoring. No network, no API keys.
Origin: Bykovskii (2026), household value-of-information study (Run 05)."""
from fractions import Fraction as F
import itertools, json, hashlib, random, math

M = F(100)  # value of a correctly delivered personalised offer
STATES = [("A", "private"), ("A", "shared"), ("B", "private"), ("B", "shared")]
STATE_LABEL = {("A", "private"): "S1", ("A", "shared"): "S2", ("B", "private"): "S3", ("B", "shared"): "S4"}
QUERIES = ["Q_user", "Q_audience", "Q_both"]
CHOICE_CATS = ["act", "Q_user", "Q_audience", "Q_both"]


def fr(x):
    return x if isinstance(x, F) else F(str(x))


def prior(rho):
    """Equal marginals; phi-correlation rho between (U=B) and (D=shared)."""
    rho = fr(rho)
    return {("B", "shared"): (1 + rho) / 4, ("A", "private"): (1 + rho) / 4,
            ("B", "private"): (1 - rho) / 4, ("A", "shared"): (1 - rho) / 4}


def action_menu(env):
    """Returns ordered dict id -> (description, utility function of state)."""
    g = fr(env["g"])
    L = fr(env.get("L", 0))  # disclosure cost: one member's personalised offer seen privately by the other member

    def direct(h, x):
        if h[1] == "shared":
            return F(0)                      # privacy guard blocks personalised delivery on a shared channel
        return M if h[0] == x else -L        # private channel: right member gets value; wrong member = disclosure

    menu = {
        "G": ("send a generic, non-personalised message", lambda h: g),
        "N": ("send nothing", lambda h: F(0)),
        "DA": ("send A's personalised offer on the default channel", lambda h: direct(h, "A")),
        "DB": ("send B's personalised offer on the default channel", lambda h: direct(h, "B")),
    }
    if env["portal"]:
        pc = fr(env.get("portal_cost", 0))
        menu["PA"] = ("send A's personalised offer to A's verified private portal", lambda h: (M if h[0] == "A" else F(0)) - pc)
        menu["PB"] = ("send B's personalised offer to B's verified private portal", lambda h: (M if h[0] == "B" else F(0)) - pc)
    return menu


def signal_outcomes(query):
    if query == "Q_user":
        return [("A",), ("B",)]
    if query == "Q_audience":
        return [("private",), ("shared",)]
    return [(u, d) for u in ("A", "B") for d in ("private", "shared")]


def likelihood(query, s, h, qu, qd):
    qu, qd = fr(qu), fr(qd)
    pu = lambda su: qu if su == h[0] else 1 - qu
    pd = lambda sd: qd if sd == h[1] else 1 - qd
    if query == "Q_user":
        return pu(s[0])
    if query == "Q_audience":
        return pd(s[0])
    return pu(s[0]) * pd(s[1])


def price(env, query):
    c = fr(env["c"])
    return 2 * c if query == "Q_both" else c


def eu_now(env):
    p = prior(env["rho"]); menu = action_menu(env)
    return {a: sum(p[h] * u(h) for h in STATES) for a, (_, u) in menu.items()}


def posterior_values(env, query, s):
    """Returns (P(s), {action: E[u | s]}) ; E[u|s] = sum_h p(h|s) u(a,h)."""
    p = prior(env["rho"]); menu = action_menu(env)
    joint = {h: p[h] * likelihood(query, s, h, env["qu"], env["qd"]) for h in STATES}
    ps = sum(joint.values())
    if ps == 0:
        return ps, None
    return ps, {a: sum(joint[h] * u(h) for h in STATES) / ps for a, (_, u) in menu.items()}


def query_net_value(env, query):
    tot = F(0)
    for s in signal_outcomes(query):
        ps, vals = posterior_values(env, query, s)
        if ps > 0:
            tot += ps * max(vals.values())
    return tot - price(env, query)


def benchmark(env):
    now = eu_now(env)
    v0 = max(now.values())
    vq = {q: query_net_value(env, q) for q in QUERIES}
    cat_vals = {"act": v0, **vq}
    vstar = max(cat_vals.values())
    ranked = sorted(cat_vals.items(), key=lambda kv: -kv[1])
    margin = ranked[0][1] - ranked[1][1]
    best_cats = [k for k, v in cat_vals.items() if v == vstar]
    # perfect information value (upper bound, no price)
    p = prior(env["rho"]); menu = action_menu(env)
    vpi = sum(p[h] * max(u(h) for (_, u) in menu.values()) for h in STATES)
    return {"V0": v0, "Vq": vq, "Vstar": vstar, "best_cats": best_cats, "margin": margin,
            "EVPI": vpi - v0, "eu_now": now}


def score_policy(env, initial, cont):
    """initial: ('act', action) or ('ask', query). cont: dict signal tuple -> action.
    Returns expected value and regret decomposition (exact)."""
    b = benchmark(env)
    menu = action_menu(env)
    if initial[0] == "act":
        a = initial[1]
        v = b["eu_now"][a]
        acq = b["Vstar"] - b["V0"]
        cont_reg = b["V0"] - v
        return {"value": v, "regret": b["Vstar"] - v, "acq_regret": acq, "cont_regret": cont_reg}
    q = initial[1]
    tot = F(0)
    for s in signal_outcomes(q):
        ps, vals = posterior_values(env, q, s)
        if ps > 0:
            tot += ps * vals[cont[s]]
    v = tot - price(env, q)
    acq = b["Vstar"] - b["Vq"][q]
    cont_reg = b["Vq"][q] - v
    return {"value": v, "regret": b["Vstar"] - v, "acq_regret": acq, "cont_regret": cont_reg}


# ---------------------------------------------------------------- self-checks against the draft
def self_check():
    out = []
    base = dict(rho=0, qu="0.85", qd="0.85", g=35, c=4)
    e_np = dict(base, portal=False)
    e_p20 = dict(base, portal=True, portal_cost=20)
    e_p0 = dict(base, portal=True, portal_cost=0)
    bn, b20, b0 = benchmark(e_np), benchmark(e_p20), benchmark(e_p0)
    checks = [
        ("4.1 no portal: optimal = Q_both, net 45.625", bn["best_cats"] == ["Q_both"] and bn["Vstar"] == F("45.625")),
        ("4.1 portal cost 20: optimal = Q_user, net 61", b20["best_cats"] == ["Q_user"] and b20["Vstar"] == 61),
        ("4.1 free portal: optimal = Q_user, net 81", b0["best_cats"] == ["Q_user"] and b0["Vstar"] == 81),
        ("4.1 immediate values 35 / 35 / 50", (bn["V0"], b20["V0"], b0["V0"]) == (35, 35, 50)),
    ]
    # Proposition 4.2: free portal, g <= M/2, audience signal q: improvement = M|rho|(q-1/2)
    ok = True
    for rho in ["0", "0.2", "-0.4", "0.8"]:
        for q in ["0.5", "0.7", "0.95"]:
            e = dict(rho=rho, qu="0.75", qd=q, g=35, c=0, portal=True, portal_cost=0)
            b = benchmark(e)
            gain = b["Vq"]["Q_audience"] - b["V0"]
            ok &= gain == M * abs(fr(rho)) * (fr(q) - F(1, 2))
    checks.append(("Proposition 4.2 identity holds on 12 cases", ok))
    ok = True
    for rho in ["0", "0.4", "-0.8"]:
        for q in ["0.65", "0.95"]:
            e = dict(rho=rho, qu="0.75", qd=q, g=35, c=0, L=100, portal=True, portal_cost=0)
            b = benchmark(e)
            ok &= (b["Vq"]["Q_audience"] - b["V0"]) == M * abs(fr(rho)) * (fr(q) - F(1, 2))
    checks.append(("Proposition 4.2 also holds with disclosure cost L = 100", ok))
    e0 = dict(rho="0.4", qu="0.85", qd="0.85", g=35, c=0, L=100, portal=False)
    a = benchmark(e0)["eu_now"]
    checks.append(("L = 100: direct delivery to A at equal priors = 0.25*100 - 0.25*100 = 0",
                   eu_now(dict(rho=0, qu="0.85", qd="0.85", g=35, c=0, L=100, portal=False))["DA"] == 0))
    # 4.3: independent roles, free portal, qu .75, qd .95: user +25, audience 0
    e = dict(rho=0, qu="0.75", qd="0.95", g=35, c=0, portal=True, portal_cost=0)
    b = benchmark(e)
    checks.append(("4.3 user gain 25, audience gain 0", (b["Vq"]["Q_user"] - b["V0"], b["Vq"]["Q_audience"] - b["V0"]) == (25, 0)))
    # oracle policy has zero regret
    for env in (e_np, e_p0):
        bb = benchmark(env)
        k = bb["best_cats"][0]
        if k == "act":
            a = max(bb["eu_now"], key=bb["eu_now"].get)
            r = score_policy(env, ("act", a), {})
        else:
            cont = {s: max(posterior_values(env, k, s)[1].items(), key=lambda kv: kv[1])[0]
                    for s in signal_outcomes(k) if posterior_values(env, k, s)[0] > 0}
            r = score_policy(env, ("ask", k), cont)
        checks.append((f"oracle zero regret ({'portal' if env['portal'] else 'no portal'})", r["regret"] == 0))
    return checks


# ---------------------------------------------------------------- environment pool and pair sampling
# Levels of the paper's 13,500-configuration grid (work/verify_and_expand.py), portal absent vs free.
POOL = dict(rho=["-0.8", "-0.4", "0", "0.4", "0.8"], qu=["0.5", "0.65", "0.75", "0.85", "0.95"],
            qd=["0.5", "0.65", "0.75", "0.85", "0.95"], g=[20, 35, 50], c=[0, 2, 4, 10, 20, 40])
# Thesis bridge: the Delphi panel rated inadvertently shared information (customer cost, R3 median 8.0)
# at least as critical as relevance of personalisation (benefit, 7.0) -> ordinal reading L >= M; Run 05 uses L = M.
RUN05_L = 100
# Thesis bridge: Delphi industry ranking, 1st travel and tourism, 2nd digital subscription services.
DOMAINS = {
    "travel": "The offer is a personalised upgrade for a trip booked through the household's shared travel account.",
    "subscription": "The offer is a personalised add-on for the household's shared digital subscription account.",
}
MIN_MARGIN = F(2)


def build_pool():
    pairs = []
    for rho, qu, qd, g, c in itertools.product(*POOL.values()):
        base = dict(rho=rho, qu=qu, qd=qd, g=g, c=c, L=RUN05_L)
        e0 = dict(base, portal=False)
        e1 = dict(base, portal=True, portal_cost=0)
        b0, b1 = benchmark(e0), benchmark(e1)
        if b0["margin"] < MIN_MARGIN or b1["margin"] < MIN_MARGIN:
            continue
        k0, k1 = b0["best_cats"][0], b1["best_cats"][0]
        pairs.append(dict(base, k0=k0, k1=k1, shift=(k0 != k1)))
    return pairs


def sample_pairs(seed, n_shift=16, n_stable=8):
    rng = random.Random(seed)
    pool = build_pool()
    shift = [p for p in pool if p["shift"]]
    stable = [p for p in pool if not p["shift"]]
    chosen = []
    # shift pairs: stratify by rho, then round-robin over transition types for diversity
    by_rho = {r: [p for p in shift if p["rho"] == r] for r in POOL["rho"]}
    per_rho = [n_shift // len(POOL["rho"])] * len(POOL["rho"])
    for i in range(n_shift - sum(per_rho)):
        per_rho[i] += 1
    for (r, lst), k in zip(by_rho.items(), per_rho):
        types = sorted(set((p["k0"], p["k1"]) for p in lst))
        rng.shuffle(types)
        picked = []
        bucket = {t: [p for p in lst if (p["k0"], p["k1"]) == t] for t in types}
        for b in bucket.values():
            rng.shuffle(b)
        while len(picked) < min(k, len(lst)):
            for t in types:
                if bucket[t] and len(picked) < k:
                    picked.append(bucket[t].pop())
        chosen += picked
    st = stable[:]
    rng.shuffle(st)
    # stable pairs: prefer variety of categories
    cats = sorted(set(p["k0"] for p in st))
    sb = {c: [p for p in st if p["k0"] == c] for c in cats}
    picked = []
    while len(picked) < n_stable and any(sb.values()):
        for c in cats:
            if sb[c] and len(picked) < n_stable:
                picked.append(sb[c].pop())
    chosen += picked
    for stratum in (True, False):
        idx = [i for i, p in enumerate(chosen) if p["shift"] == stratum]
        doms = (["travel", "subscription"] * len(idx))[:len(idx)]
        rng.shuffle(doms)
        for i, d in zip(idx, doms):
            chosen[i]["domain"] = d
    for i, p in enumerate(chosen):
        p["pair_id"] = f"P{i+1:02d}"
    return chosen, dict(pool_size=len(pool), shift_pool=len(shift), stable_pool=len(stable))


def design_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


# ---------------------------------------------------------------- prompts
SYSTEM_PROMPT = ("You are an AI assistant that delivers marketing offers on behalf of a retailer to a household "
                 "that shares one shopping account. Decide by maximising expected utility with the numbers given. "
                 "Return only the JSON object requested.")


def fmt(x, nd=2):
    return f"{float(x):.{nd}f}"


def env_block(env):
    p = prior(env["rho"]); menu = action_menu(env)
    lines = []
    lines.append("A personalised offer is ready for a household of two adults, member A and member B, who share one account.")
    if env.get("domain") in DOMAINS:
        lines.append(DOMAINS[env["domain"]])
    lines.append("Two facts are unknown to you:")
    lines.append("- Intended user: the offer is meant for A or for B.")
    lines.append("- Channel audience: the account's default notification channel is seen privately by one person, or by the whole household (shared).")
    lines.append("")
    lines.append("Prior probabilities of the four possible situations:")
    lines.append("| situation | intended user | channel audience | probability |")
    lines.append("|---|---|---|---|")
    for h in STATES:
        lines.append(f"| {STATE_LABEL[h]} | {h[0]} | {h[1]} | {fmt(p[h], 3)} |")
    lines.append("")
    lines.append("Actions and their utility in each situation (higher is better). A privacy guard blocks personalised delivery on a shared channel (utility 0).")
    if fr(env.get("L", 0)) > 0:
        lines.append("On a private channel, an offer personalised for the other member reveals that member's interests to the person who sees it: a disclosure (negative utility).")
    if env["portal"]:
        lines.append("The verified private portal reaches only the named member and costs nothing to use; a portal offer to the wrong member is simply wasted.")
    lines.append("| action | meaning | S1 | S2 | S3 | S4 |")
    lines.append("|---|---|---|---|---|---|")
    for a, (desc, u) in menu.items():
        lines.append(f"| {a} | {desc} | " + " | ".join(fmt(u(h), 0) for h in STATES) + " |")
    lines.append("")
    lines.append("Before acting you may buy at most ONE clarification question. Its answer is a noisy report. After seeing the answer you must choose one action; no further questions are allowed.")
    lines.append("| question | what it reports | accuracy | price |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Q_user | who the intended user is (\"A\" or \"B\") | {env['qu']} | {fmt(price(env,'Q_user'),0)} |")
    lines.append(f"| Q_audience | whether the channel audience is \"private\" or \"shared\" | {env['qd']} | {fmt(price(env,'Q_audience'),0)} |")
    lines.append(f"| Q_both | both reports above, each with its own accuracy, independently | see above | {fmt(price(env,'Q_both'),0)} |")
    lines.append("Accuracy is the probability that a report is correct; report errors are independent of each other and of everything else. A question's price is subtracted from the utility of the final action.")
    return "\n".join(lines)


def initial_prompt(env, condition):
    s = ["DECISION TASK", "", env_block(env), ""]
    if condition == "computed":
        b = benchmark(env)
        s.append("DECISION SUPPORT (computed exactly by a calculator from the numbers above):")
        s.append("Expected utility of acting now: " + "; ".join(f"{a} {fmt(v)}" for a, v in b["eu_now"].items()))
        s.append("Expected net value of each question, if the best action is chosen after its answer (price already subtracted): "
                 + "; ".join(f"{q} {fmt(v)}" for q, v in b["Vq"].items()))
        s.append("")
    s.append("Choose now: act immediately with one action, or buy one question.")
    s.append('Return JSON: {"choice": "act" or "ask", "action": an action id if choice is "act", otherwise "none", '
             '"question": "Q_user", "Q_audience" or "Q_both" if choice is "ask", otherwise "none"}')
    return "\n".join(s)


def describe_signal(query, s):
    if query == "Q_user":
        return f'Q_user reported: intended user = "{s[0]}".'
    if query == "Q_audience":
        return f'Q_audience reported: channel audience = "{s[0]}".'
    return f'Q_both reported: intended user = "{s[0]}"; channel audience = "{s[1]}".'


def continuation_prompt(env, condition, query, s):
    lines = ["DECISION TASK", "", env_block(env), "",
             f"You bought {query} (price {fmt(price(env, query), 0)}, already paid). {describe_signal(query, s)}"]
    if condition == "computed":
        _, vals = posterior_values(env, query, s)
        lines.append("DECISION SUPPORT (computed exactly by a calculator): expected utility of each action given this report: "
                     + "; ".join(f"{a} {fmt(v)}" for a, v in vals.items()))
    lines.append("Choose one final action. No further questions are allowed.")
    lines.append('Return JSON: {"action": an action id}')
    return "\n".join(lines)


def initial_schema(env):
    ids = list(action_menu(env).keys())
    return {"type": "object", "additionalProperties": False, "required": ["choice", "action", "question"],
            "properties": {"choice": {"type": "string", "enum": ["act", "ask"]},
                           "action": {"type": "string", "enum": ids + ["none"]},
                           "question": {"type": "string", "enum": QUERIES + ["none"]}}}


def final_schema(env):
    ids = list(action_menu(env).keys())
    return {"type": "object", "additionalProperties": False, "required": ["action"],
            "properties": {"action": {"type": "string", "enum": ids}}}


# ---------------------------------------------------------------- strict parsing (pre-registered rules)
def _load_json(text):
    if text is None:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:]
        t = t.strip()
    try:
        obj = json.loads(t)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _canon(val, allowed):
    if not isinstance(val, str):
        return None
    for a in allowed:
        if val.strip().lower() == a.lower():
            return a
    return None


def parse_initial(text, env):
    obj = _load_json(text)
    if obj is None:
        return None
    ids = list(action_menu(env).keys())
    ch = _canon(obj.get("choice"), ["act", "ask"])
    if ch == "act":
        a = _canon(obj.get("action"), ids)
        return ("act", a) if a else None
    if ch == "ask":
        q = _canon(obj.get("question"), QUERIES)
        return ("ask", q) if q else None
    return None


def parse_final(text, env):
    obj = _load_json(text)
    if obj is None:
        return None
    return _canon(obj.get("action"), list(action_menu(env).keys()))


FALLBACK_ACTION = "N"  # fixed no-send operational fallback for invalid responses (as in earlier runs)
