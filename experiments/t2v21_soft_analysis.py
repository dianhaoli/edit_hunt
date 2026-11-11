"""H1 analysis (rule pre-registered in LAB_NOTEBOOK 2026-10-02) on results/suite/t2v21/regrade.*.jsonl.
Offline only: per-item values -> hard / soft R, bootstrap by city, v2-style 16-city Iso draws, tau sensitivity.
  PYTHONPATH=. .venv/bin/python experiments/t2v21_soft_analysis.py"""
import json
import math
import random
import statistics as st
from collections import defaultdict

from edithunt.common import ROOT

D = ROOT / "results" / "suite" / "t2v21"
rows = [json.loads(l) for p in sorted(D.glob("regrade.*.jsonl")) for l in p.read_text().splitlines()]
WRONG = ["null", "random", "meandiff", "naive_uncapped"]
SHORT = ["null", "random", "meandiff", "naive_capped", "naive_uncapped"]
sig = lambda x: 1 / (1 + math.exp(-x))


def parts(it, tau=1.0, cities=None, ocities=None):
    """(hard, soft) R from per-item lists; cities/ocities = resampled city multisets (None = all)."""
    def sel(lst, key, pool):
        if pool is None:
            return lst
        by = defaultdict(list)
        for x in lst:
            by[x[0]].append(x)
        return [x for c in pool for x in by.get(c, [])]
    ca = sel(it["cause"], 0, cities); ss = sel(it["iso_state"], 0, cities); oo = sel(it["iso_other"], 0, ocities)
    m = lambda xs: sum(xs) / len(xs) if xs else 0.0
    hard = m([x[2] for x in ca]) * m([x[1] for x in ss]) * m([x[2] for x in oo]) * it["kl_factor"]
    soft = m([sig(x[3] / tau) for x in ca]) * m([x[2] for x in ss]) * m([x[3] for x in oo]) * it["kl_factor"]
    return hard, soft


def boot(it, n=200, seed=0):
    rng = random.Random(seed)
    cs = sorted({x[0] for x in it["cause"]}); os_ = sorted({x[0] for x in it["iso_other"]})
    out = [parts(it, cities=[rng.choice(cs) for _ in cs], ocities=[rng.choice(os_) for _ in os_]) for _ in range(n)]
    return st.pstdev(h for h, _ in out), st.pstdev(s for _, s in out)


def iso16(it, n=10):
    os_ = sorted({x[0] for x in it["iso_other"]})
    out = [parts(it, ocities=random.Random(f"iso16-{k}").sample(os_, min(16, len(os_)))) for k in range(n)]
    return st.pstdev(h for h, _ in out), st.pstdev(s for _, s in out)


def ranks(xs):
    o = sorted(range(len(xs)), key=lambda i: xs[i]); r = [0.0] * len(xs); i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and xs[o[j + 1]] == xs[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else float("nan")


inst = defaultdict(dict)
for r in rows:
    h, s = parts(r["items"])
    assert abs(h - r["R"]) < 1e-6 and abs(s - r["R_soft"]) < 1e-6, (r["pair"], r["layer"], r["method"], h, r["R"], s, r["R_soft"])
    inst[(r["pair"], r["layer"])][(r["method"], r["seed"])] = r
complete = {k: v for k, v in inst.items() if len(v) == 8}
print(f"rows {len(rows)}; instances complete {len(complete)} / {len(inst)}")
res = {"n_rows": len(rows), "n_instances": len(complete)}
# (a) ordering of method means, hard vs soft (oracle = per-instance mean of 3 seeds)
meth = SHORT + ["oracle"]
def val(v, m, key):
    return st.mean(v[(m, s)][key] for s in range(3)) if m == "oracle" else v[(m, 0)][key]
mm = {m: {k: st.mean(val(v, m, k) for v in complete.values()) for k in ("R", "R_soft")} for m in meth}
print("method means (all complete instances):", {m: (round(x["R"], 3), round(x["R_soft"], 3)) for m, x in mm.items()})
order_h = sorted(meth, key=lambda m: mm[m]["R"]); order_s = sorted(meth, key=lambda m: mm[m]["R_soft"])
sp = [spearman([val(v, m, "R") for m in meth], [val(v, m, "R_soft") for m in meth]) for v in complete.values()]
sp = [x for x in sp if not math.isnan(x)]
res["a"] = {"means": mm, "order_hard": order_h, "order_soft": order_s, "median_spearman": st.median(sp), "n_sp": len(sp),
            "pass": None}
# ties at 0 in the hard metric (null/random/mean-diff all 0) make the hard order of those three arbitrary: compare
# the order on methods whose hard means differ by > 0.01
distinct = [m for m in meth if all(abs(mm[m]["R"] - mm[x]["R"]) > 0.01 for x in meth if x != m)]
same_order = [m for m in order_h if m in distinct] == [m for m in order_s if m in distinct]
res["a"]["distinct_methods"] = distinct
res["a"]["pass"] = same_order and st.median(sp) >= 0.8
# (b) wrong methods stay low
res["b"] = {m: mm[m]["R_soft"] for m in WRONG}; res["b"]["pass"] = all(mm[m]["R_soft"] <= 0.10 for m in WRONG)
# (c) separation on the v2-valid instances
valid = {k: v for k, v in complete.items() if next(iter(v.values()))["valid_v2"]}
sep = {"hard": [], "soft": []}; noise = defaultdict(list)
for k, v in valid.items():
    for key, j in (("hard", 0), ("soft", 1)):
        rk = "R" if key == "hard" else "R_soft"
        orc = st.mean(v[("oracle", s)][rk] for s in range(3))
        best = max(v[(m, 0)][rk] for m in SHORT)
        bsd = st.mean(boot(v[("oracle", s)]["items"])[j] for s in range(3))
        sep[key].append((orc - best) / max(bsd, 1e-6))
        noise[key + "_boot_sd"].append(bsd)
        noise[key + "_seed_sd"].append(st.pstdev(v[("oracle", s)][rk] for s in range(3)))
        noise[key + "_iso16_sd"].append(st.mean(iso16(v[("oracle", s)]["items"])[j] for s in range(3)))
        noise[key + "_gap"].append(orc - best)
md = lambda xs: st.median(xs) if xs else float("nan")
res["c"] = {"median_sep_hard": md(sep["hard"]), "median_sep_soft": md(sep["soft"]),
            "pass": bool(sep["soft"]) and md(sep["soft"]) > md(sep["hard"]), "n": len(valid),
            **{k: md(x) for k, x in noise.items()}}
# tau sensitivity of the soft oracle/shortcut means
res["tau"] = {}
for tau in (0.5, 1.0, 2.0):
    res["tau"][tau] = {m: st.mean(st.mean(parts(v[(m, s)]["items"], tau)[1] for s in (range(3) if m == "oracle" else [0]))
                                  for v in complete.values()) for m in meth}
res["adopt_soft"] = bool(res["a"]["pass"] and res["b"]["pass"] and res["c"]["pass"])
print(json.dumps({k: res[k] for k in ("a", "b", "c", "tau", "adopt_soft")}, indent=1, default=str))
(D / "soft_analysis.json").write_text(json.dumps(res, indent=1, default=str))
