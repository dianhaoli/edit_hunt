"""Summary of the T2 v2.1 library / budget-curve run (results/suite/t2v21/library.*.jsonl) against the pre-registered
bars: reference = mean of ref_s0..2 (v2.1); agent bar = 0.8 x that; 'not script-solvable' = best script <= 0.5 x
solver. Ceiling column = the v2 cross-fitted ceiling (graded with the v2 16-city Iso draw; approximate under v2.1).
  PYTHONPATH=. .venv/bin/python experiments/t2v21_library_table.py"""
import json
import statistics as st
from collections import defaultdict

from edithunt.common import ROOT

D = ROOT / "results" / "suite" / "t2v21"
rows = [json.loads(l) for p in sorted(D.glob("library.*.jsonl")) for l in p.read_text().splitlines()]
by = defaultdict(dict)
for r in rows:
    by[r["id"]][r["method"]] = r
ceil = defaultdict(list)
for p in (ROOT / "results/suite/t2_ravel").glob("ceiling.*.jsonl"):
    for l in p.read_text().splitlines():
        r = json.loads(l)
        ceil[r["id"] + "-v21"].append(r["R"])
SCRIPTS = [m for m in ("md_mid_x1", "opt_target100", "opt_keep5", "opt_keep10", "opt_keep20", "opt_keep40", "opt_keep100",
                       "opt_keep150", "opt_keep10_lr02", "opt_keep20_lr02", "opt_keep100_s1", "opt_keep100_s2",
                       "opt_keep100_more", "ungated_keep100")]
out = {"instances": {}, "methods": {}}
for iid, d in sorted(by.items()):
    refs = [d[f"ref_s{s}"]["R"] for s in range(3) if f"ref_s{s}" in d]
    ref = st.mean(refs) if refs else None
    sc = {m: d[m]["R"] for m in SCRIPTS if m in d}
    best = max(sc, key=sc.get) if sc else None
    out["instances"][iid] = {"ref_mean": ref, "ref_seeds": refs, "ref_sd": st.pstdev(refs) if len(refs) > 1 else None,
                             "ceiling_v2": st.mean(ceil[iid]) if ceil[iid] else None, "scripts": sc, "best": best,
                             "best_R": sc.get(best), "fwd": {m: d[m]["fwd"] for m in sc}}
for m in SCRIPTS + ["ref_s0", "ref_s1", "ref_s2"]:
    xs = [(d[m]["R"], d[m]["R_soft"], d[m]["fwd"]) for d in by.values() if m in d]
    if not xs:
        continue
    bars = [0.8 * out["instances"][i]["ref_mean"] for i, d in by.items() if m in d and out["instances"][i]["ref_mean"]]
    passes = sum(d[m]["R"] >= 0.8 * out["instances"][i]["ref_mean"] for i, d in by.items()
                 if m in d and out["instances"][i]["ref_mean"])
    out["methods"][m] = {"n": len(xs), "mean_R": st.mean(x[0] for x in xs), "mean_R_soft": st.mean(x[1] for x in xs),
                         "fwd": st.mean(x[2] for x in xs), "pass_0.8ref": passes, "n_bar": len(bars)}
print(f"{'method':18s} {'n':>3s} {'R':>6s} {'R_soft':>6s} {'fwd':>6s} pass(>=0.8 ref)")
for m, v in out["methods"].items():
    print(f"{m:18s} {v['n']:3d} {v['mean_R']:6.3f} {v['mean_R_soft']:6.3f} {v['fwd']:6.0f} {v['pass_0.8ref']}/{v['n_bar']}")
print()
for i, v in out["instances"].items():
    print(f"{i[13:-4]:32s} ref {v['ref_mean'] or 0:.2f} (sd {v['ref_sd'] or 0:.2f}) ceil_v2 {v['ceiling_v2'] or 0:.2f} "
          f"best {v['best']} {v['best_R'] or 0:.2f}")
(D / "library_summary.json").write_text(json.dumps(out, indent=1))
