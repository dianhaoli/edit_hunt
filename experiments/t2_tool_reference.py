"""T2 v2: run the tool-path reference (ref_ravel_gated, through the agent tools) on the validated instances and
compare with the offline oracle. Rows -> results/suite/t2_ravel/tool_ref.jsonl.
  PYTHONPATH=. .venv/bin/python experiments/t2_tool_reference.py [--all]"""
import argparse, json, statistics as st, time
from edithunt.common import ROOT
from edithunt.model import Subject
from edithunt.env import ravel as R
from edithunt.env.tasks import run_reference

D = ROOT / "results" / "suite" / "t2_ravel"
ap = argparse.ArgumentParser(); ap.add_argument("--all", action="store_true"); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
summ = json.loads((D / "validate_summary.json").read_text())
orc = {}
for l in (D / "validate.or0.jsonl").read_text().splitlines():
    r = json.loads(l)
    if r["method"] == "oracle":
        orc.setdefault((r["pair"], r["layer"]), []).append(r["reward_mult"])
targets = sorted(orc) if a.all else sorted(tuple(x) for x in summ["valid_instances"])
out = D / "tool_ref.jsonl"
done = {(r["pair"], r["layer"], r["seed"]) for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
f = out.open("a")
S = Subject("Qwen/Qwen2.5-1.5B")
Ps = {}
for pair, L in targets:
    if (pair, L, a.seed) in done:
        continue
    s, t = pair.split("->")
    P = Ps.setdefault(pair, R.make_pair(S, s, t))
    inst = R.make_instance(S, P, L)
    t0 = time.time()
    res = run_reference(S, inst, seed=a.seed)
    g = res["grade"]
    r = {"pair": pair, "layer": L, "seed": a.seed, "error": res["error"], "budget_used": res["budget_used"],
         "valid": g.get("valid"), "reason": g.get("reason"), "oracle_mean": st.mean(orc[(pair, L)]),
         "secs": time.time() - t0}
    if g.get("valid"):
        r |= {k: g[k]["rate"] for k in ("Cause", "Iso_state", "Iso_other")} | {"kl_mean": g["kl_mean"], "reward_mult": g["reward_mult"]}
    f.write(json.dumps(r) + "\n"); f.flush()
    print(pair, L, "tool-ref R=%s oracle=%.2f" % (("%.2f" % r["reward_mult"]) if g.get("valid") else "INVALID", r["oracle_mean"]),
          {k: round(r[k], 2) for k in ("Cause", "Iso_state", "Iso_other", "kl_mean") if k in r}, r["error"] or "", r["reason"] or "",
          "budget", r["budget_used"], flush=True)
