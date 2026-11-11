"""H1 (soft metrics) + noise data: re-grade every stored T2 v2 validation edit (results/suite/t2_ravel/edits_v:
20 pairs x L8/12/16/20 x {null, random, meandiff, naive_capped, naive_uncapped, oracle s0-2}) under grader v2.1
(whole Iso pool, float32 gate) with per-item outputs, so bootstrap / Iso-draw / tau analyses run offline.
Rows -> results/suite/t2v21/regrade.<shard>.jsonl (resumable).
  PYTHONPATH=. .venv/bin/python experiments/t2v21_regrade.py --shard 0/1 [--only_valid]"""
import argparse
import json
import re
import time

from edithunt.common import ROOT
from edithunt.model import Subject
from edithunt.env import ravel as R

SRC = ROOT / "results" / "suite" / "t2_ravel"
OUT = ROOT / "results" / "suite" / "t2v21"
ap = argparse.ArgumentParser()
ap.add_argument("--shard", default="0/1")
ap.add_argument("--only_valid", action="store_true", help="only the 11 instances valid under the v2 rule")
a = ap.parse_args()
K, N = map(int, a.shard.split("/"))
OUT.mkdir(parents=True, exist_ok=True)
out = OUT / f"regrade.{K}.jsonl"
done = {(r["pair"], r["layer"], r["method"], r["seed"]) for r in map(json.loads, out.read_text().splitlines())} \
    if out.exists() else set()
f = out.open("a")
valid = {(p, L) for p, L in json.loads((SRC / "validate_summary.json").read_text())["valid_instances"]}
pat = re.compile(r"(.+)_to_(.+)-L(\d+)-(\w+)-s(\d)\.pt")
files = sorted((SRC / "edits_v").glob("*.pt"))
jobs = []
for p in files:
    s, t, L, m, sd = pat.fullmatch(p.name).groups()
    pair = f"{s.replace('_', ' ')}->{t.replace('_', ' ')}"
    if a.only_valid and (pair, int(L)) not in valid:
        continue
    jobs.append((pair, int(L), m, int(sd), p))
jobs = [j for i, j in enumerate(sorted(jobs, key=lambda j: (j[0], j[1]))) if i % N == K]
S = Subject("Qwen/Qwen2.5-1.5B")
Ps, insts = {}, {}
for pair, L, m, sd, p in jobs:
    if (pair, L, m, sd) in done:
        continue
    s, t = pair.split("->")
    P = Ps.setdefault(pair, R.make_pair(S, s, t))
    inst = insts.setdefault((pair, L), R.make_instance(S, P, L, "v2.1"))
    t0 = time.time()
    e = [x[:3] if len(x) > 3 and x[3] == "add" else x for x in R.load_edits(p)]  # stored with an explicit "add" kind
    g = R.grade_ravel(S, inst, None, edits=e, detail=True)
    r = {"pair": pair, "layer": L, "method": m, "seed": sd, "valid_v2": (pair, L) in valid,
         "R": g["reward_mult"], "R_soft": g["soft"]["reward"], "soft": g["soft"],
         **{k: g[k]["rate"] for k in ("Cause", "Iso_state", "Iso_other")}, "kl_mean": g["kl_mean"],
         "kl_factor": g["kl_factor"], "items": g["items"], "secs": round(time.time() - t0, 1)}
    f.write(json.dumps(r) + "\n"); f.flush()
    print(pair, L, m, sd, "R=%.3f soft=%.3f (%.0fs)" % (r["R"], r["R_soft"], r["secs"]), flush=True)
