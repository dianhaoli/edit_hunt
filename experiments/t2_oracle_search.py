"""Find an oracle that clearly beats the shortcuts on T2 v2 (4 pairs, L8/L12). Rows streamed to JSONL.
  PYTHONPATH=. .venv/bin/python experiments/t2_oracle_search.py --shard 0/2"""
import argparse, json, time
from edithunt.common import ROOT
from edithunt.model import Subject
from edithunt.env import ravel as R

VARIANTS = {
    "cap2": dict(cap=2.0),
    "cap4": dict(cap=4.0),
    "uncapped": dict(cap=None),
    "cap2_allkeep": dict(cap=2.0, all_keep=True),
    "cap2_allkeep_stx": dict(cap=2.0, all_keep=True, state_extra=True),
    "cap4_allkeep_stx_w3": dict(cap=4.0, all_keep=True, state_extra=True, w_state=3.0, w_other=3.0),
    "cap4_allkeep_stx_w3_300": dict(cap=4.0, all_keep=True, state_extra=True, w_state=3.0, w_other=3.0, steps=300),
}
ap = argparse.ArgumentParser(); ap.add_argument("--shard", default="0/1"); a = ap.parse_args()
k, n = map(int, a.shard.split("/"))
S = Subject("Qwen/Qwen2.5-1.5B")
out = ROOT / "results" / "suite" / "t2_ravel" / f"oracle_search.{k}.jsonl"
done = {(r["pair"], r["layer"], r["variant"]) for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
f = out.open("a")
jobs = [(pr, L, v) for pr in R.candidate_pairs()[:4] for L in (8, 12) for v in VARIANTS]
Ps = {}
for i, (pr, L, v) in enumerate(jobs):
    pair = f"{pr[0]}->{pr[1]}"
    if i % n != k or (pair, L, v) in done:
        continue
    P = Ps.setdefault(pr, R.make_pair(S, *pr))
    inst = R.make_instance(S, P, L)
    nrm = R.meandiff(S, inst, L).norm().item()
    kw = dict(VARIANTS[v]); cap = kw.pop("cap")
    t = time.time()
    e = R.train_vector(S, inst, L, 0, kw.pop("steps", 150), keep=True, max_norm=cap * nrm if cap else None, **kw)
    g = R.grade_ravel(S, inst, None, edits=e)
    r = {"pair": pair, "layer": L, "variant": v, "norm_ratio": e[0][2].norm().item() / nrm, "secs": time.time() - t,
         **{x: g[x]["rate"] for x in ("Cause", "Iso_state", "Iso_other")},
         **{x: g[x] for x in ("kl_mean", "reward", "reward_product", "reward_mult")}}
    f.write(json.dumps(r) + "\n"); f.flush()
    print(pair, L, v, " ".join(f"{x}={r[x]:.2f}" for x in ("Cause", "Iso_state", "Iso_other", "kl_mean", "reward", "reward_product", "reward_mult", "norm_ratio")), flush=True)
