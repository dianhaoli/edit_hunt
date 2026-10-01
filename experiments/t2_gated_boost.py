"""Gated oracle + extra capital wordings (fs2, zs2), 300 steps, L16/L20, 4 pairs. Rows -> oracle_search.gated.jsonl."""
import json, time
from edithunt.common import ROOT
from edithunt.model import Subject
from edithunt.env import ravel as R
S = Subject("Qwen/Qwen2.5-1.5B")
out = (ROOT / "results" / "suite" / "t2_ravel" / "oracle_search.gated.jsonl").open("a")
for pr in R.candidate_pairs()[:4]:
    P = R.make_pair(S, *pr)
    for L in (16, 20):
        inst = R.make_instance(S, P, L)
        t = time.time()
        e = R.train_gated(S, inst, L, 0, steps=300, templates=["fs1", "zs1", "fs2", "zs2"])
        g = R.grade_ravel(S, inst, None, edits=e)
        r = {"pair": f"{pr[0]}->{pr[1]}", "layer": L, "variant": "gated_boost", "secs": time.time() - t,
             **{x: g[x]["rate"] for x in ("Cause", "Iso_state", "Iso_other")},
             **{x: g[x] for x in ("kl_mean", "reward", "reward_product", "reward_mult")}}
        out.write(json.dumps(r) + "\n"); out.flush()
        print(r["pair"], L, r["variant"], " ".join(f"{x}={r[x]:.2f}" for x in ("Cause", "Iso_state", "Iso_other", "kl_mean", "reward", "reward_product")), flush=True)
