"""Choose the gate ramp (where between the other-state mean (0) and the dev-source mean (1) the gate opens) on DEV
data only, without training v: leave-one-dev-city-out, k = unit(mean of 3 dev cities - mean keep cities), ramp from
those means; score = fraction of the held dev city's prompts (fs1, zs1) with the gate fully open, and fraction of
UNUSED accessible other-state cities' prompts (keep_pool minus keep_cities) with the gate fully closed.
4 tuning pairs x L{8,12,16,20} x 4 folds."""
import json
import torch
from edithunt.common import ROOT
from edithunt.data import TEMPLATES
from edithunt.model import Subject
from edithunt.env import ravel as R

RAMPS = [(0.3, 0.6), (0.2, 0.45), (0.15, 0.35), (0.1, 0.3), (0.05, 0.2)]
S = Subject("Qwen/Qwen2.5-1.5B")
res = {str(r): {"open": [], "closed": []} for r in RAMPS}
per_layer = {}
for pr in R.candidate_pairs()[:4]:
    P = R.make_pair(S, *pr)
    for L in (8, 12, 16, 20):
        inst = R.make_instance(S, P, L)
        used = [c for cs in inst.extra["keep_cities"].values() for c in cs]
        unused = [c for cs in inst.extra["keep_pool"].values() for c in cs if c not in used]
        H = lambda cs: S.resid([S.encode(TEMPLATES[t], c) for t in R.DEV_T for c in cs], [L], "city_last")[L].float()
        Hk, Hu = H(used), H(unused)
        for held in inst.dev_source:
            tr = [c for c in inst.dev_source if c != held]
            Hs, Hh = H(tr), H([held])
            k = Hs.mean(0) - Hk.mean(0); k = k / k.norm()
            ms, mk = (Hs @ k).mean().item(), (Hk @ k).mean().item()
            for r in RAMPS:
                b, w = mk + r[0] * (ms - mk), (r[1] - r[0]) * (ms - mk)
                go = ((Hh @ k - b) / w).clamp(0, 1); gc = ((Hu @ k - b) / w).clamp(0, 1)
                res[str(r)]["open"] += (go >= 0.99).float().tolist(); res[str(r)]["closed"] += (gc <= 0.01).float().tolist()
                d = per_layer.setdefault((L, str(r)), {"open": [], "closed": []})
                d["open"] += (go >= 0.99).float().tolist(); d["closed"] += (gc <= 0.01).float().tolist()
        print(pr, L, "done", flush=True)
m = lambda x: sum(x) / len(x)
print("ramp            held-dev open   unused-others closed")
for r in RAMPS:
    print(f"{str(r):14s}  {m(res[str(r)]['open']):.3f} (n={len(res[str(r)]['open'])})   {m(res[str(r)]['closed']):.3f} (n={len(res[str(r)]['closed'])})")
for L in (8, 12, 16, 20):
    print(L, {str(r): (round(m(per_layer[(L, str(r))]['open']), 2), round(m(per_layer[(L, str(r))]['closed']), 3)) for r in RAMPS})
(ROOT / "results" / "suite" / "t2_ravel" / "ramp_select.json").write_text(json.dumps(
    {str(r): {k: m(v) for k, v in res[str(r)].items()} for r in RAMPS}, indent=1))
