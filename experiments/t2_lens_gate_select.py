"""Dev-only check of a logit-lens gate (no fitting except the threshold). At the edit site (city_last after L), score
= logit-lens log-prob of the source state's first token minus the best other state's, over the 50 state names
(final norm + unembedding applied to the residual). Held dev cities (all 4, fs1/zs1) vs EVAL accessible other-state
cities. 4 TUNING pairs x L{8,12,16,20}."""
import json
import random
import torch
from edithunt.common import ROOT
from edithunt.data import CITIES, STATES, TEMPLATES, study_states
from edithunt.model import Subject
from edithunt.env import ravel as R

S = Subject("Qwen/Qwen2.5-1.5B")
st_tok = torch.tensor([t[0] for t in S.cand_tokens(STATES)])
# first tokens collide for New */North */South */West * states: score over distinct tokens (limitation for those)
res = {}
for pr in R.candidate_pairs()[:4]:
    P = R.make_pair(S, *pr)
    priv = set(P["iso_pool"]) | set(P["decoys"])
    others = [c for s in study_states(min_cities=5) if s not in pr for c in CITIES[s] if c not in priv]
    si = STATES.index(pr[0])
    for L in (8, 12, 16, 20):
        inst = R.make_instance(S, P, L)
        def marg(cs):
            h = S.resid([S.encode(TEMPLATES[t], c) for t in R.DEV_T for c in cs], [L], "city_last")[L]
            with torch.no_grad():
                lg = S.model.lm_head(S.model.model.norm(h.to(S.device, S.dtype))).float()[:, st_tok.to(S.device)]
            o = lg.clone(); o[:, st_tok == st_tok[si]] = -1e9
            return (lg[:, si] - o.max(1).values).cpu()
        md, mo = marg(inst.dev_source), marg(others)
        r = res.setdefault(L, {"dev": [], "oth": []}); r["dev"] += md.tolist(); r["oth"] += mo.tolist()
        print(pr, L, "dev margin min %.2f mean %.2f | others max %.2f, frac>0 %.3f" % (md.min(), md.mean(), mo.max(), (mo > 0).float().mean()), flush=True)
for L, r in res.items():
    d, o = torch.tensor(r["dev"]), torch.tensor(r["oth"])
    print(L, "dev>0 %.2f | others>0 %.3f | others>1 %.3f | dev>1 %.2f" % ((d > 0).float().mean(), (o > 0).float().mean(), (o > 1).float().mean(), (d > 1).float().mean()))
(ROOT / "results" / "suite" / "t2_ravel" / "lens_gate_select.json").write_text(json.dumps({str(k): v for k, v in res.items()}))
