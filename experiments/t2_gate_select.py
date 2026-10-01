"""Dev-only gate design check (no training of v). Leave-one-dev-city-out on the 4 TUNING pairs x L{8,12,16,20}.
Accessible other-state cities (every city of a non-source/target state that is not private to the grader) are split
50/50 into FIT (key + threshold) and EVAL (closedness). Score: held dev city prompts with gate fully open; EVAL
other-state prompts with gate fully closed.
  key "diff": unit(mean dev - mean FIT others)        key "lda": unit(Sigma^-1 (mean dev - mean FIT others)),
  Sigma = shrunk covariance of FIT others (shrink to scaled identity, alpha)
  ramp: z-scores of FIT others' projection: gate 0 at mu + z0*sd, 1 at mu + z1*sd."""
import json
import random
import torch
from edithunt.common import ROOT
from edithunt.data import CITIES, TEMPLATES, study_states
from edithunt.model import Subject
from edithunt.env import ravel as R

S = Subject("Qwen/Qwen2.5-1.5B")
KEYS = [("diff", None), ("lda", 0.5), ("lda", 0.2)]
ZS = [(2, 4), (3, 5), (4, 6), (5, 8)]
tot = {}
for pr in R.candidate_pairs()[:4]:
    P = R.make_pair(S, *pr)
    priv = set(P["iso_pool"]) | set(P["decoys"])
    others = [c for s in study_states(min_cities=5) if s not in pr for c in CITIES[s] if c not in priv]
    random.Random(f"{pr}-split").shuffle(others)
    fit, ev = others[: len(others) // 2], others[len(others) // 2:]
    for L in (8, 12, 16, 20):
        inst = R.make_instance(S, P, L)
        H = lambda cs: S.resid([S.encode(TEMPLATES[t], c) for t in R.DEV_T for c in cs], [L], "city_last")[L].float()
        Hf, He = H(fit), H(ev)
        mu_o = Hf.mean(0)
        C = torch.cov(Hf.T)
        for held in inst.dev_source:
            Hs, Hh = H([c for c in inst.dev_source if c != held]), H([held])
            d = Hs.mean(0) - mu_o
            for kname, a in KEYS:
                if kname == "diff":
                    k = d
                else:
                    Cs = (1 - a) * C + a * C.diagonal().mean() * torch.eye(C.shape[0])
                    k = torch.linalg.solve(Cs, d)
                k = k / k.norm()
                pf = Hf @ k
                m, sd = pf.mean().item(), pf.std().item()
                for z0, z1 in ZS:
                    b, w = m + z0 * sd, (z1 - z0) * sd
                    go = ((Hh @ k - b) / w).clamp(0, 1); gc = ((He @ k - b) / w).clamp(0, 1)
                    t = tot.setdefault((kname, a, z0, z1), {"open": [], "closed": [], "L": {}})
                    t["open"] += (go >= 0.99).float().tolist(); t["closed"] += (gc <= 0.01).float().tolist()
                    tl = t["L"].setdefault(L, {"open": [], "closed": []})
                    tl["open"] += (go >= 0.99).float().tolist(); tl["closed"] += (gc <= 0.01).float().tolist()
        print(pr, L, "fit", len(fit), "eval", len(ev), flush=True)
m = lambda x: sum(x) / len(x)
print("key        z-ramp   held-dev open   eval-others closed   (per layer open/closed)")
out = {}
for (kn, a, z0, z1), t in tot.items():
    pl = {L: (round(m(v["open"]), 2), round(m(v["closed"]), 3)) for L, v in sorted(t["L"].items())}
    print(f"{kn}{'' if a is None else a!s:5s} ({z0},{z1})   {m(t['open']):.3f}          {m(t['closed']):.4f}        {pl}")
    out[f"{kn}{a}-{z0}-{z1}"] = {"open": m(t["open"]), "closed": m(t["closed"]), "per_layer": {str(k): v for k, v in pl.items()}}
(ROOT / "results" / "suite" / "t2_ravel" / "gate_select.json").write_text(json.dumps(out, indent=1))
