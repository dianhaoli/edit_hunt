"""Dev-only check of a multi-class probe gate (no training of v). LOO over dev cities, 4 TUNING pairs x L{8,..,20}.
Probe: multinomial logistic regression over states on residuals at the edit site (city_last after L), trained on
the 3 remaining dev source cities + FIT half of accessible other-state cities (fs1, zs1), L2 weight lam.
Gate margin = logit(source) - max other logit; gate = clamp((margin - m0)/(m1 - m0), 0, 1).
Score: held dev city prompts fully open; EVAL-half other-state prompts fully closed."""
import json
import random
import torch
from edithunt.common import ROOT
from edithunt.data import CITIES, TEMPLATES, study_states, CITY2STATE
from edithunt.model import Subject
from edithunt.env import ravel as R


def fit_probe(X, y, C, lam, steps=300):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mu) / sd
    W = torch.zeros(C, X.shape[1], requires_grad=True); b = torch.zeros(C, requires_grad=True)
    cw = torch.bincount(y, minlength=C).float().clamp(min=1); cw = (1 / cw) / (1 / cw).mean()  # class-balanced
    opt = torch.optim.LBFGS([W, b], max_iter=steps, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        l = torch.nn.functional.cross_entropy(Xn @ W.T + b, y, weight=cw) + lam * (W ** 2).sum()
        l.backward(); return l
    opt.step(closure)
    return lambda Z: ((Z - mu) / sd) @ W.detach().T + b.detach()


S = Subject("Qwen/Qwen2.5-1.5B")
LAMS = [1e-3, 1e-2]
MS = [(0, 2), (1, 3), (2, 4)]
tot = {}
for pr in R.candidate_pairs()[:4]:
    P = R.make_pair(S, *pr)
    priv = set(P["iso_pool"]) | set(P["decoys"])
    others = [c for s in study_states(min_cities=5) if s not in pr for c in CITIES[s] if c not in priv]
    random.Random(f"{pr}-split").shuffle(others)
    fit, ev = others[: len(others) // 2], others[len(others) // 2:]
    states = sorted({CITY2STATE[c] for c in fit} | {pr[0]})
    si = states.index(pr[0])
    for L in (8, 12, 16, 20):
        inst = R.make_instance(S, P, L)
        H = lambda cs: S.resid([S.encode(TEMPLATES[t], c) for t in R.DEV_T for c in cs], [L], "city_last")[L].float()
        Hf, He = H(fit), H(ev)
        yf = torch.tensor([states.index(CITY2STATE[c]) for t in R.DEV_T for c in fit])
        for held in inst.dev_source:
            tr = [c for c in inst.dev_source if c != held]
            Hs, Hh = H(tr), H([held])
            X = torch.cat([Hf, Hs]); y = torch.cat([yf, torch.full((len(Hs),), si)])
            for lam in LAMS:
                f = fit_probe(X, y, len(states), lam)
                def marg(Z):
                    lg = f(Z); o = lg.clone(); o[:, si] = -1e9
                    return lg[:, si] - o.max(1).values
                mh, me = marg(Hh), marg(He)
                for m0, m1 in MS:
                    go = ((mh - m0) / (m1 - m0)).clamp(0, 1); gc = ((me - m0) / (m1 - m0)).clamp(0, 1)
                    t = tot.setdefault((lam, m0, m1), {"open": [], "closed": [], "L": {}})
                    t["open"] += (go >= 0.99).float().tolist(); t["closed"] += (gc <= 0.01).float().tolist()
                    tl = t["L"].setdefault(L, {"open": [], "closed": []})
                    tl["open"] += (go >= 0.99).float().tolist(); tl["closed"] += (gc <= 0.01).float().tolist()
        print(pr, L, flush=True)
m = lambda x: sum(x) / len(x)
out = {}
print("lam     margin-ramp  held-dev open  eval-others closed  per layer")
for (lam, m0, m1), t in tot.items():
    pl = {L: (round(m(v["open"]), 2), round(m(v["closed"]), 3)) for L, v in sorted(t["L"].items())}
    print(f"{lam:<7} ({m0},{m1})        {m(t['open']):.3f}          {m(t['closed']):.4f}        {pl}")
    out[f"{lam}-{m0}-{m1}"] = {"open": m(t["open"]), "closed": m(t["closed"]), "per_layer": {str(k): v for k, v in pl.items()}}
(ROOT / "results" / "suite" / "t2_ravel" / "probe_gate_select.json").write_text(json.dumps(out, indent=1))
