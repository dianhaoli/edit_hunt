"""Choose ONE gated-oracle recipe for T2 v2 on DEV data only (no held-out cities, private wordings or Iso pool).

Leave-one-dev-city-out: train on 3 of the 4 public dev cities, evaluate on the 4th:
  cause   held dev city x (fs1, zs1): top capital == target
  state   held dev city, visible STATE_Q: still the source state
  other   accessible other-state cities NOT used in training (keep_pool minus keep_cities) x (fs1, zs1): unchanged
  kl      mean(country binary KL on dev cities where P(US) drops, generic KL on the agent-visible GENERIC_TOOLS)
  proxy = cause * state * other * (1 - min(1, kl))
Recipes: "gated" (150 steps, fs1+zs1) vs "gated_boost" (300 steps, + fs2/zs2). 4 tuning pairs x L{8,12,16,20} x 1 fold.
  PYTHONPATH=. .venv/bin/python experiments/t2_recipe_select.py
"""
import copy
import json
import random
import time

import torch

from edithunt.common import ROOT
from edithunt.data import CAPITALS, COUNTRY_CANDS, COUNTRY_Q, STATE_Q, STATES, TEMPLATES
from edithunt.metrics import binary_kl, kl, p_us
from edithunt.model import Subject
from edithunt.env import ravel as R
from edithunt.env.grader import CAPS, GENERIC_TOOLS, build_ivs

RECIPES = {"gated": dict(), "gated_boost": dict(steps=300, templates=["fs1", "zs1", "fs2", "zs2"])}
out = (ROOT / "results" / "suite" / "t2_ravel" / "recipe_select.jsonl")
done = {(r["pair"], r["layer"], r["recipe"], r["held"]) for r in map(json.loads, out.read_text().splitlines())} \
    if out.exists() else set()
f = out.open("a")
S = Subject("Qwen/Qwen2.5-1.5B")


def top(encs, cands, ivs=()):
    return R._scores(S, encs, cands, ivs).argmax(1).tolist()


for pr in R.candidate_pairs()[:4]:
    P = R.make_pair(S, *pr)
    for L in (8, 12, 16, 20):
        base = R.make_instance(S, P, L)
        for held in random.Random(f"{base.id}-loo").sample(base.dev_source, 2)[:1]:
            inst = copy.deepcopy(base)
            inst.dev_source = [c for c in base.dev_source if c != held]
            used = {c for cs in inst.extra["keep_cities"].values() for c in cs}
            others = [c for cs in inst.extra["keep_pool"].values() for c in cs if c not in used]
            for name, kw in RECIPES.items():
                key = (f"{pr[0]}->{pr[1]}", L, name, held)
                if key in done:
                    continue
                t = time.time()
                kw = dict(kw)
                e = R.train_gated(S, inst, L, 0, kw.pop("steps", 150), **kw)
                ce = [S.encode(TEMPLATES[tk], held) for tk in R.DEV_T]
                cause = [p == CAPS.index(inst.target_capital) for p in top(ce, CAPS, build_ivs(e, ce))]
                se = [S.encode(STATE_Q, held)]
                state = top(se, STATES, build_ivs(e, se))[0] == STATES.index(inst.source)
                oe = [S.encode(TEMPLATES[tk], c) for tk in R.DEV_T for c in others]
                other = [a == b for a, b in zip(top(oe, CAPS), top(oe, CAPS, build_ivs(e, oe)))]
                de = [S.encode(COUNTRY_Q, c) for c in base.dev_source]
                pu0, pu1 = p_us(R._clean_scores(S, de, COUNTRY_CANDS)), p_us(R._scores(S, de, COUNTRY_CANDS, build_ivs(e, de)))
                kc = (binary_kl(pu0, pu1) * (pu1 < pu0).float()).mean().item()
                ge = [S.encode_text(t) for t in GENERIC_TOOLS]
                ref = [ce[i % len(ce)] for i in range(len(ge))]
                kg = kl(S.logp_final(ge), S.logp_final(ge, build_ivs(e, ge, ref))).mean().item()
                km = 0.5 * (kc + kg)
                c_, o_ = sum(cause) / len(cause), (sum(other) / len(other) if other else 1.0)
                r = {"pair": key[0], "layer": L, "recipe": name, "held": held, "cause": c_, "state": float(state),
                     "other": o_, "n_other": len(other), "kl": km,
                     "proxy": c_ * float(state) * o_ * (1 - min(1, km)), "secs": time.time() - t}
                f.write(json.dumps(r) + "\n"); f.flush()
                print(key, " ".join(f"{x}={r[x]:.2f}" for x in ("cause", "state", "other", "kl", "proxy")), flush=True)
