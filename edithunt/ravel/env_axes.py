"""Instance axis where the recipe should break: multilingual countries (Language task).

Language_multi_s{n}: the Language_s{n} instance plus private same-country/different-language pairs (India, Canada,
Nigeria), e.g. Delhi -> Kolkata (Hindi -> Bengali, same country). On those pairs only Cause is defined (country and
continent are equal), so score = Cause(all pairs) x Iso(standard pairs) x KL factor, as in env.grade.
Cross-fitted like env_ceiling: the multilingual bases are split in halves; for fold f
  script   = the reference subspace saved by env_ceiling (dev data only) graded on fold f
  ceiling  = privileged: dev + private cities outside fold f's pairs, held-out templates, recipes as the ceiling, PLUS
             explicit same-country examples among the privileged cities (answer = source's language)
  full     = full-vector interchange (lazy)
Reported separately on the multilingual pairs (Cause) and on the combined grade.
Usage: python -m edithunt.ravel.env_axes Language_s0 ...  -> results/screen/env/axes.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch

from ..common import ROOT
from . import env as E
from .core import Model
from .env_ceiling import SUBS
from .env_recipe_check import recipes

OUT = ROOT / "results" / "screen" / "env" / "axes.json"


def multi_pairs(F, cities):
    return [(b, s) for b in cities for s in cities if b != s and F.labels[b]["Country"] == F.labels[s]["Country"]
            and F.labels[b]["Language"] != F.labels[s]["Language"]]


if __name__ == "__main__":
    M = Model()
    E.KL_BUDGET = json.loads((ROOT / "results" / "screen" / "env" / "validate.json").read_text())["kl_budget"]
    R = json.loads(OUT.read_text()) if OUT.exists() else {}
    F = E.load_survivor("city_pop")
    for iid in sys.argv[1:]:
        inst = E.load_instance(iid)
        mp = multi_pairs(F, F.ents["test"])
        mb = sorted({b for b, _ in mp})
        halves = [set(mb[0::2]), set(mb[1::2])]
        rng = random.Random(0)
        U_ref = torch.load(SUBS / f"{iid}_reference_s0.pt")
        for f, h in enumerate(halves):
            key = f"{iid}|multi|f{f}"
            if key in R:
                continue
            fold_pairs = [p for p in mp if p[0] in h]
            fold_pairs = rng.sample(fold_pairs, min(40, len(fold_pairs)))
            minst = dict(inst, grade_pairs=[list(p) for p in fold_pairs] + inst["grade_pairs"][:30])
            gb = {b for b, _ in fold_pairs} | {b for b, _ in inst["grade_pairs"][:30]}
            res = {}
            for name, sub in (("script", dict(featurizer=None, layer=inst["layer"], U=U_ref, sae_ids=None)),
                              ("full", dict(featurizer="full", layer=inst["layer"], U=None, sae_ids=None))):
                g = E.grade(M, minst, sub, items_out=True)
                res[name] = dict(combined=E.metrics_from_items(g["items"], inst["keep"], g["kl_factor"]),
                                 multi_cause=E.metrics_from_items(g["items"], inst["keep"], 1.0,
                                                                  bases={b for b, _ in fold_pairs})["cause"])
            # privileged ceiling with explicit same-country examples
            graded = {c for p in minst["grade_pairs"] for c in p}
            priv = sorted((set(inst["dev"]) | set(F.ents["test"])) - graded)
            pinst = dict(inst, dev=priv, refused=sorted(set(inst["refused"]) - set(priv)))
            env = E.CityEnv(M, pinst)
            for a in env.F.attrs:
                for i, t in enumerate(env.F.tpls.get((a, "test"), [])):
                    env.tid[f"h{a[:4].lower()}{i}"] = t
                    env.public_t[a] = env.public_t[a] + [t]
            env.heldout_t = []
            lang_t = [k for k, t in env.tid.items() if t.attr == "Language"]
            ex = [dict(base=b, source=s, template=lang_t[j % len(lang_t)], answer=F.labels[s]["Language"])
                  for j, (b, s) in enumerate(multi_pairs(F, priv)) for _ in range(3)]
            E.BUDGET.update(max_examples=6000, max_steps=900)
            out = json.loads(env.call("train_subspace", {"layer": inst["layer"], "k": 256, "steps": 900, "examples": ex[:1500],
                                                         "recipes": recipes(env, 2250, 2250, True)}))
            E.BUDGET.update(max_examples=3000, max_steps=600)
            L, U = env.subspaces[out["featurizer"]]
            torch.save(U, SUBS / f"{iid}_multiceiling_f{f}.pt")
            g = E.grade(M, minst, dict(featurizer=None, layer=L, U=U, sae_ids=None), items_out=True)
            res["ceiling"] = dict(combined=E.metrics_from_items(g["items"], inst["keep"], g["kl_factor"]),
                                  multi_cause=E.metrics_from_items(g["items"], inst["keep"], 1.0,
                                                                   bases={b for b, _ in fold_pairs})["cause"],
                                  n_same_country_examples=min(1500, len(ex)))
            R[key] = res
            OUT.write_text(json.dumps(R, indent=1))
            print(f"{key}: " + " | ".join(f"{n}: combined {r['combined']['score']:.2f}, same-country Cause {r['multi_cause']:.2f}"
                                          for n, r in res.items()), flush=True)
