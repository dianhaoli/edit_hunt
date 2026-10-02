"""Can a script recover the hidden-readout gap by guessing downstream facts? (script library for lever A)
Through the agent's tools only (dev cities): MDAS recipes + explicit keep examples on AGENT-WRITTEN prompts (worded
differently from the grader's hidden templates) for country capital and/or currency, answers from world knowledge.
  mdas+cap      keep examples on capital only  (currency stays unprotected)
  mdas+cur      keep examples on currency only (capital unprotected)
  mdas+cap+cur  both
Graded like env_hidden (per fold), compared with reference and ceiling_hid from hidden.json.
Usage: python -m edithunt.ravel.env_hidden_script Language_s0 ...  -> results/screen/env/hidden_script.json
"""
from __future__ import annotations

import json
import random
import sys

import torch

from ..common import ROOT
from . import env as E
from .core import Model
from .env_ceiling import SUBS
from .env_hidden import combined, hidden_iso
from .env_recipe_check import recipes
from .hidden_readouts import CAPITAL, CURRENCY

OUT = ROOT / "results" / "screen" / "env" / "hidden_script.json"
AGENT_T = {"cap": ("Question: What is the capital of the country where %s is located?\nAnswer:", CAPITAL),
           "cur": ("The official currency used in %s is the", CURRENCY)}


def run(M, inst, which, seed=0):
    env = E.CityEnv(M, inst)
    F = env.F
    rng = random.Random(seed)
    ex, dev = [], inst["dev"]
    per = 750 // len(which)
    for w in which:
        tpl, table = AGENT_T[w]
        n = 0
        while n < per:
            b, s = rng.sample(dev, 2)
            if F.labels[b]["Language"] == F.labels[s]["Language"] or table[F.labels[b]["Country"]] == table[F.labels[s]["Country"]]:
                continue
            ex.append(dict(base=b, source=s, template=tpl, answer=table[F.labels[b]["Country"]]))
            n += 1
    out = json.loads(env.call("train_subspace", {"layer": inst["layer"], "k": 256, "steps": 600, "seed": seed,
                                                 "examples": ex, "recipes": recipes(env, 1500, 750, True)}))
    if "error" in out:
        raise RuntimeError(out)
    L, U = env.subspaces[out["featurizer"]]
    return dict(featurizer=None, layer=L, U=U, sae_ids=None)


if __name__ == "__main__":
    M = Model()
    E.KL_BUDGET = json.loads((ROOT / "results" / "screen" / "env" / "validate.json").read_text())["kl_budget"]
    F = E.load_survivor("city_pop")
    R = json.loads(OUT.read_text()) if OUT.exists() else {}
    for iid in sys.argv[1:]:
        inst = E.load_instance(iid)
        bases = sorted({b for b, _ in inst["grade_pairs"]})
        halves = [set(bases[0::2]), set(bases[1::2])]
        for name, which in (("mdas+cap", ["cap"]), ("mdas+cur", ["cur"]), ("mdas+cap+cur", ["cap", "cur"])):
            key = f"{iid}|{name}"
            if key in R:
                continue
            sub = run(M, inst, which)
            torch.save(sub["U"], SUBS / f"{iid}_{name.replace('+', '_')}.pt")
            R[key] = [combined(E.grade(M, inst, sub, bases=h), hidden_iso(M, F, inst, sub, h), inst["keep"]) for h in halves]
            OUT.write_text(json.dumps(R, indent=1))
            print(f"{key}: " + " | ".join(f"f{f} vis {r['score_visible']:.2f} hid {r['score_hidden']:.2f} "
                                          f"({', '.join(f'{k} {v[0]:.2f}' for k, v in r['iso_hidden'].items())})"
                                          for f, r in enumerate(R[key])), flush=True)
