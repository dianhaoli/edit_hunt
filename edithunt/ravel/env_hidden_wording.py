"""Is the hidden-readout headroom reachable without the grader's exact wording? Privileged ceiling (same cities,
volume and recipes as ceiling_hid) whose hidden-fact keep examples use the AGENT-STYLE wording of env_hidden_script
instead of the grader's hidden templates. If it stays near the script, lever A's gap is wording-specific.
Usage: python -m edithunt.ravel.env_hidden_wording Language_s0 -> results/screen/env/hidden_wording.json"""
from __future__ import annotations

import json
import random
import sys

from ..common import ROOT
from . import env as E
from .core import Model
from .env_hidden import combined, hidden_iso
from .env_hidden_script import AGENT_T
from .env_recipe_check import recipes

OUT = ROOT / "results" / "screen" / "env" / "hidden_wording.json"

if __name__ == "__main__":
    M = Model()
    E.KL_BUDGET = json.loads((ROOT / "results" / "screen" / "env" / "validate.json").read_text())["kl_budget"]
    F = E.load_survivor("city_pop")
    R = json.loads(OUT.read_text()) if OUT.exists() else {}
    for iid in sys.argv[1:]:
        inst = E.load_instance(iid)
        bases = sorted({b for b, _ in inst["grade_pairs"]})
        for f, h in enumerate([set(bases[0::2]), set(bases[1::2])]):
            key = f"{iid}|f{f}"
            if key in R:
                continue
            graded = {c for b, s in inst["grade_pairs"] if b in h for c in (b, s)}
            priv = sorted((set(inst["dev"]) | set(F.ents["test"])) - graded)
            env = E.CityEnv(M, dict(inst, dev=priv, refused=sorted(set(inst["refused"]) - set(priv))))
            for a in env.F.attrs:
                for i, t in enumerate(env.F.tpls.get((a, "test"), [])):
                    env.tid[f"h{a[:4].lower()}{i}"] = t
                    env.public_t[a] = env.public_t[a] + [t]
            env.heldout_t = []
            rng = random.Random(0)
            ex = []
            while len(ex) < 1500:
                b, s = rng.sample(priv, 2)
                w = rng.choice(list(AGENT_T))
                tpl, table = AGENT_T[w]
                if F.labels[b]["Language"] == F.labels[s]["Language"]:
                    continue
                ex.append(dict(base=b, source=s, template=tpl, answer=table[F.labels[b]["Country"]]))
            E.BUDGET.update(max_examples=6000, max_steps=900)
            out = json.loads(env.call("train_subspace", {"layer": inst["layer"], "k": 256, "steps": 900, "examples": ex,
                                                         "recipes": recipes(env, 2250, 2250, True)}))
            E.BUDGET.update(max_examples=3000, max_steps=600)
            L, U = env.subspaces[out["featurizer"]]
            sub = dict(featurizer=None, layer=L, U=U, sae_ids=None)
            R[key] = combined(E.grade(M, inst, sub, bases=h), hidden_iso(M, F, inst, sub, h), inst["keep"])
            OUT.write_text(json.dumps(R, indent=1))
            r = R[key]
            print(f"{key}: privileged agent-wording ceiling vis {r['score_visible']:.2f} hid {r['score_hidden']:.2f} "
                  f"({', '.join(f'{k} {v[0]:.2f}' for k, v in r['iso_hidden'].items())})", flush=True)
