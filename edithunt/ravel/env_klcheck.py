"""KL guard check (user item 4): do damaging edits land just under the KL hinge (3.5) and pass? Edits an agent can make
through the tools, graded on Language_s0: random k=256 subspace; subspaces trained toward nonsense answers (a fixed
junk token / random cities' wrong values) on all attributes; full interchange; reference for comparison."""
from __future__ import annotations

import json
import random

import torch

from ..common import ROOT
from . import env as E
from .core import Model
from .env_recipe_check import recipes

OUT = ROOT / "results" / "screen" / "env" / "klcheck.json"

if __name__ == "__main__":
    M = Model()
    E.KL_BUDGET = json.loads((ROOT / "results" / "screen" / "env" / "validate.json").read_text())["kl_budget"]
    inst = E.load_instance("Language_s0")
    env = E.CityEnv(M, inst)
    F = env.F
    rng = random.Random(0)
    L = inst["layer"]
    cands = {}
    cands["random_k256"] = torch.linalg.qr(torch.randn(M.d, 256, generator=torch.Generator().manual_seed(0)))[0]
    tids = list(env.tid)
    for name, ans in (("junk_token", lambda b, s, a: "banana"), ("wrong_values", lambda b, s, a: rng.choice(
            [v for v in {F.labels[e][a] for e in F.labels} if v != F.labels[b][a]]))):
        ex = []
        for _ in range(2000):
            b, s = rng.sample(inst["dev"], 2)
            t = rng.choice(tids)
            ex.append(dict(base=b, source=s, template=t, answer=ans(b, s, env.tid[t].attr)))
        out = json.loads(env.call("train_subspace", {"layer": L, "k": 256, "steps": 300, "examples": ex}))
        cands[name] = env.subspaces[out["featurizer"]][1]
    R = {}
    for name, U in list(cands.items()) + [("full", None)]:
        sub = dict(featurizer=None if U is not None else "full", layer=L, U=U, sae_ids=None)
        g = E.grade(M, inst, sub)
        R[name] = {k: g[k] for k in ("score", "cause", "iso", "kl", "kl_factor")}
        print(name, {k: round(v, 3) for k, v in R[name].items()}, flush=True)
    OUT.write_text(json.dumps(R, indent=1))
