"""Cross-fitted privileged ceiling vs the reference script (user request 2026-10-02).

Per instance: the 30 private base cities are split into halves H0 / H1 (by sorted order, alternate). For fold f:
  ceiling_f: MDAS recipe trained with PRIVILEGED data: dev cities + every private city that does not appear in fold
             f's graded pairs (bases or sources), and the held-out templates as well as the public ones; bigger budget
             (6000 examples, 900 steps, k 256). Graded on fold f's pairs only.
  reference: the MDAS-via-tools script (dev data, public templates, 3000 examples, 600 steps, k 256; seed 0),
             trained once, graded on each fold.
Report reference / ceiling per fold. Every trained U is saved (~/screen_subspaces/env/) so runs can be regraded, and
per-item records are stored for noise / soft-metric analysis.
Usage: python -m edithunt.ravel.env_ceiling [instance ids]   -> results/screen/env/ceiling.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from ..common import ROOT
from . import env as E
from .core import Model
from .env_recipe_check import recipes

OUT = ROOT / "results" / "screen" / "env" / "ceiling.json"
SUBS = Path.home() / "screen_subspaces" / "env"


def train_sub(M, inst, n_t, n_k, steps, seed, privileged_cities=None, heldout_templates=False, max_examples=3000):
    inst = dict(inst)
    if privileged_cities is not None:
        inst["dev"] = sorted(privileged_cities)
        inst["refused"] = sorted(set(inst["refused"]) - set(privileged_cities))
    env = E.CityEnv(M, inst)
    if heldout_templates:
        for a in env.F.attrs:
            for i, t in enumerate(env.F.tpls.get((a, "test"), [])):
                env.tid[f"h{a[:4].lower()}{i}"] = t
                env.public_t[a] = env.public_t[a] + [t]
        env.heldout_t = []
    old = E.BUDGET["max_examples"]
    E.BUDGET.update(max_examples=max_examples, max_steps=max(steps, E.BUDGET["max_steps"]))
    try:
        out = json.loads(env.call("train_subspace", {"layer": inst["layer"], "k": 256, "steps": steps, "seed": seed,
                                                     "recipes": recipes(env, n_t, n_k, True)}))
    finally:
        E.BUDGET.update(max_examples=old, max_steps=600)
    if "error" in out:
        raise RuntimeError(out)
    L, U = env.subspaces[out["featurizer"]]
    return dict(featurizer=None, layer=L, U=U, sae_ids=None), out


if __name__ == "__main__":
    M = Model()
    E.KL_BUDGET = json.loads((ROOT / "results" / "screen" / "env" / "validate.json").read_text())["kl_budget"]
    SUBS.mkdir(parents=True, exist_ok=True)
    R = json.loads(OUT.read_text()) if OUT.exists() else {}
    save = lambda: OUT.write_text(json.dumps(R))
    ids = sys.argv[1:] or [f"{t}_s{s}" for t in E.TASKS for s in range(3)]
    for iid in ids:
        inst = E.load_instance(iid)
        bases = sorted({b for b, _ in inst["grade_pairs"]})
        halves = [set(bases[0::2]), set(bases[1::2])]
        key = f"{iid}|reference"
        if key not in R:
            sub, tr = train_sub(M, inst, 1500, 1500, 600, 0)
            torch.save(sub["U"], SUBS / f"{iid}_reference_s0.pt")
            R[key] = dict(train=tr, folds=[E.grade(M, inst, sub, bases=h, items_out=True) for h in halves],
                          full=E.grade(M, inst, sub, items_out=True))
            save()
        for f, h in enumerate(halves):
            key = f"{iid}|ceiling|f{f}"
            if key in R:
                continue
            graded = {c for b, s in inst["grade_pairs"] if b in h for c in (b, s)}
            priv = (set(inst["dev"]) | set(E.load_survivor("city_pop").ents["test"])) - graded
            sub, tr = train_sub(M, inst, 3000, 3000, 900, 0, privileged_cities=priv, heldout_templates=True,
                                max_examples=6000)
            torch.save(sub["U"], SUBS / f"{iid}_ceiling_f{f}.pt")
            R[key] = dict(train=tr, n_privileged_cities=len(priv), grade=E.grade(M, inst, sub, bases=h, items_out=True))
            save()
            ref = R[f"{iid}|reference"]["folds"][f]
            c = R[key]["grade"]
            print(f"{iid} fold {f}: reference {ref['score']:.2f} (soft {ref['soft_score']:.2f}) | ceiling {c['score']:.2f} "
                  f"(soft {c['soft_score']:.2f}; C {c['cause']:.2f} I {c['iso']:.2f}) | ratio {ref['score'] / max(1e-9, c['score']):.2f}",
                  flush=True)
