"""Lever A test (AUDIT_T2V2 §8.3 A): grade "keep" also on PRIVATE downstream readouts of the kept country (country
capital, currency; edithunt/ravel/hidden_readouts.py) that the agent never sees.

Per Language instance, cross-fitted over the 30 private bases (halves as env_ceiling):
  reference   saved MDAS-via-tools subspace (visible keep templates only)
  ceiling_vis saved privileged ceiling (env_ceiling; keep examples on visible + held-out templates)
  ceiling_hid privileged ceiling that ALSO gets keep examples on the hidden readouts (privileged cities only)
  full, das   lazy scripts (das: DAS k16 via recipes, trained here, saved)
Hidden Iso = share of hidden-readout prompts (base & source differ in that readout's value; clean answer correct) that
still answer the base's value. Score_hidden = Cause x mean(Iso over visible keep attrs and hidden readouts) x KL factor.
Usage: python -m edithunt.ravel.env_hidden Language_s0 ...  -> results/screen/env/hidden.json
"""
from __future__ import annotations

import json
import random
import sys

import torch

from ..common import ROOT
from . import env as E
from .core import Model, iv_interchange, iv_set
from .env_ceiling import SUBS, train_sub
from .env_recipe_check import recipes
from .families import Template
from .hidden_readouts import READOUTS, verbalizer
from .stage1 import flat_cands, value_scores

OUT = ROOT / "results" / "screen" / "env" / "hidden.json"
HT = {n: Template(t, n, "hidden", verbalizer(n)) for n, (t, _, _) in READOUTS.items()}


def hidden_val(name, city, F):
    return READOUTS[name][1][F.labels[city]["Country"]]


def hidden_iso(M, F, inst, sub, bases):
    """per readout: (share kept, n) over grade pairs with base in `bases`."""
    L = inst["layer"]
    pairs = [p for p in inst["grade_pairs"] if p[0] in bases]
    rng = random.Random(1)
    src_t = [t for a in F.attrs for t in F.tpls.get((a, "train"), [])]
    out = {}
    for name, tp in HT.items():
        vals, cands, owner = flat_cands(M, tp)
        rows = [(b, s) for b, s in pairs if hidden_val(name, b, F) != hidden_val(name, s, F)]
        if not rows:
            continue
        items = [M.encode(tp.text, b) for b, _ in rows]
        clean = value_scores(M.score(items, cands), owner, len(vals))
        ok_idx = [j for j, (b, _) in enumerate(rows) if vals[int(clean[j].argmax())] == hidden_val(name, b, F)]
        if not ok_idx:
            continue
        rows = [rows[j] for j in ok_idx]
        items = [items[j] for j in ok_idx]
        src = M.capture([M.encode(rng.choice(src_t).text, s) for _, s in rows], [L])[L]
        iv = iv_interchange(L, sub["U"], src) if sub["U"] is not None else iv_set(L, src)
        ed = value_scores(M.score(items, cands, iv), owner, len(vals))
        kept = [vals[int(ed[j].argmax())] == hidden_val(name, b, F) for j, (b, _) in enumerate(rows)]
        out[name] = (sum(kept) / len(kept), len(kept))
    return out


def train_hidden_ceiling(M, inst, priv, F, seed=0):
    """privileged ceiling + explicit keep examples on the hidden readouts (privileged cities only)."""
    pinst = dict(inst, dev=sorted(priv), refused=sorted(set(inst["refused"]) - set(priv)))
    env = E.CityEnv(M, pinst)
    for a in env.F.attrs:
        for i, t in enumerate(env.F.tpls.get((a, "test"), [])):
            env.tid[f"h{a[:4].lower()}{i}"] = t
            env.public_t[a] = env.public_t[a] + [t]
    env.heldout_t = []
    for n, t in HT.items():
        env.tid[f"x{n}"] = t
    rng = random.Random(seed)
    ex, pl = [], sorted(priv)
    while len(ex) < 1500:
        b, s = rng.sample(pl, 2)
        if F.labels[b]["Language"] == F.labels[s]["Language"]:
            continue
        n = rng.choice(list(HT))
        ex.append(dict(base=b, source=s, template=f"x{n}", answer=hidden_val(n, b, F)))
    E.BUDGET.update(max_examples=6000, max_steps=900)
    try:
        out = json.loads(env.call("train_subspace", {"layer": inst["layer"], "k": 256, "steps": 900, "seed": seed,
                                                     "examples": ex, "recipes": recipes(env, 2250, 2250, True)}))
    finally:
        E.BUDGET.update(max_examples=3000, max_steps=600)
    if "error" in out:
        raise RuntimeError(out)
    L, U = env.subspaces[out["featurizer"]]
    return dict(featurizer=None, layer=L, U=U, sae_ids=None)


def combined(g, hid, keep):
    isos = [g["iso_by"][a] for a in keep] + [v for v, _ in hid.values()]
    iso = sum(isos) / len(isos)
    return dict(score_visible=g["score"], score_hidden=g["cause"] * iso * g["kl_factor"], cause=g["cause"],
                iso_visible=g["iso"], iso_hidden={k: v for k, v in hid.items()})


if __name__ == "__main__":
    M = Model()
    E.KL_BUDGET = json.loads((ROOT / "results" / "screen" / "env" / "validate.json").read_text())["kl_budget"]
    F = E.load_survivor("city_pop")
    R = json.loads(OUT.read_text()) if OUT.exists() else {}
    for iid in sys.argv[1:]:
        inst = E.load_instance(iid)
        bases = sorted({b for b, _ in inst["grade_pairs"]})
        halves = [set(bases[0::2]), set(bases[1::2])]
        das_path = SUBS / f"{iid}_das_k16.pt"
        if not das_path.exists():
            env = E.CityEnv(M, inst)
            out = json.loads(env.call("train_subspace", {"layer": inst["layer"], "k": 16, "steps": 600,
                                                         "recipes": recipes(env, 3000, 0, False)}))
            torch.save(env.subspaces[out["featurizer"]][1], das_path)
        subs = {"reference": torch.load(SUBS / f"{iid}_reference_s0.pt"), "das_k16": torch.load(das_path), "full": None}
        for f, h in enumerate(halves):
            key = f"{iid}|f{f}"
            if key in R:
                continue
            res = {}
            cands = dict(subs)
            cands["ceiling_vis"] = torch.load(SUBS / f"{iid}_ceiling_f{f}.pt")
            graded = {c for b, s in inst["grade_pairs"] if b in h for c in (b, s)}
            priv = (set(inst["dev"]) | set(F.ents["test"])) - graded
            hc = train_hidden_ceiling(M, inst, priv, F)
            torch.save(hc["U"], SUBS / f"{iid}_ceiling_hidden_f{f}.pt")
            cands["ceiling_hid"] = hc["U"]
            for name, U in cands.items():
                sub = dict(featurizer=None if U is not None else "full", layer=inst["layer"], U=U, sae_ids=None)
                g = E.grade(M, inst, sub, bases=h)
                res[name] = combined(g, hidden_iso(M, F, inst, sub, h), inst["keep"])
            R[key] = res
            OUT.write_text(json.dumps(R, indent=1))
            print(f"{key}: " + " | ".join(f"{n} vis {r['score_visible']:.2f} hid {r['score_hidden']:.2f} "
                                          f"(hidden iso {', '.join(f'{k} {v[0]:.2f}' for k, v in r['iso_hidden'].items())})"
                                          for n, r in res.items()), flush=True)
