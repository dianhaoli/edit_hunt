"""Stage 1: validity. Score every (entity, template) of each family against the attribute's candidate set.

Rules (pre-specified):
- item correct = argmax value (exact log-prob, max over the value's surfaces) == entity's value;
- template kept = accuracy >= 0.8 over the family's scored entities whose value is in the candidate set;
- attribute kept (strict, decides) = accuracy over all format-valid templates x in-set entities >= 0.8 AND
  >= 2 kept train templates and >= 1 kept held-out template (needed for the Context split);
  "lenient" (reported only) = the template condition alone;
- entity kept = in-set value and correct on > 50% of kept templates, for every kept attribute;
- family survives = >= 60 kept entities and >= 2 kept attributes (one target + one Iso attribute).
Self-labeled attributes (occupation Gender Bias): label = the model's majority value over all its templates.

Output: results/screen/stage1/{family}.json (per-item predictions, per-template accuracy, survivors).
Usage: python -m edithunt.ravel.stage1 [family ...]
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter

import torch

from ..common import ROOT
from .core import Model
from .families import FAMILIES, load_family

OUT = ROOT / "results" / "screen" / "stage1"
TPL_ACC, ENT_FRAC, MIN_ENT = 0.8, 0.5, 60


def flat_cands(M: Model, tpl):
    """surfaces of all values -> token lists, and the value index of each surface."""
    vals = sorted(tpl.verb)
    surf, owner = [], []
    for vi, v in enumerate(vals):
        for s in tpl.verb[v]:
            surf.append(s); owner.append(vi)
    return vals, M.cand_ids(tpl.lead, tuple(surf)), torch.tensor(owner)


def value_scores(sc: torch.Tensor, owner: torch.Tensor, nv: int) -> torch.Tensor:
    out = torch.full((sc.shape[0], nv), -1e9)
    for vi in range(nv):
        out[:, vi] = sc[:, owner == vi].max(1).values
    return out


def predict(M: Model, tpl, ents: list[str], iv_factory=None, bs=64):
    """-> (list of predicted values or None for un-encodable items)."""
    items, keep = [], []
    for i, e in enumerate(ents):
        it = M.encode(tpl.text, e)
        if it is not None:
            items.append(it); keep.append(i)
    vals, cands, owner = flat_cands(M, tpl)
    preds = [None] * len(ents)
    if items:
        sc = M.score(items, cands, iv_factory(keep) if iv_factory else None, bs=bs)
        vs = value_scores(sc, owner, len(vals))
        for j, i in enumerate(keep):
            preds[i] = vals[int(vs[j].argmax())]
    return preds


def run_family(M: Model, name: str) -> dict:
    t0 = time.time()
    fam = load_family(name)
    ents = sorted(fam.ent_attr)
    preds = {}  # attr -> template text -> [pred per entity]
    for a in fam.attrs:
        preds[a] = {}
        for tpl in fam.templates[a]:
            preds[a][tpl.text] = predict(M, tpl, ents)
        print(f"  {name}/{a}: {len(fam.templates[a])} templates scored ({time.time() - t0:.0f}s)", flush=True)
    labels = {e: dict(fam.ent_attr[e]) for e in ents}
    for a in fam.self_labeled:
        for i, e in enumerate(ents):
            c = Counter(p[i] for p in preds[a].values() if p[i] is not None)
            labels[e][a] = c.most_common(1)[0][0] if c else None
    tpl_acc, attr_info, kept_attrs = {}, {}, []
    for a in fam.attrs:
        tpl_acc[a] = {}
        for tpl in fam.templates[a]:
            ok = [preds[a][tpl.text][i] == labels[e][a] for i, e in enumerate(ents)
                  if labels[e][a] is not None and preds[a][tpl.text][i] is not None]
            tpl_acc[a][tpl.text] = (sum(ok) / len(ok) if ok else 0.0, len(ok), tpl.split)
        kept = [t for t in fam.templates[a] if tpl_acc[a][t.text][0] >= TPL_ACC]
        n_tr, n_te = sum(t.split == "train" for t in kept), sum(t.split == "test" for t in kept)
        allacc = [preds[a][t.text][i] == labels[e][a] for t in fam.templates[a] for i, e in enumerate(ents)
                  if labels[e][a] is not None and preds[a][t.text][i] is not None]
        keptacc = [preds[a][t.text][i] == labels[e][a] for t in kept for i, e in enumerate(ents)
                   if labels[e][a] is not None and preds[a][t.text][i] is not None]
        lenient = n_tr >= 2 and n_te >= 1
        ok_attr = lenient and sum(allacc) / max(1, len(allacc)) >= TPL_ACC
        attr_info[a] = dict(n_templates=len(fam.templates[a]), skipped_format=fam.skipped.get(a, 0),
                            kept_train=n_tr, kept_test=n_te, acc_all=sum(allacc) / max(1, len(allacc)),
                            acc_kept=sum(keptacc) / max(1, len(keptacc)), n_values=len(fam.values[a]),
                            in_set=sum(labels[e][a] is not None for e in ents), kept=ok_attr, lenient_kept=lenient,
                            kept_templates=[t.text for t in kept])
        if ok_attr:
            kept_attrs.append(a)
    kept_ents = _kept_entities(ents, labels, preds, attr_info, kept_attrs)
    len_attrs = [a for a in fam.attrs if attr_info[a]["lenient_kept"]]
    len_ents = _kept_entities(ents, labels, preds, attr_info, len_attrs)
    targets = [a for a in kept_attrs if a not in fam.iso_only]
    survives = len(kept_ents) >= MIN_ENT and len(kept_attrs) >= 2 and len(targets) >= 1
    res = dict(family=name, n_entities_scored=len(ents), attrs=attr_info, kept_attrs=kept_attrs,
               kept_entities=kept_ents, n_kept_entities=len(kept_ents),
               kept_split=Counter(fam.ent_split[e] for e in kept_ents), survives=survives,
               lenient=dict(attrs=len_attrs, n_entities=len(len_ents),
                            survives=len(len_ents) >= MIN_ENT and len([a for a in len_attrs if a not in fam.iso_only]) >= 1
                            and len(len_attrs) >= 2),
               labels={e: labels[e] for e in ents}, ent_split=fam.ent_split,
               tpl_acc=tpl_acc, preds=preds, seconds=time.time() - t0)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(res, indent=1))
    return res


def _kept_entities(ents, labels, preds, attr_info, kept_attrs):
    kept_ents = []
    for i, e in enumerate(ents):
        good = True
        for a in kept_attrs:
            if labels[e][a] is None:
                good = False; break
            ps = [preds[a][t][i] for t in attr_info[a]["kept_templates"]]
            ps = [p for p in ps if p is not None]
            if not ps or sum(p == labels[e][a] for p in ps) / len(ps) <= ENT_FRAC:
                good = False; break
        if good:
            kept_ents.append(e)
    return kept_ents


def main(names):
    M = Model()
    for n in names:
        if (OUT / f"{n}.json").exists():
            print(f"{n}: checkpoint exists, skipping", flush=True)
            continue
        r = run_family(M, n)
        print(f"== {n}: survives={r['survives']} kept_attrs={r['kept_attrs']} kept_entities={r['n_kept_entities']} "
              f"{dict(r['kept_split'])} ({r['seconds']:.0f}s) | lenient: {r['lenient']}", flush=True)
        for a, d in r["attrs"].items():
            print(f"   {a}: acc_all={d['acc_all']:.2f} acc_kept={d['acc_kept']:.2f} kept_tpl={d['kept_train']}+"
                  f"{d['kept_test']}/{d['n_templates']} (format-skipped {d['skipped_format']}) in_set={d['in_set']} "
                  f"values={d['n_values']} kept={d['kept']} lenient={d['lenient_kept']}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(FAMILIES))
