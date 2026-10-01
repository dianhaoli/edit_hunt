"""Shared evaluation kit: survivors from Stage 1, (base, source) pools, interventions, Cause / Iso.

Definitions (RAVEL, Huang et al. 2024):
- site: residual at the entity's last token, layer L (output of block L).
- pool item: base prompt (entity e_b, template t of attribute P) and source prompt (entity e_s != e_b, template
  t_s drawn from all kept templates of the family in the same template split). Base items are only drawn from
  (e_b, t) the clean model answers correctly (Stage 1).
- Cause(A) = fraction of items with P = A and val_A(e_s) != val_A(e_b) whose prediction == val_A(e_s).
- Iso(A) = mean over kept A* != A of the fraction of items with P = A*, val_A(e_s) != val_A(e_b) and
  val_A*(e_s) != val_A*(e_b) (dropped when A* is constant in the family) whose prediction == val_A*(e_b).
- score = Cause x Iso; Disentangle = (Cause + Iso) / 2.
- splits: "entity" = held-out entities x train templates; "context" = train entities x held-out templates.
  Training data for every method = train entities x train templates (never the evaluation pools of "entity").
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass

import torch

from ..common import ROOT
from .core import Item, Iv, Model
from .families import Template, load_family
from .stage1 import OUT as S1, flat_cands, value_scores

SCREEN = ROOT / "results" / "screen"


@dataclass
class Fam:
    name: str
    attrs: list            # kept attributes
    targets: list          # kept attributes usable as targets
    labels: dict           # entity -> {attr: value}
    ents: dict             # split ("train"/"test") -> kept entities
    tpls: dict             # (attr, split) -> [Template] kept
    correct: dict          # (attr, template text) -> set of entities answered correctly by the clean model
    n_values: dict         # attr -> number of distinct values among kept entities


def load_survivor(name: str) -> Fam:
    r = json.loads((S1 / f"{name}.json").read_text())
    fam = load_family(name)  # same seed -> same templates/verbalizers
    attrs = r["kept_attrs"]
    kept = set(r["kept_entities"])
    labels = {e: r["labels"][e] for e in kept}
    ents = {sp: sorted(e for e in kept if r["ent_split"][e] == sp) for sp in ("train", "test")}
    tpls, correct = {}, {}
    for a in attrs:
        keep_txt = set(r["attrs"][a]["kept_templates"])
        for t in fam.templates[a]:
            if t.text in keep_txt:
                tpls.setdefault((a, t.split), []).append(t)
                ent_all = sorted(r["labels"])
                pr = r["preds"][a][t.text]
                correct[(a, t.text)] = {e for i, e in enumerate(ent_all) if e in kept and pr[i] == labels[e][a]}
    n_values = {a: len({labels[e][a] for e in kept}) for a in attrs}
    targets = [a for a in attrs if a not in fam.iso_only and n_values[a] > 1]
    return Fam(name, attrs, targets, labels, ents, tpls, correct, n_values)


def split_parts(split: str):
    """(entity split, template split) for an evaluation split / "train"."""
    return {"entity": ("test", "train"), "context": ("train", "test"), "train": ("train", "train")}[split]


@dataclass
class PItem:
    attr: str          # attribute the base prompt asks
    tpl: Template
    eb: str
    es: str
    ts: Template       # source template


def build_pool(F: Fam, split: str, n_per_attr: int, seed: int) -> list[PItem]:
    es_split, ts_split = split_parts(split)
    rng = random.Random(seed)
    ents = F.ents[es_split]
    src_tpls = [t for a in F.attrs for t in F.tpls.get((a, ts_split), [])]
    pool = []
    for a in F.attrs:
        tl = F.tpls.get((a, ts_split), [])
        combos = [(e, t) for t in tl for e in ents if e in F.correct[(a, t.text)]]
        if not combos or len(ents) < 2:
            continue
        for _ in range(n_per_attr):
            eb, t = rng.choice(combos)
            es = rng.choice(ents)
            while es == eb:
                es = rng.choice(ents)
            pool.append(PItem(a, t, eb, es, rng.choice(src_tpls)))
    return pool


def source_reps(M: Model, pool: list[PItem], layers: list[int]) -> dict:
    """{L: [N,d]} source residuals at the source entity's last token."""
    items = [M.encode(p.ts.text, p.es) for p in pool]
    assert all(it is not None for it in items)
    return M.capture(items, layers)


def base_items(M: Model, pool):
    return [M.encode(p.tpl.text, p.eb) for p in pool]


def predict_pool(M: Model, pool: list[PItem], make_iv, bs: int = 64) -> list[str]:
    """make_iv(idx: list[int]) -> Iv | None for those pool rows. Rows are grouped by template (shared candidates)."""
    by_t: dict = {}
    for i, p in enumerate(pool):
        by_t.setdefault(p.tpl.text, []).append(i)
    preds = [None] * len(pool)
    for txt, idx in by_t.items():
        tpl = pool[idx[0]].tpl
        items = [M.encode(tpl.text, pool[i].eb) for i in idx]
        vals, cands, owner = flat_cands(M, tpl)
        sc = M.score(items, cands, make_iv(idx), bs=bs)
        vs = value_scores(sc, owner, len(vals))
        for j, i in enumerate(idx):
            preds[i] = vals[int(vs[j].argmax())]
    return preds


def metrics(F: Fam, pool: list[PItem], preds: list[str], target: str) -> dict:
    L = F.labels
    cause = [preds[i] == L[p.es][target] for i, p in enumerate(pool)
             if p.attr == target and L[p.es][target] != L[p.eb][target]]
    iso_by = {}
    for a2 in F.attrs:
        if a2 == target:
            continue
        ok = [preds[i] == L[p.eb][a2] for i, p in enumerate(pool)
              if p.attr == a2 and L[p.es][target] != L[p.eb][target]
              and (F.n_values[a2] == 1 or L[p.es][a2] != L[p.eb][a2])]
        if ok:
            iso_by[a2] = (sum(ok) / len(ok), len(ok))
    c = sum(cause) / len(cause) if cause else float("nan")
    iso = sum(v for v, _ in iso_by.values()) / len(iso_by) if iso_by else float("nan")
    return dict(cause=c, n_cause=len(cause), iso=iso, iso_by=iso_by, score=c * iso, disentangle=(c + iso) / 2)


def all_metrics(F: Fam, pool, preds) -> dict:
    return {a: metrics(F, pool, preds, a) for a in F.targets}


# ------------------------------------------------------------------ training data helpers
def train_reps(M: Model, F: Fam, layers: list[int], max_items: int = 3000, seed: int = 0):
    """Residuals of train entities x train templates (all kept attrs): (entities list, {L: [N,d]})."""
    rng = random.Random(seed)
    combos = [(e, t) for a in F.attrs for t in F.tpls.get((a, "train"), []) for e in F.ents["train"]]
    rng.shuffle(combos)
    combos = combos[:max_items]
    items = [M.encode(t.text, e) for e, t in combos]
    return [e for e, _ in combos], M.capture(items, layers)


def mean_by_value(F: Fam, ents: list[str], reps: torch.Tensor, attr: str) -> dict:
    out = {}
    for v in {F.labels[e][attr] for e in ents}:
        m = torch.tensor([F.labels[e][attr] == v for e in ents])
        out[v] = reps[m].mean(0)
    return out
