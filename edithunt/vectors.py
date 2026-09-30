"""Construct steering vectors from cached residuals."""
from __future__ import annotations

import torch

from .data import CITIES, TEMPLATES
from .instances import split
from .model import Subject


def meandiff(S: Subject, V: dict, src: str, tgt: str, layers: list[int], seed: int,
             template: str = "fs1", which: str = "city_last", n_dev: int | None = None,
             templates: list[str] | None = None):
    """v_L = mean(tgt dev resid) - mean(src dev resid). If `templates` is given, averages the
    per-template vectors (each over cities valid for that template). Returns (vecs, splits)."""
    dev_s, test_s = split(V, src, seed, template, n_dev)
    dev_t, test_t = split(V, tgt, seed, template, n_dev)
    vecs = {L: torch.zeros(S.d_model) for L in layers}
    tl = templates or [template]
    for tn in tl:
        encs_s = [S.encode(TEMPLATES[tn], c) for c in dev_s if V[tn][c]] or [S.encode(TEMPLATES[tn], c) for c in dev_s]
        encs_t = [S.encode(TEMPLATES[tn], c) for c in dev_t if V[tn][c]] or [S.encode(TEMPLATES[tn], c) for c in dev_t]
        rs = S.resid(encs_s, layers, which)
        rt = S.resid(encs_t, layers, which)
        for L in layers:
            vecs[L] += (rt[L].mean(0) - rs[L].mean(0)) / len(tl)
    return vecs, dict(dev_s=dev_s, test_s=test_s, dev_t=dev_t, test_t=test_t)


def random_like(v: torch.Tensor, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(v.shape, generator=g)
    return r / r.norm() * v.norm()


def valid_cities(V: dict, state: str, template: str) -> list[str]:
    return [c for c in CITIES[state] if V[template][c]]
