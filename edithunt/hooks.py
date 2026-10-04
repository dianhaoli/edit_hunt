"""Forward hooks on decoder layers: capture / patch / add / subspace-swap.

An Intervention acts on the *output* of decoder layer `layer` (the residual
stream after that block) at per-row positions (padded coordinates).
Decoder layers may return a tensor or a tuple; both are handled.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

import torch


@dataclass
class Intervention:
    layer: int
    # list (len B) of lists of *unpadded* positions; converted at apply time
    positions: list[list[int]]
    kind: str  # "add" | "set" | "swap" | "proj" | "gate"
    # add: vec [d] or [B,d];  set: [d], [B,d] or [B,npos,d];  swap: (U [d,k], target coords [B,k])
    # proj: (U [d,k] orthonormal, center [d] or None): x <- x - ((x - center) @ U) @ U.T
    payload: object = None
    scale: float = 1.0


@dataclass
class _State:
    active: bool = True
    offsets: torch.Tensor | None = None  # left-pad offsets [B]
    captures: dict = field(default_factory=dict)


def _get_h(out):
    return out[0] if isinstance(out, tuple) else out


def _put_h(out, h):
    if isinstance(out, tuple):
        return (h,) + tuple(out[1:])
    return h


def _apply(h: torch.Tensor, iv: Intervention, offsets: torch.Tensor) -> torch.Tensor:
    rows, cols, j = [], [], []
    for b, ps in enumerate(iv.positions):
        for k, p in enumerate(ps):
            rows.append(b); cols.append(p + int(offsets[b])); j.append(k)
    if not rows:
        return h
    dev = h.device
    r = torch.tensor(rows, device=dev); c = torch.tensor(cols, device=dev)
    x = h[r, c]  # [N,d]
    if iv.kind == "add":
        v = iv.payload.to(dev, h.dtype)
        v = v[r] if v.dim() == 2 else v
        new = x + iv.scale * v
    elif iv.kind == "set":
        v = iv.payload.to(dev, h.dtype)
        if v.dim() == 1:  # one shared vector for every row/position
            new = v.expand_as(x)
        else:
            new = v[r] if v.dim() == 2 else v[r, torch.tensor(j, device=dev)]
    elif iv.kind == "swap":
        U, tgt = iv.payload  # U: [d,k] orthonormal; tgt: [B,k] (or [k]) target coords
        U = U.to(dev, h.dtype); tgt = tgt.to(dev, h.dtype)
        tgt = tgt[r] if tgt.dim() == 2 else tgt
        new = x + (tgt - x @ U) @ U.T
    elif iv.kind == "gate":  # gated key->value edit (T2 v2 oracle): x + clamp((k.x - b) / w, 0, 1) * v
        k, b, w, v = (t.to(dev, h.dtype) for t in iv.payload)
        new = x + ((x @ k - b) / w).clamp(0, 1).unsqueeze(-1) * v
    elif iv.kind == "proj":
        U, ctr = iv.payload
        U = U.to(dev, h.dtype)
        xc = x - ctr.to(dev, h.dtype) if ctr is not None else x
        new = x - (xc @ U) @ U.T
    else:
        raise ValueError(iv.kind)
    h = h.clone()
    h[r, c] = new
    return h


@contextmanager
def hooked(layers, interventions: list[Intervention] = (), capture: dict | None = None):
    """Register hooks. `capture` maps layer -> list(B) of unpadded positions;
    captured tensors land in state.captures[layer] as [B, npos, d] (float32, detached
    unless grad is enabled). Yields the shared _State (set .offsets before forward)."""
    st = _State()
    handles = []
    by_layer: dict[int, list[Intervention]] = {}
    for iv in interventions:
        by_layer.setdefault(iv.layer, []).append(iv)
    cap_layers = set(capture or {})
    for L in set(by_layer) | cap_layers:
        def hook(mod, inp, out, L=L):
            if not st.active:
                return None
            h = _get_h(out)
            if L in by_layer:
                for iv in by_layer[L]:
                    h = _apply(h, iv, st.offsets)
            if L in cap_layers:
                pos = capture[L]
                rows = []
                for b in range(h.shape[0]):
                    idx = torch.tensor([p + int(st.offsets[b]) for p in pos[b]], device=h.device)
                    rows.append(h[b, idx])
                st.captures[L] = torch.stack(rows).float()
            return _put_h(out, h) if L in by_layer else None
        handles.append(layers[L].register_forward_hook(hook))
    try:
        yield st
    finally:
        for hd in handles:
            hd.remove()
