"""Gradient-trained interventions.

C2 `train_additive`: additive vector v at (layer, position rule) maximizing log p(target capital
first token) on dev capital prompts, with optional penalties:
  - kl_weight * KL(clean || edited) on `kl_encs` (vector added at their `kl_pos`)
  - keep_weight * CE(keep-answer) on `keep_encs` (e.g. state-belief prompts must keep the source
    state; the vector is added at their city token) -> hop separation.
C3 `train_das`: rank-k orthonormal subspace U at (layer, city token); intervention sets the
coordinates of U to the mean target-state coordinates (computed from dev target cities).
"""
from __future__ import annotations

import torch

from .hooks import Intervention
from .model import Enc, Subject, pos_rule


def _first_tok(S: Subject, s: str) -> int:
    return S.tok(" " + s, add_special_tokens=False)["input_ids"][0]


def train_additive(S: Subject, layer: int, rule: str, encs: list[Enc], target: str, steps: int = 60,
                   lr: float = 2.0, kl_encs: list[Enc] = (), kl_pos: list[list[int]] | None = None,
                   kl_weight: float = 0.0, keep_encs: list[Enc] = (), keep_answers: list[str] = (),
                   keep_rule: str = "city_last", keep_weight: float = 0.0, max_norm: float | None = None,
                   init: torch.Tensor | None = None, seed: int = 0, log_every: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    v = (init.clone().float() if init is not None else torch.zeros(S.d_model)).to(S.device).requires_grad_(True)
    opt = torch.optim.Adam([v], lr=lr)
    tgt = _first_tok(S, target)
    pos = pos_rule(encs, rule)
    clean_kl = S.logp_final(kl_encs).to(S.device) if kl_weight and len(kl_encs) else None
    keep_ids = torch.tensor([_first_tok(S, a) for a in keep_answers], device=S.device) if keep_weight else None
    for step in range(steps):
        opt.zero_grad()
        lp, *_ = S.forward(encs, [Intervention(layer, pos, "add", v)], grad=True)
        loss = -lp[:, tgt].mean()
        if clean_kl is not None:
            lp2, *_ = S.forward(kl_encs, [Intervention(layer, kl_pos, "add", v)], grad=True)
            loss = loss + kl_weight * (clean_kl.exp() * (clean_kl - lp2)).sum(-1).mean()
        if keep_ids is not None:
            lp3, *_ = S.forward(keep_encs, [Intervention(layer, pos_rule(keep_encs, keep_rule), "add", v)], grad=True)
            loss = loss - keep_weight * lp3[torch.arange(len(keep_encs)), keep_ids].mean()
        loss.backward()
        opt.step()
        if max_norm is not None:
            with torch.no_grad():
                n = v.norm()
                if n > max_norm:
                    v.mul_(max_norm / n)
        if log_every and step % log_every == 0:
            print(f"  step {step} loss {loss.item():.3f} |v| {v.norm().item():.1f}", flush=True)
    return v.detach().cpu()


def train_das(S: Subject, layer: int, src_encs: list[Enc], tgt_encs: list[Enc], target: str, k: int = 1,
              steps: int = 80, lr: float = 0.05, seed: int = 0, keep_encs: list[Enc] = (),
              keep_answers: list[str] = (), keep_weight: float = 0.0, log_every: int = 0):
    """Learn U [d,k]; intervention on src prompt at city_last: set coords(U) := mean coords of target
    dev cities. Returns (U, target_coords[k])."""
    torch.manual_seed(seed)
    W = torch.randn(S.d_model, k).to(S.device).requires_grad_(True)
    opt = torch.optim.Adam([W], lr=lr)
    tgt = _first_tok(S, target)
    Rt = S.resid(tgt_encs, [layer], "city_last")[layer].to(S.device)  # [n,d]
    pos = pos_rule(src_encs, "city_last")
    keep_ids = torch.tensor([_first_tok(S, a) for a in keep_answers], device=S.device) if keep_weight else None
    for step in range(steps):
        opt.zero_grad()
        U, _ = torch.linalg.qr(W)
        coords = (Rt @ U).mean(0)
        lp, *_ = S.forward(src_encs, [Intervention(layer, pos, "swap", (U, coords))], grad=True)
        loss = -lp[:, tgt].mean()
        if keep_ids is not None:
            lp3, *_ = S.forward(keep_encs, [Intervention(layer, pos_rule(keep_encs, "city_last"), "swap", (U, coords))], grad=True)
            loss = loss - keep_weight * lp3[torch.arange(len(keep_encs)), keep_ids].mean()
        loss.backward()
        opt.step()
        if log_every and step % log_every == 0:
            print(f"  das step {step} loss {loss.item():.3f}", flush=True)
    with torch.no_grad():
        U, _ = torch.linalg.qr(W)
        coords = (Rt @ U).mean(0)
    return U.detach().cpu(), coords.detach().cpu()
