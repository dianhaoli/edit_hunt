from __future__ import annotations

import math

import torch


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def rate(flags) -> dict:
    flags = [bool(x) for x in flags]
    k, n = sum(flags), len(flags)
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": (k / n) if n else float("nan"), "ci95": [lo, hi]}


def fmt(r: dict) -> str:
    return f"{r['rate']:.2f} [{r['ci95'][0]:.2f},{r['ci95'][1]:.2f}] (n={r['n']})"


def kl(logp_clean: torch.Tensor, logp_edit: torch.Tensor) -> torch.Tensor:
    """KL(clean || edit) per row, inputs are log-softmax [B,V]."""
    return (logp_clean.exp() * (logp_clean - logp_edit)).sum(-1)


def binary_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """KL(Bern(p) || Bern(q)) elementwise."""
    p = p.clamp(eps, 1 - eps); q = q.clamp(eps, 1 - eps)
    return p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()


def p_us(scores: torch.Tensor) -> torch.Tensor:
    """scores [B, len(COUNTRY_CANDS)] exact log-probs -> P(US) among the candidate countries."""
    from .data import N_US
    return scores.softmax(-1)[:, :N_US].sum(-1)
