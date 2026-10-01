"""Gemma Scope residual SAEs (release "gemma-scope-2b-pt-res-canonical", 16k width) and L1-probe latent selection.

The canonical ids below were copied from sae_lens's pretrained_saes.yaml (release gemma-scope-2b-pt-res-canonical,
id layer_L/width_16k/canonical -> path). The weights are loaded straight from google/gemma-scope-2b-pt-res
(params.npz) instead of through sae_lens, to avoid changing torch/transformers in the shared venv; the
architecture (JumpReLU) is the one sae_lens's gemma_2 conversion uses.
"""
from __future__ import annotations

import numpy as np
import torch
from huggingface_hub import hf_hub_download

REPO = "google/gemma-scope-2b-pt-res"
CANONICAL_L0 = {0: 105, 1: 102, 2: 141, 3: 59, 4: 124, 5: 68, 6: 70, 7: 69, 8: 71, 9: 73, 10: 77, 11: 80, 12: 82,
                13: 84, 14: 84, 15: 78, 16: 78, 17: 77, 18: 74, 19: 73, 20: 71, 21: 70, 22: 72, 23: 75, 24: 73,
                25: 116}


class SAE:
    def __init__(self, layer: int, device="cuda"):
        self.layer = layer
        self.id = f"layer_{layer}/width_16k/average_l0_{CANONICAL_L0[layer]}"
        z = np.load(hf_hub_download(REPO, f"{self.id}/params.npz"))
        t = lambda k: torch.from_numpy(z[k]).float().to(device)
        self.W_enc, self.b_enc, self.W_dec, self.b_dec, self.thr = (t("W_enc"), t("b_enc"), t("W_dec"),
                                                                    t("b_dec"), t("threshold"))
        self.n = self.W_enc.shape[1]

    def enc(self, x: torch.Tensor) -> torch.Tensor:
        pre = x.float() @ self.W_enc + self.b_enc
        return pre * (pre > self.thr)

    def dec(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec


def probe_select(z: torch.Tensor, y: list, sizes=(5, 50), lam: float = 1e-3, steps: int = 400, seed: int = 0):
    """L1 multinomial logistic probe on SAE codes z [N,n] (standardised) -> {size: top latents by row L2 norm of the
    weight}; also returns train accuracy. RAVEL B.2 selects latents with an L1 classifier; we rank by weight norm
    to hit the requested sizes exactly."""
    torch.manual_seed(seed)
    classes = sorted(set(y))
    yi = torch.tensor([classes.index(v) for v in y], device=z.device)
    sd = z.std(0) + 1e-6
    zs = (z - z.mean(0)) / sd
    W = torch.zeros(z.shape[1], len(classes), device=z.device, requires_grad=True)
    b = torch.zeros(len(classes), device=z.device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=1e-2)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(zs @ W + b, yi) + lam * W.abs().sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = ((zs @ W + b).argmax(1) == yi).float().mean().item()
        norm = W.norm(dim=1)
        order = torch.argsort(norm, descending=True)
    return {k: order[:k].tolist() for k in sizes}, acc, int((norm > 1e-3).sum())


def verb_class(verb: str, form: str, attr: str) -> str:
    """Coarse label for probing verbs (each verb has its own value): the inflection pattern."""
    if attr == "Past Tense":
        if form == verb + "ed": return "ed"
        if form == verb + "d": return "d"
        if verb.endswith("y") and form == verb[:-1] + "ied": return "ied"
        if form.startswith(verb) and form.endswith("ed"): return "double"
        return "irregular"
    if form == verb + "s": return "s"
    if form == verb + "es": return "es"
    if verb.endswith("y") and form == verb[:-1] + "ies": return "ies"
    return "irregular"


def iv_sae(layer, sae: SAE, latents: list[int], src):
    """Interchange the selected latents with the source's, keeping the base's reconstruction error:
    h' = h_b + (dec(z') - dec(z_b)) = h_b + (z_s - z_b)[sel] @ W_dec[sel]."""
    from .core import Iv
    sel = torch.tensor(latents, device=sae.W_dec.device)
    Wd = sae.W_dec[sel]

    def fn(x, src):
        zb = sae.enc(x)[:, sel]
        zs = sae.enc(src.to(x.device))[:, sel]
        return (x.float() + (zs - zb) @ Wd).to(x.dtype)
    return Iv(layer, fn, src=src)
