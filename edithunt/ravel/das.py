"""DAS / MDAS (RAVEL): learn a k-dim orthonormal subspace U [d,k] at (layer L, entity last token) and interchange it.

- DAS: Cause loss only: on A-prompts, the intervened base should answer val_A(source).
- MDAS: Cause + Iso: also, on A*-prompts (A* != A), the intervened base should keep val_A*(base).
  Batches are half Cause, half Iso (RAVEL's multi-task mix, equal weight).
Loss = token-level cross-entropy of the label surface (teacher forcing). The surface used for (template, value) is
the one the clean model scores highest on train entities with that value (`surfaces`), so e.g. Gender on
"X: for" templates trains on "his"/"her", not "male".
Training data = train entities x train templates only.
"""
from __future__ import annotations

import json
import random
import time

import torch

from .core import Item, Model
from .evalkit import SCREEN, Fam, build_pool, source_reps
from .stage1 import flat_cands


def surfaces(M: Model, F: Fam, n_ent: int = 3) -> dict:
    """(template text, value) -> preferred surface string; cached in results/screen/surfaces/{family}.json."""
    path = SCREEN / "surfaces" / f"{F.name}.json"
    if path.exists():
        return {tuple(json.loads(k)): v for k, v in json.loads(path.read_text()).items()}
    out = {}
    for a in F.attrs:
        for t in F.tpls.get((a, "train"), []) + F.tpls.get((a, "test"), []):
            ents_by_v = {}
            for e in sorted(F.correct[(a, t.text)]):
                ents_by_v.setdefault(F.labels[e][a], []).append(e)
            for v, surf in t.verb.items():
                if len(surf) == 1 or v not in ents_by_v:
                    out[(t.text, v)] = surf[0]
                    continue
                es = ents_by_v[v][:n_ent]
                cands = M.cand_ids(t.lead, tuple(surf))
                sc = M.score([M.encode(t.text, e) for e in es], cands)
                out[(t.text, v)] = surf[int(sc.mean(0).argmax())]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({json.dumps(list(k)): v for k, v in out.items()}, indent=0))
    return out


class Subspace(torch.nn.Module):
    def __init__(self, d: int, k: int, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        lin = torch.nn.Linear(d, k, bias=False)
        torch.nn.init.orthogonal_(lin.weight)
        self.lin = torch.nn.utils.parametrizations.orthogonal(lin)

    def U(self) -> torch.Tensor:  # [d,k], orthonormal columns
        return self.lin.weight.T


def _examples(F: Fam, pool, target: str, mdas: bool):
    L = F.labels
    cause, iso = [], []
    for i, p in enumerate(pool):
        if L[p.es][target] == L[p.eb][target]:
            continue
        if p.attr == target:
            cause.append((i, L[p.es][target]))
        elif mdas and (F.n_values[p.attr] == 1 or L[p.es][p.attr] != L[p.eb][p.attr]):
            iso.append((i, L[p.eb][p.attr]))
    return cause, iso


def train(M: Model, F: Fam, target: str, layer: int, k: int, mdas: bool, seed: int = 0, steps: int = 150,
          bs: int = 32, lr: float = 2e-3, max_seconds: float = 300, n_train: int = 600, log=print,
          iso_weight: float = 1.0):
    """Returns (U [d,k] cpu float32, info)."""
    t0 = time.time()
    surf = surfaces(M, F)
    pool = build_pool(F, "train", n_train, seed=1000 + seed)
    cause, iso = _examples(F, pool, target, mdas)
    src = source_reps(M, pool, [layer])[layer]
    rng = random.Random(seed)
    S = Subspace(M.d, k, seed).to(M.device)
    opt = torch.optim.Adam(S.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1 - s / steps)
    lab_cache = {}

    def lab_ids(t, v):
        key = (t.text, v)
        if key not in lab_cache:
            lab_cache[key] = M.cand_ids(t.lead, (surf.get(key, t.verb[v][0]),))[0]
        return lab_cache[key]

    hist = []
    for step in range(steps):
        if time.time() - t0 > max_seconds:
            log(f"    time cap at step {step}")
            break
        batch = rng.sample(cause, min(len(cause), bs if not mdas else bs // 2))
        n_c = len(batch)
        if mdas and iso:
            batch += rng.sample(iso, min(len(iso), bs // 2))
        items, labs, rows = [], [], []
        for i, v in batch:
            p = pool[i]
            base = M.encode(p.tpl.text, p.eb)
            lab = lab_ids(p.tpl, v)
            items.append(Item(base.ids + lab[:-1], base.pos))
            labs.append(lab)
            rows.append(i)
        keep = max(len(l) for l in labs)
        U = S.U()
        s_rows = src[rows].to(M.device)

        def fn(x, U=U, s_rows=s_rows):
            xf = x.float()
            return xf + (s_rows - xf) @ U @ U.T
        logits, _, _, _ = M.forward(items, fn, layer, grad=True, keep=keep)
        if keep == 1:
            logits = logits.unsqueeze(1)
        lp = torch.log_softmax(logits, -1)
        loss = 0
        for b, lab in enumerate(labs):
            n = len(lab)
            w = 1.0 if b < n_c else iso_weight
            loss = loss - w * lp[b, keep - n:, :].gather(-1, torch.tensor(lab, device=M.device).unsqueeze(-1)).sum()
        loss = loss / len(labs)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        hist.append(loss.item())
    info = dict(steps=len(hist), loss_first=sum(hist[:10]) / max(1, len(hist[:10])),
                loss_last=sum(hist[-10:]) / max(1, len(hist[-10:])), n_cause=len(cause), n_iso=len(iso),
                seconds=time.time() - t0)
    return S.U().detach().float().cpu(), info
