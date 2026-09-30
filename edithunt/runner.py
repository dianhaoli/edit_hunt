"""Batch heterogeneous single-edit jobs (different layer / vector per row) into
shared forward passes."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .hooks import Intervention
from .model import Enc, Subject


@dataclass
class Job:
    enc: Enc
    # list of (layer, kind, positions, vec[d]); kind in {"add","set"}
    edits: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def _ivs(jobs: list[Job], d: int) -> list[Intervention]:
    layers = sorted({(e[0], e[1]) for j in jobs for e in j.edits})
    ivs = []
    for L, kind in layers:
        pos, vecs = [], torch.zeros(len(jobs), d)
        for b, j in enumerate(jobs):
            es = [e for e in j.edits if e[0] == L and e[1] == kind]
            assert len(es) <= 1, "one edit per (layer,kind) per job"
            if es:
                pos.append(list(es[0][2])); vecs[b] = es[0][3].float().cpu()
            else:
                pos.append([])
        ivs.append(Intervention(L, pos, kind, vecs))
    return ivs


def run_jobs(S: Subject, jobs: list[Job], cands: list[str], exact=None, bs: int = 32,
             return_logp: bool = False):
    """Returns scores [N,C] (and final log-probs [N,V] if requested)."""
    scs, lps = [], []
    for s in range(0, len(jobs), bs):
        chunk = jobs[s:s + bs]
        ivs = _ivs(chunk, S.d_model)
        out = S.score([j.enc for j in chunk], cands, ivs, bs=len(chunk), exact=exact, return_logp=return_logp)
        scs.append(out[0])
        if return_logp:
            lps.append(out[1])
    if return_logp:
        return torch.cat(scs), torch.cat(lps)
    return torch.cat(scs)
