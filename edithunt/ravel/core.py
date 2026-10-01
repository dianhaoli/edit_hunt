"""Model wrapper for the breadth screen: %s-template encoding, exact candidate scoring under a residual
intervention at the entity's last token, and residual capture (with early exit).

Interventions act on the output of decoder layer L (the residual stream after block L, = Gemma Scope's
hook_resid_post) at one position per row (the entity's last token).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "google/gemma-2-2b"


@dataclass
class Item:
    ids: list[int]
    pos: int   # entity last-token position (unpadded)


class Iv:
    """Residual intervention at `layer`: x_rows <- fn(x_rows, **rows_chunk). `rows` holds per-row tensors [N, ...]
    aligned with the items being scored; they are sliced per batch."""

    def __init__(self, layer: int, fn, **rows):
        self.layer, self.fn, self.rows = layer, fn, rows

    def chunk(self, a: int, b: int):
        r = {k: v[a:b] for k, v in self.rows.items()}
        return lambda x: self.fn(x, **r)


class _Stop(Exception):
    pass


class Model:
    def __init__(self, name: str = MODEL, device: str = "cuda"):
        self.name, self.device = name, device
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map=device,
                                                          attn_implementation="eager").eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.layers = self.model.model.layers
        self.n_layers, self.d = len(self.layers), self.model.config.hidden_size
        self.bos = self.tok.bos_token_id
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        self._cand: dict = {}

    # ------------------------------------------------------------ encoding
    def _ids(self, text: str) -> list[int]:
        return [self.bos] + self.tok(text, add_special_tokens=False)["input_ids"]

    def encode(self, template: str, entity: str) -> Item | None:
        """None if the entity's tokens merge with the template text after it (no clean last-token site)."""
        before = template.split("%s")[0]
        ids = self._ids(template.replace("%s", entity, 1))
        upto = self._ids(before + entity)
        if ids[:len(upto)] != upto:
            return None
        return Item(ids, len(upto) - 1)

    def cand_ids(self, lead: str, surfaces: tuple[str, ...]) -> list[list[int]]:
        key = (lead, surfaces)
        if key not in self._cand:
            self._cand[key] = [self.tok(lead + s, add_special_tokens=False)["input_ids"] for s in surfaces]
        return self._cand[key]

    # ------------------------------------------------------------ forward
    def _batch(self, items: list[Item]):
        T = max(len(it.ids) for it in items)
        B = len(items)
        ids = torch.full((B, T), self.pad, dtype=torch.long)
        mask = torch.zeros((B, T), dtype=torch.long)
        offs = []
        for i, it in enumerate(items):
            o = T - len(it.ids)
            ids[i, o:] = torch.tensor(it.ids)
            mask[i, o:] = 1
            offs.append(o)
        pos_ids = (mask.cumsum(-1) - 1).clamp(min=0)
        site = torch.tensor([o + it.pos for o, it in zip(offs, items)])
        return ids.to(self.device), mask.to(self.device), pos_ids.to(self.device), site.to(self.device)

    def forward(self, items: list[Item], iv_fn=None, iv_layer: int | None = None, capture: list[int] = (),
                stop_after_capture=False, use_cache=False, grad=False, keep: int = 1):
        """Returns (last-position logits [B,V] float32 ([B,keep,V] if keep > 1) or None if stopped, captures {L: [B,d] float32}, cache, aux)."""
        ids, mask, pos_ids, site = self._batch(items)
        ar = torch.arange(len(items), device=self.device)
        caps, handles, st = {}, [], {"on": True}
        last_cap = max(capture) if capture else -1

        def mk(L):
            def hook(mod, inp, out):
                if not st["on"]:
                    return None
                h = out[0] if isinstance(out, tuple) else out
                changed = False
                if iv_fn is not None and L == iv_layer:
                    new = iv_fn(h[ar, site])
                    h = h.clone()
                    h[ar, site] = new.to(h.dtype)
                    changed = True
                if L in capture:
                    caps[L] = h[ar, site].float()
                    if stop_after_capture and L == last_cap:
                        raise _Stop
                if changed:
                    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
                return None
            return hook

        for L in set(capture) | ({iv_layer} if iv_fn is not None else set()):
            handles.append(self.layers[L].register_forward_hook(mk(L)))
        ctx = torch.enable_grad() if grad else torch.no_grad()
        logits, cache = None, None
        try:
            with ctx:
                out = self.model(input_ids=ids, attention_mask=mask, position_ids=pos_ids, use_cache=use_cache,
                                 logits_to_keep=keep)
            logits = out.logits[:, -1].float() if keep == 1 else out.logits.float()
            cache = out.past_key_values
        except _Stop:
            pass
        finally:
            st["on"] = False
            for h in handles:
                h.remove()
        return logits, caps, cache, (mask, pos_ids)

    @torch.no_grad()
    def capture(self, items: list[Item], layers: list[int], bs: int = 64) -> dict:
        """{L: [N,d] float32 (cpu)} at the entity's last token; stops the forward after the last needed layer."""
        out = {L: [] for L in layers}
        order = sorted(range(len(items)), key=lambda i: len(items[i].ids))
        for s in range(0, len(items), bs):
            idx = order[s:s + bs]
            _, caps, _, _ = self.forward([items[i] for i in idx], capture=list(layers), stop_after_capture=True)
            for L in layers:
                out[L].append((idx, caps[L].cpu()))
        res = {}
        for L in layers:
            buf = torch.zeros(len(items), self.d)
            for idx, v in out[L]:
                buf[idx] = v
            res[L] = buf
        return res

    @torch.no_grad()
    def score(self, items: list[Item], cands: list[list[int]], iv: Iv | None = None, bs: int = 64) -> torch.Tensor:
        """Exact full-sequence log-prob of each candidate token list [N, C] (cpu), via a token trie: one forward per
        internal trie node (shared prefixes are computed once). Argmax is exact: a prefix's log-prob bounds every
        candidate below it, so a subtree is expanded only if that bound beats the best exact score in some row;
        candidates in pruned subtrees hold the bound (an upper bound, never the argmax)."""
        N, C = len(items), len(cands)
        out = torch.zeros(N, C)
        if not N:
            return out
        trie = _trie(cands)
        deep = any(len(c) > 1 for c in cands)
        for s in range(0, N, bs):
            chunk = items[s:s + bs]
            fn = iv.chunk(s, s + len(chunk)) if iv is not None else None
            logits, _, cache, (mask, pos_ids) = self.forward(chunk, fn, iv.layer if iv else None, use_cache=deep)
            B = len(chunk)
            sc = torch.full((B, C), -1e9, device=self.device)
            best = torch.full((B,), -1e9, device=self.device)
            lp0 = torch.log_softmax(logits, -1)
            zero = torch.zeros(B, device=self.device)
            ctx = {"mask": mask, "last": pos_ids[:, -1:], "cache": cache}
            self._expand(trie, lp0, zero, 0, sc, best, ctx)
            out[s:s + B] = sc.cpu()
        return out

    def _expand(self, node, lp, acc, depth, sc, best, ctx):
        kids = node[0]
        toks = list(kids)
        tt = torch.tensor(toks, device=self.device)
        s_all = acc.unsqueeze(1) + lp[:, tt]  # [B, n_kids]
        for j, t in enumerate(toks):
            for ci in kids[t][1]:
                sc[:, ci] = s_all[:, j]
                best.copy_(torch.maximum(best, s_all[:, j]))
        internal = sorted([j for j, t in enumerate(toks) if kids[t][0]], key=lambda j: -s_all[:, j].max().item())
        B = lp.shape[0]
        for j in internal:
            t, s = toks[j], s_all[:, j]
            if not bool((s > best).any()):
                for ci in _leaves(kids[t]):
                    sc[:, ci] = s
                continue
            d = depth + 1
            m2 = torch.cat([ctx["mask"], torch.ones(B, d, dtype=ctx["mask"].dtype, device=self.device)], 1)
            o = self.model(input_ids=torch.full((B, 1), t, device=self.device), attention_mask=m2,
                           position_ids=ctx["last"] + d, past_key_values=ctx["cache"], use_cache=True)
            nlp = torch.log_softmax(o.logits[:, -1].float(), -1)
            self._expand(kids[t], nlp, s, d, sc, best, ctx)
            ctx["cache"].crop(-1)


def _trie(cands):
    root = ({}, [])
    for ci, c in enumerate(cands):
        node = root
        for k, t in enumerate(c):
            if t not in node[0]:
                node[0][t] = ({}, [])
            nxt = node[0][t]
            if k == len(c) - 1:
                nxt[1].append(ci)
            node = nxt
    return root


def _leaves(node):
    out = list(node[1])
    for ch in node[0].values():
        out += _leaves(ch)
    return out


# ------------------------------------------------------------------ featurizers (as Iv factories)
def iv_set(layer, src):                      # full-vector interchange: h' = h_source
    return Iv(layer, lambda x, src: src.to(x.device, x.dtype), src=src)


def iv_add(layer, vec):                      # additive: h' = h_base + v_row
    return Iv(layer, lambda x, vec: x + vec.to(x.device, x.dtype), vec=vec)


def iv_interchange(layer, U, src):
    """k-dim interchange with orthonormal columns U [d,k] (= W^T): h' = (I - U U^T) h_b + U U^T h_s."""
    def fn(x, src):
        Uc = U.to(x.device, torch.float32)
        xf = x.float()
        return (xf + (src.to(x.device).float() - xf) @ Uc @ Uc.T).to(x.dtype)
    return Iv(layer, fn, src=src)
