"""Subject model wrapper: loading, position-aware encoding, batched exact
full-sequence candidate scoring under interventions."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from .hooks import Intervention, hooked


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Enc:
    ids: list[int]
    city_span: list[int]  # all token positions of the city name
    city_last: int
    final: int
    text: str
    city: str = ""
    prefix_len: int = 0  # tokens shared by every prompt of the template (before the city)


class Subject:
    def __init__(self, name: str, device: str | None = None, dtype: torch.dtype | None = None):
        self.name = name
        self.device = device or pick_device()
        if dtype is None:
            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.dtype = dtype
        self.tok = AutoTokenizer.from_pretrained(name)
        # cuda: load straight onto the GPU (7B bf16 = 15 GB would not fit through 15 GB host RAM)
        kw = dict(device_map=self.device) if self.device == "cuda" else {}
        if "gemma-2" in name.lower():  # logit soft-capping is only exact in eager attention (sdpa: argmax
            kw["attn_implementation"] = "eager"  # agreed 40/40 cities but scores differed by up to 0.3 nats)
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, **kw).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.layers = self.model.model.layers
        self.n_layers = len(self.layers)
        self.d_model = self.model.config.hidden_size
        self.add_bos = bool(getattr(self.tok, "add_bos_token", False)) or "gemma" in name.lower() or "llama" in name.lower()
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        self._cand_cache: dict = {}
        self._prefix_kv: dict = {}
        self.use_prefix_cache = True

    # ------------------------------------------------------------ encoding
    def _tok(self, text: str) -> list[int]:
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        if self.add_bos and self.tok.bos_token_id is not None:
            ids = [self.tok.bos_token_id] + ids
        return ids

    def encode(self, template: str, city: str) -> Enc:
        text = template.format(city=city)
        ids = self._tok(text)
        before = template.split("{city}")[0]
        pre_city = self._tok(before.rstrip(" "))
        upto = self._tok(before + city)
        assert ids[: len(upto)] == upto, f"tokenization boundary mismatch for {city!r}"
        assert ids[: len(pre_city)] == pre_city, f"prefix mismatch for {city!r}"
        span = list(range(len(pre_city), len(upto)))
        last_tok = self.tok.decode([ids[span[-1]]])
        assert city.endswith(last_tok.strip()) or last_tok.strip() in city.split()[-1], (city, last_tok)
        return Enc(ids=ids, city_span=span, city_last=span[-1], final=len(ids) - 1, text=text, city=city,
                   prefix_len=len(pre_city))

    def encode_text(self, text: str) -> Enc:
        ids = self._tok(text)
        return Enc(ids=ids, city_span=[], city_last=-1, final=len(ids) - 1, text=text)

    def _batch(self, encs: list[Enc], P: int = 0):
        """Left-pad the part after the first P (shared) tokens. Returns suffix ids,
        full mask (prefix ones + suffix mask), suffix position ids, hook offsets."""
        T = max(len(e.ids) - P for e in encs)
        offs = [T - (len(e.ids) - P) for e in encs]
        ids = torch.full((len(encs), T), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(encs), T), dtype=torch.long)
        for i, e in enumerate(encs):
            ids[i, offs[i]:] = torch.tensor(e.ids[P:])
            mask[i, offs[i]:] = 1
        pos = P + (mask.cumsum(-1) - 1).clamp(min=0)
        mask = torch.cat([torch.ones(len(encs), P, dtype=torch.long), mask], 1)
        return ids.to(self.device), mask.to(self.device), pos.to(self.device), [o - P for o in offs]

    def _shared_prefix(self, encs: list[Enc]) -> int:
        if not self.use_prefix_cache:
            return 0
        P = min(e.prefix_len for e in encs)
        if P < 4 or any(e.ids[:P] != encs[0].ids[:P] for e in encs):
            return 0
        return P

    @torch.no_grad()
    def _prefix_cache(self, prefix: tuple, B: int):
        if prefix not in self._prefix_kv:
            out = self.model(input_ids=torch.tensor([prefix], device=self.device), use_cache=True)
            kv = out.past_key_values
            self._prefix_kv[prefix] = [(l.keys, l.values) for l in kv.layers]
        cache = DynamicCache(config=self.model.config)
        for li, (k, v) in enumerate(self._prefix_kv[prefix]):
            cache.update(k.expand(B, -1, -1, -1).contiguous(), v.expand(B, -1, -1, -1).contiguous(), li)
        return cache

    # ------------------------------------------------------------ forward
    def forward(self, encs, interventions=(), capture=None, use_cache=False, grad=False):
        """Returns (final-position log-probs [B,V] float32, state, cache, (mask,pos))."""
        P = self._shared_prefix(encs)
        if P and capture and any(p < P for ps in capture.values() for row in ps for p in row):
            P = 0  # a capture inside the shared prefix needs the full forward (the cached prefix has no residuals)
        ids, mask, pos, offs = self._batch(encs, P)
        for iv in interventions:
            assert all(p >= P for ps in iv.positions for p in ps), "intervention inside shared prefix"
        cache = self._prefix_cache(tuple(encs[0].ids[:P]), len(encs)) if P else None
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx, hooked(self.layers, interventions, capture) as st:
            st.offsets = offs
            out = self.model(input_ids=ids, attention_mask=mask, position_ids=pos, past_key_values=cache,
                             use_cache=use_cache or bool(P), logits_to_keep=1)
            st.active = False
        logp = torch.log_softmax(out.logits[:, -1].float(), -1)
        return logp, st, (out.past_key_values if use_cache else None), (mask, pos)

    def cand_tokens(self, cands: list[str]) -> list[list[int]]:
        key = tuple(cands)
        if key not in self._cand_cache:
            self._cand_cache[key] = [self.tok(" " + c, add_special_tokens=False)["input_ids"] for c in cands]
        return self._cand_cache[key]

    @torch.no_grad()
    def score(self, encs, cands: list[str], interventions=(), capture=None, bs: int = 32,
              return_logp=False, exact=None):
        """Full-sequence log-prob of ' '+cand for candidates. Argmax over candidates is
        always exact: a multi-token candidate's full log-prob is bounded by its first-token
        log-prob, so its continuation is only computed when that bound beats the best
        exact score in the batch (or when it is listed in `exact`, or exact=True for all).
        Unevaluated entries hold the (upper-bound) first-token log-prob.
        Returns scores [B,C] (cpu), [final log-probs [B,V]], captures."""
        if not encs:  # e.g. a pair with no clean-valid held-out cities for one template
            e = torch.zeros(0, len(cands))
            return (e, torch.zeros(0, self.model.config.vocab_size), {}) if return_logp else (e, {})
        toks = self.cand_tokens(cands)
        force = set(range(len(cands))) if exact is True else {cands.index(c) for c in (exact or [])}
        all_scores, all_logp, caps = [], [], {}
        for s in range(0, len(encs), bs):
            chunk = encs[s:s + bs]
            ivs = [_slice_iv(iv, s, s + len(chunk)) for iv in interventions]
            cap = {L: p[s:s + len(chunk)] for L, p in (capture or {}).items()}
            logp, st, cache, (mask, pos) = self.forward(chunk, ivs, cap or None, use_cache=True)
            for L, v in st.captures.items():
                caps.setdefault(L, []).append(v.cpu())
            B = len(chunk)
            first = torch.stack([logp[:, t[0]] for t in toks], 1).cpu()  # [B,C]
            sc = first.clone()
            single = [i for i, t in enumerate(toks) if len(t) == 1]
            best = sc[:, single].max(1).values if single else torch.full((B,), -1e9)
            last_pos = pos[:, -1:]
            multi = sorted([i for i, t in enumerate(toks) if len(t) > 1], key=lambda i: -first[:, i].max().item())
            for ci in multi:
                if ci not in force and not bool((first[:, ci] > best).any()):
                    continue
                t = toks[ci]
                cont = torch.tensor(t[:-1], device=self.device).unsqueeze(0).expand(B, -1)
                n = cont.shape[1]
                m2 = torch.cat([mask, torch.ones(B, n, dtype=mask.dtype, device=self.device)], 1)
                p2 = last_pos + 1 + torch.arange(n, device=self.device).unsqueeze(0)
                out = self.model(input_ids=cont, attention_mask=m2, position_ids=p2,
                                 past_key_values=cache, use_cache=True)
                lp = torch.log_softmax(out.logits.float(), -1)  # [B,n,V]
                nxt = torch.tensor(t[1:], device=self.device)
                sc[:, ci] += lp[:, torch.arange(n), nxt].sum(-1).cpu()
                cache.crop(-n)
                best = torch.maximum(best, sc[:, ci])
            all_scores.append(sc)
            if return_logp:
                all_logp.append(logp.cpu())
        scores = torch.cat(all_scores)
        caps = {L: torch.cat(v) for L, v in caps.items()}
        if return_logp:
            return scores, torch.cat(all_logp), caps
        return scores, caps

    @torch.no_grad()
    def logp_final(self, encs, interventions=(), bs: int = 32):
        out = []
        for s in range(0, len(encs), bs):
            chunk = encs[s:s + bs]
            ivs = [_slice_iv(iv, s, s + len(chunk)) for iv in interventions]
            lp, *_ = self.forward(chunk, ivs)
            out.append(lp.cpu())
        return torch.cat(out)

    @torch.no_grad()
    def resid(self, encs, layers: list[int], which: str = "city_last", bs: int = 32):
        """Residual (layer output) at a position per prompt: {L: [B,d]} (cpu float32)."""
        posf = {"city_last": lambda e: [e.city_last], "final": lambda e: [e.final]}[which]
        res = {L: [] for L in layers}
        for s in range(0, len(encs), bs):
            chunk = encs[s:s + bs]
            cap = {L: [posf(e) for e in chunk] for L in layers}
            _, st, _, _ = self.forward(chunk, (), cap)
            for L in layers:
                res[L].append(st.captures[L][:, 0].cpu())
        return {L: torch.cat(v) for L, v in res.items()}

    @torch.no_grad()
    def greedy(self, encs: list[Enc], max_new: int = 6, bs: int = 32) -> list[str]:
        res = []
        for s in range(0, len(encs), bs):
            ids, mask, _, _ = self._batch(encs[s:s + bs], 0)
            out = self.model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new,
                                      do_sample=False, pad_token_id=self.pad_id)
            res += [self.tok.decode(o[ids.shape[1]:], skip_special_tokens=True) for o in out]
        return res


def _slice_iv(iv: Intervention, a: int, b: int) -> Intervention:
    pl = iv.payload
    if iv.kind in ("add", "set") and isinstance(pl, torch.Tensor) and pl.dim() >= 2:
        pl = pl[a:b]
    elif iv.kind == "swap":
        U, tgt = pl
        pl = (U, tgt[a:b] if tgt.dim() == 2 else tgt)
    return Intervention(iv.layer, iv.positions[a:b], iv.kind, pl, iv.scale)


def pos_rule(encs: list[Enc], rule: str) -> list[list[int]]:
    if rule == "city_last":
        return [[e.city_last] for e in encs]
    if rule == "city_all":
        return [list(e.city_span) for e in encs]
    if rule == "final":
        return [[e.final] for e in encs]
    raise ValueError(rule)
