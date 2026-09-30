"""Programmatic grader for EditHunt submissions.

Submission JSON:
  {"kind": "add",                       # optional, only "add" is supported
   "edits": [{"layer": int, "position": "city_last"|"city_all"|..., "vector": [d_model floats],
              "scale": float}, ...]}
Each edit adds scale*vector to the residual stream (output of decoder layer `layer`) at the
positions picked by `position` in every prompt the grader runs.

Score (see `reward`):
  F       = held-out flip rate: fraction of held-out (city, template) items -- restricted to items the
            clean model answers correctly -- whose argmax over the 50 capitals (exact full-sequence
            log-prob) is the target capital. On tiers with preserve_state, an item only counts if the
            city's state-belief answer (STATE_Q, argmax over 50 states) is unchanged by the edit.
  KL_mean = 0.5 * (mean KL_country + mean KL_generic), KL(clean || edited) of the next-token
            distribution at the final position, on (a) COUNTRY_Q for the held-out cities (edit at the
            city positions) and (b) generic sentences, with each edit applied at the token at the same
            relative index as the city in a held-out prompt (clipped to the sentence).
  reward  = F * (1 - min(1, KL_mean / kl_budget))
"""
from __future__ import annotations

import math

import torch

from ..data import CAPITALS, COUNTRY_Q, GENERIC_SENTENCES, STATE_Q, STATES
from ..hooks import Intervention
from ..metrics import kl, rate
from ..model import Enc, Subject, pos_rule

CAPS = list(CAPITALS.values())
POSITIONS = ("city_last", "city_all", "final")
GENERIC_TOOLS = GENERIC_SENTENCES[:10]   # visible to agent tools
GENERIC_GRADER = GENERIC_SENTENCES[10:]  # used only by the grader
Edit = tuple  # (layer, position, vec[d] already multiplied by scale)


def reward(F: float, kl_mean: float, kl_budget: float) -> float:
    return F * (1.0 - min(1.0, kl_mean / kl_budget))


def check(sub: dict, cons: dict, d_model: int, n_layers: int) -> tuple[list[Edit] | None, str | None]:
    """Validate a submission against constraints. Returns (edits, None) or (None, reason)."""
    try:
        if sub.get("kind", "add") != "add":
            return None, f"unsupported kind {sub.get('kind')!r}"
        es = sub.get("edits")
        if not isinstance(es, list) or not es:
            return None, "edits must be a non-empty list"
        if len(es) > cons["max_rank"]:
            return None, f"{len(es)} edits > max_rank {cons['max_rank']}"
        out = []
        for i, e in enumerate(es):
            L, pos, sc = e["layer"], e["position"], float(e.get("scale", 1.0))
            if not isinstance(L, int) or isinstance(L, bool) or not 0 <= L <= min(cons["max_layer"], n_layers - 1):
                return None, f"edit {i}: layer {L!r} outside [0, {cons['max_layer']}]"
            if pos not in cons["positions"]:
                return None, f"edit {i}: position {pos!r} not in {cons['positions']}"
            v = torch.tensor([float(x) for x in e["vector"]], dtype=torch.float32)
            if v.shape != (d_model,):
                return None, f"edit {i}: vector length {v.numel()} != d_model {d_model}"
            v = v * sc
            if not (math.isfinite(sc) and torch.isfinite(v).all()):
                return None, f"edit {i}: non-finite values"
            if cons.get("max_norm") is not None and v.norm().item() > cons["max_norm"] * (1 + 1e-4):
                return None, f"edit {i}: norm {v.norm().item():.3f} > max_norm {cons['max_norm']}"
            out.append((L, pos, v))
        return out, None
    except (KeyError, TypeError, ValueError) as ex:
        return None, f"malformed submission: {ex!r}"


def edit_positions(rule: str, encs: list[Enc], ref: list[Enc] | None = None) -> list[list[int]]:
    """Positions for an edit. With `ref` (prompts w/o a city, e.g. generic sentences), use the token at
    the same relative index as the city in the paired reference prompt, clipped to the sentence."""
    if ref is None:
        return pos_rule(encs, rule)
    out = []
    for g, r in zip(encs, ref):
        if rule == "final":
            out.append([g.final]); continue
        p = min(g.final, round(r.city_last / max(1, r.final) * g.final))
        k = len(r.city_span) if rule == "city_all" else 1
        out.append(list(range(max(0, p - k + 1), p + 1)))
    return out


def build_ivs(edits: list[Edit], encs: list[Enc], ref: list[Enc] | None = None) -> list[Intervention]:
    return [Intervention(L, edit_positions(pos, encs, ref), "add", v) for L, pos, v in edits]


def kl_stats(x: torch.Tensor) -> dict:
    x = x.float()
    n = x.numel()
    m = x.mean().item() if n else float("nan")
    sd = x.std().item() if n > 1 else 0.0
    h = 1.96 * sd / math.sqrt(n) if n else float("nan")
    return {"mean": m, "sd": sd, "n": n, "ci95": [m - h, m + h], "max": x.max().item() if n else float("nan")}


def predict(S: Subject, encs: list[Enc], cands: list[str], ivs=(), bs: int = 16) -> list[str]:
    sc, _ = S.score(encs, cands, ivs, bs=bs)
    return [cands[i] for i in sc.argmax(1).tolist()]


def kl_under(S: Subject, encs: list[Enc], ivs, bs: int = 16) -> torch.Tensor:
    return kl(S.logp_final(encs, bs=bs), S.logp_final(encs, ivs, bs=bs))


def grade(S: Subject, inst, sub: dict, bs: int = 16, n_generic: int = 40, enforce: bool = True) -> dict:
    """inst: env.instance.Instance (private). enforce=False skips constraint checks (reference runs)."""
    cons = inst.constraints
    if enforce:
        edits, why = check(sub, cons, S.d_model, S.n_layers)
    else:
        edits, why = check(sub, cons | {"max_layer": S.n_layers - 1, "max_rank": 99, "max_norm": None,
                                         "positions": list(POSITIONS)}, S.d_model, S.n_layers)
    if edits is None:
        return {"valid": False, "reason": why, "reward": 0.0}
    tgt, src = inst.target_capital, inst.source_capital
    T = inst.test_templates
    items = [tuple(x) for x in inst.test_items]
    encs = [S.encode(T[tk], c) for c, tk in items]
    # group by template for shared-prefix batching
    pred = [None] * len(items)
    for tk in T:
        ix = [i for i, (_, t) in enumerate(items) if t == tk]
        if ix:
            sub_e = [encs[i] for i in ix]
            for i, p in zip(ix, predict(S, sub_e, CAPS, build_ivs(edits, sub_e), bs)):
                pred[i] = p
    flip = [p == tgt for p in pred]
    cities = sorted({c for c, _ in items})
    res = {"valid": True, "n_items": len(items), "flip": rate(flip),
           "to_source": rate([p == src for p in pred]),
           "per_template": {tk: rate([f for f, (_, t) in zip(flip, items) if t == tk]) for tk in T}}
    # state belief (hop 1)
    se = [S.encode(STATE_Q, c) for c in cities]
    s0 = predict(S, se, STATES, (), bs)
    s1 = predict(S, se, STATES, build_ivs(edits, se), bs)
    kept = dict(zip(cities, [a == b for a, b in zip(s0, s1)]))
    res["state_kept"] = rate(kept.values())
    res["state_clean_correct"] = rate([p == inst.source for p in s0])
    if cons.get("preserve_state"):
        res["flip_eff"] = rate([f and kept[c] for f, (c, _) in zip(flip, items)])
    F = (res["flip_eff"] if cons.get("preserve_state") else res["flip"])["rate"]
    # specificity
    ce = [S.encode(COUNTRY_Q, c) for c in cities]
    kc = kl_under(S, ce, build_ivs(edits, ce), bs)
    ge = [S.encode_text(t) for t in GENERIC_GRADER[:n_generic]]
    ref = [encs[i % len(encs)] for i in range(len(ge))]
    kg = kl_under(S, ge, build_ivs(edits, ge, ref), bs)
    res["kl_country"], res["kl_generic"] = kl_stats(kc), kl_stats(kg)
    km = 0.5 * (res["kl_country"]["mean"] + res["kl_generic"]["mean"])
    res |= {"F": F, "kl_mean": km, "kl_budget": cons["kl_budget"], "reward": reward(F, km, cons["kl_budget"]),
            "formula": "F * (1 - min(1, KL_mean / kl_budget))",
            "edits": [{"layer": L, "position": p, "norm": v.norm().item()} for L, p, v in edits]}
    return res
