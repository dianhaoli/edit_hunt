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
  KL_mean = 0.5 * (mean KL_country + mean KL_generic).
            KL_country: binary KL on P(US) from the few-shot COUNTRY_Q probe (exact scores over
            COUNTRY_CANDS) for the held-out cities, edit at the city positions. (Full-vocab KL on a
            zero-shot country prompt was state-entangled: working edits scored ~1-3.5 nats there because the
            model answers the *state*. See LAB_NOTEBOOK 2026-09-30.)
            KL_generic: full-vocab KL(clean || edited) at the final position of generic sentences, each edit
            applied at the token at the same relative index as the city in a held-out prompt.
  leak    = fraction of private third-state (city, template) items, clean-valid, whose top capital changes
            under the edit (the main collateral damage of mean-diff-like edits).
  reward  = F * (1 - leak_weight * leak) * (1 - min(1, KL_mean / kl_budget))

Other task kinds (inst.task; see env/tasks.py) reuse the same pieces:
  consistency  reward = (F_cap * F_state * F_hidden)^(1/3) * KL factor; F_hidden from private readouts
  minimal      pass = F >= min_flip and KL_mean <= kl_budget; reward = min(1, r_ref / r), r = |v| / |resid_L|
  erase        "proj" edits; reward = mean(behaviour erasure, probe erasure) * KL factor
  detective / handoff   {"report": {...}} graded against private ground truth
"""
from __future__ import annotations

import math

import torch

from ..data import CAPITALS, CITY2STATE, COUNTRY_CANDS, COUNTRY_Q, GENERIC_SENTENCES, STATE_Q, STATES
from ..hooks import Intervention
from ..metrics import binary_kl, kl, p_us, rate
from ..model import Enc, Subject, pos_rule

CAPS = list(CAPITALS.values())
POSITIONS = ("city_last", "city_all", "final")
GENERIC_TOOLS = GENERIC_SENTENCES[:10]   # visible to agent tools
GENERIC_GRADER = GENERIC_SENTENCES[10:]  # used only by the grader
Edit = tuple  # (layer, position, vec[d] already multiplied by scale) or (layer, position, (U, center), "proj")


def reward(F: float, kl_mean: float, kl_budget: float, leak: float = 0.0, leak_weight: float = 0.0) -> float:
    return F * (1.0 - leak_weight * leak) * (1.0 - min(1.0, kl_mean / kl_budget))


def check(sub: dict, cons: dict, d_model: int, n_layers: int) -> tuple[list[Edit] | None, str | None]:
    """Validate a submission against constraints. Returns (edits, None) or (None, reason)."""
    try:
        kind = sub.get("kind", "add")
        if kind != cons.get("edit_kind", "add"):
            return None, f"unsupported kind {kind!r} (this task takes {cons.get('edit_kind', 'add')!r})"
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
            if cons.get("layer") is not None and L != cons["layer"]:  # fixed intervention site (T2 v2, RAVEL/MIB)
                return None, f"edit {i}: layer must be {cons['layer']} (the task's intervention site)"
            if pos not in cons["positions"]:
                return None, f"edit {i}: position {pos!r} not in {cons['positions']}"
            if kind == "proj":
                B = torch.tensor([[float(x) for x in b] for b in e["basis"]], dtype=torch.float32)
                if B.dim() != 2 or B.shape[1] != d_model or not 1 <= B.shape[0] <= cons.get("max_basis", 8):
                    return None, f"edit {i}: basis must be 1..{cons.get('max_basis', 8)} vectors of length {d_model}"
                c = e.get("center")
                c = torch.tensor([float(x) for x in c], dtype=torch.float32) if c is not None else None
                if not torch.isfinite(B).all() or (c is not None and (c.shape != (d_model,) or not torch.isfinite(c).all())):
                    return None, f"edit {i}: non-finite or malformed basis/center"
                Q, R = torch.linalg.qr(B.T)
                Q = Q[:, R.diagonal().abs() > 1e-6 * R.diagonal().abs().max()]
                out.append((L, pos, (Q, c), "proj"))
                continue
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
    return [Intervention(e[0], edit_positions(e[1], encs, ref), e[3] if len(e) > 3 else "add", e[2]) for e in edits]


def edit_summary(edits: list[Edit]) -> list[dict]:
    return [{"layer": e[0], "position": e[1], "kind": "proj", "rank": int(e[2][0].shape[1])} if len(e) > 3
            else {"layer": e[0], "position": e[1], "norm": e[2].norm().item()} for e in edits]


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


def grade(S: Subject, inst, sub: dict | None, bs: int = 16, n_generic: int = 40, enforce: bool = True) -> dict:
    """inst: env.instance.Instance (private). enforce=False skips constraint checks (reference runs)."""
    if sub is None:
        return {"valid": False, "reason": "no submission", "reward": 0.0}
    task = getattr(inst, "task", "edit")
    if task in ("detective", "handoff"):
        from .tasks import grade_report
        return grade_report(inst, sub)
    if task != "edit":
        from .tasks import GRADERS
        return GRADERS[task](S, inst, sub, bs, enforce)
    return grade_edit(S, inst, sub, bs, n_generic, enforce)


def parse(S: Subject, inst, sub: dict, enforce: bool = True):
    cons = inst.constraints
    if enforce:
        return check(sub, cons, S.d_model, S.n_layers)
    return check(sub, cons | {"max_layer": S.n_layers - 1, "max_rank": 99, "max_norm": None, "max_basis": 64,
                              "positions": list(POSITIONS)}, S.d_model, S.n_layers)


def side_effects(S: Subject, edits, cities: list[str], ref_encs, bs: int = 16, n_generic: int = 40) -> dict:
    """P(US) binary KL on the country probe for `cities` and generic-text KL (edit at the token matching the
    city's relative position in ref_encs)."""
    ce = [S.encode(COUNTRY_Q, c) for c in cities]
    pu0 = p_us(S.score(ce, COUNTRY_CANDS, bs=bs, exact=True)[0])
    pu1 = p_us(S.score(ce, COUNTRY_CANDS, build_ivs(edits, ce), bs=bs, exact=True)[0])
    ge = [S.encode_text(t) for t in GENERIC_GRADER[:n_generic]]
    ref = [ref_encs[i % len(ref_encs)] for i in range(len(ge))]
    kg = kl_under(S, ge, build_ivs(edits, ge, ref), bs)
    res = {"p_us": {"clean": pu0.mean().item(), "edited": pu1.mean().item()},
           "kl_country": kl_stats(binary_kl(pu0, pu1)), "kl_generic": kl_stats(kg)}
    res["kl_mean"] = 0.5 * (res["kl_country"]["mean"] + res["kl_generic"]["mean"])
    return res


def grade_edit(S: Subject, inst, sub: dict, bs: int = 16, n_generic: int = 40, enforce: bool = True) -> dict:
    cons = inst.constraints
    edits, why = parse(S, inst, sub, enforce)
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
    pu0 = p_us(S.score(ce, COUNTRY_CANDS, bs=bs, exact=True)[0])
    pu1 = p_us(S.score(ce, COUNTRY_CANDS, build_ivs(edits, ce), bs=bs, exact=True)[0])
    kc = binary_kl(pu0, pu1)
    res["p_us"] = {"clean": pu0.mean().item(), "edited": pu1.mean().item()}
    # third-state leakage (private items; absent on instances generated before 2026-09-30 -> leak 0, n=0)
    li = [tuple(x) for x in getattr(inst, "leak_items", [])]
    changed, to_tgt = [], []
    for tk in T:
        cs = [c for c, t in li if t == tk]
        if cs:
            le = [S.encode(T[tk], c) for c in cs]
            pl = predict(S, le, CAPS, build_ivs(edits, le), bs)
            changed += [p != CAPITALS[CITY2STATE[c]] for c, p in zip(cs, pl)]
            to_tgt += [p == tgt for p in pl]
    res["leak"], res["leak_to_target"] = rate(changed), rate(to_tgt)
    ge = [S.encode_text(t) for t in GENERIC_GRADER[:n_generic]]
    ref = [encs[i % len(encs)] for i in range(len(ge))]
    kg = kl_under(S, ge, build_ivs(edits, ge, ref), bs)
    res["kl_country"], res["kl_generic"] = kl_stats(kc), kl_stats(kg)
    km = 0.5 * (res["kl_country"]["mean"] + res["kl_generic"]["mean"])
    lk = res["leak"]["rate"] if res["leak"]["n"] else 0.0
    lw = cons.get("leak_weight", 0.0)
    res |= {"F": F, "kl_mean": km, "kl_budget": cons["kl_budget"], "leak_weight": lw,
            "reward": reward(F, km, cons["kl_budget"], lk, lw),
            "formula": "F * (1 - leak_weight * leak) * (1 - min(1, KL_mean / kl_budget))",
            "edits": edit_summary(edits)}
    return res
