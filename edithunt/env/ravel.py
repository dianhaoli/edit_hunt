"""T2_keepstate v2: an RL-environment version of RAVEL.

RAVEL (Huang et al., "RAVEL: Evaluating Interpretability Methods on Disentangling Language Model
Representations", ACL 2024) asks whether an intervention on an entity's representation changes one attribute
(Cause) while leaving the entity's other attributes and other entities alone (Iso), scored as
Disentangle = (Cause + Iso) / 2. MIB's RAVEL track (Mueller et al., "MIB: A Mechanistic Interpretability
Benchmark", 2025) fixes the intervention site: the entity's last token, at a given layer. We follow both:

  entity     a US city;  attribute to change: the capital of its state ("hop 2");  attribute to keep: the
             state itself (hop 1), and other cities' capitals.
  site       position = the city's last token (fixed), layer = given per instance (constraints["layer"]),
             sampled from the model's mid band (Qwen2.5-1.5B: LAYERS). The layer is a difficulty dial.
  edit       one additive vector at that site (the agent's submission). The grader can also score internal
             "swap" edits (DAS) for the offline references.

Grader v2 (`grade_ravel`; v1 = grader.grade_edit, kept for comparison):
  Cause      held-out source cities (>= MIN_TEST per instance, all valid source cities not in dev) x held-out
             capital wordings (ho_fs, ho_zs), clean-correct items only: top capital (exact log-prob over the 50)
             == target capital.
  Iso_state  per held-out source city whose clean answers are right on both: the held-out state question
             (STATE_Q_HO) AND the hidden postal-abbreviation readout (READOUTS["abbr"]) still give the source state.
  Iso_other  N_ISO cities sampled at grade time from a large private pool of other states' cities x (ho_fs,
             ho_zs): top capital unchanged w.r.t. the clean prediction computed at grade time; items whose
             clean top-1 vs top-2 margin is < MIN_MARGIN nats are dropped (bf16 near-ties).
  Iso        = (Iso_state + Iso_other) / 2   (equal weight per family; the two item counts differ)
  Disentangle = (Cause + Iso) / 2
  KL_mean    = mean(country binary KL, generic-text KL) as in v1 (grader.side_effects)
  reward     = Disentangle * (1 - min(1, KL_mean / kl_budget))
  Also reported: continuous margins (mean log-prob margins, see `_margins`), the old product-form reward
  F_keep * (1 - leak_weight * leak) * KL factor (F_keep = Cause items whose city keeps both state readouts), and
  v1 aliases flip / state_kept / leak.

Follow-up (2026-10-01, FINDINGS §11): the averaged reward above gives a null edit 0.5, so the validation uses the
product reward_mult = Cause * Iso_state * Iso_other * KL factor (null = 0). KL_mean's country term counts binary KL
only where the edit lowers P(US), over held-out cities not in data.AMBIG_NONUS (kl_mean_oldprobe = the v1 probe).
A single additive vector cannot isolate (it moves every city); the offline oracle is a gated edit
(`train_gated`, hooks kind "gate": x + clamp((k.x - b)/w, 0, 1) v), fit on dev + accessible keep cities only.
"""
from __future__ import annotations

import json
import math
import random
import re

import torch

from ..common import ROOT
from ..data import (ABBR, AMBIG_NONUS, CAPITALS, COUNTRY_CANDS, COUNTRY_Q, CITIES, CITIES_EXT, CITIES_EXTRA, CITY2STATE_EXT, HELDOUT_TEMPLATES, READOUTS, STATE_Q,
                    STATE_Q_HO, STATES, TEMPLATES, study_states)
from ..hooks import Intervention
from ..metrics import binary_kl, p_us, rate
from ..model import Subject, pos_rule
from .grader import CAPS, build_ivs, edit_summary, parse, side_effects
from .instance import Instance

LAYERS = {"Qwen/Qwen2.5-1.5B": [6, 8, 10, 12, 15]}
MIN_TEST, N_DEV, N_ISO, MIN_MARGIN = 10, 4, 16, 0.1
DEV_T = ["fs1", "zs1"]
ABBR_T = READOUTS["abbr"][0]
ABBRS = [ABBR[s] for s in STATES]
CONS = dict(positions=["city_last"], max_rank=1, max_norm=None, kl_budget=1.0, fwd_budget=4000,
            preserve_state=True, leak_weight=0.5, generic_refusal=True)
_CLEAN: dict = {}  # (model, prompt, cands) -> exact clean scores [C]


# ------------------------------------------------------------------ scoring helpers
def _scores(S: Subject, encs, cands, ivs=(), bs=16) -> torch.Tensor:
    """Exact full-sequence log-probs [B, C] (all candidates, so margins are exact)."""
    return S.score(encs, cands, ivs, bs=bs, exact=True)[0] if encs else torch.zeros(0, len(cands))


def _clean_scores(S: Subject, encs, cands, bs=16) -> torch.Tensor:
    key = lambda e: (S.name, e.text, tuple(cands))
    todo = [e for e in encs if key(e) not in _CLEAN]
    if todo:
        for e, s in zip(todo, _scores(S, todo, cands, (), bs)):
            _CLEAN[key(e)] = s
    return torch.stack([_CLEAN[key(e)] for e in encs]) if encs else torch.zeros(0, len(cands))


def _margins(sc: torch.Tensor, idx: list[int]) -> torch.Tensor:
    """score of candidate idx[i] minus the best other candidate (row-wise)."""
    if not len(sc):
        return torch.zeros(0)
    r = torch.arange(len(sc))
    a = sc[r, idx]
    o = sc.clone(); o[r, idx] = -float("inf")
    return a - o.max(1).values


def _top2_margin(sc: torch.Tensor) -> torch.Tensor:
    t = sc.topk(2, dim=1).values
    return t[:, 0] - t[:, 1]


def _mean(x) -> float:
    x = torch.as_tensor(x, dtype=torch.float32)
    return x.mean().item() if x.numel() else float("nan")


def _grouped(S, items, edits, cands, bs, templates):
    """Encode (city, template key) items, score clean + edited per template (shared-prefix batching)."""
    encs, sc0, sc1 = [], [], []
    order = []
    for tk in templates:
        ix = [i for i, (_, t) in enumerate(items) if t == tk]
        if not ix:
            continue
        e = [S.encode(TEMPLATES[tk] if tk in TEMPLATES else tk, items[i][0]) for i in ix]
        encs += e; order += ix
        sc0.append(_clean_scores(S, e, cands, bs))
        sc1.append(_scores(S, e, cands, build_ivs(edits, e), bs))
    inv = sorted(range(len(order)), key=lambda j: order[j])
    cat = lambda xs: torch.cat(xs)[inv] if xs else torch.zeros(0, len(cands))
    return [encs[j] for j in inv], cat(sc0), cat(sc1)


# ------------------------------------------------------------------ grader v2
def iso_sample(inst, iso_seed: int = 0, n: int = N_ISO) -> list[str]:
    pool = inst.extra["iso_pool"]
    return random.Random(f"{inst.id}-iso-{iso_seed}").sample(pool, min(n, len(pool)))


def grade_ravel(S: Subject, inst, sub: dict | None, bs: int = 16, enforce: bool = True, edits=None,
                iso_seed: int = 0, n_iso: int = N_ISO, n_generic: int = 40) -> dict:
    """`edits` (internal, offline references): pre-built grader edits, e.g. a DAS swap, bypassing `sub`."""
    if edits is None:
        edits, why = parse(S, inst, sub, enforce)
        if edits is None:
            return {"valid": False, "reason": why, "reward": 0.0}
    tgt, src = inst.target_capital, inst.source
    ti = CAPS.index(tgt)
    # Cause
    items = [tuple(x) for x in inst.test_items]
    cenc, c0, c1 = _grouped(S, items, edits, CAPS, bs, HELDOUT_TEMPLATES)
    pred = c1.argmax(1)
    cause = (pred == ti).tolist()
    cause_m = _margins(c1, [ti] * len(items))
    # Iso_state: both readouts, per city
    cities = sorted({c for c, _ in items})
    se = [S.encode(STATE_Q_HO, c) for c in cities]
    ae = [S.encode(ABBR_T, c) for c in cities]
    s0, a0 = _clean_scores(S, se, STATES, bs), _clean_scores(S, ae, ABBRS, bs)
    s1, a1 = _scores(S, se, STATES, build_ivs(edits, se), bs), _scores(S, ae, ABBRS, build_ivs(edits, ae), bs)
    si, ai = STATES.index(src), ABBRS.index(ABBR[src])
    ok0 = (s0.argmax(1) == si) & (a0.argmax(1) == ai)
    kept_q, kept_a = s1.argmax(1) == si, a1.argmax(1) == ai
    st_items = [i for i in range(len(cities)) if ok0[i]]
    iso_state = [bool(kept_q[i] and kept_a[i]) for i in st_items]
    unchanged = {c: bool(s1[i].argmax() == s0[i].argmax() and a1[i].argmax() == a0[i].argmax())
                 for i, c in enumerate(cities)}
    sm_q, sm_a = _margins(s1, [si] * len(cities))[st_items], _margins(a1, [ai] * len(cities))[st_items]
    # Iso_other
    oc = iso_sample(inst, iso_seed, n_iso)
    oitems = [(c, tk) for c in oc for tk in HELDOUT_TEMPLATES]
    _, o0, o1 = _grouped(S, oitems, edits, CAPS, bs, HELDOUT_TEMPLATES)
    keep = (_top2_margin(o0) >= MIN_MARGIN).tolist() if len(o0) else []
    top0 = o0.argmax(1) if len(o0) else torch.zeros(0, dtype=torch.long)
    iso_other = [bool(o1[i].argmax() == top0[i]) for i in range(len(oitems)) if keep[i]]
    om = _margins(o1, top0.tolist())[[i for i in range(len(oitems)) if keep[i]]] if keep else torch.zeros(0)
    to_tgt = [bool(o1[i].argmax() == ti) for i in range(len(oitems)) if keep[i]]
    # specificity. Country term (fixed 2026-10-01, before any agent run): binary KL on P(US) counted only where the
    # edit LOWERS P(US), over held-out cities whose name is not also a non-US place (AMBIG_NONUS). The v1 probe
    # (any change, all cities) is kept as kl_mean_oldprobe for comparison.
    se_ = side_effects(S, edits, cities, cenc, bs, n_generic)
    cc = [c for c in cities if c not in AMBIG_NONUS]
    ce = [S.encode(COUNTRY_Q, c) for c in cc]
    pu0 = p_us(_clean_scores(S, ce, COUNTRY_CANDS, bs))
    pu1 = p_us(_scores(S, ce, COUNTRY_CANDS, build_ivs(edits, ce), bs))
    kc = binary_kl(pu0, pu1) * (pu1 < pu0).float()
    kc_mean = kc.mean().item() if len(cc) else 0.0
    km, kb = 0.5 * (kc_mean + se_["kl_generic"]["mean"]), inst.constraints["kl_budget"]
    klf = 1.0 - min(1.0, km / kb)
    R = lambda f: rate(f)["rate"] if f else 0.0
    Cause, Is, Io = R(cause), R(iso_state), R(iso_other)
    Iso = 0.5 * (Is + Io)
    D = 0.5 * (Cause + Iso)
    f_keep = [f and unchanged[c] for f, (c, _) in zip(cause, items)]
    lw = inst.constraints.get("leak_weight", 0.5)
    old = R(f_keep) * (1 - lw * (1 - Io)) * klf
    res = {"valid": True, "grader": "v2",
           "Cause": rate(cause), "Iso_state": rate(iso_state), "Iso_other": rate(iso_other),
           "Iso": Iso, "Disentangle": D, "kl_mean": km, "kl_budget": kb, "kl_factor": klf,
           "reward": D * klf, "reward_product": old, "reward_mult": Cause * Is * Io * klf,
           "kl_country_fixed": {"mean": kc_mean, "n": len(cc), "excluded_ambiguous": sorted(set(cities) - set(cc)),
                                "p_us_clean": pu0.mean().item() if len(cc) else None,
                                "p_us_edited": pu1.mean().item() if len(cc) else None},
           "kl_mean_oldprobe": se_["kl_mean"],
           "continuous": {"cause_margin": _mean(cause_m), "state_q_margin": _mean(sm_q),
                          "abbr_margin": _mean(sm_a), "iso_other_margin": _mean(om),
                          "note": "mean log-prob margin of the wanted answer over the best other candidate; "
                                  "Iso_other: clean top answer under the edit"},
           "Iso_state_parts": {"state_q_ho": rate(kept_q[st_items].tolist()), "abbr": rate(kept_a[st_items].tolist())},
           "iso_other_to_target": rate(to_tgt), "iso_other_dropped_near_ties": len(oitems) - len(iso_other),
           "iso_cities": oc, "layer": inst.constraints.get("layer"),
           "formula": "reward = (Cause + (Iso_state + Iso_other)/2)/2 * (1 - min(1, KL_mean/kl_budget))",
           "edits": edit_summary(edits) if not any(len(e) > 3 and e[3] in ("swap", "gate") for e in edits) else
           [{"layer": e[0], "position": e[1], "kind": e[3] if len(e) > 3 else "add"} for e in edits]}
    # v1 aliases
    res |= {"flip": res["Cause"], "F": Cause, "F_keep": rate(f_keep), "state_kept": res["Iso_state"],
            "leak": {"rate": 1 - Io, "n": len(iso_other)}, "kl_country": se_["kl_country"],
            "kl_generic": se_["kl_generic"], "p_us": se_["p_us"]}
    return res


# ------------------------------------------------------------------ instances
def _vpath(S):
    return ROOT / "results" / S.name.split("/")[-1] / "ravel_validity.json"


def validity(S: Subject, cities: list[str], bs: int = 16) -> dict:
    """city -> {template key / 'state_q' / 'state_q_ho' / 'abbr': clean argmax correct}. Cached on disk."""
    p = _vpath(S)
    tab = json.loads(p.read_text()) if p.exists() else {}
    todo = [c for c in cities if c not in tab]
    if todo:
        for c in todo:
            tab[c] = {}
        for k, (tpl, cands, ans) in {
                **{tk: (TEMPLATES[tk], CAPS, lambda c: CAPITALS[CITY2STATE_EXT[c]]) for tk in DEV_T + HELDOUT_TEMPLATES},
                "state_q": (STATE_Q, STATES, lambda c: CITY2STATE_EXT[c]),
                "state_q_ho": (STATE_Q_HO, STATES, lambda c: CITY2STATE_EXT[c]),
                "abbr": (ABBR_T, ABBRS, lambda c: ABBR[CITY2STATE_EXT[c]])}.items():
            ok, enc = [], []
            for c in todo:
                try:
                    enc.append((c, S.encode(tpl, c)))
                except AssertionError:
                    tab[c][k] = False
            sc = _clean_scores(S, [e for _, e in enc], cands, bs)
            for (c, _), i in zip(enc, sc.argmax(1).tolist()):
                tab[c][k] = cands[i] == ans(c)
        p.write_text(json.dumps(tab, indent=0))
    return {c: tab[c] for c in cities}


def candidate_pairs(seed: int = 0, n: int | None = None) -> list[tuple[str, str]]:
    """Sources = states with extended city lists (enough held-out cities); one random target each."""
    rng = random.Random(f"ravel-pairs-{seed}")
    tg = study_states(min_cities=5)
    out = [(s, rng.choice([t for t in tg if t != s])) for s in sorted(s for s in tg if s in CITIES_EXTRA)]
    return out[:n] if n else out


def make_pair(S: Subject, source: str, target: str, seed: int = 0, bs: int = 16) -> dict | None:
    """Layer-independent part of a T2 v2 instance (splits, private pool, decoys). None if too few cities."""
    rng = random.Random(f"ravel-{seed}-{source}-{target}")
    V = validity(S, CITIES_EXT[source] + CITIES_EXT[target], bs)
    src_ok = [c for c in CITIES_EXT[source] if V[c]["fs1"] and V[c]["zs1"]]
    tgt_ok = [c for c in CITIES_EXT[target] if V[c]["fs1"] and V[c]["zs1"]]
    rng.shuffle(src_ok); rng.shuffle(tgt_ok)
    dev_s, dev_t = src_ok[:N_DEV], tgt_ok[:N_DEV]
    test = [c for c in CITIES_EXT[source] if c not in dev_s]
    items = [[c, tk] for c in test for tk in HELDOUT_TEMPLATES if V[c][tk]]
    test = sorted({c for c, _ in items})
    if len(dev_s) < N_DEV or len(dev_t) < N_DEV or len(test) < MIN_TEST:
        return None
    others = sorted(s for s in study_states(min_cities=5) if s not in (source, target))
    keep_states = rng.sample(others, 8)  # oracle keep-own-capital states: accessible, disjoint from the Iso pool
    rest = [s for s in others if s not in keep_states]
    private, pool = [], []
    for s in rest:
        cs = [c for c in CITIES[s]]
        rng.shuffle(cs)
        priv = cs[: max(1, len(cs) // 2)]  # half of every remaining state's cities are private
        private += priv
    rng.shuffle(private)
    pool = sorted(private[: int(0.6 * len(private))])  # Iso_other pool; the other 40% are decoys
    decoys = sorted(set(private) - set(pool))
    Vp = validity(S, pool + [c for s in keep_states for c in CITIES[s]], bs)
    leak_v1 = [[c, tk] for c in random.Random(f"{seed}-leak").sample(pool, 8) for tk in HELDOUT_TEMPLATES if Vp[c][tk]]
    keep_ok = {s: [c for c in CITIES[s] if Vp[c]["fs1"] and Vp[c]["zs1"]] for s in keep_states}
    keep_cities = {s: cs[:2] for s, cs in keep_ok.items()}
    keep_pool = {s: cs[:4] for s, cs in keep_ok.items()}
    return dict(source=source, target=target, seed=seed, dev_s=dev_s, dev_t=dev_t, test=test, items=items,
                keep_states=keep_states, keep_cities=keep_cities, keep_pool=keep_pool, iso_pool=pool, decoys=decoys, leak_v1=leak_v1,
                state_ok=[c for c in test if V[c]["state_q_ho"] and V[c]["abbr"]])


def make_instance(S: Subject, P: dict, layer: int) -> Instance:
    cons = CONS | {"layer": layer, "max_layer": layer}
    iid = re.sub(r"\s+", "_", f"T2_keepstate-{P['source']}-{P['target']}-L{layer}-s{P['seed']}")
    return Instance(iid, S.name, "ravel", P["source"], P["target"], CAPITALS[P["source"]], CAPITALS[P["target"]],
                    P["dev_s"], P["dev_t"], {t: TEMPLATES[t] for t in DEV_T}, cons, P["test"],
                    {t: TEMPLATES[t] for t in HELDOUT_TEMPLATES}, P["items"],
                    {"source": P["dev_s"], "target": P["dev_t"]}, leak_items=P["leak_v1"], task="ravel",
                    suite_task="T2_keepstate",
                    extra={"iso_pool": P["iso_pool"], "private_cities": P["iso_pool"] + P["decoys"],
                           "decoys": P["decoys"], "keep_states": P["keep_states"], "keep_cities": P["keep_cities"],
                           "keep_pool": P["keep_pool"], "layer": layer,
                           "grader": "v2", "extended_cities": True})


# ------------------------------------------------------------------ offline methods (no tool limits)
def _first(S, a: str) -> int:
    return S.cand_tokens([a])[0][0]


def meandiff(S: Subject, inst, L: int, bs: int = 16) -> torch.Tensor:
    """Plain mean-diff from the public dev cities: mean over fs1/zs1 of mean(target) - mean(source), city_last."""
    vs = []
    for tk in DEV_T:
        m = {k: S.resid([S.encode(TEMPLATES[tk], c) for c in cs], [L], "city_last", bs)[L].mean(0)
             for k, cs in (("s", inst.dev_source), ("t", inst.dev_target))}
        vs.append(m["t"] - m["s"])
    return torch.stack(vs).mean(0)


def _keep_cities(inst, seed: int) -> list[str]:
    ks = random.Random(f"{inst.id}-keep-{seed}").sample(inst.extra["keep_states"], 4)
    return [c for s in ks for c in inst.extra["keep_cities"][s]]


STATE_T_EXTRA = ["{city} is a city in the US state of", "Q: In which US state is {city}?\nA: It is in"]


def train_vector(S: Subject, inst, L: int, seed: int = 0, steps: int = 150, keep: bool = True,
                 max_norm: float | None = None, lr: float = 0.05, das: bool = False, all_keep: bool = False,
                 state_extra: bool = False, w_state: float = 1.0, w_other: float = 1.0):
    """Oracle (keep=True): target-capital NLL on dev cities (fs1+zs1) + keep-state NLL on STATE_Q (dev cities)
    + keep-own-capital NLL for dev cities of 4 other states (seeded choice from inst.extra['keep_states'],
    disjoint from the Iso pool). keep=False: naive gradient (target term only). Adam, relative lr as the tool.
    das=True: rank-1 DAS (interchange: set the coordinate on a learned direction u to the dev-target mean)."""
    torch.manual_seed(seed)
    dev = inst.dev_source
    E = [S.encode(TEMPLATES[tk], c) for tk in DEV_T for c in dev]
    groups = [(E, [_first(S, inst.target_capital)] * len(E), 1.0)]
    if keep:
        for tpl in [STATE_Q] + (STATE_T_EXTRA if state_extra else []):
            Es = [S.encode(tpl, c) for c in dev]
            groups.append((Es, [_first(S, inst.source)] * len(Es), w_state))
        kc = [c for cs in inst.extra["keep_cities"].values() for c in cs] if all_keep else _keep_cities(inst, seed)
        Ek = [S.encode(TEMPLATES[tk], c) for tk in DEV_T for c in kc]
        groups.append((Ek, [_first(S, CAPITALS[CITY2STATE_EXT[e.city]]) for e in Ek], w_other))
    rn = S.resid(E, [L], "city_last")[L].norm(dim=-1).mean().item()
    if das:
        Rt = torch.cat([S.resid([S.encode(TEMPLATES[tk], c) for c in inst.dev_target], [L], "city_last")[L]
                        for tk in DEV_T]).to(S.device)
        w = torch.randn(S.d_model, 1, device=S.device).requires_grad_(True)
        opt = torch.optim.Adam([w], lr=lr)
    else:
        w = (torch.randn(S.d_model, device=S.device) * 1e-3 * rn / math.sqrt(S.d_model)).requires_grad_(True)
        opt = torch.optim.Adam([w], lr=lr * rn / math.sqrt(S.d_model))
    for _ in range(steps):
        opt.zero_grad()
        if das:
            u = w / w.norm()
            iv = lambda es: Intervention(L, pos_rule(es, "city_last"), "swap", (u, (Rt @ u).mean(0)))
        else:
            iv = lambda es: Intervention(L, pos_rule(es, "city_last"), "add", w)
        loss = 0.0
        for es, ids, wt in groups:
            lp, *_ = S.forward(es, [iv(es)], grad=True)
            loss = loss - wt * lp[torch.arange(len(es)), torch.tensor(ids, device=S.device)].mean()
        loss.backward()
        opt.step()
        if max_norm is not None and not das:
            with torch.no_grad():
                w.mul_(min(1.0, max_norm / max(w.norm().item(), 1e-12)))
    if das:
        with torch.no_grad():
            u = (w / w.norm()).detach()
            return [(L, "city_last", (u.cpu(), (Rt @ u).mean(0).cpu()), "swap")]
    return [(L, "city_last", w.detach().float().cpu())]


def random_vec(S: Subject, norm: float, L: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(S.d_model, generator=g)
    return [(L, "city_last", v / v.norm() * norm)]


def train_gated(S: Subject, inst, L: int, seed: int = 0, steps: int = 150, lr: float = 0.05,
                all_keep: bool = True, state_extra: bool = False, w_keep: float = 1.0, templates=None,
                keep_sample: bool = False, ramp: tuple = (0.3, 0.6)):
    """Gated key->value oracle (conditional edit): x + clamp((k.x - b) / w, 0, 1) * v at city_last.
    k (fixed) = unit(mean source dev - mean other-state keep cities); the gate ramps from 0 at b (30% of the way
    from the keep mean to the source mean) to fully open at b + w (60%), so held-out source cities get the full v
    and other states' cities none. Only v is trained (a learned threshold overfit to the 4 dev cities). Same losses
    as train_vector."""
    torch.manual_seed(seed)
    dev = inst.dev_source
    E = [S.encode(TEMPLATES[tk], c) for tk in DEV_T for c in dev]
    groups = [(E, [_first(S, inst.target_capital)] * len(E), 1.0)]
    for tk in [t for t in (templates or []) if t not in DEV_T]:  # extra non-held-out capital wordings
        Ex = [S.encode(TEMPLATES[tk], c) for c in dev]
        groups.append((Ex, [_first(S, inst.target_capital)] * len(Ex), 1.0))
    for tpl in [STATE_Q] + (STATE_T_EXTRA if state_extra else []):
        Es = [S.encode(tpl, c) for c in dev]
        groups.append((Es, [_first(S, inst.source)] * len(Es), w_keep))
    if keep_sample:  # seed picks 2 keep cities per keep state from up to 4 accessible ones (seed variation)
        rk = random.Random(f"{inst.id}-gkeep-{seed}")
        kc = [c for cs in inst.extra["keep_pool"].values() for c in rk.sample(cs, min(2, len(cs)))]
    elif all_keep:
        kc = [c for cs in inst.extra["keep_cities"].values() for c in cs]
    else:
        kc = _keep_cities(inst, seed)
    # fit only on public dev cities + accessible keep cities: never held-out source cities or the Iso pool
    assert not (set(dev) | set(kc)) & (set(inst.test_cities) | set(inst.extra["private_cities"])), "held-out leak"
    Ek = [S.encode(TEMPLATES[tk], c) for tk in DEV_T for c in kc]
    groups.append((Ek, [_first(S, CAPITALS[CITY2STATE_EXT[e.city]]) for e in Ek], w_keep))
    Rs = S.resid(E, [L], "city_last")[L].float()
    Rk = S.resid(Ek, [L], "city_last")[L].float()
    k = Rs.mean(0) - Rk.mean(0); k = k / k.norm()
    ms, mk = (Rs @ k).mean().item(), (Rk @ k).mean().item()
    b, w = torch.tensor(mk + ramp[0] * (ms - mk)), torch.tensor((ramp[1] - ramp[0]) * (ms - mk))
    v = meandiff(S, inst, L).float() * 1e-3
    if keep_sample:
        v = v + torch.randn(S.d_model, generator=torch.Generator().manual_seed(seed)) * 1e-3 * v.norm() / math.sqrt(S.d_model)
    v = v.to(S.device).requires_grad_(True)
    rn = Rs.norm(dim=-1).mean().item()
    opt = torch.optim.Adam([v], lr=lr * rn / math.sqrt(S.d_model))
    kd, bd, wd = k.to(S.device), b.to(S.device), w.to(S.device)
    for _ in range(steps):
        opt.zero_grad()
        loss = 0.0
        for es, ids, wt in groups:
            lp, *_ = S.forward(es, [Intervention(L, pos_rule(es, "city_last"), "gate", (kd, bd, wd, v))], grad=True)
            loss = loss - wt * lp[torch.arange(len(es)), torch.tensor(ids, device=S.device)].mean()
        loss.backward()
        opt.step()
    return [(L, "city_last", (k.cpu(), b, w, v.detach().cpu()), "gate")]


def save_edits(path, edits):
    torch.save([(e[0], e[1], e[2], e[3] if len(e) > 3 else "add") for e in edits], path)


def load_edits(path):
    return [tuple(e) for e in torch.load(path)]
