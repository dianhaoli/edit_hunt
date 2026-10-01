"""EditHunt task suite: seven interpretability tasks on shared infrastructure (subject model, city/state data,
hooks, scoring). Each task = instance generator + allowed tool subset (+ optional tool-call budget) + grader +
reference solver that runs THROUGH the agent tools. An instance is kept only if the reference reward >= 0.5.

  T1_easy         flip SOURCE-state cities to the TARGET capital; no leakage penalty (edit tier "easy")
  T2_keepstate    flip the capital while "which state is X in?" still gives SOURCE (edit tier "hard"; v2 = the
                  RAVEL-style version at a fixed per-instance site, env/ravel.py)
  T3_consistency  make SOURCE cities act as TARGET-state cities everywhere, graded partly on hidden readouts
  T4_detective    a steering edit is planted; report its source state, target state and layer
  T5_erase        make which-of-K-states a city is in unrecoverable (behaviour + a fresh linear probe)
  T6_handoff      report the layer where city-token state edits stop working (the handoff)
  T7_minimal      flip >= 0.8 of held-out items with the smallest edit (relative to the residual norm)

Task descriptions state the goal and the reward, never the method.

CLI (generate + validate instances):
  python -m edithunt.env.tasks --task T3_consistency --n 8 [--seed 0] [--model Qwen/Qwen2.5-1.5B]
"""
from __future__ import annotations

import json
import math
import random
import re

import torch

from ..common import ROOT
from ..data import (ABBR, CAPITALS, CITIES, CITY2STATE, READOUTS, STATE_Q, STATE_Q_HO, STATES, TEMPLATES,
                    study_states)
from ..metrics import rate
from ..model import Subject, pos_rule
from .grader import CAPS, build_ivs, edit_summary, grade_edit, parse, predict, side_effects
from .instance import Instance, generate, load_validity, reference_vector, save_instance, validity

EDIT_TOOLS = ["describe_task", "run_prompts", "logit_lens", "cache_mean", "vec_op", "vec_info",
              "eval_intervention", "optimize_vector", "submit"]
REPORT_TOOLS = ["describe_task", "run_prompts", "logit_lens", "cache_mean", "vec_op", "vec_info",
                "eval_intervention", "submit_report"]
# Phase 1 handoff layers (first layer after the mean-diff band; FINDINGS §1)
HANDOFF = {"Qwen/Qwen2.5-1.5B": 22, "Qwen/Qwen2.5-3B": 31, "google/gemma-2-2b": 18, "Qwen/Qwen2.5-7B": 22}

SPECS = {
    "T1_easy": dict(task="edit", tier="easy", tools=EDIT_TOOLS),
    "T2_keepstate": dict(task="edit", tier="hard", tools=EDIT_TOOLS),
    "T3_consistency": dict(task="consistency", tier="consistency", tools=EDIT_TOOLS),
    "T4_detective": dict(task="detective", call_budget=20,
                         tools=["describe_task", "run_prompts", "logit_lens", "cache_mean", "act_diff", "vec_op",
                                "vec_info", "submit_report"],
                         blackbox_tools=["describe_task", "run_prompts", "submit_report"]),
    "T5_erase": dict(task="erase", tools=EDIT_TOOLS),
    "T6_handoff": dict(task="handoff", call_budget=25, tools=REPORT_TOOLS,
                       blackbox_tools=["describe_task", "run_prompts", "submit_report"]),
    "T7_minimal": dict(task="minimal", tier="minimal", tools=EDIT_TOOLS),
}


def spec_of(inst) -> dict:
    return SPECS[inst.suite_task] if getattr(inst, "suite_task", "") else SPECS["T1_easy"]


# ------------------------------------------------------------------ descriptions (what the agent sees)
_POSITIONS = {"city_last": "last token of the city name", "city_all": "all city-name tokens",
              "final": "last prompt token"}
_KLTXT = ("KL_mean = average of (a) the binary KL of the model's probability that a held-out city is in the United "
          "States (on a country prompt) and (b) the KL divergence (clean || edited) of the next-token distribution on "
          "unrelated sentences, with the edit applied at a comparable token.")


def describe(env) -> dict:
    i, S = env.inst, env.S
    subj = {"model": i.model, "n_layers": S.n_layers, "d_model": S.d_model}
    base = {"subject": subj, "registers": sorted(env.regs), "budget_total": env.budget,
            "constraints": i.constraints}
    if env.call_budget is not None:
        base["tool_call_budget"] = env.call_budget
    t = i.task
    if t in ("edit", "consistency", "minimal", "ravel"):
        base |= {"source": i.source, "target": i.target, "source_capital": i.source_capital,
                 "target_capital": i.target_capital, "dev_source_cities": i.dev_source,
                 "dev_target_cities": i.dev_target, "templates": env.templates, "positions": _POSITIONS}
    if t == "edit":
        keep = (" An item only counts if the model's answer to 'which US state is <city> in?' is unchanged by your "
                "edit." if i.constraints.get("preserve_state") else "")
        base["task"] = (f"Make prompts about cities in {i.source} answer the capital of {i.target} ({i.target_capital}) "
                        f"instead of {i.source_capital}, on held-out {i.source} cities and held-out prompt wordings.{keep}")
        base["reward"] = ("F = fraction of held-out (city, wording) items whose top answer among the 50 capitals is the "
                          "target capital" + (" (and whose state answer is unchanged)" if keep else "") + ". " + _KLTXT +
                          " leak = fraction of held-out cities from other states whose top capital changes. "
                          "reward = F * (1 - leak_weight * leak) * (1 - min(1, KL_mean / kl_budget)).")
    elif t == "ravel":
        c = i.constraints
        base["task"] = (f"Make held-out cities in {i.source} answer the capital of {i.target} ({i.target_capital}) instead "
                        f"of {i.source_capital}, on held-out prompt wordings, while the model still knows these cities "
                        f"are in {i.source} (a held-out state question and hidden state-level readouts that you cannot "
                        f"see or query) and cities of other states keep their own capitals. Submit one additive vector "
                        f"at the fixed intervention site: after layer {c['layer']}, at the last token of the city name. "
                        "Some cities and prompt wordings are private to the grader; tools answer 'not accessible' for them.")
        if c.get("allow_scale_from"):
            base["edit_format"] = ("Additive edit {register, layer, position, scale}, optionally with scale_from {key: "
                                   "register, lo, hi}: the edited residual h then gets register * scale * "
                                   "clamp((dot(key, h) - lo) / (hi - lo), 0, 1). The project tool reports dot(register, "
                                   "h) for given cities.")
        base["reward"] = ("Cause = fraction of held-out (source city, wording) items whose top answer among the 50 capitals "
                          "is the target capital. Iso_state = fraction of held-out source cities whose held-out state "
                          "question AND hidden state readout still give " + i.source + ". Iso_other = fraction of "
                          "private cities of other states (sampled at grading time, held-out wordings) whose top capital "
                          "is unchanged. Iso = (Iso_state + Iso_other) / 2. Disentangle = (Cause + Iso) / 2. "
                          "KL_mean = average of (a) the binary KL of the model's probability that a held-out city is in "
                          "the United States (country prompt), counted only where your edit lowers that probability, over "
                          "held-out cities whose name is not also a well-known non-US place, and (b) the KL divergence "
                          "(clean || edited) of the next-token distribution on unrelated sentences, with the edit applied "
                          "at a comparable token."
                          " reward = Disentangle * (1 - min(1, KL_mean / kl_budget)).")
    elif t == "consistency":
        base["task"] = (f"Make the model treat held-out cities in {i.source} as if they were in {i.target}: the capital "
                        f"question, the question 'which US state is <city> in?', and further hidden state-level "
                        f"readouts that you cannot see or query should all give the {i.target} answer.")
        base["reward"] = ("F_cap, F_state, F_hidden = fraction of held-out items answered with the target state's answer "
                          "on the capital question (held-out wordings), on the state question (two wordings, one "
                          "held out), and on the hidden readouts. " + _KLTXT +
                          " reward = (F_cap * F_state * F_hidden)^(1/3) * (1 - min(1, KL_mean / kl_budget)).")
    elif t == "minimal":
        base["task"] = (f"Make prompts about held-out {i.source} cities answer {i.target_capital} (the capital of "
                        f"{i.target}) with the smallest possible edit.")
        base["reward"] = (f"Passed if F (held-out flip rate to {i.target_capital}, held-out wordings) >= "
                          f"{i.extra['min_flip']} and KL_mean <= kl_budget. " + _KLTXT + " r = |scale * vector| / "
                          "(mean norm of the residual stream after that layer at the city's last token). "
                          "reward = min(1, r_ref / r) if passed else 0, where r_ref is a private reference edit's r.")
    elif t == "erase":
        pe = i.public_extra
        base |= {"states": pe["states"], "dev_cities": pe["dev_cities"], "templates": env.templates,
                 "positions": _POSITIONS}
        K = len(pe["states"])
        base["task"] = (f"Make it impossible to recover which of these {K} states ({', '.join(pe['states'])}) a city is "
                        "in, for held-out cities of these states: both from the model's behaviour (asked which state "
                        "the city is in, choosing among these states) and from its internal activations (a fresh linear "
                        "probe that the grader trains AFTER your edit on held-out cities, reading the residual stream "
                        f"at the city's last token after a private layer <= {i.constraints['max_layer']}). Keep the "
                        "city recognised as a US city and leave unrelated text unchanged.")
        base["edit_format"] = ("Edits are orthogonal projections: each removes the span of `basis` (1.."
                               f"{i.constraints['max_basis']} registers, orthonormalised by the grader) from the residual "
                               "stream after `layer` at `position`: h <- h - P(h - center), center optional (register). "
                               "Use submit(edits=[{basis, center?, layer, position}]); eval_intervention accepts the "
                               "same edit format (answers='states' ranks the 50 states).")
        base["reward"] = ("behaviour_erasure = clip((acc_clean - acc_edited) / (acc_clean - chance), 0, 1) with acc = "
                          "accuracy on the state question (two wordings, one held out, argmax over these states); "
                          "probe_erasure = the same for the probe's leave-one-city-out accuracy. " + _KLTXT +
                          " reward = 0.5 * (behaviour_erasure + probe_erasure) * (1 - min(1, KL_mean / kl_budget)).")
    elif t == "detective":
        base |= {"templates": env.templates, "positions": _POSITIONS,
                 "known_cities": {s: CITIES[s] for s in STATES}}
        clean = "clean_access" in i.constraints.get("features", [])
        base["task"] = ("Someone planted an edit in this model: a fixed vector is added to the residual stream after one "
                        "layer, at the last token of the name of every city of one US state (the SOURCE state) listed in "
                        "known_cities (the edit fires only on those city names, wherever they appear in a prompt). It was "
                        "made to push those cities towards another state (the TARGET). Find the source state, the target "
                        "state and the layer. "
                        + ("You can query the planted model (default) and the clean model (model='clean')."
                           if clean else "You can only query the planted model."))
        base["reward"] = ("reward = 0.4 * [source correct] + 0.4 * [target correct] + 0.2 * [|layer error| <= 2]. "
                          "Submit with submit_report(report={source, target, layer}).")
    elif t == "handoff":
        pe = i.public_extra
        base |= {"example_pair": pe, "templates": env.templates, "positions": _POSITIONS}
        base["task"] = ("Adding (mean residual of target-state cities - mean residual of source-state cities), taken and "
                        "added at the city's last token after layer L, makes source-state cities answer the target "
                        "state's capital when L is in a band of early/middle layers. After some layer this stops "
                        "working (the state has already been read out of the city token). Report the handoff layer: "
                        "the first layer after the band at which this edit flips fewer than half of the source cities "
                        "(few-shot capital template fs1, averaged over many state pairs; an example pair is given).")
        base["reward"] = "reward = 1 if |reported layer - true handoff| <= 1 else 0. Submit with submit_report(report={layer})."
    return base


def check_report(inst, report) -> str | None:
    if not isinstance(report, dict):
        return "report must be an object"
    if inst.task == "detective":
        for k in ("source", "target"):
            if report.get(k) not in STATES:
                return f"{k} must be a US state name"
    elif inst.task != "handoff":
        return "this task takes submit, not submit_report"
    L = report.get("layer")
    if not isinstance(L, int) or isinstance(L, bool):
        return "layer must be an integer"
    return None


def grade_report(inst, sub: dict) -> dict:
    r = sub.get("report") if isinstance(sub, dict) else None
    if r is None or check_report(inst, r):
        return {"valid": False, "reason": "missing or malformed report", "reward": 0.0}
    x = inst.extra
    if inst.task == "detective":
        comp = {"source": r["source"] == inst.source, "target": r["target"] == inst.target,
                "layer": abs(r["layer"] - x["plant"]["layer"]) <= 2}
        rw = 0.4 * comp["source"] + 0.4 * comp["target"] + 0.2 * comp["layer"]
        return {"valid": True, "reward": rw, "components": comp, "report": r,
                "truth": {"source": inst.source, "target": inst.target, "layer": x["plant"]["layer"]}}
    err = r["layer"] - x["handoff"]
    return {"valid": True, "reward": float(abs(err) <= 1), "components": {"layer_error": err}, "report": r,
            "truth": {"layer": x["handoff"]}}


# ------------------------------------------------------------------ graders
def _kl_factor(km: float, budget: float) -> float:
    return 1.0 - min(1.0, km / budget)


def grade_consistency(S: Subject, inst, sub: dict, bs: int = 16, enforce: bool = True) -> dict:
    edits, why = parse(S, inst, sub, enforce)
    if edits is None:
        return {"valid": False, "reason": why, "reward": 0.0}
    T = inst.test_templates
    items = [tuple(x) for x in inst.test_items]
    cap_encs, flips = [], []
    for tk in T:
        cs = [c for c, t in items if t == tk]
        if cs:
            e = [S.encode(T[tk], c) for c in cs]
            cap_encs += e
            flips += [p == inst.target_capital for p in predict(S, e, CAPS, build_ivs(edits, e), bs)]
    cities = sorted({c for c, _ in items})
    st = []
    for tpl in (STATE_Q, STATE_Q_HO):
        e = [S.encode(tpl, c) for c in cities]
        p0, p1 = predict(S, e, STATES, (), bs), predict(S, e, STATES, build_ivs(edits, e), bs)
        st += [b == inst.target for a, b in zip(p0, p1) if a == inst.source]
    hid, per = [], {}
    for key in sorted({k for _, k in inst.extra["hidden_items"]}):
        tpl, ans = READOUTS[key]
        cs = [c for c, k in inst.extra["hidden_items"] if k == key]
        cands = [ans[s] for s in STATES]
        e = [S.encode(tpl, c) for c in cs]
        f = [p == ans[inst.target] for p in predict(S, e, cands, build_ivs(edits, e), bs)]
        per[key] = rate(f)
        hid += f
    res = {"valid": True, "F_cap": rate(flips), "F_state": rate(st), "F_hidden": rate(hid), "hidden_by_readout": per}
    res |= side_effects(S, edits, cities, cap_encs, bs)
    Fs = [res[k]["rate"] if res[k]["n"] else 0.0 for k in ("F_cap", "F_state", "F_hidden")]
    res["F"] = (Fs[0] * Fs[1] * Fs[2]) ** (1 / 3)
    res["reward"] = res["F"] * _kl_factor(res["kl_mean"], inst.constraints["kl_budget"])
    res["edits"] = edit_summary(edits)
    return res


def grade_minimal(S: Subject, inst, sub: dict, bs: int = 16, enforce: bool = True) -> dict:
    g = grade_edit(S, inst, sub, bs, enforce=enforce)
    if not g.get("valid"):
        return g
    x = inst.extra
    (e,) = g["edits"]
    r = e["norm"] / x["rho"][e["layer"]]
    passed = g["flip"]["rate"] >= x["min_flip"] and g["kl_mean"] <= inst.constraints["kl_budget"]
    g |= {"r": r, "r_ref": x.get("r_ref"), "passed": passed,
          "reward": (min(1.0, x["r_ref"] / r) if x.get("r_ref") else 1.0) if passed else 0.0}
    return g


def _probe_acc(X: torch.Tensor, y: torch.Tensor, groups: list[str]) -> float:
    """Leave-one-city-out accuracy of a linear (kernel ridge, one-hot targets) classifier on standardized X."""
    K = int(y.max()) + 1
    correct = 0
    for g in sorted(set(groups)):
        te = torch.tensor([x == g for x in groups])
        Xtr, Xte = X[~te], X[te]
        mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-4)
        A, B = (Xtr - mu) / sd, (Xte - mu) / sd
        G = A @ A.T
        lam = 0.1 * G.diagonal().mean()
        Y = torch.nn.functional.one_hot(y[~te], K).double() - 1.0 / K
        alpha = torch.linalg.solve(G.double() + lam.double() * torch.eye(len(A), dtype=torch.float64), Y)
        pred = ((B @ A.T).double() @ alpha).argmax(1)
        correct += int((pred == y[te]).sum())
    return correct / len(y)


def grade_erase(S: Subject, inst, sub: dict, bs: int = 16, enforce: bool = True) -> dict:
    edits, why = parse(S, inst, sub, enforce)
    if edits is None:
        return {"valid": False, "reason": why, "reward": 0.0}
    x = inst.extra
    states = x["states"]
    cities = [c for s in states for c in x["test_by_state"][s]]
    y = torch.tensor([states.index(CITY2STATE[c]) for c in cities])
    res = {"valid": True}
    acc = {}
    for tag, ivf in (("clean", lambda e: ()), ("edited", lambda e: build_ivs(edits, e))):
        ok = []
        for tpl in (STATE_Q, STATE_Q_HO):
            e = [S.encode(tpl, c) for c in cities]
            ok += [p == CITY2STATE[c] for c, p in zip(cities, predict(S, e, states, ivf(e), bs))]
        acc[tag] = sum(ok) / len(ok)
        feats, grp, ys = [], [], []
        P = x["probe_layer"]
        for tk in x["probe_templates"]:
            e = [S.encode(TEMPLATES[tk] if tk in TEMPLATES else {"state_q": STATE_Q}[tk], c) for c in cities]
            _, st, _, _ = S.forward(e, ivf(e), {P: pos_rule(e, "city_last")})
            feats.append(st.captures[P][:, 0].cpu().float())
            grp += cities; ys.append(y)
        acc["probe_" + tag] = _probe_acc(torch.cat(feats), torch.cat(ys), grp)
    chance = max((y == k).float().mean().item() for k in range(len(states)))
    beh = max(0.0, min(1.0, (acc["clean"] - acc["edited"]) / max(1e-6, acc["clean"] - chance)))
    prb = max(0.0, min(1.0, (acc["probe_clean"] - acc["probe_edited"]) / max(1e-6, acc["probe_clean"] - chance)))
    enc_ref = [S.encode(TEMPLATES["fs1"], c) for c in cities]
    res |= {"acc": acc, "chance": chance, "behaviour_erasure": beh, "probe_erasure": prb, "n_cities": len(cities)}
    res |= side_effects(S, edits, cities, enc_ref, bs)
    res["F"] = 0.5 * (beh + prb)
    res["reward"] = res["F"] * _kl_factor(res["kl_mean"], inst.constraints["kl_budget"])
    res["edits"] = edit_summary(edits)
    return res


def _grade_ravel(S, inst, sub, bs=16, enforce=True):
    from .ravel import grade_ravel
    return grade_ravel(S, inst, sub, bs, enforce)


GRADERS = {"consistency": grade_consistency, "minimal": grade_minimal, "erase": grade_erase, "ravel": _grade_ravel}


# ------------------------------------------------------------------ reference solvers (through the tools)
def _call(env, tool: str, **kw) -> dict:
    out = json.loads(env.call(tool, kw))
    if "error" in out:
        raise RuntimeError(f"{tool}: {out['error']}")
    return out


def ref_meandiff(env, **_):
    from .baselines import agent_meandiff
    agent_meandiff(env)


def _accessible(env, cities: list[str]) -> list[str]:
    from .tools import _letters
    return [c for c in cities if _letters(c) not in env._held]


def ref_keepstate(env, seed: int = 0, steps: int = 50, **_):
    """Hop-2 recipe with generic extra_examples: raise the target capital on dev cities; keep other states' cities
    on their own capital and the source cities' state answer unchanged; norm capped at the mean-diff norm."""
    d = _call(env, "describe_task")
    c = d["constraints"]
    L = c.get("layer", c["max_layer"])  # T2 v2: the fixed intervention layer
    tpls = [k for k in d["templates"] if k not in ("state_q", "country_q")]
    src = d["dev_source_cities"]
    _call(env, "cache_mean", name="s", cities=src, template=tpls[0], layer=L)
    _call(env, "cache_mean", name="t", cities=d["dev_target_cities"], template=tpls[0], layer=L)
    nrm = _call(env, "vec_op", name="md", expr="t - s")["norm"]
    rng = random.Random(f"{seed}-{d['source']}-{d['target']}")
    others = [s for s in study_states(min_cities=5) if s not in (d["source"], d["target"])]
    keep = [x for st in rng.sample(others, 6) for x in _accessible(env, CITIES[st])[:1]][:4]
    ex = [{"template": t, "city": k, "answer": CAPITALS[CITY2STATE[k]], "weight": 1.0} for k in keep for t in tpls]
    ex += [{"template": "state_q", "city": s, "answer": d["source"], "weight": 1.0} for s in src]
    _call(env, "optimize_vector", name="g", layer=L, position="city_last", dev_cities=src, templates=tpls,
          steps=steps, max_norm=nrm, extra_examples=ex[:16])
    _call(env, "submit", edits=[{"register": "g", "layer": L, "position": "city_last", "scale": 1.0}])


def ref_ravel_gated(env, seed: int = 0, steps: int = 150, ramp=(0.3, 0.6), **_):
    """T2 v2 reference through the agent tools: key = normalize(mean dev source - mean accessible other-state cities)
    (fs1 + zs1, cache_mean/vec_op), gate ramp from `project` means, push v from optimize_vector with that scale_from
    (target capital on dev cities, keep-state on state_q, keep-own-capital on other cities), then submit."""
    d = _call(env, "describe_task")
    L = d["constraints"]["layer"]
    tpls = [k for k in d["templates"] if k not in ("state_q", "country_q")]
    src = d["dev_source_cities"]
    rng = random.Random(f"{seed}-{d['source']}-{d['target']}")
    others = [s for s in study_states(min_cities=5) if s not in (d["source"], d["target"])]
    oth = [x for st in rng.sample(others, 12) for x in _accessible(env, CITIES[st])[:1]][:8]
    for t in tpls:
        _call(env, "cache_mean", name=f"s_{t}", cities=src, template=t, layer=L)
        _call(env, "cache_mean", name=f"o_{t}", cities=oth, template=t, layer=L)
    _call(env, "vec_op", name="k", expr="normalize(" + " + ".join(f"s_{t}" for t in tpls) + " - (" +
          " + ".join(f"o_{t}" for t in tpls) + "))")
    ms = sum(_call(env, "project", name="k", cities=src, template=t, layer=L)["mean"] for t in tpls) / len(tpls)
    mo = sum(_call(env, "project", name="k", cities=oth, template=t, layer=L)["mean"] for t in tpls) / len(tpls)
    gate = {"key": "k", "lo": mo + ramp[0] * (ms - mo), "hi": mo + ramp[1] * (ms - mo)}
    ex = [{"template": "state_q", "city": c, "answer": d["source"], "weight": 1.0} for c in src]
    ex += [{"template": t, "city": c, "answer": CAPITALS[CITY2STATE[c]], "weight": 1.0} for c in oth[:6] for t in tpls]
    _call(env, "optimize_vector", name="v", layer=L, position="city_last", dev_cities=src, templates=tpls,
          steps=steps, extra_examples=ex[:16], scale_from=gate)
    _call(env, "submit", edits=[{"register": "v", "layer": L, "position": "city_last", "scale": 1.0,
                                 "scale_from": gate}])


def ref_minimal(env, **_):
    """Mean-diff and gradient directions at 3 layers; per direction, bisect the smallest scale that flips every dev
    prompt (both dev templates); submit the smallest relative norm."""
    d = _call(env, "describe_task")
    top = d["constraints"]["max_layer"]
    tpls = [k for k in d["templates"] if k not in ("state_q", "country_q")]
    src, tgt = d["dev_source_cities"], d["dev_target_cities"]
    best = None
    for L in sorted({top, (2 * top) // 3, top // 3}):
        _call(env, "cache_mean", name=f"s{L}", cities=src, template=tpls[0], layer=L)
        _call(env, "cache_mean", name=f"t{L}", cities=tgt, template=tpls[0], layer=L)
        _call(env, "vec_op", name=f"md{L}", expr=f"normalize(t{L} - s{L})")
        rho = _call(env, "vec_op", name="tmp", expr=f"s{L}")["norm"]
        _call(env, "optimize_vector", name=f"g{L}", layer=L, dev_cities=src, templates=tpls, steps=40, kl_weight=1.0)
        _call(env, "vec_op", name=f"gd{L}", expr=f"normalize(g{L})")
        for u in (f"md{L}", f"gd{L}"):
            def ok(sc):
                return all(_call(env, "eval_intervention", template=t, cities=src, vector=u, layer=L, scale=sc,
                                 n_generic=0)["flip_rate"] >= 1.0 for t in tpls)
            hi = rho
            if not ok(hi):
                continue
            lo = 0.0
            for _ in range(6):
                mid = 0.5 * (lo + hi)
                lo, hi = (lo, mid) if ok(mid) else (mid, hi)
            sc = hi * 1.1  # margin for held-out cities / wordings
            if best is None or sc / rho < best[0]:
                best = (sc / rho, u, L, sc)
    if best is None:
        raise RuntimeError("no direction flips the dev prompts")
    _, u, L, sc = best
    _call(env, "submit", edits=[{"register": u, "layer": L, "position": "city_last", "scale": sc}])


def ref_erase(env, layers: list[int] | None = None, **_):
    """LEACE-style: per layer, project out the span of (state mean - mean of state means) around that center,
    estimated from dev cities on two templates."""
    d = _call(env, "describe_task")
    c = d["constraints"]
    states = d["states"]
    top = c["max_layer"]
    layers = layers or [x for x in (top // 4, top // 2, (3 * top) // 4, top)][: c["max_rank"]]
    for s in states:
        for t in ("fs1", "state_q"):
            _call(env, "cache_mean", name=f"m_{t}_{s.replace(' ', '_')}", cities=d["dev_cities"][s], template=t,
                  layers=layers, position="city_all")
    edits = []
    for L in layers:
        names = []
        for t in ("fs1", "state_q"):
            ms = [f"m_{t}_{s.replace(' ', '_')}_L{L}" for s in states]
            _call(env, "vec_op", name=f"c_{t}_L{L}", expr="(" + " + ".join(ms) + f") / {len(ms)}")
            for i, m in enumerate(ms[:-1]):
                _call(env, "vec_op", name=f"b_{t}_{i}_L{L}", expr=f"{m} - c_{t}_L{L}")
                names.append(f"b_{t}_{i}_L{L}")
        _call(env, "vec_op", name=f"c_L{L}", expr=f"(c_fs1_L{L} + c_state_q_L{L}) / 2")
        edits.append({"basis": names[: c["max_basis"]], "center": f"c_L{L}", "layer": L, "position": "city_all"})
    _call(env, "submit", edits=edits)


def ref_detective(env, **_):
    """Scan one city per state for a planted-vs-clean activation difference at a late layer (-> source), scan layers
    for that city (-> layer), compare planted vs clean capital scores on source cities (-> target)."""
    d = _call(env, "describe_task")
    n = d["subject"]["n_layers"]
    probe_L = min(n - 1, 20)
    reps = {s: CITIES[s][0] for s in STATES}
    cities = list(reps.values())
    diffs = {}
    for k in range(0, len(cities), 16):
        r = _call(env, "act_diff", template="zs1", cities=cities[k:k + 16], layer=probe_L)
        diffs |= {c: rel for c, _, rel in r["diff_by_layer"][str(probe_L)]}
    src_city = max(diffs, key=diffs.get)
    src = CITY2STATE[src_city]
    r = _call(env, "act_diff", template="zs1", cities=[src_city], layers=list(range(probe_L + 1)))
    L = min(int(k) for k, v in r["diff_by_layer"].items() if v[0][2] > 1e-3)
    cs = CITIES[src][:8]
    # target = the capital the planted model most often answers (top-1) on source cities instead of the source capital
    # (2026-09-30 fix: summed log-prob gains picked a neighbouring capital whose score rose most on 3/8 instances)
    o = _call(env, "run_prompts", template="fs1", cities=cs, top_k=50, model="planted")
    votes = {}
    for res in o["results"]:
        top = max(res["top_capitals"], key=lambda cv: cv[1])[0]
        if top != CAPITALS[src]:
            votes[top] = votes.get(top, 0) + 1
    tcap = max(votes, key=votes.get)
    tgt = next(s for s, cp in CAPITALS.items() if cp == tcap)
    _call(env, "submit_report", report={"source": src, "target": tgt, "layer": L})


def ref_handoff(env, **_):
    """Mean-diff at the city's last token; bisect the first layer after the band where the dev flip rate < 0.5."""
    d = _call(env, "describe_task")
    n = d["subject"]["n_layers"]
    pe = d["example_pair"]
    src, tgt = pe["source_cities"], pe["target_cities"]
    Ls = list(range(n))
    _call(env, "cache_mean", name="s", cities=src, template="fs1", layers=Ls)
    _call(env, "cache_mean", name="t", cities=tgt, template="fs1", layers=Ls)
    cache = {}

    def flip(L):
        if L not in cache:
            _call(env, "vec_op", name=f"d{L}", expr=f"t_L{L} - s_L{L}")
            cache[L] = _call(env, "eval_intervention", template="fs1", cities=src, vector=f"d{L}", layer=L,
                             n_generic=0)["flip_rate"]
        return cache[L]

    lo, hi = n // 3, n - 1  # assume flip(lo) >= 0.5 > flip(hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if flip(mid) >= 0.5 else (lo, mid)
    _call(env, "submit_report", report={"layer": hi})


REFERENCES = {"T1_easy": ref_meandiff, "T2_keepstate": ref_keepstate, "T3_consistency": ref_meandiff,
              "T4_detective": ref_detective, "T5_erase": ref_erase, "T6_handoff": ref_handoff,
              "T7_minimal": ref_minimal}


def make_env(S: Subject, inst, bs: int = 16, blackbox: bool = False):
    from .tools import ToolEnv
    sp = spec_of(inst)
    tools = sp.get("blackbox_tools") if blackbox else sp["tools"]
    if getattr(inst, "task", "") == "ravel" and not blackbox:
        tools = list(tools) + ["project"]  # T2 v2: read-only projection tool (other tasks unchanged)
    if blackbox:
        import copy
        inst = copy.deepcopy(inst)
        inst.constraints["features"] = [f for f in inst.constraints.get("features", []) if f != "clean_access"]
    return ToolEnv(S, inst, bs, tools=tools, call_budget=sp.get("call_budget"))


def run_reference(S: Subject, inst, bs: int = 16, **kw) -> dict:
    from .grader import grade
    env = make_env(S, inst, bs)
    err = None
    try:
        (ref_ravel_gated if getattr(inst, "task", "") == "ravel" else REFERENCES[inst.suite_task])(env, **kw)
    except RuntimeError as e:
        err = str(e)
    g = grade(S, inst, env.submission, bs)
    return {"grade": g, "error": err, "calls": env.calls, "budget_used": env.used, "submission": env.submission}


# ------------------------------------------------------------------ instance generators
def _valid_cities(S, tpl: str, cities: list[str], answer_fn, cands, bs=16) -> list[str]:
    e = [S.encode(tpl, c) for c in cities]
    return [c for c, p in zip(cities, predict(S, e, cands, (), bs)) if p == answer_fn(c)]


def gen_edit(S, name: str, n: int, seed: int, handoff: int, bs: int = 16, max_tries: int = 60) -> list:
    """T1/T2/T3/T7: edit instances (mean-diff prefilter), then the task's reference must pass."""
    sp = SPECS[name]
    valid = load_validity(S.name)
    ro = json.loads((ROOT / "results" / S.name.split("/")[-1] / "phase7_readouts.json").read_text())["result"]["valid"] \
        if sp["task"] == "consistency" else None
    rng = random.Random(f"{seed}-{name}")
    states = study_states(min_cities=5)
    out, tries = [], 0
    while len(out) < n and tries < max_tries:
        tries += 1
        src, tgt = rng.sample(states, 2)
        cand = generate(S, sp["tier"], 1, seed, handoff, valid, pairs=[(src, tgt)], max_tries=1, bs=bs)
        if not cand:
            continue
        inst = cand[0]
        inst.task, inst.suite_task = sp["task"], name
        inst.id = f"{name}-{src}-{tgt}-s{seed}".replace(" ", "_")
        if sp["task"] == "consistency":
            inst.extra["hidden_items"] = [[c, k] for c in inst.test_cities for k in ("abbr", "abbr_addr") if ro[k].get(c)]
        if sp["task"] == "minimal":
            pool = [S.encode(TEMPLATES["fs1"], c) for c in inst.ref_pool["source"] + inst.ref_pool["target"]]
            R = S.resid(pool, list(range(S.n_layers)), "city_last", bs)
            inst.extra |= {"rho": [R[L].norm(dim=-1).mean().item() for L in range(S.n_layers)], "min_flip": 0.8}
        r = run_reference(S, inst, bs, seed=seed)
        g = r["grade"]
        if sp["task"] == "minimal" and g.get("valid") and g.get("passed"):
            inst.extra["r_ref"] = g["r"]
            g["reward"] = 1.0
        ok = g.get("reward", 0) >= 0.5
        inst.extra["reference"] = {"reward": g.get("reward"), "F": g.get("F"), "kl_mean": g.get("kl_mean"),
                                   "error": r["error"], "edits": g.get("edits")}
        print(f"{inst.id}: reference reward {g.get('reward', 0):.3f} F {g.get('F')} {r['error'] or ''} -> "
              f"{'keep' if ok else 'drop'}", flush=True)
        if ok:
            out.append(inst)
    return out


def gen_erase(S, n: int, seed: int, handoff: int, K: int = 4, bs: int = 16, max_tries: int = 30) -> list:
    rng = random.Random(f"{seed}-erase")
    states = study_states(min_cities=7)
    out, tries = [], 0
    while len(out) < n and tries < max_tries:
        tries += 1
        sts = rng.sample(states, K)
        dev, test = {}, {}
        for s in sts:
            ok = _valid_cities(S, STATE_Q, CITIES[s], lambda c: CITY2STATE[c], STATES, bs)
            rng.shuffle(ok)
            dev[s], test[s] = ok[:4], ok[4:]
        if any(len(test[s]) < 2 for s in sts):
            continue
        cons = {"max_layer": handoff - 1, "positions": ["city_last", "city_all"], "max_rank": 4, "max_basis": 8,
                "max_norm": None, "kl_budget": 1.0, "fwd_budget": 4000, "edit_kind": "proj", "preserve_state": False}
        iid = f"T5_erase-{'-'.join(sts)}-s{seed}".replace(" ", "_")
        inst = Instance(iid, S.name, "erase", "", "", "", "", [], [], {t: TEMPLATES[t] for t in ("fs1", "zs1")}, cons,
                        [c for s in sts for c in test[s]], {}, [], {}, task="erase", suite_task="T5_erase",
                        extra={"states": sts, "test_by_state": test, "probe_layer": handoff - 1,
                               "probe_templates": ["fs1", "zs1", "ho_fs", "ho_zs"]},
                        public_extra={"states": sts, "dev_cities": dev})
        r = run_reference(S, inst, bs)
        g = r["grade"]
        ok = g.get("reward", 0) >= 0.5
        inst.extra["reference"] = {k: g.get(k) for k in ("reward", "acc", "behaviour_erasure", "probe_erasure", "kl_mean")}
        inst.extra["reference"]["error"] = r["error"]
        print(f"{iid}: reference reward {g.get('reward', 0):.3f} acc {g.get('acc')} kl {g.get('kl_mean')} "
              f"{r['error'] or ''} -> {'keep' if ok else 'drop'}", flush=True)
        if ok:
            out.append(inst)
    return out


def gen_detective(S, n: int, seed: int, handoff: int, scale: float = 1.0, band=(4, 16), bs: int = 16,
                  max_tries: int = 30) -> list:
    valid = load_validity(S.name)
    rng = random.Random(f"{seed}-detective")
    states = study_states(min_cities=5)
    out, tries = [], 0
    while len(out) < n and tries < max_tries:
        tries += 1
        src, tgt = rng.sample(states, 2)
        L = rng.randint(*band)
        cand = generate(S, "easy", 1, seed, handoff, valid, pairs=[(src, tgt)], validate=False, max_tries=1, bs=bs)
        if not cand:
            continue
        base = cand[0]
        v = scale * reference_vector(S, base, L, bs)
        cons = {"max_layer": S.n_layers - 1, "positions": [], "max_rank": 0, "max_norm": None, "kl_budget": 1.0,
                "fwd_budget": 3000, "features": ["clean_access"]}
        iid = f"T4_detective-{src}-{tgt}-L{L}-s{seed}".replace(" ", "_")
        inst = Instance(iid, S.name, "detective", src, tgt, CAPITALS[src], CAPITALS[tgt], [], [],
                        {t: TEMPLATES[t] for t in ("fs1", "zs1")}, cons, task="detective", suite_task="T4_detective",
                        extra={"plant": {"layer": L, "vector": v.tolist(), "source": src, "scale": scale}})
        # behavioural strength of the plant on source cities (fs1), for the record
        cs = [c for c in CITIES[src] if valid["fs1"].get(c)]
        e = [S.encode(TEMPLATES["fs1"], c) for c in cs]
        from ..hooks import Intervention
        pr = predict(S, e, CAPS, [Intervention(L, pos_rule(e, "city_last"), "add", v)], bs)
        inst.extra["plant"]["flip_fs1"] = rate([p == CAPITALS[tgt] for p in pr])
        r = run_reference(S, inst, bs)
        g = r["grade"]
        inst.extra["reference"] = {"reward": g.get("reward"), "report": g.get("report"), "calls": r["calls"],
                                   "error": r["error"]}
        ok = g.get("reward", 0) >= 0.5
        print(f"{iid}: plant flip {inst.extra['plant']['flip_fs1']['rate']:.2f} reference {g.get('reward')} "
              f"{g.get('report')} calls {r['calls']} {r['error'] or ''} -> {'keep' if ok else 'drop'}", flush=True)
        if ok:
            out.append(inst)
    return out


def gen_handoff(S, n: int, seed: int, bs: int = 16, max_tries: int = 20) -> list:
    valid = load_validity(S.name)
    truth = HANDOFF[S.name]
    rng = random.Random(f"{seed}-handoff-{S.name}")
    states = study_states(min_cities=5)
    out, tries = [], 0
    while len(out) < n and tries < max_tries:
        tries += 1
        src, tgt = rng.sample(states, 2)
        sc = [c for c in CITIES[src] if valid["fs1"].get(c)][:8]
        tc = [c for c in CITIES[tgt] if valid["fs1"].get(c)][:8]
        if len(sc) < 4 or len(tc) < 4:
            continue
        cons = {"max_layer": S.n_layers - 1, "positions": ["city_last", "city_all", "final"], "max_rank": 2,
                "max_norm": None, "kl_budget": 1.0, "fwd_budget": 3000}
        iid = f"T6_handoff-{S.name.split('/')[-1]}-{src}-{tgt}-s{seed}".replace(" ", "_")
        inst = Instance(iid, S.name, "handoff", src, tgt, CAPITALS[src], CAPITALS[tgt], sc, tc,
                        {t: TEMPLATES[t] for t in ("fs1", "zs1")}, cons, task="handoff", suite_task="T6_handoff",
                        extra={"handoff": truth},
                        public_extra={"source": src, "target": tgt, "source_capital": CAPITALS[src],
                                      "target_capital": CAPITALS[tgt], "source_cities": sc, "target_cities": tc})
        r = run_reference(S, inst, bs)
        g = r["grade"]
        inst.extra["reference"] = {"reward": g.get("reward"), "report": g.get("report"), "calls": r["calls"],
                                   "error": r["error"]}
        ok = g.get("reward", 0) >= 0.5
        print(f"{iid}: truth {truth} reference {g.get('report')} calls {r['calls']} {r['error'] or ''} -> "
              f"{'keep' if ok else 'drop'}", flush=True)
        if ok:
            out.append(inst)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(SPECS))
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max_tries", type=int, default=None)
    a = ap.parse_args()
    S = Subject(a.model)
    h = HANDOFF[a.model]
    kw = {"max_tries": a.max_tries} if a.max_tries else {}
    if a.task == "T5_erase":
        insts = gen_erase(S, a.n, a.seed, h, bs=a.bs, **kw)
    elif a.task == "T4_detective":
        insts = gen_detective(S, a.n, a.seed, h, bs=a.bs, **kw)
    elif a.task == "T6_handoff":
        insts = gen_handoff(S, a.n, a.seed, bs=a.bs, **kw)
    else:
        insts = gen_edit(S, a.task, a.n, a.seed, h, bs=a.bs, **kw)
    d = ROOT / "results" / "suite" / "instances" / a.task
    for inst in insts:
        save_instance(inst, d)
    print(f"saved {len(insts)} instances to {d}")


if __name__ == "__main__":
    main()
