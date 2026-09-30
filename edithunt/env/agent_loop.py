"""Minimal Claude tool-use loop for EditHunt.

  python -m edithunt.env.agent_loop --instance results/instances/<model>/<tier>/<id>.private.json
         [--model claude-sonnet-5-5] [--max_turns 40] [--max_tokens 16000] [--effort high] [--dry_run]

API key: env var ANT_KEY (loaded from .env via python-dotenv at runtime). --dry_run uses a scripted
fake client (mean-difference strategy) and never touches the network.
Output: results/agent_runs/<tier>/<instance_id>.json (transcript, tool log, usage, grade).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

from ..common import ROOT
from ..model import Subject
from .grader import grade
from .instance import Instance, load_instance
from .tools import ToolEnv

_EDITS = {"type": "array", "items": {"type": "object", "properties": {
    "register": {"type": "string"}, "layer": {"type": "integer"},
    "position": {"type": "string", "enum": ["city_last", "city_all", "final"]},
    "scale": {"type": "number"},
    "basis": {"type": "array", "items": {"type": "string"}}, "center": {"type": "string"}},
    "required": ["layer", "position"]}}
_STRS = {"type": "array", "items": {"type": "string"}}
_INTS = {"type": "array", "items": {"type": "integer"}}
_POS = {"type": "string", "enum": ["city_last", "city_all", "final"]}
_MODEL = {"type": "string", "enum": ["planted", "clean"]}
_ANS = {"type": "string", "enum": ["capitals", "states"]}


def _tool(name, desc, props, req=()):
    return {"name": name, "description": desc,
            "input_schema": {"type": "object", "properties": props, "required": list(req)}}


_MODEL_NOTE = " `model` ('planted'|'clean') only matters in tasks with a planted edit."
TOOL_SCHEMAS = {t["name"]: t for t in [
    _tool("describe_task", "Return the task, reward, constraints, cities, named templates, subject model shape, "
          "register names and budgets.", {}),
    _tool("run_prompts", "Run the unedited subject model. Either raw `prompts`, or a `template` (key or string "
          "with {city}) plus `cities`. Returns top next tokens and the ranking of the 50 US state capitals (or the 50 "
          "states, answers='states') by exact full-sequence log-prob; top_k controls how many (max 50)." + _MODEL_NOTE,
          {"prompts": _STRS, "template": {"type": "string"}, "cities": _STRS, "top_k": {"type": "integer"},
           "answers": _ANS, "model": _MODEL}),
    _tool("logit_lens", "Decode the residual stream after `layer` at `position` (city_last, final, or token "
          "index) through the final norm + unembedding." + _MODEL_NOTE,
          {"template": {"type": "string"}, "city": {"type": "string"}, "layer": {"type": "integer"},
           "position": {"type": ["string", "integer"]}, "top_k": {"type": "integer"}, "model": _MODEL},
          ["template", "city", "layer"]),
    _tool("cache_mean", "Store in register `name` the mean residual (output of `layer`) at `position` over the "
          "given cities in `template` (city_all averages over the city tokens). With `layers` (list) instead of "
          "`layer`, stores one register per layer named <name>_L<layer>." + _MODEL_NOTE,
          {"name": {"type": "string"}, "cities": _STRS, "template": {"type": "string"},
           "layer": {"type": "integer"}, "layers": _INTS, "position": _POS, "model": _MODEL},
          ["name", "cities", "template"]),
    _tool("act_diff", "Per city: norm of (planted - clean) residual after `layer` (or each of `layers`) at "
          "`position`, absolute and relative to the clean residual norm. With `name`, stores the mean difference "
          "in a register (<name>_L<layer> for several layers).",
          {"template": {"type": "string"}, "cities": _STRS, "layer": {"type": "integer"}, "layers": _INTS,
           "position": _POS, "name": {"type": "string"}}, ["template", "cities"]),
    _tool("vec_op", "Compute `expr` over registers and store the result in register `name`. Grammar: numbers, "
          "register names, + - * / (scalar only), parentheses, normalize(v), proj(v,u), proj_out(v,u), randn(seed) "
          "(unit vector), and scalar-valued dot(u,v), norm(v), cos(u,v). A scalar result is returned, not stored.",
          {"name": {"type": "string"}, "expr": {"type": "string"}}, ["name", "expr"]),
    _tool("vec_info", "Norm of a register, cosine to other registers, top tokens of its direct unembedding.",
          {"name": {"type": "string"}, "top_k": {"type": "integer"}}, ["name"]),
    _tool("eval_intervention", "Apply edits and report clean vs edited predictions (among the 50 capitals, or 50 "
          "states with answers='states') on template x cities, the changed rate, the flip rate to the target capital "
          "(if the task has one), KL on these prompts and on a few generic sentences. Edits: additive {register, "
          "layer, position, scale} (adds register*scale after `layer`) or projection {basis: [registers], center?, "
          "layer, position}. Give `edits` (list) or vector/layer/position/scale for one additive edit.",
          {"template": {"type": "string"}, "cities": _STRS, "edits": _EDITS, "vector": {"type": "string"},
           "layer": {"type": "integer"}, "position": _POS, "scale": {"type": "number"},
           "n_generic": {"type": "integer"}, "answers": _ANS}, ["template", "cities"]),
    _tool("optimize_vector", "Gradient-optimize an additive vector (Adam) added at `layer`/`position`. Loss = mean "
          "NLL of the target capital's first token on dev_cities x templates (tasks with a target capital; optional) "
          "+ weighted NLL of arbitrary (template, city) -> answer examples in extra_examples (first token of the "
          "answer; negative weight pushes the answer down) + kl_weight * KL on generic sentences. max_norm caps the "
          "vector norm during training; init starts from a register; lr is relative to the residual norm. Costs "
          "steps x prompts forward passes. Result stored in register `name`.",
          {"name": {"type": "string"}, "layer": {"type": "integer"}, "position": _POS, "dev_cities": _STRS,
           "templates": _STRS, "steps": {"type": "integer"}, "kl_weight": {"type": "number"},
           "init": {"type": "string"}, "lr": {"type": "number"}, "max_norm": {"type": "number"},
           "extra_examples": {"type": "array", "items": {"type": "object", "properties": {
               "template": {"type": "string"}, "city": {"type": "string"}, "answer": {"type": "string"},
               "weight": {"type": "number"}}, "required": ["template", "city", "answer"]}}},
          ["name", "layer"]),
    _tool("submit", "Submit the final edits (ends the episode). Checked against the constraints; a rejected "
          "submission returns an error and you may retry.", {"edits": _EDITS}, ["edits"]),
    _tool("submit_report", "Submit your final answer (ends the episode), e.g. {source, target, layer} or {layer} as "
          "the task says.", {"report": {"type": "object"}}, ["report"]),
]}
TOOLS = [TOOL_SCHEMAS[n] for n in ("describe_task", "run_prompts", "logit_lens", "cache_mean", "vec_op", "vec_info",
                                   "eval_intervention", "optimize_vector", "submit")]


def tools_for(env) -> list[dict]:
    return [TOOL_SCHEMAS[n] for n in TOOL_SCHEMAS if n in env.tools]


SYSTEM = """You are working on an interpretability task on a small open-weights language model (the subject), using \
tools that run the subject, read and edit its residual stream, and store vectors in named registers on the server.

The first message is the output of describe_task: the goal, how you will be rewarded, the constraints, and your \
budgets. An additive edit adds (register vector * scale) to the output of decoder layer `layer` at a position rule \
(city_last = last token of the city name, city_all = all city tokens, final = last prompt token); the grader applies \
your edits to every prompt it runs. The grader uses held-out cities and prompt wordings that the tools refuse.

Budgets: forward passes (and, in some tasks, tool calls); every tool result reports what is left. End the episode by \
calling submit (edit tasks) or submit_report (report tasks) exactly once; you will not see the grade."""


# ------------------------------------------------------------ fake client (dry run)
def _block(**kw):
    return SimpleNamespace(**kw)


class FakeClient:
    """Scripted mean-difference agent speaking the Messages API shape."""

    def __init__(self):
        self.messages = self
        self.step = 0

    def create(self, **kw):
        msgs = kw["messages"]
        task = json.loads(msgs[0]["content"].split("\n", 1)[1])
        c = task["constraints"]
        L, tpl = c["max_layer"], next(iter(task["templates"]))
        src, tgt = task["dev_source_cities"], task["dev_target_cities"]
        use = lambda i, n, a: _block(type="tool_use", id=f"tu_{self.step}_{i}", name=n, input=a)
        script = [
            [_block(type="text", text="Cache source and target means."),
             use(0, "cache_mean", {"name": "s", "cities": src, "template": tpl, "layer": L}),
             use(1, "cache_mean", {"name": "t", "cities": tgt, "template": tpl, "layer": L})],
            [use(0, "vec_op", {"name": "d", "expr": "t - s"})],
            [use(0, "eval_intervention", {"template": tpl, "cities": src, "vector": "d", "layer": L,
                                          "position": "city_last", "scale": 1.0})],
            [use(0, "submit", {"edits": [{"register": "d", "layer": L, "position": "city_last", "scale": 1.0}]})],
        ]
        content = script[self.step] if self.step < len(script) else [_block(type="text", text="Done.")]
        self.step += 1
        stop = "tool_use" if any(b.type == "tool_use" for b in content) else "end_turn"
        return SimpleNamespace(content=content, stop_reason=stop, id=f"fake_{self.step}",
                               usage=SimpleNamespace(input_tokens=0, output_tokens=0))


def _dump(b) -> dict:
    return b.model_dump(exclude_none=True) if hasattr(b, "model_dump") else dict(vars(b))


# ------------------------------------------------------------ loop
# $/token. Sonnet 5.5: input 2, output 10, cache read 0.20 per MTok; cache write assumed 1.25x input (5-min TTL).
# gpt-6.1-sol: input 2, cached input 0.10, output 10 per MTok (reasoning tokens are billed as output).
PRICES = {"anthropic": {"input_tokens": 2e-6, "output_tokens": 10e-6, "cache_read_input_tokens": 0.2e-6,
                        "cache_creation_input_tokens": 2.5e-6},
          "openai": {"input_tokens": 2e-6, "output_tokens": 10e-6, "cache_read_input_tokens": 0.1e-6,
                     "cache_creation_input_tokens": 0.0}}
PRICE = PRICES["anthropic"]


def cost(usage: dict, provider: str = "anthropic") -> float:
    return sum(v * usage.get(k, 0) for k, v in PRICES[provider].items())


def provider_of(model: str) -> str:
    return "openai" if model.startswith(("gpt", "o3", "o4")) else "anthropic"


def _first_msg(env) -> str:
    return "Task (output of describe_task):\n" + env.call("describe_task", {})


def run_episode(client, env: ToolEnv, model: str, max_turns: int = 40, max_tokens: int = 16000,
                effort: str | None = None, fallbacks: bool = True, max_nudges: int = 2,
                max_cost: float | None = None) -> dict:
    messages = [{"role": "user", "content": _first_msg(env)}]
    tools = tools_for(env)
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    nudges, stop, end, t0 = 0, None, "max_turns", time.time()
    served = set()
    for turn in range(max_turns):
        # auto-caching: the history is append-only, so each turn reads the previous turns from cache
        kw = dict(model=model, max_tokens=max_tokens, system=SYSTEM, tools=tools, messages=messages,
                  cache_control={"type": "ephemeral"})
        if effort:
            kw["output_config"] = {"effort": effort}
        if fallbacks:  # server-side refusal fallback (beta)
            kw["extra_headers"] = {"anthropic-beta": "server-side-fallback-2026-07-01"}
            kw["extra_body"] = {"fallbacks": "default"}
        resp = client.messages.create(**kw)
        served.add(getattr(resp, "model", None))
        for k in usage:
            usage[k] += getattr(resp.usage, k, 0) or 0
        messages.append({"role": "assistant", "content": resp.content})  # echo blocks unchanged
        stop = resp.stop_reason
        capped = max_cost is not None and cost(usage) >= max_cost
        if stop == "refusal":
            end = "refusal"
            break
        uses = [b for b in resp.content if b.type == "tool_use"]
        if capped and not uses:
            end = "cost_cap"
            break
        if not uses:
            if nudges >= max_nudges:
                end = "no_tool_calls"
                break
            nudges += 1
            messages.append({"role": "user", "content": "Continue. Submit when you are done."})
            continue
        results = []
        for b in uses:
            out = env.call(b.name, b.input)
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": out,
                            "is_error": out.startswith('{"error"')})
        messages.append({"role": "user", "content": results})
        if env.done:
            end = "submitted"
            break
        if capped:  # the capping turn's tool calls were run (so a submit in it counts); stop now
            end = "cost_cap"
            break
    log = [{"role": m["role"], "content": m["content"] if isinstance(m["content"], str)
            else [x if isinstance(x, dict) else _dump(x) for x in m["content"]]} for m in messages]
    return {"messages": log, "turns": turn + 1, "stop_reason": stop, "end": end,
            "served_models": sorted(m for m in served if m), "usage": usage,
            "cost_usd": round(cost(usage), 4),
            "secs": round(time.time() - t0, 1)}


def run_episode_openai(client, env: ToolEnv, model: str, max_turns: int = 40, max_tokens: int = 16000,
                       effort: str | None = None, max_nudges: int = 2, max_cost: float | None = None) -> dict:
    """Same loop over the OpenAI Responses API (function tools + reasoning; turns chained with
    previous_response_id; prompt caching is automatic)."""
    # strict=False: optional properties stay optional (strict schemas make the model fill every field)
    tools = [{"type": "function", "name": t["name"], "description": t["description"],
              "parameters": t["input_schema"], "strict": False} for t in tools_for(env)]
    log = [{"role": "user", "content": _first_msg(env)}]
    new_input = list(log)
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    nudges, stop, end, prev, t0 = 0, None, "max_turns", None, time.time()
    served = set()
    for turn in range(max_turns):
        kw = dict(model=model, instructions=SYSTEM, input=new_input, tools=tools, max_output_tokens=max_tokens)
        if effort:
            kw["reasoning"] = {"effort": effort}
        if prev:
            kw["previous_response_id"] = prev
        resp = client.responses.create(**kw)
        prev = resp.id
        u = resp.usage
        cached = getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0
        usage["input_tokens"] += u.input_tokens - cached
        usage["cache_read_input_tokens"] += cached
        usage["output_tokens"] += u.output_tokens
        usage["reasoning_tokens"] = usage.get("reasoning_tokens", 0) + (
            getattr(getattr(u, "output_tokens_details", None), "reasoning_tokens", 0) or 0)  # logged, not priced
        served.add(getattr(resp, "model", None))
        stop = resp.status
        items = [it.model_dump(exclude_none=True) for it in resp.output]
        log.append({"role": "assistant", "content": [it for it in items if it.get("type") != "reasoning"]})
        capped = max_cost is not None and cost(usage, "openai") >= max_cost
        calls = [it for it in resp.output if it.type == "function_call"]
        if capped and not calls:
            end = "cost_cap"
            break
        if not calls:
            if nudges >= max_nudges:
                end = "no_tool_calls"
                break
            nudges += 1
            new_input = [{"role": "user", "content": "Continue. Submit when you are done."}]
            log.append(new_input[0])
            continue
        new_input = []
        for c in calls:
            try:
                args = json.loads(c.arguments or "{}")
                out = env.call(c.name, args if isinstance(args, dict) else {})
            except json.JSONDecodeError as e:
                out = json.dumps({"error": f"arguments are not valid JSON: {e}"})
            new_input.append({"type": "function_call_output", "call_id": c.call_id, "output": out})
        log.append({"role": "user", "content": new_input})
        if env.done:
            end = "submitted"
            break
        if capped:
            end = "cost_cap"
            break
    return {"messages": log, "turns": turn + 1, "stop_reason": stop, "end": end,
            "served_models": sorted(m for m in served if m), "usage": usage,
            "cost_usd": round(cost(usage, "openai"), 4), "secs": round(time.time() - t0, 1)}


def make_client(model: str):
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    if provider_of(model) == "openai":
        import openai
        return openai.OpenAI(api_key=os.environ["OPENAI_KEY"])
    import anthropic
    return anthropic.Anthropic(api_key=os.environ["ANT_KEY"])


def episode(S, inst, model: str, client=None, blackbox: bool = False, max_turns: int = 25, max_tokens: int = 16000,
            effort: str | None = "medium", max_cost: float | None = 0.10, bs: int = 16, dry_run: bool = False) -> dict:
    """One full episode on a suite instance: env with the task's tool subset -> agent loop -> grade -> record."""
    from .tasks import make_env
    env = make_env(S, inst, bs, blackbox=blackbox)
    if dry_run:
        ep = run_episode(FakeClient(), env, model, max_turns, max_tokens, None, False, max_cost=None)
    elif provider_of(model) == "openai":
        ep = run_episode_openai(client, env, model, max_turns, max_tokens, effort, max_cost=max_cost)
    else:
        ep = run_episode(client, env, model, max_turns, max_tokens, effort, True, max_cost=max_cost)
    g = grade(S, inst, env.submission, bs)
    return {"instance": inst.id, "suite_task": getattr(inst, "suite_task", ""), "tier": inst.tier,
            "agent_model": "dry_run" if dry_run else model, "blackbox": blackbox,
            "config": {"max_turns": max_turns, "max_tokens": max_tokens, "effort": effort, "max_cost": max_cost},
            "budget_used": env.used, "budget_total": env.budget, "tool_calls": env.calls,
            "call_budget": env.call_budget, "tool_log": env.log,
            "submission": {k: v for k, v in (env.submission or {}).items() if k != "edits"},
            "submission_edits": [{k: v for k, v in e.items() if k not in ("vector", "basis", "center")}
                                 for e in (env.submission or {}).get("edits", [])],
            "grade": g, **ep}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True, help="path to <id>.private.json")
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--max_turns", type=int, default=25)
    ap.add_argument("--max_tokens", type=int, default=16000)
    ap.add_argument("--effort", default="medium", help="low|medium|high|... ('' = API default)")
    ap.add_argument("--blackbox", action="store_true", help="prompting-only tool subset (report tasks)")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--max_cost", type=float, default=0.10, help="stop the episode once its cost reaches this ($)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--out_dir", default=str(ROOT / "results" / "suite" / "runs"))
    a = ap.parse_args()
    inst: Instance = load_instance(a.instance)
    S = Subject(inst.model, device=a.device)
    client = None if a.dry_run else make_client(a.model)
    rec = episode(S, inst, a.model, client, a.blackbox, a.max_turns, a.max_tokens, a.effort or None, a.max_cost, a.bs,
                  a.dry_run)
    d = Path(a.out_dir) / rec["agent_model"] / (rec["suite_task"] or inst.tier)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{inst.id}{'.bb' if a.blackbox else ''}.json"
    p.write_text(json.dumps(rec, indent=1, default=str))
    g = rec["grade"]
    print(f"{inst.id}: reward {g['reward']:.3f} turns {rec['turns']} calls {rec['tool_calls']} budget "
          f"{rec['budget_used']}/{rec['budget_total']} cost ${rec['cost_usd']:.4f} stop {rec['stop_reason']} -> {p}")


if __name__ == "__main__":
    main()
