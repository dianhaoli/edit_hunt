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
    "scale": {"type": "number"}}, "required": ["register", "layer", "position"]}}
_STRS = {"type": "array", "items": {"type": "string"}}
_POS = {"type": "string", "enum": ["city_last", "city_all", "final"]}


def _tool(name, desc, props, req=()):
    return {"name": name, "description": desc,
            "input_schema": {"type": "object", "properties": props, "required": list(req)}}


TOOLS = [
    _tool("describe_task", "Return the task, constraints, dev cities, named templates, subject model shape, "
          "register names and budget.", {}),
    _tool("run_prompts", "Run the unedited subject model. Either raw `prompts`, or a `template` (key or string "
          "with {city}) plus `cities`. Returns top next tokens and the ranking of the 50 US state capitals "
          "(exact full-sequence log-prob). Max 16 prompts.",
          {"prompts": _STRS, "template": {"type": "string"}, "cities": _STRS, "top_k": {"type": "integer"}}),
    _tool("logit_lens", "Decode the residual stream after `layer` at `position` (city_last, final, or token "
          "index) through the final norm + unembedding.",
          {"template": {"type": "string"}, "city": {"type": "string"}, "layer": {"type": "integer"},
           "position": {"type": ["string", "integer"]}, "top_k": {"type": "integer"}},
          ["template", "city", "layer"]),
    _tool("cache_mean", "Store in register `name` the mean residual (output of `layer`) at `position` over the "
          "given cities in `template` (city_all averages over the city tokens).",
          {"name": {"type": "string"}, "cities": _STRS, "template": {"type": "string"},
           "layer": {"type": "integer"}, "position": _POS}, ["name", "cities", "template", "layer"]),
    _tool("vec_op", "Compute `expr` over registers and store the result in register `name`. Grammar: numbers, "
          "register names, + - * / (scalar only), parentheses, normalize(v), proj(v,u), proj_out(v,u), randn(seed) "
          "(unit vector), and scalar-valued dot(u,v), norm(v), cos(u,v). A scalar result is returned, not stored.",
          {"name": {"type": "string"}, "expr": {"type": "string"}}, ["name", "expr"]),
    _tool("vec_info", "Norm of a register, cosine to other registers, top tokens of its direct unembedding.",
          {"name": {"type": "string"}, "top_k": {"type": "integer"}}, ["name"]),
    _tool("eval_intervention", "Apply additive edits (register * scale added after `layer` at `position`) and "
          "report clean vs edited capital predictions on template x cities, flip rate to the target capital, KL on "
          "these prompts and on a few generic sentences. Give `edits` (list) or vector/layer/position/scale.",
          {"template": {"type": "string"}, "cities": _STRS, "edits": _EDITS, "vector": {"type": "string"},
           "layer": {"type": "integer"}, "position": _POS, "scale": {"type": "number"},
           "n_generic": {"type": "integer"}}, ["template", "cities"]),
    _tool("optimize_vector", "Gradient-optimize an additive vector (Adam) at `layer`/`position` to raise the "
          "log-prob of the target capital's first token on dev_cities x templates, optionally penalizing KL on "
          "generic sentences, and optionally (keep_cities) rewarding each keep city's OWN state capital with the "
          "vector applied to it; max_norm caps the vector norm during training; keep_state_cities rewards each such city's OWN state on "
          "the state_q prompt with the vector applied. Costs steps x prompts forward passes. Result stored in register `name`.",
          {"name": {"type": "string"}, "layer": {"type": "integer"}, "position": _POS, "dev_cities": _STRS,
           "templates": _STRS, "steps": {"type": "integer"}, "kl_weight": {"type": "number"},
           "init": {"type": "string"}, "lr": {"type": "number"}, "keep_cities": _STRS,
           "keep_weight": {"type": "number"}, "max_norm": {"type": "number"}, "keep_state_cities": _STRS},
          ["name", "layer", "position", "dev_cities", "templates"]),
    _tool("submit", "Submit the final edits (ends the episode). Checked against the constraints; a rejected "
          "submission returns an error and you may retry.", {"edits": _EDITS}, ["edits"]),
]

SYSTEM = """You are working on an interpretability task on a small open-weights language model (the subject).

Goal: find an intervention on the subject's residual stream such that, for cities in the SOURCE state, prompts \
asking for the capital of the state containing the city are answered with the TARGET state's capital. It will be \
graded on held-out SOURCE-state cities and held-out prompt wordings that you cannot access, and penalized for \
changing the model's behavior elsewhere.

Intervention format: up to `max_rank` edits; each adds (register vector * scale) to the output of decoder layer \
`layer` (<= max_layer) at one of the allowed position rules (city_last = last token of the city name, city_all = \
all city tokens, final = last prompt token). The same edits are applied to every prompt the grader runs.

Grading: F = fraction of held-out items whose top answer among the 50 state capitals is the target capital \
(if constraints.preserve_state is true, an item only counts if the model's answer to "which state is <city> in" \
is unchanged). KL_mean = average of (a) the binary KL of the model's probability that the held-out city is in the United States \
(on a country prompt) and (b) the KL divergence (clean || edited) of the next-token distribution on unrelated \
sentences. leak = fraction of held-out cities from OTHER states (neither source nor target) whose top capital \
answer changes under your edit. reward = F * (1 - leak_weight * leak) * (1 - min(1, KL_mean / kl_budget)); \
leak_weight and kl_budget are in the constraints.

You have a limited forward-pass budget; tool outputs report what is left. Vectors live in named registers on the \
server. Call submit exactly once when you are done; you will not see the grade."""


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
def run_episode(client, env: ToolEnv, model: str, max_turns: int = 40, max_tokens: int = 16000,
                effort: str | None = None, fallbacks: bool = True, max_nudges: int = 2) -> dict:
    messages = [{"role": "user", "content": "Task (output of describe_task):\n" + env.call("describe_task", {})}]
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    nudges, stop, t0 = 0, None, time.time()
    for turn in range(max_turns):
        # auto-caching: the history is append-only, so each turn reads the previous turns from cache
        kw = dict(model=model, max_tokens=max_tokens, system=SYSTEM, tools=TOOLS, messages=messages,
                  cache_control={"type": "ephemeral"})
        if effort:
            kw["output_config"] = {"effort": effort}
        if fallbacks:  # server-side refusal fallback (beta)
            kw["extra_headers"] = {"anthropic-beta": "server-side-fallback-2026-07-01"}
            kw["extra_body"] = {"fallbacks": "default"}
        resp = client.messages.create(**kw)
        for k in usage:
            usage[k] += getattr(resp.usage, k, 0) or 0
        messages.append({"role": "assistant", "content": resp.content})  # echo blocks unchanged
        stop = resp.stop_reason
        if stop == "refusal":
            break
        uses = [b for b in resp.content if b.type == "tool_use"]
        if not uses:
            if nudges >= max_nudges:
                break
            nudges += 1
            messages.append({"role": "user", "content": "Continue. Call submit when you are done."})
            continue
        results = []
        for b in uses:
            out = env.call(b.name, b.input)
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": out,
                            "is_error": out.startswith('{"error"')})
        messages.append({"role": "user", "content": results})
        if env.done:
            break
    log = [{"role": m["role"], "content": m["content"] if isinstance(m["content"], str)
            else [x if isinstance(x, dict) else _dump(x) for x in m["content"]]} for m in messages]
    return {"messages": log, "turns": turn + 1, "stop_reason": stop, "usage": usage,
            "secs": round(time.time() - t0, 1)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True, help="path to <id>.private.json")
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--max_turns", type=int, default=40)
    ap.add_argument("--max_tokens", type=int, default=16000)
    ap.add_argument("--effort", default=None, help="low|medium|high|xhigh|max (default: API default)")
    ap.add_argument("--no_fallbacks", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--out_dir", default=str(ROOT / "results" / "agent_runs"))
    a = ap.parse_args()
    inst: Instance = load_instance(a.instance)
    if a.dry_run:
        client = FakeClient()
    else:
        import anthropic
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        client = anthropic.Anthropic(api_key=os.environ["ANT_KEY"])
    S = Subject(inst.model, device=a.device)
    env = ToolEnv(S, inst, a.bs)
    ep = run_episode(client, env, a.model, a.max_turns, a.max_tokens, a.effort, not a.no_fallbacks and not a.dry_run)
    g = grade(S, inst, env.submission, a.bs) if env.done else {"valid": False, "reason": "no submission", "reward": 0.0}
    rec = {"instance": inst.id, "tier": inst.tier, "agent_model": "dry_run" if a.dry_run else a.model,
           "config": vars(a), "budget_used": env.used, "budget_total": env.budget, "tool_log": env.log,
           "submission_edits": [{k: v for k, v in e.items() if k != "vector"} for e in (env.submission or {}).get("edits", [])],
           "grade": g, **ep}
    d = Path(a.out_dir) / inst.tier
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{inst.id}{'.dryrun' if a.dry_run else ''}.json"
    p.write_text(json.dumps(rec, indent=1, default=str))
    print(f"{inst.id}: reward {g['reward']:.3f} F {g.get('F')} KL {g.get('kl_mean')} turns {ep['turns']} "
          f"budget {env.used}/{env.budget} -> {p}")


if __name__ == "__main__":
    main()
