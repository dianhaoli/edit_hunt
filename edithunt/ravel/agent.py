"""Agent loop for the Stage-5 city environment (OpenAI Responses API; same structure as edithunt/env/agent_loop.py).

  python -m edithunt.ravel.agent --model gpt-6-luna --instances Continent_s0,Language_s0 [--sae] [--effort medium]

Spend: results/screen/env/ledger.json, hard cap CAP_USD (user budget $15 for this test). Each episode reserves its
per-episode cap before it starts and settles the measured cost after. Prices: gpt-6.1-sol $2 / $0.10 cached / $10 per
MTok; gpt-6-luna is billed at the same rates (upper bound; its price list was not available here).
Records: results/screen/env/runs/{model}/{instance}{_sae}.json (transcript, tool log, usage, cost, grade).
"""
from __future__ import annotations

import argparse
import json
import os
import time

from ..common import ROOT
from . import env as E
from .core import Model

CAP_USD = 15.0
LEDGER = ROOT / "results" / "screen" / "env" / "ledger.json"
RUNS = ROOT / "results" / "screen" / "env" / "runs"
PRICE = {"input": 2e-6, "cached": 0.1e-6, "output": 10e-6}
SYSTEM = ("You are an AI research agent with tools for a mechanistic-interpretability task on a language model. "
          "Read the task, use the tools to investigate and build a solution, and call submit with your final answer. "
          "Tool calls and compute are limited; budgets are listed in the task.")

_S = {"type": "string"}
_STRS = {"type": "array", "items": _S}
_PAIRS = {"type": "array", "items": {"type": "object", "properties": {"base": _S, "source": _S}, "required": ["base", "source"]}}
_EX = {"type": "array", "items": {"type": "object", "properties": {
    "base": _S, "source": _S, "template": _S, "answer": _S, "weight": {"type": "number"}, "source_template": _S},
    "required": ["base", "source", "template", "answer"]}}


_REC = {"type": "array", "items": {"type": "object", "properties": {
    "n": {"type": "integer"}, "templates": _STRS, "answer": _S, "differ": _STRS, "weight": {"type": "number"}},
    "required": ["n", "templates", "answer"]}}


def _fn(name, desc, props, req=()):
    return {"type": "function", "name": name, "description": desc, "strict": False,
            "parameters": {"type": "object", "properties": props, "required": list(req)}}


def tools(sae: bool):
    t = [_fn("describe_task", "Task statement, grading, attributes and values, dev cities with labels, template ids, budgets.", {}),
         _fn("run_prompts", "Run the unedited model on a template (id or text with %s) for cities. Returns top next tokens "
             "and, if the template's attribute is known (or `attribute` is given), the answer by exact log-prob over that "
             "attribute's values.", {"template": _S, "cities": _STRS, "attribute": _S, "top_k": {"type": "integer"}},
             ["template", "cities"]),
         _fn("cache_activations", "Residual-stream statistics at a city's last token after `layer` (norms, mean cosine "
             "similarity). Vectors stay server-side.", {"template": _S, "cities": _STRS, "layer": {"type": "integer"}},
             ["template", "cities", "layer"]),
         _fn("interchange", "Run the model on base prompts with the featurizer's subspace interchanged from source cities "
             "(at the featurizer's layer, base city's last token). featurizer: 'full' (needs `layer`), a train_subspace id"
             + (", or 'sae:[latent ids]' (SAE latents at the task layer; reconstruction error kept)" if sae else "")
             + ". `template`: one template or a list of up to 6 (each run on all pairs). Returns per template the clean, "
             "edited and source-city answers per pair and a summary over pairs whose base and source answers differ "
             "(kept_base / took_source / other).",
             {"featurizer": _S, "pairs": _PAIRS, "template": {"anyOf": [_S, _STRS]}, "attribute": _S, "source_template": _S,
              "layer": {"type": "integer"}}, ["featurizer", "pairs", "template"]),
         _fn("train_subspace", "Train an orthonormal k-dim subspace U at `layer` by gradient descent so that, with U "
             "interchanged from each example's source city, the model answers each example's `answer` on its base prompt "
             "(cross-entropy, optional per-example weight). Give explicit `examples` and/or `recipes`; a recipe generates n "
             "examples over random pairs of distinct dev cities: {n, templates: [template ids], answer: 'source:<attribute>' "
             "or 'base:<attribute>' (the answer is that city's value), differ: [attributes the pair must differ in], "
             "weight}. 10% of all examples are held out; their loss and answer accuracy are reported. Returns a featurizer id.",
             {"layer": {"type": "integer"}, "k": {"type": "integer"}, "examples": _EX, "recipes": _REC,
              "steps": {"type": "integer"}, "lr": {"type": "number"}, "seed": {"type": "integer"}}, ["layer", "k"]),
         _fn("submit", "Submit a featurizer (must be at the task layer). Ends the episode.", {"featurizer": _S}, ["featurizer"])]
    if sae:
        t += [_fn("sae_top_latents", "Most active Gemma Scope SAE latents (16k, task layer) at a city's last token.",
                  {"template": _S, "city": _S, "top_k": {"type": "integer"}}, ["template", "city"]),
              _fn("sae_latent_info", "Dev cities on which an SAE latent fires most.", {"latent": {"type": "integer"},
                  "n": {"type": "integer"}}, ["latent"])]
    return t


def _ledger():
    return json.loads(LEDGER.read_text()) if LEDGER.exists() else {"spent": 0.0, "episodes": []}


def cost(u):
    return u["input"] * PRICE["input"] + u["cached"] * PRICE["cached"] + u["output"] * PRICE["output"]


def episode(M, client, model, inst, sae, max_turns=40, max_cost=0.75, effort="medium", max_tokens=16000):
    env = E.CityEnv(M, inst, sae_tools=sae)
    first = "Task (output of describe_task):\n" + env.call("describe_task", {})
    log = [{"role": "user", "content": first}]
    new_input, prev = list(log), None
    u = {"input": 0, "cached": 0, "output": 0, "reasoning": 0}
    end, nudges, t0 = "max_turns", 0, time.time()
    for turn in range(max_turns):
        kw = dict(model=model, instructions=SYSTEM, input=new_input, tools=tools(sae), max_output_tokens=max_tokens)
        if effort:
            kw["reasoning"] = {"effort": effort}
        if prev:
            kw["previous_response_id"] = prev
        for attempt in range(10):  # rate limits (429) / transient errors: back off and retry
            try:
                resp = client.responses.create(**kw)
                break
            except Exception as e:
                if attempt == 9 or not any(x in str(e) for x in ("429", "rate_limit", "502", "503", "timeout", "overloaded")):
                    raise
                time.sleep(min(60, 8 * (attempt + 1)))
        prev = resp.id
        cached = getattr(getattr(resp.usage, "input_tokens_details", None), "cached_tokens", 0) or 0
        u["input"] += resp.usage.input_tokens - cached
        u["cached"] += cached
        u["output"] += resp.usage.output_tokens
        u["reasoning"] += getattr(getattr(resp.usage, "output_tokens_details", None), "reasoning_tokens", 0) or 0
        items = [it.model_dump(exclude_none=True) for it in resp.output]
        log.append({"role": "assistant", "content": [it for it in items if it.get("type") != "reasoning"]})
        capped = cost(u) >= max_cost
        calls = [it for it in resp.output if it.type == "function_call"]
        if not calls:
            if capped or nudges >= 2:
                end = "cost_cap" if capped else "no_tool_calls"
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
    g = E.grade(M, inst, env.submission, items_out=True)
    sub = {k: v for k, v in (env.submission or {}).items() if k != "U"}
    if env.submission and env.submission.get("U") is not None:  # keep the subspace so the run can be regraded
        import torch
        from pathlib import Path
        d = Path.home() / "screen_subspaces" / "agent" / model
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{inst['id']}{'_sae' if sae else ''}_{int(time.time())}.pt"
        torch.save(env.submission["U"], path)
        sub["U_path"] = str(path)
    return dict(instance=inst["id"], model=model, sae_tools=sae, effort=effort, end=end, turns=turn + 1,
                usage=u, cost_usd=round(cost(u), 4), secs=round(time.time() - t0), submission=sub,
                train_calls=env.train_calls, tool_calls=env.calls, tool_log=env.log, grade=g, messages=log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-6-luna")
    ap.add_argument("--instances", required=True)
    ap.add_argument("--sae", action="store_true")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--max_cost", type=float, default=0.75)
    ap.add_argument("--samples", type=int, default=1)
    a = ap.parse_args()
    from dotenv import load_dotenv
    import openai
    load_dotenv(ROOT / ".env") or load_dotenv(ROOT.parent / "edit_hunt" / ".env")  # worktree has no .env
    client = openai.OpenAI(api_key=os.environ["OPENAI_KEY"])
    val = json.loads((ROOT / "results" / "screen" / "env" / "validate.json").read_text())
    E.KL_BUDGET = val["kl_budget"]
    M = Model()
    jobs = [(iid, r) for r in range(a.samples) for iid in a.instances.split(",")]
    for iid, r in jobs:
        path = RUNS / a.model / f"{iid}{'_sae' if a.sae else ''}{f'_r{r}' if a.samples > 1 else ''}.json"
        if path.exists():
            print(f"{path.name}: exists, skipping", flush=True)
            continue
        L = _ledger()
        if L["spent"] + a.max_cost > CAP_USD:
            print(f"budget: spent ${L['spent']:.2f}, next episode could exceed ${CAP_USD}; stopping", flush=True)
            break
        L["spent"] += a.max_cost  # reserve worst case
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        LEDGER.write_text(json.dumps(L, indent=1))
        rec = None
        try:
            rec = episode(M, client, a.model, E.load_instance(iid), a.sae, max_cost=a.max_cost, effort=a.effort)
        finally:
            L = _ledger()
            L["spent"] += (rec["cost_usd"] if rec else a.max_cost) - a.max_cost  # settle (worst case if it crashed)
            L["episodes"].append(dict(model=a.model, instance=iid, sae=a.sae, cost=rec["cost_usd"] if rec else None))
            LEDGER.write_text(json.dumps(L, indent=1))
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, indent=1, default=str))
        tmp.rename(path)
        g = rec["grade"]
        print(f"{a.model} {iid}{' sae' if a.sae else ''}: score {g['score']:.2f} (C {g.get('cause', 0):.2f} "
              f"I {g.get('iso', 0):.2f} KLf {g.get('kl_factor', 0):.2f}) passed={g['passed']} end={rec['end']} "
              f"turns={rec['turns']} train_calls={rec['train_calls']} ${rec['cost_usd']:.3f} {rec['secs']}s", flush=True)


if __name__ == "__main__":
    main()
