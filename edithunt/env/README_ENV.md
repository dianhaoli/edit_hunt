# EditHunt environment (Phase 6 prototype)

The agent must build a constrained additive residual-stream edit on the subject model (Qwen2.5) so that
two-hop prompts about SOURCE-state cities ("The state containing Dallas has its capital in") answer the
TARGET state's capital, on held-out cities and held-out templates, with little collateral damage.
Grading is fully programmatic.

All commands run from the repo root with the project venv:
`PYTHONPATH=. .venv/bin/python -m edithunt.env.<module> ...`

## Files
| file | role |
|---|---|
| `instance.py` | `Instance` dataclass, `TIERS` presets, generator, `validate_instance` (reference check), JSON IO |
| `grader.py` | `check` (constraints), `grade` (score), `reward` (formula) |
| `tools.py` | `ToolEnv`: agent tool API with forward-pass budget and server-side vector registers |
| `agent_loop.py` | Claude tool-use loop (Anthropic SDK) + `--dry_run` fake client |
| `baselines.py` | scripted agents (mean-difference, gradient, random) + pass-rate harness |

## Instances
```
python -m edithunt.env.instance --model Qwen/Qwen2.5-1.5B --tier easy --n 5 --handoff <L> [--pairs Texas:California]
```
Writes `results/instances/<model>/<tier>/<id>.public.json` (agent view) and `<id>.private.json` (grader view).
- Validity: uses `results/<model>/phase0_validity.json` if present; missing (template, city) entries are
  computed on the fly (clean 50-way argmax == true capital). `--max_cities k` truncates city lists (smoke tests).
- Split: dev source cities = `n_dev` source cities valid on the first dev template; held-out test cities =
  all other source cities; test items = test city x held-out template (`ho_fs`, `ho_zs`), kept only if the
  clean model answers correctly. Dev target cities are given when `target_examples`.
- `validate_instance`: reference edit = mean over dev templates of [mean(target pool) - mean(source pool)] of
  the city_last residual, using the grader's private full pool (all valid cities of both states, incl. held-out).
  Tries up to 3 layers <= min(max_layer, handoff-1), keeps the instance iff the best held-out flip rate >= 0.8.
  The full grade of that reference is stored in `private.reference` (useful to calibrate `kl_budget`).
- Tier presets (`TIERS`, placeholders): `max_layer` is an int, `"last"`, or `"handoff±k"` (resolved with
  `--handoff`, default n_layers//2).

| tier | max_layer | positions | max_rank | kl_budget | fwd_budget | n_dev | notes |
|---|---|---|---|---|---|---|---|
| easy | handoff-1 | city_last, city_all | 2 | 1.0 | 3000 | 4 (+4 target) | dev templates fs1, zs1 |
| medium | 5 | city_last | 1 | 0.3 | 800 | 2 (+2 target) | dev template fs1 |
| hard | last | city_last, city_all, final | 2 | 0.3 | 1500 | 3 (+3 target) | preserve_state: STATE_Q answer must not change |

## Submission format
```json
{"kind": "add",
 "edits": [{"layer": 7, "position": "city_last", "vector": [/* d_model floats */], "scale": 1.0}]}
```
Each edit adds `scale * vector` to the output of decoder layer `layer` at the positions chosen by `position`
(`city_last`, `city_all`, `final`) in every prompt the grader runs. `check` rejects (reward 0, with a reason):
more than `max_rank` edits, `layer > max_layer`, a disallowed position, wrong vector length, non-finite values,
`||scale*vector|| > max_norm` (if set), `kind != "add"`.

## Grader formula (`grader.reward`)
- **F** = held-out flip rate: fraction of test items whose exact argmax over the 50 capitals is the target
  capital. With `preserve_state` (hard), an item counts only if the city's STATE_Q answer (argmax over 50 states)
  is unchanged by the edit (`flip_eff`).
- **KL_mean** = 0.5 * (mean KL_country + mean KL_generic); KL(clean || edited) of the next-token distribution at
  the final position on (a) `COUNTRY_Q` for the held-out cities (edit at the city positions), (b) 40 generic
  sentences (`GENERIC_SENTENCES[10:]`, the first 10 are what the tools use), with each edit placed at the token
  at the same relative index as the city in a paired held-out prompt (clipped; `final` -> last token).
- **reward = F * (1 - min(1, KL_mean / kl_budget))**. Pass = reward >= threshold (default 0.5).
- `grade` returns every component with n and Wilson CIs (rates) or mean/sd/95% normal CI (KL), plus
  `to_source`, `per_template`, `state_kept`, `state_clean_correct`.

## Tools (`ToolEnv.call(name, args) -> JSON text`)
`describe_task`, `run_prompts(prompts | template+cities, top_k)`, `logit_lens(template, city, layer, position)`,
`cache_mean(name, cities, template, layer, position)`, `vec_op(name, expr)`, `vec_info(name)`,
`eval_intervention(template, cities, edits | vector+layer+position+scale, n_generic)`,
`optimize_vector(name, layer, position, dev_cities, templates, steps, kl_weight, init, lr)`, `submit(edits)`.
- Templates: dev keys, `state_q`, `country_q`, or any string containing `{city}` (except held-out template strings).
- Held-out guard: rejects held-out cities, raw prompts mentioning one, and prompts matching a held-out template.
  Any other city (including source-state cities the agent knows that are not in the pool) is allowed.
- Budget: 1 unit per prompt sequence run; clean runs cached per prompt; `optimize_vector` costs
  steps x (prompts + 4 generic if kl_weight > 0). At most 16 prompts per call.
- `vec_op` grammar (hand-written parser, no eval): numbers, registers, `+ - * /` (scalar mult/div only),
  parentheses, `normalize(v)`, `proj(v,u)`, `proj_out(v,u)`, `randn(seed)` (unit vector), scalar
  `dot(u,v)`, `norm(v)`, `cos(u,v)`.
- `submit` validates constraints (rejected -> error, agent may retry); a valid submit ends the episode. The agent
  never sees the grade.

## Baselines
```
python -m edithunt.env.baselines --model Qwen/Qwen2.5-1.5B --tiers easy,medium,hard --n 5 --threshold 0.5 \
    [--instances_dir results/instances/Qwen2.5-1.5B] [--handoff L] [--steps 30]
```
Agents: `meandiff` (C1: dev mean difference at city_last, best of 3 layers x scale {1, 1.5} by dev flip - generic
KL), `gradient` (C2: `optimize_vector` with kl_weight 1), `random` (norm-matched random direction). Saves
`results/<model>/phase6_baselines.json` and the instances it generates.

## Agent loop
```
python -m edithunt.env.agent_loop --instance results/instances/<model>/<tier>/<id>.private.json \
    [--model claude-sonnet-5-5] [--max_turns 40] [--max_tokens 16000] [--effort high] [--no_fallbacks]
python -m edithunt.env.agent_loop --instance ... --dry_run     # scripted fake client, no network
```
- Key: `ANT_KEY` from `.env` (python-dotenv). Server-side refusal fallback (`fallbacks: "default"`, beta header
  `server-side-fallback-2026-07-01`) is on by default; disable with `--no_fallbacks`.
- The first user message is the `describe_task` output; the loop runs tool calls until `submit`, `max_turns`,
  a refusal, or a third reply without tool calls (2 "continue" nudges max). Log: `results/agent_runs/<tier>/<id>.json` (transcript,
  tool log, budget, usage, grade).
