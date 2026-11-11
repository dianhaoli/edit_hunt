# T8_undo: overview and results so far (2026-11-11)

Companion files in this folder:
- `T8_TRANSCRIPT_ANALYSIS.md`: merged failure and success report.
- `failure_report.md` / `success_report.md`: the two reports separately.
- `failure_episodes.json`: per-episode root-cause analyses for all 32 episodes, each with an independent skeptic verdict.
- `success_episodes.json`: success anatomies for 13 episodes and 6 paired success-vs-failure contrasts.

These were produced by reading the run JSONs under `results/suite/runs/*/T8_undo*/` against the private instance files. **Agent reasoning is not logged:** `edithunt/env/agent_loop.py:328` drops `reasoning` items. So all motives are inferred from tool calls and tool results.

## 1. What T8 is

T2 could be imitated from visible data ("proxy + optimise"). T8 is designed so that it cannot (`edithunt/env/undo.py`).

### The plant

A hidden edit is planted in Qwen2.5-1.5B so that source-state cities answer as if they were in the target state. It is added at layer L, at every token position:

    h += clamp((k·h − b)/w, 0, 1) · v

The gate fires on the model's own representation of the source state, so it also spills onto neighbouring states. There are four kinds:

| Kind | What is planted |
|---|---|
| **hop2 L8** | v is trained so that the capital changes while the state and abbreviation answers stay the same |
| **state L16** | v = mean(target) − mean(source) |
| **spread** | split across L8, L9 and L10 |
| **two** | two plants, e.g. MN→TN L8 + CA→PA L14, or NC→KY L8 + WI→UT L14 |

### What the agent has

- **Tools:** run_prompts, logit_lens, cache_mean, project, vec_op, optimize_vector, eval_intervention.
- **Budget:** 12k forward passes, rank ≤ 2 edits.
- **No access to:** the original model, the test cities (private), the eval prompts, or what was planted.
- **Submission:** an additive edit, optionally gated.

### Scoring and pass criteria

- **Restoration:** 1 − Σ KL(orig‖fixed) / Σ KL(orig‖planted). It is computed over held-out source cities, other cities and generic text, each in 9 private contexts. A null edit scores 0 and an exact undo scores 1. Damage to cities the plant did not touch counts against the agent.
- **Solver:** the privileged inverse, scoring 0.84–0.98.
- **Pass:** ≥ 0.8 × solver.
- **Library:** 4 scripts (T2-style patch, detect v1, v2, v3).
- **Hard-instance filter:** best script ≤ 0.5 × solver and solver ≥ 0.6.

## 2. Runs

| Phase | What | Outcome |
|---|---|---|
| v0.1 A5 r1 | sol, 4 MN→TN instances, hidden 60-turn cap | 0/4; all capped, 3 holding a passing candidate |
| v0.1 A5b | same instances, 150 turns | 1/4 (hop2 0.961) |
| v0.2 | turns_left shown; private cities skipped instead of refusing the batch; opaque ids; +4 NC→KY instances | the filter keeps 1 of 8 (NC two-plant) |
| P1 | sol, 8 × 2 runs | 4/16 pass |
| P2 | luna, 8 × 1 run | 0/8 |
| P3 red-team | – | never ran: the $32 OpenAI ledger cap was reached |

Overall: 5 of 32 episodes passed (sol 5/24, luna 0/8). No state or two-plant episode has passed yet.

## 3. v0.2 results by instance

"held" = the best candidate the agent had evaluated when it was cut off. ✓ = pass.

| Instance | Solver | Best script | sol r1 | sol r2 | luna |
|---|---|---|---|---|---|
| MN→TN hop2 L8 | 0.983 | 0.93 | 0.921 ✓ | 0.978 ✓ | 0 (cap) |
| NC→KY hop2 L8 | 0.965 | 0.88 | 0.928 ✓ | 0.762 (held 0.964) | 0 (cap) |
| NC→KY spread | 0.956 | 0.82 | 0, cap (held 0.783) | 0.796 ✓ (thin margin) | 0 (cap) |
| MN→TN spread | 0.911 | 0.84 | 0.027 | 0, cap (0.478) | 0 |
| NC→KY state L16 | 0.905 | 0.85 | 0, cap (held 0.768) | 0, cap (0.665) | 0 (cap) |
| MN→TN state L16 | 0.841 | 0.90 | 0, cap (held 0.910) | 0, cap (held 0.931) | 0 |
| MN two-plant | 0.867 | 0.49 | 0, cap (0.609) | 0.514 | 0 |
| NC two-plant (kept by filter) | 0.935 | 0.42 | 0.298 | 0, cap (0.384) | 0 |

## 4. Findings from the transcripts

### How successes work

Every sol pass follows the same recipe:

1. Probe the spill-over cities, e.g. Fargo → "Nashville".
2. Do per-layer model diffing of affected vs control residual updates, and look for the one-layer spike (L8 jump norm 38 vs 13).
3. Take the diff as the plant vector (cos ≈ 0.97 with the true plant).
4. Undo it with a self-keyed clamp gate, sized from the projection plateau.
5. Validate: affected cities flip back and generic KL stays ≈ 0.001.

The best run had a passing edit at turn 10 for about $0.07. Sol found this route unprompted, and it beat the solver on 3 instances.

### Why failures happen (ranked)

1. **The hidden $0.90 cost cap.** turns_left is shown but cost is not, and context growth makes later turns more expensive. The cap therefore binds at around turns 80–120, while the agent still sees 27–48 turns left. 7 sol runs were holding a passing candidate when this happened. With environment fixes, sol v0.2 would plausibly score 6–8 of 16 instead of 4.
2. **Two-plant instances: 0 of 8.** Agents fix the first plant and never look for a second. In addition, the missed plant's state was only ever probed through private cities, which are silently skipped. Even a perfect single-plant fix caps out at about 0.60.
3. **Selection without the original model.** The agent's proxies measure distance from the planted model, not from the original. Polishing with optimize_vector lowered true R every time (0.964 → 0.762 in 651dcd13.r2), because it trains toward real-world answers rather than the original model's answers.
4. **Spread instances.** There is no single-layer spike to find. MN spread went 0/5: agents anchored on L5 (natural state formation) or patched downstream.
5. **Luna (0/8).** It does little model diffing (7.5 cache_mean calls per episode vs 51 for sol). It mistakes noise or spill for the plant. In 5 of 8 runs it never queried an accessible source-state city.

### Measurement gaps

- Reasoning is not logged.
- The audit regrades only the last 6 candidates.
- Restoration is floored at 0, so harmful submissions look the same as no-ops.
- n = 2 per instance for sol and n = 1 for luna. Several pass margins are inside bootstrap noise.

## 5. Why the budget was so strict

Two reasons, both recorded in LAB_NOTEBOOK.md:

1. **Money.** The project started from a $5 Anthropic credit (l.~420), and later a $50 OpenAI ledger. T8 v0.2 was approved for an extra $20, giving a $32 stop (l.1028). The runner reserves the per-episode cap from the ledger before each episode (l.468). So the per-episode cap is what guarantees that the total stays under the stop.
2. **Hidden on purpose, for comparability.** The Sonnet calibration runs (25 turns / $0.10) did not show their limits. They were kept hidden for sol too, so the two could be compared; rerunning Sonnet with visible limits was not affordable (l.619). v0.2 fixed the turn cap but not the cost cap.

Nobody anticipated that every turn re-sends the full transcript. As a result, $0.90 binds before 150 turns.

The work itself fits the budget: a pass needs about $0.07–0.40. The problem is that the cap is invisible, not that it is too small.

## 6. Expected cost per rollout

Estimated from the 24 v0.2 runs plus 12 submitted sol runs:

| Model | Typical | Worst case (all 150 turns) |
|---|---|---|
| gpt-6.1-sol | ~$0.55–0.80 (submitted runs: $0.36–0.91) | ~$1.3–2.0 (it reaches $0.90 at about turn 100; cost grows superlinearly) |
| gpt-6-luna | ~$0.37–0.53 when it submits (rare) | ~$2–5 (it reaches $0.90 by about turn 63; priced at sol rates as an upper bound) |

A rerun of the 16 sol v0.2 episodes would cost about $10–13 with cost_left visible. With a $1.50 per-episode cap it would cost about $12–16, and the ledger would reserve $24 as the worst case.

## 7. Next steps

1. **T8 v0.3 environment changes:**
   - Show cost_left, or allow a provisional submit.
   - Log reasoning summaries.
   - Remove private cities from known_cities, or list accessible ones first.
   - Add a batched least-squares tool.
   - Remove "clean" from the T8 model enum.
   - Grade every candidate in the audit.
2. **Check that a second plant can be discovered at all.** Run a script that sweeps the accessible cities of every state; it needs no API spend.
3. **Build more two-plant and spread instances,** which are the ones that discriminate. Keep single-plant instances as warm-ups (detect v3 solves 6 of 8).
4. **Rerun sol** on the v0.3 environment.
