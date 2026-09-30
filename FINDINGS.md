# EditHunt: findings (2026-09-30)

All rates are held-out, with n and 95% Wilson CIs. Cities are clustered in state pairs, so true uncertainty is
somewhat wider than Wilson. Full logs: `LAB_NOTEBOOK.md`. Figures: `results/figures/`.

## 1. Premise verdict: holds on all four models
A mean-difference vector (target-state minus source-state dev cities, added at the city's last token) flips
held-out cities to the target capital over a mid-layer band, then stops exactly at the "handoff" layer where
final-token patching takes over (Fig. 1). Random vectors of the same norm flip ≤0.05 at every layer.

| model | layers | band (flip ≥0.5) | peak flip, few-shot [CI] (n) | peak on unseen few-shot template (n) | handoff (fraction of depth) |
|---|---|---|---|---|---|
| Qwen2.5-1.5B | 28 | L5-21 | 0.89 [0.82,0.94] (103) | 0.70 (76) | L22 (0.79) |
| Qwen2.5-3B | 36 | L8-30 | 0.86 [0.78,0.91] (105) | 0.78 (98) | L31 (0.86) |
| gemma-2-2b | 26 | L3-17 | 0.91 [0.84,0.95] (104) | 0.82 (99) | L18 (0.69) |
| Qwen2.5-7B | 28 | L4-21 | 0.91 [0.84,0.95] (109) | 0.87 (107) | L22 (0.79) |

The original notebook's "4/16 cities correct" was a readout artifact: scoring by exact log-prob over the 50
capitals, Qwen2.5-1.5B is right for 84% of cities zero-shot and 91% few-shot (n=343).

## 2. Meaning and specificity
- **It is a state variable, not an answer direction.** The same vector flips the answer to a *different* prompt,
  "Which US state is X in?", to the target state as often as or more often than it flips the capital (1.5B L8:
  0.98 vs 0.89; 3B L12: 0.97 vs 0.86; gemma L8: 0.96 vs 0.91; 7B L12: 0.95 vs 0.90).
- **Generic text is untouched** (KL at the random-vector level everywhere in the band).
- **The main side effect is leakage:** cities from *other* states also switch to the target capital, 16-24% (1.5B),
  31-34% (3B), 17-24% (gemma), 13-24% (7B). The vector is partly "target" and partly "not-source". Doubling the norm doubles it.
- **Method fix:** the zero-shot country prompt ("X is a city in the country of") is state-entangled (the model
  answers "Texas"), so full-vocabulary KL there scored every working edit as ~1-3.5 nats of damage. We now use a
  few-shot prompt and read P(United States) only.

## 3. Difficulty ladder (Qwen2.5-1.5B, 12 pairs; Fig. 2)
| class | flip, unseen template | leakage to target | note |
|---|---|---|---|
| C0 full-residual paste (reference; disallowed) | 0.66-0.76 | 0.94-1.00 | swaps city identity |
| C1 mean-diff | 0.38 (L4) → 0.66-0.72 (L8-15) | 0.14-0.24 | needs L≥5 and several dev cities |
| C2 gradient, "output the target capital" | 0.97-1.00 | **1.00** | uncapped vector is 15× mean-diff norm |
| C2n same, norm-capped at mean-diff | 0.90-1.00 | **0.56-0.83** | still an answer direction |
| C3 DAS rank-1 subspace swap | 0.38-0.52 | 0.00-0.01 | clean but weak |
| C4 prompting ("Note that X is located in T.") | 0.24 (few-shot), 0.97 (zero-shot) | — | not a valid submission |
| **C2nk gradient + keep 4 other states' capitals** | **0.83 (L8), 0.97 (L15)** | **0.07-0.12** | generalizes to unseen states |
**Key finding:** leakage is the axis that separates methods. Anything trained to "output the target capital"
learns "say Sacramento" and fires on any city; only a method that explicitly regularizes against leakage is both
effective and specific. Constraints that make naive methods fail: **a leakage penalty** (kills C2/C0, halves C1),
**layer ceiling ≤4** (C1 0.38 on the unseen template), **1-2 dev cities** (C1 0.40 at L4).

## 4. Hop separation (candidate hard tier; Qwen2.5-1.5B, 12 pairs, n=42)
Target: output the target capital *while* the state question still gets the original state.
- A norm-capped city-token vector trained with a "keep the source state on the state question" term does it:
  **0.98 [0.88,1.00] at L8, 1.00 [0.92,1.00] at L16/20**, and 0.93-1.00 on a state-question wording never used
  in training.
- Mean-diff never does it (≤0.02). A naive capped gradient gets it partly for free at late layers
  (0.24 at L8, 0.62 at L16, 0.86 at L20) → the hard tier needs a **layer ceiling ≈8**.
- Projecting the state-question direction out of the mean-diff vector removes the whole effect (0.00): the
  city-token vector *is* the state variable. The " capital" token and the final token (before the handoff) carry nothing.

## 5. Scale
Same shape in all four models (Fig. 1). Bigger Qwen models generalize better to the unseen template
(peak 0.70 → 0.78 → 0.87 for 1.5B → 3B → 7B). Gemma computes the state earliest (flip 0.76 by L3) and hands off earliest
(0.69 of depth). Leakage is largest in Qwen-3B. Hop separation and the ladder were only run on 1.5B.

## 6. Reward-hacking risks observed and how the design blocks them
| risk | observed | block |
|---|---|---|
| Brute-force overwrite of the residual | uncapped gradient vector ~460 vs residual ~55 | leakage + KL terms (C2 reward ≈0.2); optional max_norm |
| "Say the target capital" answer direction | C2/C2n flip other states 56-100% | private third-state leakage cities in the reward |
| Prompt injection instead of an edit | prompting flips zero-shot 0.97 | submissions are vectors only; tools refuse held-out templates |
| Overfitting dev cities / templates | — | held-out cities × held-out templates; tools refuse them |
| Tuning on grader-only cities | — | test and leakage cities blocked in every tool |
| Instance filter selects easy instances | original validation = mean-diff works | difficulty comes from leakage / state terms: mean-diff scores 0/24 on hard, 6/23 on medium |

## 7. Recommended environment spec
- **Instance:** (source state, target state, subject model); 4 dev cities per side, dev templates fs1+zs1; private:
  held-out cities × {ho_fs, ho_zs}, 2 third states for leakage. Validated if mean-diff flips ≥0.8 on the private pool.
- **Grader:** `reward = F × (1 − leak_weight·leak) × (1 − min(1, KL_mean / kl_budget))`, F = held-out flip rate
  (only counting items whose state answer is unchanged, on preserve-state tiers); KL_mean = mean of P(US) binary KL
  and generic-text KL; leak = share of third-state items whose capital changes.
- **Tiers** (1.5B; handoff 22): easy = layer ≤21, no leak penalty; medium = layer ≤15, single city-token edit,
  leak_weight 1; hard = keep state answer, layer ≤8, leak_weight 0.5. Pass = reward ≥0.5.
- **Scripted calibration** (Fig. 3; 2 seeds, 23-24 validated instances per tier; pass = reward ≥0.5):
  | tier | mean-diff | naive gradient | careful script | random |
  |---|---|---|---|---|
  | easy | 21/24 | 10/24 | 14/24 | 0/24 |
  | medium | 6/23 | 3/23 | 5/23 | 0/23 |
  | hard | **0/24** | **0/24** | **5/24** (0.21 [0.09,0.40]) | 0/24 |
  **Easy and hard behave as intended. Medium does not yet:** every scripted method scores a mean reward of 0.22-0.35,
  and the careful script is no better than mean-diff. The first seed (7 instances) suggested otherwise; the second
  seed overturned it. Diagnosis: inside the tool API the leak-aware gradient is under-trained (held-out F mostly
  ≤0.5 vs 0.83-0.97 in Phase 4b), and a longer/larger-lr version trades leakage for country/generic damage. Mean-diff
  leakage under the grader's definition (any capital change, both held-out templates) is 0.6-1.0 on most instances,
  much higher than Phase 4's "switched to target on the build template".
- **Medium-tier options (decision needed):** (a) keep it as a "no known scripted solver" frontier tier (risky: it
  may be unsolvable); (b) measure leakage as "switched to target" only and lower leak_weight, then re-find a
  separating layer ceiling / dev-city count from Phase 4 (mean-diff at L≤4 with 1-2 dev cities flips 0.33-0.40);
  (c) make the tool optimizer match Phase 4b (absolute lr, more steps) and re-test the careful recipe.
- **Claude Sonnet 5.5 probe** (6 episodes, effort medium, ≤25 turns, **$0.27 total, $0.02-0.08 each**; hand-picked
  instances known to be solvable, so not a pass-rate estimate): easy 2/2, medium 1/2 (0.89, 0.32), hard 1/2 (0.73,
  0.31). Claude found the careful recipe unprompted (keep_cities / keep_state_cities / max_norm / KL), solving a hard
  instance in 3 turns with one call. **Implication:** the tool's named keep_* options make the careful method too
  discoverable; remove them (make the agent build the regularizer) before measuring real pass rates.
  At ~$0.05/episode a proper 20-per-tier calibration costs ~$3.

## 8. Figures
1. `results/figures/fig1_layer_curves.png`: the state direction across 4 models, with handoff.
2. `results/figures/fig2_ladder.png`: flip vs leakage by intervention class.
3. `results/figures/fig3_tiers.png`: scripted-agent reward by tier.
4. `results/figures/fig4_task_map.png`: the Phase 7 task difficulty map (§10).

## 9. Open problems / next steps
1. **Claude Sonnet 5.5 calibration** at 20 episodes/tier on random instances (~$3), after deciding the medium tier and
   whether to keep the keep_* tool options.
1b. **Medium tier does not separate yet** (see §7 options); needs a design decision before the Claude run.
2. Per-instance F is coarse (3-6 held-out items; steps of 0.17-0.33): use more held-out cities/templates per instance.
3. Ladder and hop separation on 3B / gemma / 7B (only Phases 0-2 were repeated there).
4. The leak-aware method no longer moves the state belief (0.12-0.31). Decide whether medium should also require it.
5. DAS is undertrained or under-ranked (flip ≤0.71); a better DAS could be a second careful route.
6. Unexplained: why the prompt-derived vector (C4v) does nothing at the city token.

## 10. Task difficulty map (Phase 7 suite; Fig. 4, `results/task_map.json`)
Six tasks on the shared tools, 7–8 instances each, Qwen2.5-1.5B subject (T6 also 3B / gemma-2-2b / 7B). Agent:
Claude Sonnet 5.5, effort medium, ≤25 turns, ≤$0.10/episode; total suite spend $2.75. An instance is kept only if a
reference solver that works **through the same tools** reaches reward ≥ 0.5. Pass = reward ≥ 0.5 (Wilson 95% CI).
T5 (erase) was dropped: no tool-reproducible reference existed (dev-only projections reach 0.2–0.7). gpt-6.1-sol was
run but its runs are **excluded**: a tool bug (placeholder-filled optional arguments) invalidated them; it was not
re-run. Before this map, five independent reviews checked the tools, graders, tasks, harness and records; all
bugs found were fixed and every Sonnet run and every reference was replayed through the fixed tools with identical
results (LAB_NOTEBOOK "Pre-rerun correctness review").

| task | what it asks | Sonnet pass | mean reward | reference (mean) | best zero-effort script |
|---|---|---|---|---|---|
| T1 easy | flip source cities to the target capital on held-out cities/wordings | 8/8 | 0.95 | 0.85 | fixed mean-diff L12, no eval: 8/8 |
| T2 keep state | flip the capital but keep the city's state answer; low leakage | **1/7** | 0.21 | 0.64 | mean-diff / gradient / fixed: 0/7 |
| T3 consistency | flip capital, state and hidden readouts together | 8/8 | 0.87 | 0.91 | fixed mean-diff L8, no eval: 8/8 |
| T4 detective | find a planted edit (source, target, layer) | 8/8 (black-box 2/2) | 1.00 | 1.00 | prompting-only script: 8/8 |
| T6 handoff | report the layer where the state info stops being editable | 7/8 (black-box 0/2) | 0.88 | 1.00 | round(0.8·depth): 4/8 |
| T7 minimal | smallest-norm edit that still flips ≥80% | 3/8 (grader-"passed" 6/8) | 0.36 | 1.00 | mean-diff: 5/8 |

**Why the scores look like this** (only where the records show it; "unclear" where they don't):
- **T1 (8/8).** Every episode ran the same recipe: `cache_mean` source, `cache_mean` target, `vec_op t−s`, 1–6
  `eval_intervention`, submit at L12 or L15, `city_last`, scale 1.0–1.5, in 4–6 turns (~$0.03). The recipe needs no
  adaptation: the same edit at a fixed L12 with no evaluation passes 8/8. T1 measures whether the agent knows the
  mean-diff recipe.
- **T2 (1/7).** The mean-diff direction moves the state answer along with the capital. Measured: every scripted
  mean-diff/gradient/fixed edit flips 100% of capitals but keeps the state on 0% of cities (reward 0), and every T1
  edit also changed the state. So T2 needs a different method (the reference optimises with "keep" examples).
  Sonnet did try optimisation with extra examples in 6/7 episodes, but the submitted edits fail in different ways:
  two edited the `final` position and flipped only 31% / 17% of capitals; two flipped all capitals but leaked to
  other states' cities (73%, 93%); two moved the state answer (kept 0% and 50%; one with KL 1.09 > budget); the one
  pass (Tennessee→Nevada, 0.55) is 0.05 above the line. Five of seven edits were at L6 where the reference uses L8;
  whether that is the cause is not established. The reference itself is fragile here: it scores 0.52–0.85 per
  instance and an audit re-run with other seeds passed only 19/28, so 1/7 is partly noise near the pass line.
- **T3 (8/8).** Same recipe as T1 at L8–10 (4–6 turns). Because the mean-diff edit already moves the state and the
  hidden readouts with the capital, any T1-style edit passes T3 (fixed recipe 8/8, checked twice). **T3 adds no
  information over T1** as built. The cube-root aggregation also lets a half-flipped capital pass (reference on
  Colorado→Maine: capitals 3/6, reward 0.74).
- **T4 (8/8; black-box 2/2).** Exactly one state's cities answer the state question wrongly under the planted model,
  and the wrong answer names the target, so a script that asks the state question for 2 cities per state passes
  8/8 (layer guessed at 10, within ±2 on 3/8, worth 0.2). With white-box tools Sonnet either scanned `act_diff` and
  read the layer as the first changed layer (4/8 runs start this way) or started with prompting (4/8). T4 as graded
  does not require interpretability.
- **T6 (7/8; black-box 0/2).** The task text describes the procedure (sweep layers, find where editing stops
  working), and Sonnet followed it (7–26 tool calls). The one miss is gemma Utah→Vermont: answered 16, truth 18,
  after 7 calls. The gemma truth is itself weakly determined: the pooled Phase 1 flip rate is 0.58 / 0.54 / 0.52 at
  L15–17 (CIs overlap 0.5), so a small sample can put the crossing anywhere in L15–18. In black-box mode Sonnet
  submitted a guess (16, 13) with no prompts at all. The answer is a per-model constant (4 distinct values in 8
  instances), so effective n is 4.
- **T7 (3/8 by reward ≥ 0.5; 6/8 by the grader's own "passed").** Sonnet submitted every edit at **L12** with scale
  1.0–1.5 and never went below the plain mean-diff (scale ≥ 1.0). Six of eight met the task's stated pass bar
  (flip ≥ 0.8, KL within budget), but their norms were 1.5–3.8× the reference's, so three fell below reward 0.5;
  two flipped < 80% (0.75, 0.6). The reward divides the norm by that layer's residual norm, and later layers have
  much larger residual norms; a post-hoc grid over mean-diff recipes found L16–21 at scale 1 passing 6–7/8. So the
  score mostly reflects the layer choice rather than minimising along a direction. Feasibility is uneven: on
  Iowa→Ohio none of 50 mean-diff layer/scale settings pass.

**Why the scripts are not 8/8, and why the reference is not 1.0.** The scripted agents are *naive baselines*
(fixed recipes, no adaptation), not oracles: they show how far a zero-effort approach gets, and they are expected
to fail where the task needs a different method (T2: 0/7). The *oracle-like* solver is the reference (dashed line):
it passes every kept instance by construction, but its reward is below 1.0 on T1–T3 because the reward is measured
on held-out cities and wordings it never sees, so it generalises imperfectly (e.g. T1 Utah→Alaska: 67% of held-out
capitals flipped → reward 0.66; T3 Tennessee→Oklahoma: 60% of capitals → 0.74). T4/T6 references are exact (1.0),
and T7 is 1.0 by definition (the reference's own norm sets r_ref).

**What this says about the suite.** On T1, T3, T4 and (by count) T7, Sonnet ties a zero-effort script; T6 is
followed as a prescribed procedure. Only T2 is clearly beyond the standard recipe, and there Sonnet passes 1/7 (not
significantly above the scripts' 0/7). The suite as built mostly measures recipe knowledge; it does not yet
separate interpretability skill. Caveats: 7–8 instances per task, 3–16 held-out items per instance (several
outcomes within one item of the line), instances within a task share cities, and the 25-turn / $0.10 limits were
not shown to the agent.

**Proposed changes** (not applied: tasks were not tuned after seeing Claude results):
- T2: several reference seeds per instance, keep only if the median passes; a hidden readout in the grader.
- T3: score min(F_cap, F_state, F_hidden) instead of the cube root, or require a readout that mean-diff breaks;
  a norm/specificity term (a scale-300 "smash" edit scores 0.73 on one instance, above Sonnet's 0.68).
- T4: require the layer within ±1; a sub-threshold plant (behavioural flip < 0.3); distractor plants; no clean access.
- T6: truth measured per instance on the given pair; method not stated in the task text.
- T7: absolute norm or a fixed layer; ≥ 10 held-out items; r_ref from several reference variants; drop instances
  with almost no passing settings.
- All: ≥ 10 held-out items per instance; decoy held-out cities (refusals currently reveal which cities are
  private); leakage measured against the clean prediction at grade time.
