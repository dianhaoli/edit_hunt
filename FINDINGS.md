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

## 11. T2 as RAVEL-style RL env (T2_keepstate v2; Qwen2.5-1.5B; no agent spend)
**Framing.** T2 is rebuilt as an RL-environment version of RAVEL (Huang et al., ACL 2024) with MIB's RAVEL-track
convention (fixed intervention site: the entity's last token at a layer given per instance). Entity = a US city;
attribute to change = its state's capital (Cause, held-out cities x held-out wordings); attributes that must not
change = the city's state (Iso_state: held-out state question AND a hidden postal-abbreviation readout) and other
states' cities' capitals (Iso_other: >= 12 private cities sampled at grade time from a large pool, compared with
the clean prediction, bf16 near-ties < 0.1 nats dropped). Code: `edithunt/env/ravel.py` (grader v2; v1 kept),
tools give a generic "not accessible" refusal and the private set includes decoy cities. >= 10 held-out source
cities per instance required an extended city list (`data.CITIES_EXTRA`, T2 v2 only).

**1. The plan's reward did not discriminate.** With Disentangle = (Cause + Iso)/2, a zero edit scores exactly 0.5
(Cause 0, Iso 1) and "flip everything" scores ~0.5 (Cause 1, Iso 0). First 4 pairs, L6-20: null 0.50, random
0.44-0.47, mean-diff 0.43-0.52, naive gradient 0.04-0.63, plan's oracle 0.56. Pass >= 0.5 passes the null edit.
We switched to the product reward **R = Cause x Iso_state x Iso_other x (1 - min(1, KL_mean))**: null = 0.

**2. A single additive vector cannot isolate (structural).** The plan's oracle (norm-capped vector, target NLL +
keep-state + keep-own-capital, 150 steps) reached Cause ~0.4-0.56, Iso_state ~0.7, Iso_other ~0.6. Loosening the
norm cap changed nothing (the trained norm stays 1.23-1.38 x |mean-diff|, the cap never binds); heavier keep terms
or extra state wordings raise Iso_state to 0.86-0.93 but drop Cause to 0.12-0.16. Iso_other stays 0.5-0.7 in every
variant: one fixed vector added to every city moves other states' cities too. RAVEL avoids this by construction
(interchange interventions depend on the base entity).

**3. A gated (conditional) edit is the existence proof.** x <- x + clamp((k.x - b)/w, 0, 1) v at the city's last
token: k = unit(mean dev source - mean accessible other-state cities), the ramp b..b+w placed 30%-60% of the way
between the two group means (other states project ~0, source cities ~25), only v trained (a learned threshold
overfit to the 4 dev cities: held-out activations 2-6 vs dev ~20, Cause 0). Isolation becomes ~perfect everywhere
(Iso_state >= 0.71, Iso_other >= 0.94, mostly 1.00 over 16 runs); Cause is the limit and is pair- and
layer-dependent (Alabama->Maryland L20 0.92-0.96, Indiana->New Mexico L8-16 0.89-0.95, California->Idaho <= 0.52).

**Fit on dev data only (step 2).** The gate direction k, threshold b and ramp w come from the 4 public dev cities
(fs1, zs1) and accessible keep cities of 8 other states (disjoint from the Iso pool states); v is trained on the
dev cities (target capital), the visible STATE_Q on dev cities (keep state) and the keep cities (keep own
capital). Held-out source cities, private wordings (ho_fs, ho_zs, STATE_Q_HO, abbreviation readout) and the Iso
pool are never used; `train_gated` asserts this at run time (`held-out leak`).

**Grader fix (step 1, decided before any agent run).** The country term now counts binary KL on P(US) only where
the edit LOWERS P(US), over held-out cities whose name is not also a well-known non-US place (`data.AMBIG_NONUS`).
Reason: on Colorado->Idaho L20 the gated edit (Cause 0.87, Iso 1/1) had KL_mean 0.63, all from the country probe
(1.26; generic 0.002), because it moved Colorado cities from P(US) 0.63 to 0.93 (Durango, Pueblo are also Mexican
places), i.e. toward the correct answer. All 123 earlier runs were regenerated (122/123 reproduce Cause/Iso
exactly; 1 random-vector run differs by 0.06 on Iso_other) and re-graded (`results/suite/t2_ravel/regrade_summary.json`):

| method (runs) | KL old -> new | mean R old -> new | max R old -> new | R >= 0.6 old -> new |
|---|---|---|---|---|
| gated oracle (27) | 0.202 -> 0.096 | 0.46 -> 0.53 | 0.86 -> 0.91 | 8 -> 12 |
| additive oracle (14) | 0.224 -> 0.162 | 0.17 -> 0.19 | 0.34 -> 0.36 | 0 -> 0 |
| naive gradient capped (16) | 0.242 -> 0.178 | 0.07 -> 0.08 | 0.28 -> 0.34 | 0 -> 0 |
| naive gradient uncapped (15) | 0.412 -> 0.359 | 0.01 -> 0.01 | 0.07 -> 0.07 | 0 -> 0 |
| mean-diff (17) | 0.047 -> 0.024 | 0.01 -> 0.01 | 0.03 -> 0.03 | 0 -> 0 |
| random (17) / null (17) | 0.03 / 0 | 0.00 | 0.00 | 0 |

**Recipe choice on dev data only.** Default gated (150 steps, fs1+zs1) vs boosted (300 steps, + fs2/zs2), scored
leave-one-dev-city-out (held dev city's capital and visible state answer, unused accessible other-state cities,
KL on agent-visible generic sentences): 16 comparisons (4 tuning pairs x L8/12/16/20), boosted better 0, default
better 7, tied 9; mean proxy 0.377 vs 0.260 (boosted raises KL). Fixed recipe: default gated.

**Validity bar.** Instance valid = oracle R >= 0.6 on >= 2 of 3 seeds at its layer (fixed before step-3 oracle
results; the best of 82 shortcut runs is 0.34). 0.5 and 0.7 reported for sensitivity.

**Step 3: validation on fresh pairs** (`experiments/t2_validate.py`, `results/suite/t2_ravel/validate_summary.json`).
20 pairs were drawn (none of them a tuning pair; 9 candidates skipped for < 4 usable target/dev cities, mostly
West Virginia / Georgia targets and Missouri as source). Shortcuts ran on all 20 pairs x L{8,12,16,20}; the oracle
(default gated recipe, 3 seeds, seed = which accessible keep cities + init noise) ran on 10 of them (the user chose
to stop after half: the answer was clear). Seeds agree closely (e.g. 0.70/0.70/0.68).

| layer | valid @0.6 (oracle R >= 0.6 on >= 2/3 seeds) | oracle mean R / Cause / Iso_state / Iso_other / KL (all 30 runs) |
|---|---|---|
| 8 | 2/10 | 0.35 / 0.40 / 0.90 / 0.96 / 0.06 |
| 12 | 0/10 | 0.27 / 0.31 / 0.88 / 0.98 / 0.13 |
| 16 | 4/10 | 0.53 / 0.66 / 0.93 / 0.95 / 0.16 |
| 20 | 5/10 | 0.63 / 0.79 / 0.93 / 0.93 / 0.09 |
| all | **11/40 = 27.5% (Wilson 95% CI 16-43%)**; @0.5: 19/40, @0.7: 9/40 | |

On the 11 valid instances the oracle averages R 0.69-0.84 (Cause 0.77-0.93, Iso ~0.95, KL <= 0.03) and **every
shortcut passes 0** (null, random, mean-diff, naive gradient capped and uncapped). Over all 400 shortcut runs (20
pairs) none reaches 0.6; the best is 0.51 (naive capped gradient, Oregon->Idaho L12: Cause 0.88, Iso_state 0.86,
Iso_other 0.73), which is why the bar is 0.6 and not 0.5. Shortcut profiles: mean-diff flips (Cause ~0.8) but moves
the state (Iso_state 0.00) and other cities (Iso_other ~0.35); naive gradient flips everything (Cause 0.9-1.0) and
breaks both Iso terms; random/null change nothing (Cause 0).

**Verdict.** The task is possible (an offline gated edit solves it with near-perfect isolation, and no shortcut
comes close), but the oracle validates only ~28% of instances (50% at L20), below the 60% target. The bottleneck
is Cause: held-out source cities that the dev-fitted edit does not flip. Validity is pair-dependent more than
seed-dependent (e.g. Minnesota->Tennessee, North Carolina->Virginia pass at L16-20; Colorado->New Jersey fails at
every layer). If T2 v2 is used now, use only validated (pair, layer) instances and prefer L16-20.

**Why Cause is limited (option 3; diagnosed on the 4 tuning pairs only, never on validation pairs).**
Per held-out item, the gate value and the effect with the gate forced open (`experiments/t2_cause_diag.py`):
- *Late layers: the push works, the gate does not open.* At L20, forcing the gate open gives Cause 0.96-1.00 on all
  4 pairs (California->Idaho: Cause 0.50, gate fully open on only 9/42 items, 1.00 if forced open).
- *Early/mid layers: the push itself does not generalise.* Alabama L8/12 and Colorado L8-16: Cause ~0.4-0.5 even
  with the gate open (the layer effect seen throughout).
So at late layers the bottleneck is recognising held-out source cities from 4 dev cities. Dev-only attempts to
improve the detector, none of which helped:

| detector (fit on dev + accessible cities only) | held-out dev city opens | unused other-state cities stay closed |
|---|---|---|
| mean-difference key, current ramp (30-60% of the gap) | 0.38 | 0.91 |
| same, ramp 10-30% / 5-20% | 0.95 / 0.99 | 0.54 / 0.40 |
| mean-difference key, threshold at others' mean + 2-4 sd | 0.33 | 0.96 |
| LDA key (shrunk covariance of other-state cities) | 1.00 | 0.55-0.80 (overfits the fit half) |
| multi-class state probe (logistic, 3 source examples) | 0.01-0.31 | 0.99-1.00 |
| logit lens: source state's name top among 50 states | 0.00-0.16 (the state is not readable as a word at the city token) | |

Effect-based check of lower ramps with v trained (`experiments/t2_ramp_effect.py`, leave-one-dev-city-out, 16
comparisons each): ramp 30-60% proxy 0.585 (cause 0.69, other 0.96); 20-45% 0.569 (0.81, 0.87); 15-35% 0.551
(0.84, 0.80). Lower thresholds trade isolation for Cause about one for one, so the ramp is unchanged.
**Conclusion:** with one edit site and a detector fitted from 4 public dev cities, ~28% of instances validate
(~50% at L20). Raising it needs more information about the source state than 4 cities give: more public dev
cities per source (needs >= 18 usable cities per state, i.e. more data), or validating instances per (pair, layer)
and keeping only those (what step 3 does).

**Step 4 (proposal, not implemented): how an agent builds a gated edit from primitive tools.** No named "gated"
option; instead two generic primitives that also cover ordinary steering:
1. A read-only `project(register, template, cities, layer)` tool returning k.h at the city's last token (h = the
   residual after `layer`), so the agent can see how a candidate key separates source cities from others and pick
   a threshold. The key itself is built with existing tools (cache_mean on dev source vs other-state cities,
   vec_op difference).
2. An optional input-dependent scale on any additive edit: `scale_from = {key: register, lo: float, hi: float}`
   meaning scale = clamp((k.h - lo)/(hi - lo), 0, 1) (constant scale when absent). The same field in
   `optimize_vector` lets the agent train v with the gate fixed. Wide lo/hi gives a linear rank-one (ROME-like)
   edit v (k.h); a constant key gives the current additive edit, so the primitive is not a disguised solution.
Grader: validate the key's shape/finiteness and lo < hi; KL is measured as now, so a gate that opens on unrelated
text is penalised. Rejected alternatives: firing only on a listed set of cities (the agent would tell the grader
which cities are the source; it must generalise to held-out cities from activations), and an MLP weight edit
(linear in h; isolation then needs centring so other cities read ~0, weaker than a clamp).
