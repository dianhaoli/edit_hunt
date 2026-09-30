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
| Instance filter selects easy instances | original validation = mean-diff works | difficulty now comes from leakage / state terms, so mean-diff still fails medium/hard |

## 7. Recommended environment spec
- **Instance:** (source state, target state, subject model); 4 dev cities per side, dev templates fs1+zs1; private:
  held-out cities × {ho_fs, ho_zs}, 2 third states for leakage. Validated if mean-diff flips ≥0.8 on the private pool.
- **Grader:** `reward = F × (1 − leak_weight·leak) × (1 − min(1, KL_mean / kl_budget))`, F = held-out flip rate
  (only counting items whose state answer is unchanged, on preserve-state tiers); KL_mean = mean of P(US) binary KL
  and generic-text KL; leak = share of third-state items whose capital changes.
- **Tiers** (1.5B; handoff 22): easy = layer ≤21, no leak penalty; medium = layer ≤15, single city-token edit,
  leak_weight 1; hard = keep state answer, layer ≤8, leak_weight 0.5. Pass = reward ≥0.5.
- **Scripted calibration** (Fig. 3; 7-8 instances per tier): easy — mean-diff 7/8, naive gradient 5/8, careful 7/8;
  medium — mean-diff 1/7, naive gradient 0/7, **careful 3/7**; hard — mean-diff 0/8, naive gradient 0/8,
  **careful 3/8**; random 0 everywhere.
- **Expected frontier-agent band:** easy ≈ solved; medium and hard between the naive (≤0.15) and careful-script
  (≈0.4) pass rates or above. **Not yet measured with Claude:** the calibration run is blocked on API credits.

## 8. Figures
1. `results/figures/fig1_layer_curves.png`: the state direction across 4 models, with handoff.
2. `results/figures/fig2_ladder.png`: flip vs leakage by intervention class.
3. `results/figures/fig3_tiers.png`: scripted-agent reward by tier.

## 9. Open problems / next steps
1. **Claude Sonnet 5.5 calibration** (10-20 episodes per tier): blocked on API credit balance; code dry-run tested.
2. More instances per tier (currently 7-8; per-instance F moves in steps of 0.17-0.33).
3. Ladder and hop separation on 3B / gemma / 7B (only Phases 0-2 were repeated there).
4. The leak-aware method no longer moves the state belief (0.12-0.31). Decide whether medium should also require it.
5. DAS is undertrained or under-ranked (flip ≤0.71); a better DAS could be a second careful route.
6. Unexplained: why the prompt-derived vector (C4v) does nothing at the city token.
