# Audit: T2_keepstate v2 (RAVEL-style RL task), FINDINGS §11 and LAB_NOTEBOOK (2026-10-01)

Auditor: independent Claude session, 2026-10-01/02. No API spend. Nothing in the repo was edited except this file.
GPU runs (Qwen2.5-1.5B, shared A10G) were used for a scripted gated baseline, an offline replay of every agent
episode, and `verify_references.py`. Scripts and raw rows from this audit are listed in §6.

## 1. Verdict

**Not yet a valid basis for the §11 agent claims.** The grader is implemented correctly and nothing gives an agent the
graded data wholesale. But: (i) **the agents were told a different reward than the one they were scored on.** The task
text says the reward is the averaged `Disentangle x KL`, where a null edit gets 0.5. The audit table uses the product
reward. (ii) **The task can be solved by a fixed script.** A 100-step script that uses only the documented tools
passes 9/10 pilot instances under the pre-registered rule (mean R 0.82, 93% of ceiling, 2,456 of 12,000 forward
passes). gpt-6.1-sol gets 0.53 mean R and passes 5/20. (iii) **The held-out guard can be bypassed with a spelling
alias** ("Saint Cloud" reaches the graded city "St. Cloud"). The numbers in §11 recompute exactly from the raw files.
What does not hold up is their interpretation: "agents discover the gate", "60% of ceiling", "discriminates", and the
RAVEL framing.

## 2. Findings, ranked by severity

### CRITICAL

**C1. Agents were optimising a different reward from the one they were graded on.**
- Where: `edithunt/env/tasks.py:104-114` (`describe`, task "ravel") tells the agent
  `reward = Disentangle * (1 - min(1, KL_mean / kl_budget))` with `Disentangle = (Cause + Iso)/2`. `grade_ravel`
  returns that same averaged value as `"reward"` (`edithunt/env/ravel.py:185`). Every §11 number, the pass rule and
  the "% of ceiling" use `reward_mult = Cause x Iso_state x Iso_other x KL factor`.
- Checked: the first user message of every one of the 30 T2 v2 episode records
  (`results/suite/runs/*/T2_keepstate_v2/*.json`, `messages[0]`) contains the averaged formula. Neither FINDINGS §11
  nor the notebook mentions the mismatch. The system prompt says the agent "will not see the grade", so this text is
  the agent's only statement of the objective.
- Why it matters: under the stated objective, a null edit scores 0.5. A too-strict gate (Cause 0.3, Iso 1) scores
  0.65 under the stated reward and 0.3 under the graded one. That is exactly the "typical failure" §11 attributes to
  agent skill. Re-scoring the recorded grades with the reward the agents were given, against 0.9 x the reference's
  averaged reward: **sol 12/20 pass (mean 0.74) instead of 5/20 (0.53); Sonnet 1/10.** The agent audit therefore
  says nothing reliable about agents on the product-reward task.
- Fix: in `describe`, state `reward = Cause * Iso_state * Iso_other * (1 - min(1, KL_mean/kl_budget))`. Make
  `grade_ravel`'s `"reward"` field equal `reward_mult` (keep the averaged value under another name). Re-run the agent
  audit. Also fix "sampled at grading time": the Iso sample is fixed (`iso_seed=0`, seeded by instance id,
  `ravel.py:120-122`).

### MAJOR

**M1. A fixed, zero-adaptation script solves the task (shortcut solvability).**
- Checked: §4. I wrote 5 scripted pipelines that use only agent tools and public information (`describe_task`,
  accessible cities) and graded them with `grade_ravel` on the 10 pilot instances. I also ran the tool-path reference
  with seeds 1 and 2.
- Evidence: `opt_keep100` = key from dev-source minus 8 accessible other-state cities, gate 40-60% of the way between
  the two projection means, 100 steps of `optimize_vector` with that `scale_from` and 16 keep examples (state_q on dev
  cities, own capital on 6 other cities). It **passes 9/10 under the pre-registered rule (R >= 0.9 x tool-ref R),
  mean R 0.82, 93% of ceiling, 2,456 forward passes**. sol: 5/20, mean 0.53, 60% of ceiling. The tool reference is
  itself a fixed script (`tasks.py:363-387`) and passes its own rule by construction.
- What does *not* pass: a gate with a plain mean-diff push (0/10, mean 0.02), the gate with a target-only trained push
  (0/10, mean 0.03), and the mean-diff push at x2 or at an auto-chosen scale (0/5 each). These flip the source cities
  (Cause 0.6-1.0) but drive Iso_state to ~0. **The ingredient that decides success is the keep-state term in the
  push, not the gate alone.**
- Why it matters: the shortcut set behind "no shortcut >= 0.6 across 400 runs" (`experiments/t2_validate.py:26`:
  null, mean-diff, random, naive gradient x2) contains no gated baseline. It does not test script-solvability of the
  intended solution family. As built, the task measures whether an agent assembles a known recipe from primitives it
  is handed (M2), not open-ended interpretability.
- Fix: add `opt_keep100`-style scripted baselines to the validation and report them next to agents. Either accept
  "recipe recall" as the construct or harden the task: no `scale_from`/`project` (agents must build conditionality
  from more generic tools), harder negatives (M7), or instances where the simple recipe fails.

**M2. The method is effectively handed to the agent.**
- Where: `tasks.py:99-103` (`edit_format` spells out `clamp((dot(key, h) - lo)/(hi - lo), 0, 1)`), `tasks.py:515-516`
  (the `project` tool is added only for this task), and `agent_loop.py:32-35, 82, 86, 98-107` (schemas and notes).
- The only tools new to T2 v2 are exactly the three steps of the reference solution: `project` to place the
  threshold, `scale_from` on `optimize_vector` to train behind the gate, and `scale_from` on `submit`. The gate is a
  known model-editing primitive: GRACE's key with a deferral radius ([Hartvigsen et al., NeurIPS 2023](https://arxiv.org/abs/2211.11031)),
  i.e. a ROME-style key-value edit with a threshold.
- §11's "Agents discover the conditional edit without being told (sol 17/20 gated submissions)" is overstated. They
  are told the operator exists and how it is computed. 17/20 recomputes correctly. Fix: reword; to measure discovery,
  ablate (run with `allow_scale_from=False` and `project` removed) or expose the primitive in every task.

**M3. The held-out guard has an alias hole: graded cities can be reached by alternate spellings.**
- Where: `tools.py:707-716` (`_norm`/`_letters` normalise case, punctuation, spacing and zero-width characters, but
  not St./Saint, Ft./Fort, Mt./Mount). Guard checks are at `tools.py:186-189, 255-271`.
- Evidence (ToolEnv built from the stored private instance, no GPU): on `Minnesota-Tennessee-L20`, `St. Cloud` is a
  graded test city and is refused, but **`Saint Cloud` is accessible** via both `cities=` and raw prompts. The same
  holds on all 3 Minnesota instances. Iso-pool cities are reachable via `Saint Louis`, `Ft. Collins`, `Ft. Worth`,
  `Ft. Smith`, `Ft. Wayne`, `Saint George`, `Saint Joseph` on all 10 instances. Sonnet actually tried `Saint Cloud`
  and `Saint Louis` (MN-TN-L20). Those calls were refused only because the same batch also contained a held-out city.
- Impact on the current numbers is small: at most 1 of 11-15 test cities, and no recorded call got through. But under
  RL a policy can learn the exploit. Fix: canonicalise these abbreviations in `_letters`/`_norm` (both directions) and
  add a regression test over an alias list.

**M4. The pass rule is fragile, and the ceiling and "% of ceiling" were defined after the agent results.**
- Reference noise: the tool reference with seeds 0/1/2 gives NC->Kentucky L20 0.70 / 1.00 / 0.75 and NC->Virginia
  L20 0.61 / 0.76 / 0.75. The reference passes its own seed-0-defined rule only 9/10 (seed 1) and 7/10 (seed 2). Two
  of sol's five passes are on NC->KY L20, where the seed-0 reference was a low draw (bar 0.63).
- Sol's pass count under reasonable alternatives: 0.9 x fair-oracle mean **3/20**; 0.9 x ceiling **3/20**; 0.9 x seed-0
  reference **5/20** (the reported number); 0.9 x 3-seed reference mean **6/20**; 0.8 x ceiling (the "proposed" rule)
  **10/20**. The headline 5/20 depends on the choice of reference.
- Timeline from file mtimes, all 2026-10-01 UTC: Sonnet runs 17:55-18:06, sol runs 18:25-19:26 plus two re-runs at
  21:07 and 21:12, ceiling rows written at 21:02. The pass rule itself was committed before any agent run (verified:
  it is in 24fdb5e at 17:53). But the ceiling oracle, the "% of ceiling" column and the proposed 0.8 x ceiling rule were
  designed after agent scores were visible. The sol budget ($0.25 / 40 turns) was chosen after Sonnet's pilot (§11
  says so).
- Fix: define the bar against the mean of >= 3 reference seeds (or a fixed fraction of a pre-registered ceiling),
  report every alternative, and label the ceiling analysis as post hoc.

**M5. Statistics and comparisons overstate what n supports.**
- The 10 pilot instances come from 6 (source, target) pairs and 4 source states. North Carolina is the source in 4
  instances and in **4 of the 5 sol passes**. Instances of the same pair at L16/L20 share dev/test cities. The Wilson
  CI (11-47%) treats 20 correlated runs as independent. The effective n is closer to 6 pairs, so the interval should be
  much wider (cluster bootstrap by pair).
- Run-to-run variance is large (0.00 vs 0.89 on NC->VA L16; 2/20 sol runs hit max_turns and score 0).
- Sonnet vs sol: different budgets ($0.10/25 turns vs $0.25/40 turns), 1 vs 2 runs, 5/10 Sonnet episodes ended at the
  cost cap without submitting, and both were scored on a reward they were not told (C1). "Discriminates" is not
  supported; at most "sol > Sonnet at unequal budget".
- "60% of ceiling" recomputes (mean of per-run ratios 59.9%; ratio of means 60.1%). But it averages 0-scored
  non-submissions, uses a ceiling that is below the fair oracle on 2/10 instances, and was defined post hoc (M4).

**M6. "RAVEL-style" framing: the task is closer to CounterFact-style knowledge editing than to RAVEL.**
- In [RAVEL](https://arxiv.org/html/2402.17700), Cause and Iso are **interchange interventions**: the feature takes
  the value it has for a counterfactual entity. Iso means the *other attributes of the same entity* stay put. The
  attributes (country, continent, language, ...) are parallel. [MIB's RAVEL track](https://arxiv.org/abs/2504.13151)
  scores interchange-intervention accuracy for a featurizer at the entity's last token.
- T2 v2 differs on three counts. (a) The edit is a constant-target counterfactual (every source city -> one capital),
  not an interchange. (b) Iso_other ("other states' cities keep their capitals") is a neighbourhood-specificity term
  as in ROME/CounterFact; RAVEL has no such term. RAVEL never needs one, because an interchange intervention only
  touches the base entity, so §11's "a single additive vector cannot isolate (structural)" is a property of this task
  design, not of RAVEL. (c) The changed attribute (the state's capital) is a function of the kept attribute (the
  state). So "disentangling" here means breaking hop 2 while keeping hop 1, not separating two independent attributes.
- What does match RAVEL/MIB: the intervention site (the entity's last token at a fixed layer), filtering to items the
  clean model answers correctly, held-out entities and held-out templates, and Disentangle = (Cause + Iso)/2 as the
  original metric. Fix: describe it as "RAVEL-inspired counterfactual editing with a fixed site" and drop
  comparisons that rely on RAVEL semantics.

### MINOR

**m1. Selection effects (claim C).** The pilot set is the 11 instances where the gated oracle reached R >= 0.6 on
>= 2/3 seeds, minus the one where the seed-0 tool reference fell below 0.6. Both were measured on the same graded
items. Since oracle and reference share the recipe, this mainly inflates the reference, which makes the 0.9 x ref bar
harder (conservative for agents). It also excludes every L12 instance and over-represents NC. The ~28% validity
figure comes from 10 pairs and is a property of these pairs. The seed-1/seed-2 reference runs above give the size of
the noise: per-instance SD 0.01-0.13, mean of 3 seeds 0.79 vs seed 0 0.76.

**m2. Timing and wording of design decisions (claim C).**
- Validity bar: the code comment (`t2_validate.py:22`) says the bar was fixed "before step-3 oracle results ... (max
  0.34)". FINDINGS §11 says the 0.51 shortcut seen in step 3 "is why the bar is 0.6 and not 0.5". These rationales
  conflict. Everything is in one commit (0b0e4d9), so the order cannot be verified. `validate.sc0.jsonl` (with the
  0.51 row) was last written at 06:42, before the oracle rows (08:04).
- The country-KL fix and AMBIG_NONUS were motivated by one gated-oracle instance (Colorado->Idaho). They raised oracle
  R >= 0.6 from 8 to 12 of 27 (`regrade_summary.json`). The change is principled: raising P(US) toward the right
  answer is not damage. "Before any agent run" is true, but it came after oracle results. On the pilot set,
  AMBIG_NONUS removes only Bethlehem and Easton (PA->CA), so its effect there is tiny.
- The gated recipe (ramp 30-60%, 150 steps) was chosen on 4 tuning pairs by a dev-only leave-one-out proxy. I found no
  use of validation pairs in that choice (`t2_recipe_select.py`, `t2_ramp_effect.py` run on the tuning pairs).

**m3. Shortcut set and "400 runs".** The 400 runs are 80 null + 80 random (both trivially 0) + 80 mean-diff (scale 1
only) + 160 naive gradient (one seed each). They include no gated baseline and no keep-trained additive vector on the
validation pairs. "Best 0.51, none >= 0.6" recomputes exactly (Oregon->Idaho L12, naive_capped). "Every shortcut passes
0 on the 11 valid instances" also recomputes (best there: 0.39, NC->VA L20). See M1 for why this set does not settle
script-solvability.

**m4. The "4 dev cities" bottleneck applies to the offline oracle, not to agents.** Agents may name any source-state
city that is not private. On MN instances, `Rochester`, `Bloomington`, `Minneapolis` and `Duluth` are accessible
(checked with ToolEnv), and sol/Sonnet did use unlisted towns. §11's conclusion that raising validity "needs more
public dev cities per source (more data)" describes `train_gated`'s self-imposed restriction. A fair oracle could use
self-named cities too. Likewise, the private wordings and the "hidden" readouts are guarded only against exact
(normalised) copies. Trivial paraphrases of `STATE_Q_HO` or of the abbreviation readout (`"Chicago, IL\n...{city},"`)
are allowed, so "readouts you cannot see or query" overstates it.

**m5. Iso_other is mostly easy negatives.** The 16-city Iso sample spans 12-15 states with 0-4 cities from states
bordering the source (e.g. WI->MS: none). No non-city entities of the source state are tested (a source-state gate
key may fire on them). A gate keyed on "source-state-ness" therefore gets Iso_other ~0.9-1.0 almost for free (scripted
gates: 0.81-1.00). Fix: stratify the Iso sample to include neighbouring-state and same-name cities.

**m6. Gate numerics.** The hook computes `(x @ k - b)/w` in the model dtype (bf16) (`hooks.py:69-71`), while `project`
returns float32 dot products. For narrow ramps, the agent's threshold and the grader's gate can disagree near the
edge. Not quantified. Fix: compute the gate scalar in float32.

**m7. Episode records omit the submitted vector** (`agent_loop.py:341-343` strips `vector`). Agent grades cannot be
re-graded from the record, only replayed (§5).

**m8. Spend and bookkeeping.** The Anthropic pilot was $0.896 (10 entries, ledger total $3.91 of $4.30), confirmed.
OpenAI for the audit: episode costs sum to $2.22. The ledger charged $2.72, including two crashed attempts at a
worst-case $0.25 each (MN->TN L16 r1 and PA->CA L20 r1). Neither re-run passes, so counting the crashes as 0 changes
nothing (5/20 either way). §11 says "oracle shard 1 paused at 0 rows", but `validate.or1.jsonl` has 1 row
(California->Indiana L8).

**m9. Shared-code side effects on other tasks (claim F).** Grading and gating are unaffected (no other task's
constraints contain `layer`, `allow_scale_from` or `generic_refusal`; checked all 57 stored instances). But
`_SCALE_FROM_NOTE` and the `scale_from` schema fields are added to `eval_intervention`, `optimize_vector` and `submit`
for **every** task (`agent_loop.py:28-35, 82, 98-107`). Future T1/T3/T7 runs therefore see different tool
descriptions than the §10 runs, plus a feature those tasks reject. Fix: add the note and schema fields only when
`allow_scale_from` is set. The describe text also still shows stale v1 constraints (`leak_weight`, `preserve_state`)
that play no role in the v2 reward.

## 3. Claims table

| # | Claim (FINDINGS §11) | Status | Recomputed |
|---|---|---|---|
| 1a | 11/40 valid (oracle R >= 0.6 on >= 2/3 seeds), 10 fresh pairs | **Confirmed** | 11/40 from `validate.or0.jsonl` (40 instances x 3 seeds); @0.5 19, @0.7 9 |
| 1b | Per layer L8 2, L12 0, L16 4, L20 5 | **Confirmed** | 2 / 0 / 4 / 5 |
| 1c | No shortcut >= 0.6 in 400 runs, best 0.51 | **Confirmed numerically; not representative** | 400 rows, max 0.507 (Oregon->Idaho L12 naive_capped), 1 run >= 0.5. No gated shortcut in the set; a scripted gated baseline passes 9/10 (M1) |
| 2a | Tool-path reference passes 10/11 | **Confirmed** | 10/11 >= 0.6; fails WI->MS L8 (0.487); mean 0.74 vs oracle 0.80 |
| 2b | verify_references reproduces 51/51 | **Confirmed** | 51/51 re-run by this audit; 25/30 agent episodes also replay exactly, 5 within +0.02 (§5) |
| 3a | Sonnet 0/10, sol 5/20 under R >= 0.9 x tool ref | **Confirmed numerically; invalidated as an agent measure by C1** | 0/10, 5/20. Under the reward the agents were told: sol 12/20, Sonnet 1/10. Alternative bars give sol 3-10/20 (M4) |
| 3b | Mean % of ceiling: sol 60%, Sonnet 15% | **Confirmed numerically** | 59.9% / 15.3%. A fixed script reaches 93.3% |
| 3c | 17/20 sol gated submissions; Sonnet 2/10 | **Confirmed** | 17/20 (18 submitted), 2/10 (5 submitted). "Discover without being told" overstated (M2) |
| 3d | Crash handling (2 sol crashes re-run) | **Confirmed** | MN->TN L16 r1 and PA->CA L20 r1 charged $0.25 each, re-run at 21:07/21:12; neither passes |
| 3e | Spend: OpenAI ~$2.2, Anthropic $0.90 ($3.91 of $4.30) | **Confirmed** (OpenAI ledger charge $2.72 incl. crashes) | |
| 4a | Ceiling 2-fold cross-fitted; graded only on the never-trained half | **Confirmed** | `t2_ceiling.py`: fit = dev + half A, graded on `restrict(inst, B)`; `restrict` correctly limits Cause items, Iso_state cities, country-KL cities and generic-KL anchors |
| 4b | Never trained on the grader's Iso sample, the grader's KL sentences or the country probe | **Confirmed** | pool_train = iso_pool minus `iso_sample(inst, 0)`; KL on `GENERIC_TOOLS` only; no COUNTRY_Q in `train_ceiling` |
| 4c | Legal edit format | **Confirmed** | `as_submission` -> `grade_ravel(enforce=True)` -> `check` |
| 4d | Ceiling mean 0.88 is "the ceiling" | **Partially** | 0.88 recomputes; below the fair oracle on 2/10; Cause/Iso_state are measured on 5-8 cities per fold (noisy); trained on the graded wordings/readouts (other cities); config chosen post hoc (M4) |

## 4. Scripted gated baseline (claim B)

`audit` scripts use only tool calls and public info (`describe_task`, accessible cities chosen by the reference's own
rule, seeded per pair). The gate key is the normalised (fs1 + zs1 dev-source mean) minus (8 accessible other-state
cities' mean). lo/hi = 40%/60% of the way from the others' mean projection to the source mean. Grader: `grade_ravel`,
`reward_mult`. Pass = R >= 0.9 x seed-0 tool-ref R (the pre-registered rule). Cells give R (Cause / Iso_state /
Iso_other). `*` = pass.

| instance | tool ref s0 | md_mid_x1 (mean-diff push, 56 fwd) | opt_target100 (target-only gated push) | **opt_keep100** (gated push + keep examples, 2,456 fwd) | ref seed 1 | ref seed 2 | sol r1, r2 |
|---|---|---|---|---|---|---|---|
| Minnesota->Nebraska L8 | 0.62 | 0.13 (0.76/0.17/1.00) | 0.07 (0.88/0.08/0.97) | **0.68*** (0.71/1.00/1.00) | 0.62* | 0.54 | 0.29, 0.11 |
| Minnesota->Tennessee L16 | 0.88 | 0.00 (0.64/0.00/1.00) | 0.00 (1.00/0.00/0.97) | **0.88*** (1.00/0.91/0.97) | 0.92* | 0.88* | 0.74, 0.74 |
| Minnesota->Tennessee L20 | 0.88 | 0.00 (0.64/0.00/1.00) | 0.00 (1.00/0.00/0.97) | **0.88*** (1.00/0.91/0.97) | 0.76 | 0.88* | 0.61, 0.69 |
| North Carolina->Kentucky L16 | 0.78 | 0.00 (1.00/0.00/0.87) | 0.00 (1.00/0.00/0.81) | **0.80*** (1.00/0.92/0.87) | 0.93* | 0.97* | 0.36, 0.83 |
| North Carolina->Kentucky L20 | 0.70 | 0.00 (1.00/0.00/1.00) | 0.00 (1.00/0.00/1.00) | **0.91*** (1.00/0.92/1.00) | 1.00* | 0.75* | 0.79, 0.82 |
| North Carolina->Virginia L16 | 0.91 | 0.00 (0.88/0.00/1.00) | 0.00 (1.00/0.00/1.00) | **0.97*** (1.00/1.00/1.00) | 0.92* | 0.94* | 0.00, 0.89 |
| North Carolina->Virginia L20 | 0.61 | 0.00 (0.88/0.00/0.94) | 0.00 (1.00/0.00/0.94) | **0.79*** (1.00/0.92/0.94) | 0.76* | 0.75* | 0.28, 0.32 |
| Pennsylvania->California L20 | 0.66 | 0.07 (0.48/0.14/0.97) | 0.15 (0.74/0.29/0.94) | 0.53 (0.65/1.00/0.94) | 0.70* | 0.58 | 0.56, 0.28 |
| Wisconsin->Mississippi L16 | 0.75 | 0.00 (0.52/0.00/1.00) | 0.00 (0.97/0.00/0.97) | **0.83*** (0.83/1.00/1.00) | 0.69* | 0.78* | 0.84, 0.62 |
| Wisconsin->Mississippi L20 | 0.87 | 0.00 (0.45/0.00/1.00) | 0.12 (0.93/0.13/0.97) | **0.93*** (0.93/1.00/1.00) | 0.80* | 0.78 | 0.00, 0.76 |
| **pass / mean R** | | **0/10, 0.02** | **0/10, 0.03** | **9/10, 0.82** | 9/10, 0.81 | 7/10, 0.79 | 5/20, 0.53 |

Also run on the first 5 instances only (stopped to save GPU time, all 0 so far): mean-diff push x2 (0/5, mean 0.01)
and an auto-scaled mean-diff push (smallest of {1, 1.5, 2, 3, 4} flipping all dev prompts; 0/5, mean 0.03).

Reading: a gate alone (any push without keep-state examples) flips the source cities but destroys the state readouts.
Adding 4 `state_q` keep examples and 6 own-capital examples to a 100-step gated `optimize_vector` solves 9/10, with
no inspection and no instance-specific choices. All agents need is that recipe. The reference's seed-to-seed spread
(SD up to 0.13 per instance) is as large as many agent-vs-reference gaps.

## 5. Replay of agent episodes and reference regression (claims 2b, F)

**Replay (grader pipeline check).** The records omit the submitted vector (m7), so I replayed every T2 v2 episode's
`tool_log` (10 Sonnet + 20 sol) through the current `ToolEnv` and re-graded the resulting submission:
- **25/30 reproduce the stored `reward_mult` exactly.** The other 5 differ by +0.009 to +0.022 (WI->MS L20 Sonnet,
  PA->CA L20 r1/r2, WI->MS L16 r2, WI->MS L20 r2), consistent with bf16 nondeterminism. **No pass/fail changes**; the
  closest is WI->MS L20 r2 (0.776 replayed vs a bar of 0.784). The same 7 episodes have no submission (5 Sonnet cost
  caps, 2 sol max_turns). The grading pipeline and the recorded grades are sound.
- **Iso-sample sensitivity.** Re-grading the 23 submissions with `iso_seed=1` (a different 16-city Iso sample) moves R
  by 0.037 on average and by up to 0.14. Examples: Sonnet MN->NE L8 0.28 -> 0.42 (Iso_other 0.45 -> 0.68); sol MN->TN
  L20 r1 0.61 -> 0.73; sol NC->KY L20 r1 0.79 -> 0.68. Because the grader always uses `iso_seed=0`, R carries a fixed
  per-instance draw of this size. That matters when bars sit within ~0.05 of agent scores. Fix: average over several
  Iso samples or enlarge `N_ISO`.

**Reference regression (claim 2b / F).** `experiments/verify_references.py`, re-run by this audit after the scripted
baseline and the replay: **"references: 51/51 reproduce the stored reward"**. That is 41 T1/T2-v1/T3/T4/T6/T7
references exactly (tolerance 1e-6) and the 10 T2 v2 tool references within 0.02. Together with the constraint check
in m9, grading of the other tasks is unaffected by the shared-code changes. Only their tool descriptions changed (m9).

## 6. What I could not check, and why

- **The pre-registration order inside commit 0b0e4d9.** The KL fix, the bar, AMBIG_NONUS and the validation all
  landed in one commit with no intermediate history. Only file mtimes order them, and mtimes record the last write,
  not the decision.
- **Whether agents reasoned about the (wrong) reward.** Sonnet's thinking is stored encrypted and sol's reasoning is
  not in the record. Visible text never quotes the formula. The effect of C1 on behaviour can only be measured by
  re-running agents with the corrected text (API spend, out of scope).
- **The counterfactual of a correct task text, equal budgets, or the alias fix on agent scores.** All need API runs.
- **Generalisation beyond these 10 instances / 6 pairs**, and the bf16 gate-threshold mismatch (m6), were not
  quantified.
- The crashed sol attempts left no transcript, so I could not see how far they had got.

Audit artefacts (scratchpad, not committed): `scripted_gate.py` / `scripted_gate.jsonl` (§4),
`replay_t2v2.py` / `replay.jsonl` (§5), `verify_refs.log`.
Sources: RAVEL (https://arxiv.org/html/2402.17700), MIB (https://arxiv.org/abs/2504.13151), GRACE
(https://arxiv.org/abs/2211.11031).

## 7. Is T2 v2 a suitable RL environment? (auditor's judgement)

**Not in its current form. It can become a useful *unit* in a curriculum, but not a stand-alone training env.**

What is already good, and is the hard part of building an env: the grader is correct and deterministic (replays
reproduce, 51/51 references reproduce), a null edit scores 0, the reward gives partial credit, and the private data is
mostly well protected.

Quick fixes (about a day): the reward text (C1, one line, but the current agent results must be discarded), the alias
leak (M3), and the fixed Iso sample (§5): draw a fresh or larger sample per episode so a policy cannot overfit to
16 fixed cities over many episodes on 10 instances.

The structural problem is that **the reward measures whether the agent knows one recipe**:
- A fixed 100-step script gets mean R 0.82 (93% of ceiling, 20% of the budget). Anything without the recipe gets
  0-0.3. The reward is close to all-or-nothing.
- Once the recipe is found, the room left (about 0.06 to the ceiling) is smaller than the grading noise (reference
  seed SD up to 0.13; Iso-sample re-draw shifts R by 0.04 on average). There is nothing left to learn.
- The task-only tools (`project`, `scale_from`) map one-to-one onto the solution, so most of the "discovery" is done
  for the agent.
- 10 instances from 6 pairs, 4 source states and 1 model: a policy can memorise them, and generalisation cannot be
  measured.

Today T2 v2 is a decent **eval** of whether an agent knows a GRACE-style gated-edit recipe. As **RL signal** it
teaches one trick in a few updates and then the reward stops improving. This judgement rests on 10 instances and one
model; a larger instance set might show more room above the recipe.

## 8. Fixing the structural problems: design options

### 8.1 Why RL learns a recipe here, and what an oracle is for

RL learns whatever is common across training instances and earns reward. If one fixed procedure scores ~0.8 on almost
every instance, the cheapest policy is "always run that procedure". Investigating never decides the reward, so
nothing pushes the agent to investigate better. **To teach discovery, the right action must depend on facts about
the specific instance that can only be learned by experimenting, and those facts must vary across instances.**

Keep the oracle, but change its role:
- **Keep it** as proof that each instance is solvable, as a calibration point, and as a filter when generating
  instances (8.4).
- **Don't** make "match the oracle's recipe" the target. A pass rule like R >= 0.9 x reference rewards copying the
  reference's method and caps how far the agent can improve.

"Make it harder" is not the fix on its own. Harder in the same direction (more layers, more steps of the same recipe)
gives instances nobody solves, and an always-zero reward teaches nothing. It must be harder in the sense that **the
answer differs per instance**.

Learning recipes is not useless: gating plus keep-constraints is a real technique, and human researchers learn
recipes too. The problem is only that a single recipe saturates, so it must not be the whole curriculum. Many recipe
units with different structures, plus a held-out task family to test composition, is a legitimate design.

### 8.2 Why the recipe problem is general, not specific to T2

Every term of T2's grader (Cause, Iso_state, Iso_other, KL) can be approximated with data the agent can access: dev
cities, `state_q`, accessible cities of other states, generic sentences. So the universal strategy is **write a proxy
for the grader's objective, then optimise it**. The 100-step script does exactly that, and so does the oracle. Any
task whose answer is "the minimiser of an objective you can build from accessible data" collapses to that recipe.
This is why §10 found T1/T3/T4 pass by script as well. Adjusting T2's knobs will not escape it. There are three ways
out:

1. **Break the proxy:** the grader measures something the agent cannot imitate with more examples, because doing well
   requires predicting how the model generalises, i.e. understanding the mechanism.
2. **Make optimisation unaffordable:** the budget is too small for training loops, so the agent must get there by
   analysis.
3. **Make the output non-optimisable:** the agent submits a claim, a location or a program, not a vector.

### 8.3 Ways to fix T2 itself without training a model

**A. Break the proxy: hide what "keep the state" means.** Iso_state currently checks two readouts, and the agent can
copy both with `state_q` plus a paraphrase. Instead, grade "the model still knows the city is in <state>" on a large
private pool of downstream uses of that knowledge: governor, neighbouring states, time zone, area code, sports
teams, "state of the Twin Cities". Only a few are visible to the agent. Keep-examples on visible readouts would then
overfit; the edit that generalises is the one that touches only the "capital" information and leaves "state" intact,
which requires analysis. *Best chance of the three.*

**B. Budget squeeze.** Cut the budget from 12,000 forward passes to about 300-500. Gradient training no longer fits
(the script used 2,456), so the agent must build the edit from mean-differences, projections and logit lens.
Caveat: a cheap closed-form recipe may still exist (for example, push = mean-diff with the state direction projected
out). *Uncertain; must be tested.*

**C. A mechanism zoo from natural knowledge.** Vary the instance structure, not only the state pair: other
attributes and entity types (the RAVEL dataset already has cities, Nobel laureates, objects and verbs with several
attributes each), fixed or free edit site, several models (gemma-2-2b is accessible; Qwen 7B). Some attribute pairs
are separable (a plain vector works), some are tangled (a gate is needed), some only at certain layers, so diagnosing
the instance type matters. Risk: "build a proxy and optimise" may still win everywhere; the script filter (8.4) would
show this quickly. *Probably does not escape the recipe on its own.*

Also: return to RAVEL's actual formulation (interchange interventions with a featurizer, per-pair targets). This
removes the constant-target gate shortcut, but DAS/MDAS-style optimisation is itself a recipe that gets close to the
best known numbers (MIB leaderboard), so it does not escape 8.2 either.

### 8.4 Script filter in instance generation (works for any task)

When generating instances, run a library of scripted baselines: mean-diff, gated recipes, the reference, and later
the best trained policies. Keep only instances the oracle solves and no script does. Periodically add the current
policy's typical strategies to the library and regenerate, so a learned trick stops paying. This is the same idea as
curriculum methods like PAIRED and POET: always generate instances that are solvable but not yet solved.

### 8.5 Planted mechanisms: cheaper than they look

Planting does not need fine-tuning:
- **Hook planting already exists:** `tools.py` plants an additive vector for the T4 detective. Anything can be planted
  the same way: a gated edit, a projection, a small random nonlinear function keyed on a feature, edits at several
  sites.
- **Weight planting without training:** a gated edit is essentially one rank-one write into an MLP (the idea behind
  ROME's closed-form edit), and `ravel.train_gated` already computes such edits. Write them into the weights at a
  random layer with a random trigger: unlimited instances with known ground truth.
- **The real difficulty is making them not trivially detectable.** The agent must never see a clean model to diff
  against (T4 blackbox variant), never get the weights, and the plant must cause no obvious damage elsewhere. §10
  found T4 script-solvable too; find out why (likely clean-model access via `act_diff`) before building on it.
- Task: given an edited model, find what changed (which entities, which attribute, where) and undo it with minimal
  side effects. The hypothesis space is large and the investigation is the skill. Precedent: trojan/backdoor
  detection challenges, and auditing games with planted hidden objectives.

### 8.6 Explanations scored by prediction (most open-ended)

The agent submits a short program or rule that predicts a behaviour ("this feature fires when...", "patching here
flips the answer when..."). The grader runs it on held-out inputs; reward = accuracy minus a length penalty. A short,
accurate rule requires insight, while a memorised lookup table is long. Every feature, neuron and behaviour is a new
instance, with no training. It is the most research-like option and the hardest to grade well.

### 8.7 How to tell it is working

- **Hold out whole task families** (train on some, test on another). Improvement there shows the agent learned to
  investigate rather than memorise.
- **Report the scripted-baseline library's score next to every agent result**; the agent should stay clearly above.
- **Reward should keep rising during training**, not jump once and flatten.

## 9. Recommended next steps

1. **This week:** the quick fixes (C1, M3, Iso sampling, pass rule against several reference seeds) plus the script
   filter (8.4) on T2. This makes the current task honest.
2. **Decision experiments (GPU only, no API, about half a day each):**
   - *Proxy-break test (8.3 A):* add about 10 private state readouts to Iso_state; re-run the 100-step script
     (`opt_keep100`) and the ceiling. If the script drops a lot while the ceiling stays high, T2 is fixable along
     lever A.
   - *Budget-squeeze test (8.3 B):* run closed-form scripts at about 500 forward passes and compare with the oracle.
3. **If neither opens real room above the recipe:** keep T2 as a calibration and skill unit, and put the structural
   effort into planted weight edits (8.5), a one-to-two-day prototype using the existing edit code.
4. **Longer term:** explanations scored by prediction (8.6); it scales furthest and teaches the most transferable skill.
