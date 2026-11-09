# EditHunt: Grading Interpretability Skill with Verifiable Model-Editing Tasks

**Final project report.** Research period 2026-09-29 to 2026-10-04. Status: complete. This report summarises the whole
project in one place. Detailed logs and per-run data are in the files cited throughout.

## Abstract

EditHunt asks whether a frontier agent's interpretability skill can be graded by a program, with no human judge. The
agent must edit a small open model's internal state so that a fact changes in a controlled way. The core task uses a
two-hop fact: city -> US state -> state capital. The project had three parts. (1) **Science.** On four open models
(Qwen2.5-1.5B, 3B, 7B and gemma-2-2b), a mean-difference "state" vector added at the city token flips held-out two-hop
answers (peak 0.86-0.91), over a mid-layer band that ends at a sharp handoff layer (0.69-0.86 of depth). The vector
acts as a state variable, not an answer direction. Its main side effect is leakage onto other states' cities, and
leakage turned out to be the axis that separates naive edits from careful ones. (2) **Environments.** We built a
grader-outside-the-sandbox RL environment, a six-task agent suite, a RAVEL-style "keep the state, change the capital"
environment (T2 v2), and a RAVEL breadth screen plus city environment on gemma-2-2b. (3) **Audits.** Replays,
zero-effort script baselines, a five-reviewer correctness review, an independent audit and privileged ceiling oracles.
The audits produced mostly negative results about the tasks: 5 of 6 suite tasks can be passed by a standard recipe or
a script. T2 v2 has no shortcut (best of 400 shortcut runs 0.51 vs a 0.6 bar), but only 11/40 fresh instances
validate, and the fixed tool-path reference already averages 0.76 against a privileged ceiling of 0.88. The RAVEL Language task is honest, but a fixed script already reaches about 89% of a privileged ceiling. Final
status: the science results are validated; the environment designs are partly validated and come with clearly stated
negative results.

## 1. Motivation and goal

Interpretability claims are hard to check. A task where the agent must *change* a model's internal computation, and
where a program can measure whether the change worked, would make interpretability skill gradeable at scale. That is
also what an RL environment needs. The project brief (`BRIEF.md` §3) set these questions:

- Does a shared, causally used state direction exist at the city token, with robust n, several templates and several
  models?
- Is it a state variable or an answer direction?
- What collateral damage does editing it cause?
- Can the second hop (state -> capital) be redirected alone?
- Which constraints make naive methods (mean-diff, gradient, prompting) fail while a careful method still succeeds, so
  that a medium-difficulty tier exists?

The working criteria for a "good RL environment" were:

- **Verifiable reward.** It is computed by a grader the agent cannot see or touch.
- **Existence proof.** A reference solver passes, using only the tools the agent has.
- **No shortcut.** Zero-effort scripts (mean-diff, plain gradient, prompting, random, untrained subspaces) fail.
- **Headroom.** A fixed script does not already saturate the reward, so there is room to learn.
- **Validity on fresh instances.** Instances are validated on fresh pairs, and the rules are fixed before agents run.

## 2. What was built

| Component | Path | Size / contents |
|---|---|---|
| Core library | `edithunt/` (`data.py`, `model.py`, `hooks.py`, `runner.py`, `metrics.py`, `instances.py`, `vectors.py`, `optim.py`, `common.py`) | 994 lines. 50 states and capitals, 343 hand-curated cities, capital templates (fs1, fs2, zs1, zs2, held-out ho_fs and ho_zs), state probes, a few-shot P(United States) probe, 50 generic sentences. Exact full-sequence log-prob scoring over all 50 capitals with a safe pruning bound, a shared-prefix KV cache, and raw forward hooks for capture, patch, add, subspace swap and (later) gated edits. |
| Phase 0-4b experiments | `experiments/phase0_validity.py` ... `phase4b_careful.py`, `plots.py`, `findings_plots.py` | 771 lines for the phase scripts |
| Agent environment and task suite | `edithunt/env/` (`tools.py` 735, `tasks.py` 715, `ravel.py` 501, `agent_loop.py` 377, `grader.py` 257, `instance.py` 237, `baselines.py` 195, `suite_run.py` 143) | 3,160 lines. Instance generator with reference validation; grader outside the agent sandbox; 12 agent tools that keep vectors in server-side registers and refuse held-out cities and templates; scripted baselines; an agent loop for the Anthropic SDK and the OpenAI Responses API with per-episode cost caps; a file-locked spend ledger that reserves the worst case before each episode. |
| T2 v2 RAVEL-style environment | `edithunt/env/ravel.py`, `experiments/t2_*.py`, `results/suite/t2_ravel/` | Product-reward grader v2, gated edit hook, `train_gated`, cross-fitted ceiling oracle, a 10-instance pilot set |
| Audit and verification scripts | `experiments/audit_phase7.py`, `verify_references.py`, `verify_review_fixes.py`, `verify_tools_fix.py`, `suite_baselines.py`, `suite_shortcuts.py`, `task_map.py` | Replays, shortcut baselines and the difficulty map |
| RAVEL breadth screen and city environment (gemma-2-2b) | `edithunt/ravel/` (`stage1.py`-`stage4.py`, `das.py`, `sae.py`, `families.py`, `pointer.py`, `gap_hunt.py`, `boost.py`, `env.py`, `agent.py`, `env_*.py`) | 28 Python files, 3,694 lines. DAS/MDAS training, a Gemma Scope SAE baseline, an 8-tool city environment, and saturation and learnability tests |
| Results | `results/` | 385 files (352 json, 18 jsonl, 9 png, 5 log, 1 lock) |
| Figures | `results/figures/fig1_layer_curves.png` ... `fig5_breadth_screen.png` | 5 figures |
| Write-ups | `FINDINGS.md`, `LAB_NOTEBOOK.md` (append-only), `AUDIT.md`, `FINDINGS_RAVEL_SCREEN.md`, `LAB_NOTEBOOK_RAVEL_SCREEN.md`, `PROGRESS.md` | Detailed findings and logs |

Overall: 10,276 lines of Python (7,848 in `edithunt/`, 2,428 in `experiments/`), 42 commits.

**Architecture of the core environment.** An *instance* is (source state, target state, subject model). The agent
gets 4 public dev cities per side. It works through a tool API (run prompts, cache means, project, optimise a vector,
evaluate an intervention, submit). It can only submit an *edit* (a vector, a projection or a gate at a layer), never
a prompt. The grader runs outside the sandbox on private held-out cities, two held-out templates, private third-state
"leakage" cities and damage probes. Reward:

    reward = F x (1 - leak_weight * leak) x (1 - min(1, KL_mean / kl_budget))

Here F is the held-out flip rate, leak is the change on third-state cities, and KL_mean measures damage on the country
probe and on generic text. Pass = reward >= 0.5.

## 3. Methods

**Subject models.** Qwen2.5-1.5B (main), Qwen2.5-3B, Qwen2.5-7B and gemma-2-2b. Qwen2.5-0.5B was a smoke test only.
Llama-3.2 was gated and skipped.

**Agents.** Claude Sonnet 5.5, gpt-6.1-sol and gpt-6-luna, called through their APIs with fixed turn and cost caps.

**Scoring.** Exact full-sequence log-probability over all 50 capitals (19 of the 50 are multi-token in the Qwen
tokenizer), not the greedy token. For RAVEL, an exact scorer runs over every surface form of every candidate value
using a token trie.

**Interventions compared** (the "ladder"):

| Class | Method |
|---|---|
| C0 | Full residual paste |
| C1 | Mean-difference vector (target minus source dev cities) |
| C2 / C2n / C2kl / C2nkl | Gradient-trained steering vector, plain or norm-capped, with or without a KL term |
| C2nk | Norm-capped gradient plus "keep 4 other states' capitals" (leak-aware) |
| C3 | DAS, rank 1 and rank 4 |
| C4 / C4v | Prompting; prompt-derived vector |
| Gated edit (T2 v2) | x <- x + clamp((k.x - b)/w, 0, 1) v, fit on dev data only |
| MDAS / DAS / SAE (RAVEL) | Multi-attribute DAS, plain DAS, Gemma Scope 16k SAE latents (5 or 50), mean-diff, full vector, random-16 |

**Graders and baselines.** Every environment has a private grader, a reference solver and a set of zero-effort
shortcut scripts (mean-diff at a fixed layer, naive gradient, prompting-only, random, the round(0.8 x depth) prior,
untrained DAS or SAE). In the Phase 7 suite, the T2 v2 pilot set and the RAVEL environment, an instance is kept only
if a reference solver, working through the *same tools as the agent*, passes (Phase 6 instances were validated by a
mean-diff flip >= 0.8; T2 v2 validity was first set by an offline oracle).

**Statistical practice.**
- All rates are on held-out cities, with n and 95% Wilson CIs.
- Held-out cities, templates and readouts are private to the grader. The tools refuse them, including aliases and
  near-copies (NFKC, zero-width and punctuation normalisation; about 14k refused keys in the RAVEL environment).
- Thresholds were fixed before the agent runs. Examples: the T2 v2 validity bar (oracle R >= 0.6 on >= 2/3 seeds),
  the T2 v2 agent pass rule (R >= 0.9 x tool-reference R) and the country-KL grader fix. The RAVEL KL budget of 3.5
  was fixed before the learnability episodes, but the hinge form was chosen after seeing reference KLs and a pilot.
- RAVEL seed checks used 4 seeds on fresh pools; T2 v2 oracles used 3 seeds. The Phase 6 calibration pooled 2 seeds
  after the second seed overturned the first.
- Audits: tool-log replays, re-seeded references, a five-reviewer correctness review before the reruns, an independent
  audit (`AUDIT.md`) and cross-fitted privileged ceilings.
- There is no unit-test suite. Correctness was checked with replay and verification scripts and 16 inline asserts.
  Library checks: city-token position had 0 failures over 343 cities x 6 templates; pruned vs exhaustive scoring gave
  the same argmax on 172 prompts; KV cache vs padding differed by at most 1.4e-4.

**Hardware.** Work started on an Apple M4 laptop (MPS, fp32) and moved to one shared NVIDIA A10G 23 GB (CUDA, bf16).
No total GPU-hours figure was recorded.

## 4. Results

### 4.1 Phase 0: validity of the readout (Qwen2.5-1.5B, n=343 cities)

| Template | Correct [95% CI] (bf16 rerun) |
|---|---|
| fs1 (few-shot) | 0.91 [0.87, 0.94] |
| zs1 (zero-shot) | 0.83 [0.79, 0.87] |
| zs2 | 0.79 [0.74, 0.83] (greedy-token readout: 0.06) |
| ho_fs (held-out few-shot) | 0.72 [0.67, 0.76] |
| ho_zs | 0.87 [0.83, 0.90] |
| State question | 0.99 [0.98, 1.00] |

- The original notebook's "4/16 cities correct" was a readout artefact from reading the greedy token.
- Hop 1 (city -> state) is near-perfect, so most errors are in hop 2.
- The other models were also valid. Examples: gemma-2-2b fs1 0.96 and zs1 0.94; Qwen2.5-7B fs1 0.96; Qwen2.5-3B fs1
  0.90.
- Source: `LAB_NOTEBOOK.md` "Phase 0 rerun in bf16" and "Phase 5". The earlier fp32 run in `PROGRESS.md` gives zs1 0.84
  and ho_fs 0.71. The bf16 run is the later one.

### 4.2 Phases 1-2: a state direction at the city token, in four models

| Model | Layers | Band (flip >= 0.5) | Peak flip, few-shot [CI] (n) | Peak on unseen template (n) | Handoff (fraction of depth) |
|---|---|---|---|---|---|
| Qwen2.5-1.5B | 28 | L5-21 | 0.89 [0.82, 0.94] (103) | 0.70 (76) | L22 (0.79) |
| Qwen2.5-3B | 36 | L8-30 | 0.86 [0.78, 0.91] (105) | 0.78 (98) | L31 (0.86) |
| gemma-2-2b | 26 | L3-17 | 0.91 [0.84, 0.95] (104) | 0.82 (99) | L18 (0.69) |
| Qwen2.5-7B | 28 | L4-21 | 0.91 [0.84, 0.95] (109) | 0.87 (107) | L22 (0.79) |

Meaning test (same vector; state-question flip vs capital flip): 1.5B 0.98 vs 0.89; 3B 0.97 vs 0.86; gemma 0.96 vs
0.91; 7B 0.95 vs 0.90.

- The premise holds on all four models. Random vectors of the same norm flip <= 0.05 at every layer. Shuffled labels
  flip 0.00-0.17 (1.5B).
- The handoff is sharp. On 1.5B, pasting the full city-token residual flips 0.95-0.98 over L3-21, then 0.16 at L22 and
  0.00 after. Final-token patching goes from 0.00 to 0.66 at L22 and 0.99 at L23.
- It is a state variable, not an answer direction: it flips the answer to "Which US state is X in?" at least as often
  as the capital. Projecting the state-question direction out of the vector removes the whole effect (flip 0.00).
- Generic-text KL stays at the random-vector level. The main side effect is leakage: 16-24% (1.5B), 31-34% (3B), 17-24%
  (gemma) and 13-24% (7B) of other states' cities also switch to the target capital. Doubling the norm roughly doubles
  leakage (1.5B: 0.48 at 2x).
- Method correction: the zero-shot country probe was state-entangled (the model answers "Texas" to "X is a city in the
  country of"). It counted the intended edit as 1-3.5 nats of damage, so it was replaced by a few-shot P(United States)
  binary-KL probe.
- Scale: bigger Qwen models generalise better to the unseen template (0.70 -> 0.78 -> 0.87). Gemma computes the state
  earliest and hands off earliest.

### 4.3 Phases 3-4b: the difficulty ladder and hop separation (Qwen2.5-1.5B only, 12 pairs, n=42)

| Class | Flip on unseen template | Leakage to target |
|---|---|---|
| C0 full residual paste | 0.66-0.76 | 0.94-1.00 |
| C1 mean-diff | 0.38 (L4) -> 0.66-0.72 (L8-15) | 0.14-0.24 |
| C2 gradient | 0.97-1.00 | 1.00 (vector 15x mean-diff norm) |
| C2n gradient, norm-capped | 0.90-1.00 | 0.56-0.83 |
| C3 DAS rank 1 | 0.38-0.52 | 0.00-0.01 |
| C4 prompting | 0.24 few-shot, 0.97 zero-shot | not a valid submission |
| **C2nk leak-aware gradient** | **0.83 (L8), 0.97 (L15)** | **0.07-0.12** |

Hop separation (output the target capital while the state question keeps the original state):

| Method | HOP2 success |
|---|---|
| Norm-capped city-token vector with a keep-source-state term | 0.98 [0.88, 1.00] at L8; 1.00 [0.92, 1.00] at L16/20; 0.93-1.00 on an unseen state-question wording |
| Mean-diff | <= 0.02 |
| Naive capped gradient | 0.24 (L8), 0.62 (L16), 0.86 (L20) |
| Uncapped gradient | 0.00 / 0.07 / 0.14 (L8/16/20) |

- Leakage is the axis that separates methods. Gradient vectors trained to "output the target capital" learn an answer
  direction ("say Sacramento") that fires on any city, even when norm-capped.
- Only the leak-aware gradient (C2nk) is both effective and specific, and it generalises to unseen states. It is no
  longer a state edit, though: the state answer moves only 0.12-0.31. It costs about 2.5x the country damage of
  mean-diff (binary KL 0.20-0.22 vs 0.08).
- Constraints that make naive methods fail: a leakage penalty (kills C0 and C2, halves C1), a layer ceiling <= 4 (C1
  0.38) and 1-2 dev cities (C1 0.40 at L4).
- The Phase 4b reward figures (about 0.66 for C2nk at L15, 0.40 for C1 at L8) are rough hand calculations, not grader
  runs. They are **suggestive only**.

### 4.4 Phase 6: reward-hacking checks and tier calibration (Qwen2.5-1.5B)

Reward-hacking risks observed, and what blocks each one:

| Risk | Evidence | Block |
|---|---|---|
| Brute-force overwrite | Uncapped vector norm ~460 vs residual ~55 | Leakage + KL terms bring C2 to ~0.2 |
| "Say the target capital" | C2/C2n flip other states 56-100% | Private third-state leakage cities |
| Prompt injection | Prompting flips zero-shot 0.97 | Submissions are edits only |
| Easy instances | Mean-diff 0/24 on hard | Instance filter and tiers |

Scripted calibration (2 seeds pooled, 23-24 validated instances per tier, pass = reward >= 0.5):

| Tier | Mean-diff | Naive gradient | Careful | Random |
|---|---|---|---|---|
| Easy | 21/24 | 10/24 | 14/24 | 0/24 |
| Medium | 6/23 | 3/23 | 5/23 | 0/23 |
| Hard | 0/24 | 0/24 | **5/24 (0.21 [0.09, 0.40])** | 0/24 |

- The easy and hard tiers behave as intended. On hard, every naive method scores 0 and the careful script sometimes
  passes.
- **Negative result:** the medium tier does not separate (careful 5/23 vs mean-diff 6/23). A seed-0 result that
  suggested it did (3/7 vs 1/7) was noise and was overturned by seed 1.
- Diagnosis: inside the tool API the leak-aware gradient was under-trained (held-out F mostly <= 0.5 vs 0.83-0.97
  offline). The grader's leakage definition ("any capital change", both held-out templates) is also much stricter than
  the Phase 4 metric ("switched to target"). Mean-diff leakage under the grader is 0.6-1.0.
- Claude Sonnet 5.5 probe: 6 episodes, $0.27 total. Easy 2/2, medium 1/2, hard 1/2; one hard instance solved in 3
  turns. This is a **cost and behaviour probe only**: the instances were hand-picked because the careful script solved
  them, so it is not a pass-rate estimate. It showed that named `keep_*` tool options made the careful method too easy
  to find.

### 4.5 Phase 7: six-task interpretability suite (Qwen2.5-1.5B; agent Claude Sonnet 5.5)

Setup: 7-8 instances per task, at most 25 turns and $0.10 per episode, $2.75 suite spend. Instances were kept only if
a tool-only reference reached reward >= 0.5.

| Task | Sonnet pass | Sonnet mean reward | Reference | Best zero-effort script |
|---|---|---|---|---|
| T1 easy edit | 8/8 | 0.95 | 0.85 | fixed mean-diff L12: 8/8 |
| T2 keep state | 1/7 (CI [0.03, 0.51]) | 0.21 | 0.64 (0.54 re-seeded) | 0/7 |
| T3 consistency (hidden readouts) | 8/8 | 0.87 | 0.91 | fixed mean-diff L8: 8/8 |
| T4 detective (planted edit) | 8/8 (black-box 2/2) | 1.00 | 1.00 | prompting-only: 8/8 |
| T6 handoff | 7/8 (black-box 0/2) | 0.88 | 1.00 | round(0.8 x depth): 4/8 |
| T7 minimal edit | 3/8 (grader-"passed" 6/8) | 0.36 | 1.00 | mean-diff: 5/8 |

- **Negative result:** as built, the suite mostly measures recipe knowledge. Sonnet ties a zero-effort script on T1,
  T3, T4 and (by count) T7. T6 is a prescribed procedure with effective n = 4 (the answer is a per-model constant).
- T3 adds nothing over T1: mean-diff already moves the hidden readouts, and the cube-root reward passes weak edits
  (F_cap = 0.125 already gives 0.5).
- T4 passes without interpretability: asking the state question for 2 known cities per state is worth 0.8 of 1.0.
- T7 mostly rewards the layer choice, because the norm is divided by that layer's residual norm.
- Only T2 is clearly beyond the standard recipe. Every scripted edit there flips 100% of capitals but keeps the state
  on 0% of cities.
- A fifth task, T5 (erase the state), was dropped: no tool-reproducible reference reached 0.5 (dev-only 0.20-0.59 vs an
  all-cities oracle of 0.956).
- **Invalid numbers:** all gpt-6.1-sol Phase 7 runs are invalid. Strict tool schemas made it fill optional arguments
  with placeholders, which the tools treated as real input (193 tool errors in 1,545 results vs 38 in 519 for Sonnet).
  The 44 files are archived in `results/suite/runs_sol_bug` and were not rerun. The gpt-6.1-sol Phase 7 numbers in
  `AUDIT.md` should not be cited.
- A security finding: the refusal message acted as a membership oracle for grader-private cities. It came up in 16
  runs and revealed leakage cities in 3, but no result depended on it.

### 4.6 T2 v2: RAVEL-style keep-state environment (Qwen2.5-1.5B)

Reward redesign. The plan's averaged reward did not discriminate (the null edit scored 0.50). It was replaced by a
product reward under which the null edit scores 0:

    R = Cause x Iso_state x Iso_other x (1 - min(1, KL_mean))

| Quantity | Value | Source |
|---|---|---|
| Additive oracle (best single vector) | Cause ~0.4-0.56, Iso_state ~0.7, Iso_other ~0.6 | `FINDINGS.md` §11 |
| Gated oracle isolation | Iso_state >= 0.71, Iso_other >= 0.94 (mostly 1.00) | `FINDINGS.md` §11 |
| Fresh-pair validation (bar fixed in advance) | 11/40 = 27.5% valid (Wilson 16-43%); L8 2/10, L12 0/10, L16 4/10, L20 5/10 | `results/suite/t2_ravel/validate_summary.json` |
| Oracle on the 11 valid instances | R 0.69-0.84, Cause 0.77-0.93, KL <= 0.03 | `FINDINGS.md` §11 |
| Shortcut runs | 400; none >= 0.6; best 0.51 | `validate.sc0.jsonl` |
| Tool-path reference | passes 10/11 valid instances, R 0.49-0.91 (mean 0.74) | `FINDINGS.md` §11 |
| Regrade after country-KL fix | 122/123 runs reproduce exactly | `regrade_summary.json` |

Agent audit on the 10-instance pilot set (pass rule fixed in advance: R >= 0.9 x tool-reference R):

| Agent | Budget | Pass (Wilson) | Mean R | Share of ceiling | Gated submissions |
|---|---|---|---|---|---|
| gpt-6.1-sol | $0.25, 40 turns | 5/20 (11-47%) | 0.53 | 60% | 17/20 |
| Claude Sonnet 5.5 | $0.10, 25 turns | 0/10 (0-28%) | 0.14 | 15% | 2/10 (5 hit the cost cap) |

Reference means: tool reference 0.76, fair oracle 0.81, cross-fitted privileged ceiling 0.88.

- A single additive vector cannot isolate the edit. One fixed vector moves every city, so Iso_other stays at 0.5-0.7 in
  every variant.
- A gated (conditional) edit is the existence proof. Isolation is near-perfect, and Cause is the limit.
- The reward has no shortcut, and the tool path nearly reproduces the oracle (mean R 0.74 vs 0.80; 10/11 pass).
  Agents found the conditional edit without being told.
- **Negative result (headroom):** the fixed tool-path reference script averages 0.76 against a ceiling of 0.88
  (about 86%). A later T2 v2 audit, `AUDIT_T2V2.md` (not in the repo), is quoted in `LAB_NOTEBOOK_RAVEL_SCREEN.md` as
  finding that a fixed 100-step script reached 93% of the ceiling.
- **Negative result:** only 27.5% of fresh instances validate (the target was 60%). Diagnosis: at late layers the push
  works (forcing the gate open at L20 gives Cause 0.96-1.00), but a gate fit on 4 dev cities does not open on held-out
  source cities. Five dev-only detector variants did not fix it.
- The agent comparison is a pilot at unequal budgets, so the two agents are not directly comparable.

### 4.7 RAVEL breadth screen and city environment (gemma-2-2b)

Screen: 7 RAVEL-style families plus 2 pointer families. Methods: MDAS, DAS, mean-diff, full vector, SAE and random.
Rule: the trained reference scores >= 0.5 and beats every shortcut by >= 0.2, robustly across seeds.

| Candidate | MDAS reference | Best shortcut | Seeds passing |
|---|---|---|---|
| city_pop Language, first screen (L15, 150 steps) | 0.58 | 0.27 mean-diff | Entity split 0/4 (0.46-0.48) -> fails |
| city_pop Language @L12, gap hunt (k128, 300 steps) | 0.52-0.56 / 0.64-0.66 | 0.22 SAE-50 | 4/4 both splits (thin margin) |
| **city_pop Language @L12, boosted (k256, 600 steps)** | **0.65-0.68 / 0.72-0.76** | 0.22 SAE-50; DAS 0.13 | 4/4 both splits, gap ~0.45 |
| city_pop Continent @L10, boosted | 0.80-0.86 / 0.74-0.77 | 0.42 mean-diff; DAS 0.24-0.28 | 4/4, gap ~0.40 |
| US state/capital/abbr (inseparable) | 0.24-0.37 | - | - |

Stage-5 environment stress tests (Language task):

| Test | Result |
|---|---|
| Reference through the tools | passes 4/4 seeds on all 6 instances, 3 Language and 3 Continent (0.51-0.68) |
| Lazy scripts (DAS k1-k256, no keep examples) | all <= 0.18 |
| Script vs cross-fitted privileged ceiling | mean ratio 0.885; headroom 0.05-0.11 |
| Grading noise | ~0.05 SD combined; headroom ~1.5 SD |
| KL guard | inert (damaging edits have KL <= 0.054 and already fail on Iso) |
| gpt-6-luna, 12 episodes | 0.32-0.53, pass 2/12 (pass@1 ~0.17), keep examples used in 12/12, $1.43 |

- No candidate passed the strict rule at the original screen budget. A candidate appeared only after the gap hunt and
  the boost. More training made DAS leak more, so the gap is not caused by an under-trained baseline.
- **Continent is not a valid task:** lazy DAS with no keep examples reaches 0.50 at k=8. Its gap was a k-gap, not a
  method gap.
- Inseparable pairs replicate RAVEL's country/language result and extend it to two-hop US state/capital/abbreviation
  and to verb forms.
- **Negative result for RL use:** the Language task is honest, but a fixed MDAS script reaches ~89% of the ceiling. The
  remaining spread is about the size of grading noise, and gpt-6-luna already knew the recipe. Its score tracked
  training volume. The task is usable as an evaluation or calibration unit, not as a stand-alone RL training
  environment.
- Instance axes meant to break the recipe (multilingual same-country pairs, hidden downstream readouts) opened no
  learnable room: ratio ~0.87 and reachable headroom 0.01-0.13.

## 5. Final status and conclusions

**Validated:**
- The state-direction premise and its handoff, on 4 models, with controls.
- The state-variable meaning.
- The leakage axis on the ladder, and hop separation (Qwen2.5-1.5B only).
- The hard tier of the Phase 6 environment (scripted calibration only).
- T2 v2 as a no-shortcut environment with a gated existence proof and a working tool path.
- The boosted RAVEL Language gap.

**Negative results:**
- The Phase 6 medium tier does not separate.
- 5 of 6 Phase 7 tasks are passed by a recipe or a script.
- T2 v2 has low instance validity (27.5%), and its fixed tool-path reference already reaches about 86% of the
  ceiling (0.76 vs 0.88), so headroom is limited.
- RAVEL Continent is invalid.
- RAVEL Language is saturated by a script, so it does not work as an RL training environment.

**Invalid or excluded:** all gpt-6.1-sol Phase 7 runs, 4 Sonnet runs affected by a T4 bug, the Sonnet Phase 6 probe as a
pass rate, the seed-0 medium-tier result, and the Phase 4b hand-computed rewards.

Conclusions:

1. Two-hop recall in these models runs through a causally used state variable at the subject token. It is readable and
   editable over a mid-layer band and hands off sharply to the final token at 0.69-0.86 of depth. This holds in four
   models from two families.
2. Leakage onto other subjects, not flip rate, is what separates naive edits from careful ones. A grader that checks
   private third-state cities blocks the main reward hacks: answer directions, brute-force overwrites and prompting.
3. Tasks that look like interpretability tasks are often passable by a fixed recipe or a behavioural script. Zero-effort
   baselines and construct-validity audits are needed before agent scores mean anything. Here they overturned 5 of 6
   tasks.
4. Requiring isolation ("keep the state, change the capital") creates a task beyond the standard recipe. Isolation needs
   a conditional (gated) edit, which no single additive vector can provide. Validity on fresh instances is limited by
   how well a gate fit on a few dev cities recognises held-out ones.
5. An honest task (no shortcut, sound grader) can still be a poor RL environment if a fixed script reaches the ceiling.
   Script-vs-ceiling ratios and grading-noise estimates should be part of environment validation.
6. Replays, re-seeded references, shortcut baselines and spend ledgers caught real bugs. One tool bug invalidated a whole
   agent's runs. Another grader issue (country KL) was fixed before any agent run, and all earlier runs were
   re-graded.

## 6. Limitations and threats to validity

- **Clustered data.** Wilson CIs treat cities as independent, but cities cluster in state pairs (3-6 per pair in Phases
  3-4). True uncertainty is wider.
- **One model for the deeper studies.** The ladder, hop separation, the Phase 7 suite and T2 v2 ran only on
  Qwen2.5-1.5B. The RAVEL work ran only on gemma-2-2b.
- **Small instance counts.** 7-8 instances per Phase 7 task, 10 T2 v2 pilot instances, and 2 RAVEL Language
  instances for the learnability test. Per-instance F is coarse (Phase 6: 3-6 held-out items, in steps of 0.17-0.33;
  Phase 7: 3-16 items).
- **Different leakage definitions.** The Phase 4 metric ("switched to target") understates the Phase 6 grader's
  ("any change").
- **Unequal agent budgets.** In T2 v2, Sonnet had $0.10 and 25 turns, and 5/10 episodes hit the cost cap. sol had $0.25
  and 40 turns, a budget chosen after the Sonnet pilot.
- **Partial runs.** The T2 v2 validation oracle ran on 10 of 20 drawn pairs. The cause diagnosis used only the 4 tuning
  pairs.
- **Ceilings are not proven maxima.** They are the best edit found. In some cases a ceiling came out below the fair
  oracle or the tool reference.
- **Circular reference scores.** A 100% reference pass on kept instances is circular, because instances were selected on
  it. Re-seeded, the T2 v1 reference passes 19/28 (mean 0.54).
- **Post-hoc settings.** The RAVEL KL hinge was set after seeing reference KLs. It turned out inert.
- **Shared GPU.** The GPU was shared with other jobs. Two sol episodes crashed with CUDA errors and were re-run. One luna
  episode crashed on a 429 and was charged its cap.
- **Small document disagreements.** Phase 0 zs1 is 0.84 (fp32) vs 0.83 (bf16). The 3B unseen-template peak is 0.78
  (`FINDINGS.md`) vs 0.77 (notebook); the raw value is 76/98 = 0.7755. The Sonnet probe cost is $0.33 in a commit
  message vs $0.27 after correction. `PROGRESS.md` §3-5 are stale.
- **Spend totals.** No document states a single total API spend. The $12.02 figure below is the sum of three ledgers.
  It includes worst-case charges and the invalidated sol runs. It does not include the cost of the research sessions
  themselves.
- **Missing files.** The audit cites generation logs under `logs/`, which is gitignored and absent. `AUDIT_T2V2.md`,
  referenced in the RAVEL notebook, is not in the repo. `FINDINGS_RAVEL_SCREEN.md` says the code lives on branch
  `ravel-screen`, but that branch was merged with main (c8e6ed8, "Merge main into ravel-screen", which is now main's
  head), so the code is on the main line.

## 7. Future work (in order)

1. **Raise T2 v2 instance validity.** Give the gate more public dev cities per source state (>= 18 usable cities), or
   ship only validated (pair, layer) instances, preferring L16-20. Test whether the tool's 16-example cap explains the
   gap between the tool reference and the oracle.
2. **Rerun the T2 v2 agent audit with fixed rules.** Fix in advance a pass rule of R >= 0.8 x ceiling. Use equal
   budgets for all agents and more instances, and drop or flag low-ceiling instances.
3. **Apply the proposed suite redesigns** (`FINDINGS.md` §10, `AUDIT.md` §10). T3: min(F) instead of cube root. T4:
   layer within +/-1, distractor plants, no clean access. T6: per-instance truth, method not stated. T7: absolute norm
   or fixed layer. All tasks: >= 10 held-out items and decoy cities.
4. **Rerun gpt-6.1-sol on the fixed Phase 7 suite** (about $4 estimated).
5. **Decide the Phase 6 medium tier.** Options: keep it as a frontier tier, use "switched to target" leakage with a
   lower weight, or match the tool optimiser to Phase 4b. Then run a proper calibration at 20 episodes per tier (about
   $3).
6. **Extend the ladder and hop separation** to Qwen2.5-3B, 7B and gemma-2-2b.
7. **Build tasks with per-instance ground truth.** Planted mechanisms, a budget squeeze with no gradient training, and a
   large varied pool of hidden downstream readouts, to find tasks a fixed script cannot saturate.
8. **Open questions:** why the prompt-derived vector (C4v) does nothing at the city token; better DAS training as a
   second careful route; T5 (erase) as a frontier task.

## 8. Key numbers at a glance

| Item | Value | Source |
|---|---|---|
| Subject models | 4 studied (Qwen2.5-1.5B/3B/7B, gemma-2-2b) | `FINDINGS.md` §1 |
| Dataset | 50 states/capitals, 343 curated cities | `LAB_NOTEBOOK.md` 2026-09-29 setup |
| Readout fix (1.5B) | 91% few-shot / 83-84% zero-shot correct (greedy readout had suggested 4/16) | `FINDINGS.md` §1; `LAB_NOTEBOOK.md` Phase 0 bf16 |
| Peak state-vector flip | 0.86-0.91 across 4 models (random <= 0.05) | `FINDINGS.md` §1 |
| Handoff layer | 0.69-0.86 of depth | `FINDINGS.md` §1 |
| State-question flip vs capital flip | 0.95-0.98 vs 0.86-0.91 | `FINDINGS.md` §2 |
| Leakage, plain gradient vs leak-aware | 1.00 -> 0.07-0.12 (flip 0.83-0.97) | `FINDINGS.md` §3 |
| Hop separation | 0.98-1.00 vs mean-diff <= 0.02 | `FINDINGS.md` §4 |
| Phase 6 hard tier | careful 5/24 vs all naive 0/24 | `FINDINGS.md` §7 |
| Phase 6 medium tier (negative) | careful 5/23 vs mean-diff 6/23 | `FINDINGS.md` §7 |
| Phase 7 tasks passed by a recipe or script | 5 of 6 (only T2 not; T6 via the procedure stated in the task) | `FINDINGS.md` §10; `results/task_map.json` |
| Invalidating tool bug | 193 errors / 1,545 tool results; 44 runs excluded | `LAB_NOTEBOOK.md` "gpt-6.1-sol results INVALID" |
| T2 v2 shortcut runs | 400, none >= 0.6 (best 0.51) | `results/suite/t2_ravel/validate.sc0.jsonl`; `FINDINGS.md` §11 |
| T2 v2 valid instances | 11/40 = 27.5% (CI 16-43%) | `results/suite/t2_ravel/validate_summary.json` |
| T2 v2 gated oracle isolation | Iso_other >= 0.94 | `FINDINGS.md` §11 |
| T2 v2 tool-path reference | 10/11, mean R 0.74 (pilot-set mean 0.76 vs ceiling 0.88) | `FINDINGS.md` §11 |
| T2 v2 agents | sol 5/20 (60% of ceiling); Sonnet 0/10 (15%), unequal budgets | `FINDINGS.md` §11; `results/suite/t2_ravel/ceiling_summary.json` |
| RAVEL Language @L12 boosted | MDAS 0.65-0.76 vs best shortcut 0.22, 4/4 seeds | `FINDINGS_RAVEL_SCREEN.md` §A |
| RAVEL script saturation | ~89% of privileged ceiling (ratio 0.885) | `FINDINGS_RAVEL_SCREEN.md` §B |
| gpt-6-luna on RAVEL Language | 0.32-0.53, pass 2/12 | `FINDINGS_RAVEL_SCREEN.md` §B |
| Replays reproducing | 42/42 (audit), 31/31 (tool fix), 45/45 (review fixes), 51/51 references (10 within bf16 tolerance), 122/123 regraded exactly | `AUDIT.md`; `LAB_NOTEBOOK.md`; `FINDINGS.md` §11 |
| Python code | 10,276 lines (edithunt/ 7,848; experiments/ 2,428) | `wc -l` over `edithunt/**/*.py`, `experiments/*.py` |
| Agent tools | 12 (suite env), 8 (RAVEL city env) | `edithunt/env/tools.py`; `edithunt/ravel/agent.py` |
| Result files | 385 | `results/` |
| API spend (ledgers) | Anthropic $3.91, OpenAI suite $5.63, OpenAI screen $2.48; sum $12.02 (computed) | `results/suite/ledger.json`; `results/screen/env/ledger.json` |
| Commits | 42 (all branches) | `git log --all` |

## 9. Reproducing

- **Code:** core library `edithunt/`; agent environments `edithunt/env/` and `edithunt/ravel/`; experiment scripts
  `experiments/` (Phases 0-4b: `phase*.py`; suite: `suite_*.py`, `task_map.py`, `audit_phase7.py`; T2 v2:
  `t2_*.py`; checks: `verify_*.py`); RAVEL stages: `edithunt/ravel/stage1.py`-`stage4.py`, `gap_hunt.py`,
  `boost.py`, `env_*.py`. Environment notes: `edithunt/env/README_ENV.md`.
- **Results:** per model `results/Qwen2.5-1.5B/`, `results/Qwen2.5-3B/`, `results/Qwen2.5-7B/`,
  `results/gemma-2-2b/`; Phase 6 agent runs `results/agent_runs/`; suite `results/suite/` (instances, runs,
  `runs_sol_bug/` for the invalid runs, `t2_ravel/`, `ledger.json`); audit `results/audit/`; task map
  `results/task_map.json`; RAVEL `results/screen/` and `results/screen/env/`.
- **Figures:** `results/figures/fig1_layer_curves.png` (state direction across 4 models), `fig2_ladder.png` (flip vs
  leakage by intervention class), `fig3_tiers.png` (scripted reward by tier), `fig4_task_map.png` (Phase 7 difficulty
  map), `fig5_breadth_screen.png` (RAVEL screen).
- **Detailed write-ups:** `FINDINGS.md`, `LAB_NOTEBOOK.md`, `AUDIT.md`, `FINDINGS_RAVEL_SCREEN.md`,
  `LAB_NOTEBOOK_RAVEL_SCREEN.md`.
- **Requirements:** runs need a CUDA GPU (results here used one NVIDIA A10G 23 GB in bf16). Agent runs also need
  Anthropic or OpenAI API keys and are capped by the spend ledger.
