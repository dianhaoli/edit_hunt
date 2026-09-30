# EditHunt: progress report (as of 2026-09-30, evening)

**Start with `FINDINGS.md`** (2-page summary, 3 figures in `results/figures/`).

This covers what has been built, what has been measured, what it means, and what remains.
All numbers below are real measurements from this repo. Nothing is extrapolated.
Every rate is given with its sample size n and a 95% Wilson confidence interval.

---

## 0. TL;DR

| Item | Status |
|---|---|
| Core library (dataset, hooks, exact answer scoring, batching) | **Done**, verified with explicit correctness checks |
| Phase 0: which prompts the subject model already answers correctly | **Done** for Qwen2.5-1.5B (rerun in bf16 on A10G, 2026-09-30) |
| Phase 1: replicate and scale experiments A and B (does a "state direction" exist?) | **Done** (1.5B, 30 pairs): premise holds, mean-diff flip 0.82-0.89 over L6-21, handoff at L22. See LAB_NOTEBOOK 2026-09-30 |
| Phase 2 (meaning & specificity) | **Done** (1.5B, 3B, gemma, 7B; 30 pairs each): a state variable (STATE_Q flips 0.97-0.98 at L8-21); main collateral damage is third-state leakage 0.16-0.24. Country probe redesigned (state-entangled) |
| Phase 3 (hop separation) | **Done** (1.5B, 12 pairs, n=42): achievable only deliberately. Norm-capped city-token vector trained with a keep-state term: HOP2 0.98-1.00 (0.93-1.00 on unseen state-question wording). Mean-diff ≤0.02; naive capped gradient 0.24 at L8 but 0.86 at L20, so the hard tier needs a layer ceiling around 8 |
| Phase 4 (difficulty ladder) + 4b | **Done** (1.5B, 12 pairs): leakage separates methods. Gradient 'output target' edits leak 56-100%; mean-diff 14-24%; DAS ≈0 but flips ≤0.71; leak-aware gradient (4b) flips 0.83-0.97 on the unseen template with 7-12% leakage |
| Phase 5 (other models) | **Phases 0-2 done** on Qwen2.5-3B, gemma-2-2b, Qwen2.5-7B: premise and 'state variable' meaning hold in all four models; handoff at 0.69-0.86 of depth. Ladder/hop separation only on 1.5B |
| Phase 6 (environment) | **Rebuilt from data**: leakage term, P(US) country probe, tiers, keep_cities/keep_state_cities tool options. Scripted calibration (23-24 instances/tier): easy mean-diff 21/24; **hard separates** (naive 0/24, careful 5/24); **medium does not** (mean-diff 6/23, careful 5/23), decision needed (FINDINGS §7). **Claude run blocked: API credit balance too low** |

**Headline so far (Phase 0):** the zero-shot failure in the original notebook ("only 4/16 Texas cities
correct") is mostly an artifact of how the answer was read. When you ask "which of the 50 state capitals
does the model rank highest?", Qwen2.5-1.5B gets the zero-shot prompt right for **84% of cities** (n=343).
It just prefers to *say* " the" or " which" first. With a few-shot prompt it gets **91%** right and also
says the answer. The model knows which state a city is in **99%** of the time, so almost all failures are
in the second hop (state → capital), not the first (city → state).

---

## 1. What "the model gets the capital right" means (Phase 0 explained)

There are two models in EditHunt:

- The **subject**: a small open model (here **Qwen2.5-1.5B**) that we inspect and edit.
- The **agent**: a frontier LLM (e.g. Claude) that will later try to edit the subject.

**Phase 0 only involves the subject.** We give the *unmodified* Qwen2.5-1.5B a prompt such as

> The state containing **Dallas** has its capital in

and check whether it answers **" Austin"**. "It gets the capital right" means that **Qwen itself, with
no edits, already produces the correct capital** for that city.

**Why this matters.** The whole task is "edit the model so it answers Sacramento *instead of* Austin". That
is only meaningful if the model answered Austin to begin with. If it already answers wrongly, a "flip"
proves nothing. So Phase 0 is a **validity filter**: we keep only the (city, prompt template) instances
the clean model gets right. Every later experiment uses only those instances.

### How the answer is read (the scoring method)
We do **not** just look at the single most likely next token. Many capitals are several tokens long
("Salt Lake City" = " Salt" + " Lake" + " City"; "Tallahassee" = " Tall" + "ahas" + "see"), and 19 of the
50 are multi-token for Qwen's tokenizer. Instead, for each prompt we compute the model's full probability
of each of the 50 capital strings, and the **answer = the capital with the highest probability**.
We also run ordinary greedy text generation and report how often the two methods agree.

### Results: Qwen2.5-1.5B, 343 cities (all 50 states)

| Template | Example (for Dallas) | Valid (top capital correct) | Greedy generation starts with the correct capital | Agreement between the two |
|---|---|---|---|---|
| `fs1` few-shot (build template) | "The state containing Chicago has its capital in Springfield. [2 more examples] The state containing Dallas has its capital in" | **0.91** [0.87, 0.93] | 0.87 | 0.96 |
| `fs2` few-shot, second wording | "The capital of the state where Buffalo is located is Albany. … Dallas is located is" | **0.92** [0.89, 0.95] | 0.88 | 0.96 |
| `ho_fs` **held-out** few-shot | "Chicago is in a state whose capital is Springfield. … Dallas is in a state whose capital is" | 0.71 [0.66, 0.75] | 0.68 | 0.97 |
| `zs1` zero-shot | "The state containing Dallas has its capital in" | 0.84 [0.80, 0.88] | 0.45 | 0.61 |
| `zs2` zero-shot | "The capital of the state where Dallas is located is" | 0.78 [0.73, 0.82] | **0.07** | 0.29 |
| `ho_zs` **held-out** zero-shot | "Q: What is the capital of the state that Dallas is in?\nA:" | 0.87 [0.83, 0.90] | 0.13 | 0.26 |
| `STATE_Q` (first hop only) | "Q: Which US state is Dallas in?\nA: The state of" | **0.99** [0.98, 1.00] | 0.99 | 1.00 |

n = 343 for every row. "Held-out" templates are never used to build any intervention. They exist only
to test whether an edit generalizes.

### What this tells us
1. **Zero-shot "failures" are mostly a readout problem.** On `zs2` the model generates the correct capital
   first only 7% of the time, yet the correct capital is its top-ranked capital 78% of the time. With
   few-shot prompts the two readouts agree 96–97% of the time. **So few-shot templates are the main
   workhorse**, which fixes the tiny-sample problem from the original notebook.
2. **Hop 1 (city → state) is near-perfect (99%); hop 2 (state → capital) is where errors happen.** Some of
   the failures are revealing:
   - Georgia cities (Savannah, Macon, Statesboro) → **"Augusta"**. Augusta *was* Georgia's capital in the
     1700s, so the model links Georgia to a historical capital.
   - Marietta → Columbus (there is a Marietta, Ohio), Hoover → Des Moines (Herbert Hoover was from Iowa),
     Lowell → Salem (Massachusetts ↔ Salem association). These are ambiguity or association errors.
   - Small Delaware towns → Annapolis/Trenton (neighboring states).
3. **36 states** have ≥ 5 valid cities under `fs1` (excluding the 6 states used in the few-shot
   examples). That is enough to run ≥ 10 source→target state pairs with separate dev and test cities,
   which fixes the "only 2 dev + 3 held-out cities" problem.

Raw data: `results/Qwen2.5-1.5B/phase0_validity.json` (per city, per template: valid flag, predicted
capital, greedy correctness).

---

## 2. What has been built

```
edithunt/
  data.py        50 states + capitals; 343 hand-curated cities; 6 capital templates; state/country probes; 50 generic sentences
  model.py       Subject wrapper: loading, city-position indexing, batching, exact 50-way scoring, residual capture
  hooks.py       forward hooks on decoder layers: capture / set (patch) / add (steer) / subspace swap
  runner.py      packs many different edits (different layer or vector per row) into one forward pass
  metrics.py     Wilson CIs, KL divergence
  instances.py   eligible states, deterministic dev/test city splits, pair selection
  vectors.py     mean-difference vectors, norm-matched random controls
  optim.py       C2 gradient-trained steering vector (optional KL / keep-state penalty); C3 DAS-style rank-k subspace
  common.py      CLI args, seeding, result JSON saving (config + git hash + timestamp)
  env/           Phase 6 environment prototype (see §5)
experiments/
  phase0_validity.py   done
  phase1_patching.py   exps A + B + controls + cross-pair transfer (was running)
  phase2_meaning.py    state variable vs answer direction, collateral damage
  phase3_hopsep.py     hop separation (edit the capital without changing the state belief)
  phase4_ladder.py     difficulty ladder: intervention classes C0–C4 x constraints
  plots.py             layer-sweep plot for Phase 1
results/<model>/       JSON per experiment (+ logs, PNGs)
LAB_NOTEBOOK.md        append-only log
```

### Dataset hygiene (per the brief)
- Removed capital cities, any city whose name contains a state name ("Oklahoma City", "Virginia Beach",
  "Iowa City", …), and ~85 names that exist prominently in several states (Portland, Springfield,
  Columbus, Aurora, Decatur, Lancaster, Quincy, Carmel, Moscow, …).
- Few-shot example states are **reserved**: Illinois, Florida, Washington (in `fs1`/`ho_fs`) and New York,
  Arizona, Michigan (in `fs2`) are never used as source or target states. That way the answer
  string never appears in the prompt.

### Correctness checks performed (all passed)
| Check | Result |
|---|---|
| City-token position: tokenize the prefix up to and including the city, assert it is a prefix of the full prompt, and assert the decoded token ends the city name | 0 failures over 343 cities × 6 templates |
| Fast scoring vs exhaustive scoring. The fast version skips a multi-token capital's continuation when even its first-token probability can't beat the current best, which is mathematically safe. | Identical top answer on 172 prompts; the bound was never violated |
| Shared-prefix KV cache (the few-shot examples are computed once and reused) vs plain padding | Max score difference 1.4e-4; identical answers; captured activations match to 1e-4, with and without an edit applied |
| fp16 vs fp32 on Apple MPS | 100% identical answers, median score difference 0.008 (fp32 used anyway, per the brief) |

---

## 3. Phase 1: in progress (no results yet)

**Question:** is there a *shared* "this city is in state X" direction inside the model, sitting at the
city's token, that causally drives the capital answer?

**Design (running when paused):**
- 12 source→target state pairs: Texas→California plus 11 random pairs, e.g. Oklahoma→Pennsylvania,
  Arkansas→Mississippi, Oregon→Nevada, Indiana→Hawaii.
- Each state's valid cities are split into **dev** (used to build vectors) and **test** (held out).
- **Exp A (full residual patch):** copy the entire internal state at one layer from a target-state city
  into a source-state city, at (i) the city token or (ii) the final token. This locates the "handoff"
  layer where information moves from the city token to the answer position.
- **Exp B (mean-difference vector):** v = mean(target-state dev cities) − mean(source-state dev cities) at
  the city token. Add v to *held-out* source cities and measure the fraction that flip to the target
  capital. Evaluated on the build template and on held-out templates, at every layer 0–27.
- **Controls:** a random vector of the same length, and mean-difference with shuffled state labels.
- **Cross-pair transfer:** does a Texas→California vector also push *Ohio* cities toward Sacramento?
  This tells "means California" apart from "means Texas-minus-California".

**What to expect / look for:** the original notebook suggested flip rate ≈ 1.0 for layers ~5–21 and a
handoff at ~22, but on n = 3. Phase 1 will confirm or refute that with n ≈ 40 per layer.

---

## 4. Phases 2–4: designed, scripts written, not run
- **Phase 2 (meaning and specificity):** apply the Phase 1 vectors and ask "Which US state is X in?".
  If the answer flips to the target state, the vector is a *state variable*. If only the capital
  changes, it is an *answer direction*. Also measures collateral damage (KL divergence) on the country
  prompt, on 50 unrelated sentences, and on third-state cities.
- **Phase 3 (hop separation, the candidate hard tier):** find an edit that makes the model output
  Sacramento **while still saying Dallas is in Texas**. It compares mean-difference edits at the city
  token, the final token and the "capital" token; a city-token vector with the state direction
  projected out; and a gradient-trained vector with a "keep the state answer" penalty.
- **Phase 4 (difficulty ladder):**
  - Intervention classes:
    - C0: paste one city's activation
    - C1: mean-difference
    - C2: gradient-trained vector, with or without a KL penalty
    - C3: DAS rank-1/rank-4 subspace
    - C4: prompt-only, plus a prompt-derived vector
  - Constraints swept:
    - layer ceiling
    - city-last vs all city tokens
    - number of dev examples
    - build template vs averaged templates
  - Output: which settings make naive methods fail while careful ones succeed. That split is the
    medium-difficulty tier for the pitch.

---

## 5. Phase 6: environment prototype (built by a subagent, smoke-tested on Qwen2.5-0.5B)

Files: `edithunt/env/` (`instance.py`, `grader.py`, `tools.py`, `agent_loop.py`, `baselines.py`,
`README_ENV.md`).

- **Instance:** (subject model, source state, target state, dev cities, *private* held-out cities and
  templates, constraints: max layer, allowed positions, max number of edits, KL budget, forward-pass
  budget, and whether the state answer must be preserved). Instances are kept only if a reference
  mean-difference edit reaches a held-out flip rate ≥ 0.8. That proves the intermediate is causally
  used in that instance.
- **Grader (no LLM judge):** `reward = F × (1 − min(1, KL_mean / kl_budget))`.
  - F = held-out flip rate, using held-out cities × held-out templates.
  - KL_mean = the average of KL on country prompts and on generic sentences.
  - Hard tier: a flip counts only if the state answer is unchanged.
  - The grader also verifies the submission's constraints (layer, position, number of edits, finite values).
- **Agent tools** (the vectors stay server-side in named registers):
  - `run_prompts`, `logit_lens`
  - `cache_mean`, `vec_op` (safe mini-language, no `eval`), `vec_info`
  - `eval_intervention`, `optimize_vector`
  - `submit`

  Tools refuse held-out cities and templates, and enforce a forward-pass budget.
- **Claude agent loop:** Anthropic SDK with tool use, model `claude-sonnet-5-5`, key read from `ANT_KEY`.
  So far only tested with a scripted fake client. **No real API calls yet.**
- **Smoke-test numbers (0.5B, tiny n, not meaningful as results):** easy tier: mean-difference agent
  reward 0.45, gradient agent 0.54, random 0.
- **Known issue to fix:** on 0.5B, edits that work cause KL ≈ 0.8–1.1 on the country prompt but only
  ≈ 0.01 on generic text, so the country term dominates the penalty. With the placeholder budget of
  0.3 the medium tier is unwinnable. Tier settings must be tuned from Phases 2–4 data.

---

## 6. Environment constraints hit on this laptop
- **Hardware:** Apple M4, 24 GB unified memory, about 18 GB of swap already in use by other apps. Raw GPU
  matmul benchmarks at only about 0.8 TFLOPS, so runs are compute-bound at about 5 prompts/s. Phase 0
  took about 20 minutes, and Phase 1 was estimated at about 45 minutes.
- **Qwen2.5-7B** doesn't fit.
- **No `HF_TOKEN`**, so gated models (gemma-2-2b, Llama-3.2) are skipped.
- **Python:** the base conda env has a broken torch/torchvision pairing, so the project uses its own
  `.venv` (torch 2.14, transformers 5.17).

## 7. How to resume on a cloud GPU
```bash
git clone https://github.com/dianhaoli/edit_hunt && cd edit_hunt
python -m venv .venv && .venv/bin/pip install torch transformers accelerate matplotlib numpy anthropic python-dotenv scipy
echo "ANT_KEY=..." > .env          # plus HF_TOKEN=... to unlock gemma-2-2b / Llama
export PYTHONPATH=.
# CUDA is auto-detected (bf16). Consider --bs 128 on a big GPU.
.venv/bin/python experiments/phase1_patching.py --model Qwen/Qwen2.5-1.5B
.venv/bin/python experiments/plots.py        --model Qwen/Qwen2.5-1.5B
.venv/bin/python experiments/phase2_meaning.py --model Qwen/Qwen2.5-1.5B
.venv/bin/python experiments/phase3_hopsep.py  --model Qwen/Qwen2.5-1.5B
.venv/bin/python experiments/phase4_ladder.py  --model Qwen/Qwen2.5-1.5B
# Phase 5: repeat phase0 -> phase2 (+ key phase4 cells) with --model Qwen/Qwen2.5-3B, Qwen/Qwen2.5-7B, google/gemma-2-2b
```
Notes for resuming:
- Phase 0 must be run per model before anything else, because it writes the validity table.
- Phases 2–4 have not been executed yet, so expect first-run bugs.
- On CUDA, the brief asks for bf16. If results look noisy, rerun with fp32 and compare a small check.
- Budget for the Claude calibration: one-off, small (10–20 instances per tier), Sonnet 5.5.

## 8. Next steps (in order)
1. Finish Phase 1. Verify the premise (flip rate ≥ 50% on held-out cities for most pairs) against the
   random and shuffled controls.
2. Run Phases 2 and 3, and decide whether hop separation is achievable and where.
3. Run Phase 4 to find the naive-fails / careful-succeeds constraint settings, then set the tier values
   in `edithunt/env/instance.py` from that data, and rebalance the KL terms.
4. Phase 5 on 3B, 7B and Gemma.
5. One small Claude Sonnet 5.5 calibration run per tier. Write `FINDINGS.md` with the three plots.
