# EditHunt lab notebook (append-only)

## 2026-09-29 — setup
- Hardware: Apple M4 (10-core GPU), 24 GB unified memory, MPS. No CUDA. Machine is under heavy memory
  pressure from other apps (~18 GB swap in use); raw MPS fp16 matmul benchmarks at ~0.8 TFLOPS, so
  everything is compute-bound. Qwen2.5-7B is not feasible here (brief: only with >=20 GB GPU).
- No `HF_TOKEN` in `.env` -> gated models (gemma-2-2b, Llama-3.2) are skipped.
- Env: project venv `.venv` (torch 2.14, transformers 5.17). The base conda env has a torchvision/torch
  mismatch that breaks transformers imports, so it is not used.
- dtype: fp32 on MPS (brief default for <=3B). fp16 check on 86 fs1 prompts: argmax over 50 capitals
  agrees 100%, median |score diff| 0.008 nats, max 0.07.
- Dataset (`edithunt/data.py`): 50 states, 343 curated cities after removing capitals, state-name
  substrings, and ~85 multi-state names. 32 non-reserved states have >=6 cities. Few-shot demo states
  (IL, FL, WA for fs1/ho_fs; NY, AZ, MI for fs2) are reserved and never source/target.
- Scoring: exact full-sequence log-prob of each of the 50 capital strings (19 are multi-token in the Qwen
  tokenizer, e.g. " Tall|ahas|see", " B|ism|ar|ck"). Continuations run off the KV cache. Pruning: a
  multi-token candidate's full log-prob <= its first-token log-prob, so continuations are computed only
  when that bound beats the current best -> the argmax is exact. Verified: pruned argmax == full-exact
  argmax on 86x2 prompts, and the bound is never violated.
- Shared-prefix KV cache for few-shot templates (pad inserted between prefix and suffix). Verified vs
  plain left-padding: max |score diff| 1.4e-4, identical argmax, captures match to 1e-4, with and
  without interventions.
- Observation (smoke test): Coeur d'Alene/fs1 greedy-decodes to " Lewiston" (not a capital) while the
  50-way argmax is Boise. "Valid" (50-way argmax) is therefore more lenient than greedy correctness;
  both are reported.

## 2026-09-30 — moved to cloud GPU (NVIDIA A10G 23 GB, CUDA, bf16)
- Env: fresh `.venv` (torch 2.8.0+cu128, transformers 4.57.6). `.env` has HF_TOKEN (ANT_KEY pending). No push
  credentials on this box, so commits are local until pushed from elsewhere.
- Fix (phase4_ladder.py, before first run): C2/C3/C4v keys used `ndev=len(dev_s_all)`, which differs per pair,
  so those cells were never pooled across pairs; the C1 `min(ndev,3)` hack also merged "all" with nd=2 when a
  state had only 2 dev cities. Now ndev is a label: 1, 2, or "all" (C4: "none").

### Phase 0 rerun in bf16 (Qwen2.5-1.5B), 22 s, n=343 cities per template
| template | valid (50-way argmax) | greedy-correct | agree |
|---|---|---|---|
| zs1 | 0.83 [0.79,0.87] | 0.44 | 0.61 |
| zs2 | 0.79 [0.74,0.83] | 0.06 | 0.27 |
| fs1 | 0.91 [0.87,0.94] | 0.87 | 0.96 |
| fs2 | 0.92 [0.89,0.95] | 0.88 | 0.95 |
| ho_fs | 0.72 [0.67,0.76] | 0.69 | 0.96 |
| ho_zs | 0.87 [0.83,0.90] | 0.13 | 0.26 |
| state_q | 0.99 [0.98,1.00] | 0.99 | 1.00 |
- bf16 vs laptop fp32 (MPS), per-instance valid-flag disagreements: fs1 1/343, fs2 2, zs1 4, zs2 6, ho_fs 10,
  ho_zs 1, state_q 0. Eligible fs1 states unchanged (36). fp32 table kept as `phase0_validity_fp32_mps.json`.

### Phase 1 (Qwen2.5-1.5B, bf16): 30 pairs (increased from 12; runtime was only 3.7 min), all 28 layers
- Build template fs1, city_last; disjoint dev/test cities; n = 103 held-out (city) instances on fs1 per layer.
- Exp A (full residual paste from one target test city): city-token flip 0.95-0.98 for L3-21
  (L8 0.97 [0.92,0.99]), L22 0.16 [0.10,0.24], L23+ 0.00. Final-token flip 0.00 [0.00,0.04] for L0-21,
  L22 0.66 [0.56,0.74], L23 0.99. Mean logit-diff restoration: city L10 0.97, L21 0.92, L22 0.31, L23 -0.01;
  final L21 0.07, L22 0.70, L23 1.01. **Handoff = layer 22** (same as the original notebook).
- Exp B (mean-diff added at held-out city_last):
  - fs1: L0 0.07, L3 0.43, L5 0.76, **L6-21 0.82-0.89** (L8 0.88 [0.81,0.93], L15 0.89 [0.82,0.94]), L22 0.11, L23+ 0.
  - ho_fs (held-out template, n=76): peaks 0.70 [0.59,0.79] at L14-15; 0.45 at L21.
  - ho_zs (n=92): 0.83-0.86 over L7-21. zs1: 0.68-0.82. fs2: 0.78-0.91.
- Controls: random norm-matched ≤0.04 at every layer (fs1 L8 0.00 [0.00,0.04]). Shuffled-label mean-diff
  0.00-0.17 (L8 0.09 [0.05,0.16]); nonzero because shuffled groups still mix src/tgt cities.
- Per-pair premise check (fixed layer L10, chosen from pooled curve): 29/30 pairs ≥0.5 on fs1, 20/27 on ho_fs.
  Weakest pair Maine->Vermont (n=4). Per-pair n is 1-6 cities, so per-pair rates are noisy.
- Cross-pair transfer (T->C vector added to third-state cities), L12: to target capital 0.33 [0.28,0.38],
  kept own capital 0.43 [0.38,0.48] (n=360). The vector is partly "target" and partly "not-source".
- Caveat: Wilson CIs treat cities as independent. Cities are clustered in 30 pairs, so the true uncertainty is wider.
- **Premise verdict: holds on 1.5B.** A shared, causally used state direction at the city token over L6-21,
  handoff at L22. Plot: results/Qwen2.5-1.5B/phase1_layers.png.

### Diagnosis: the country probe was state-entangled (method change)
- `"{city} is a city in the country of"`: base Qwen2.5-1.5B answers the *state* (Dallas -> " Texas" p=0.33,
  Galveston -> " Texas" 0.66). A working T->C edit moves " Texas" to " California" there, so full-vocab KL (~1.0)
  measured the intended state change, not collateral damage. A few-shot version (Lyon/France, Osaka/Japan,
  Toronto/Canada) makes top-1 " the" for 198/343, but full-vocab KL stays ~1.0 (state-token mass still moves).
- New probe (edithunt/data.py COUNTRY_Q + COUNTRY_CANDS; metrics.p_us / binary_kl): few-shot prompt, exact
  scores of 16 country strings, P(US) = mass on {the United States, United States, America, the USA}.
  Damage = binary KL on P(US) and "US-majority kept". Clean P(US) over 343 cities: mean 0.74, >0.5 for 82%.
  Old zero-shot full-vocab KL still recorded as `country_zs` for reference only.

### Phase 2 (Qwen2.5-1.5B, bf16): 30 pairs, mean-diff built on fs1 dev cities at city_last, n=103 held-out
| L | capital flip fs1 | state-belief -> target | P(US) clean->edit | binKL country (rand) | generic KL (rand) | 3rd-state capital -> target |
|---|---|---|---|---|---|---|
| 2 | 0.24 [0.17,0.33] | 0.30 [0.22,0.40] | 0.72->0.39 | 0.53 (0.35) | 0.037 (0.036) | 0.06 |
| 3 | 0.43 [0.34,0.52] | 0.55 [0.46,0.65] | 0.72->0.44 | 0.43 (0.31) | 0.032 (0.037) | 0.09 |
| 4 | 0.60 [0.51,0.69] | 0.78 [0.69,0.85] | 0.72->0.57 | 0.21 (0.15) | 0.030 (0.033) | 0.13 |
| 5 | 0.76 [0.67,0.83] | 0.93 [0.87,0.97] | 0.72->0.63 | 0.12 (0.10) | 0.029 (0.038) | 0.22 |
| 8 | 0.89 [0.82,0.94] | 0.98 [0.93,0.99] | 0.72->0.66 | 0.086 (0.034) | 0.015 (0.029) | 0.24 |
| 15 | 0.89 [0.82,0.94] | 0.97 [0.92,0.99] | 0.72->0.63 | 0.086 (0.031) | 0.003 (0.008) | 0.19 |
| 21 | 0.82 [0.73,0.88] | 0.98 [0.93,0.99] | 0.72->0.66 | 0.060 (0.017) | 0.002 (0.002) | 0.16 |
| 22 | 0.11 [0.06,0.18] | 0.22 [0.15,0.31] | 0.72->0.70 | 0.011 (0.001) | 0.002 (0.002) | 0.00 |
- Random norm-matched: capital flip ≤0.05, state flip ≤0.04 at every layer; 3rd-state -> target 0.00.
- **Meaning: a state variable.** The fs1-built vector flips the STATE_Q answer (a different prompt format) to
  the target state more often than it flips the capital (L8 0.98 vs 0.89). It is not an "answer" direction.
- **Specificity:** generic-text KL is at the random-vector level. On the country probe, mid-layer damage is
  small but above random (binKL 0.086 vs 0.03). Early layers (≤4) are fragile: even random vectors of matched
  norm cut P(US) (0.72->0.47 at L2). **The main collateral damage is leakage:** third-state cities go to the
  target capital 0.16-0.24 (n=180) at L5-21 and keep their own capital only 0.60-0.68 (random 0.98). This fits
  the Phase 1 transfer result: the vector is partly "target" and partly "not-source".
- **Norm sweep** (scales 0.25/0.5/1/2 at L4, L8, L15): flip 0.04/0.25/0.89/0.95 at L8. At 2x, country binKL
  0.27 (P(US) 0.48), leakage 0.48, zero-shot full-vocab KL 3.3. Damage grows with norm, so a grader penalty
  scaled to a reference edit will punish overshooting. The old ~1.0 country KL was entanglement, not norm.

## 2026-09-30 (session 2) — reflection before Phase 4, and plan changes

### What the Phase 4 debug run (1 pair Texas->California, L4/L8, 10 steps, n=6) exposed
Two results were implementation artifacts, not findings. Diagnosed on the same pair (scratch script):
- **C3 DAS gave exactly 0 flips and ~0 KL.** Not broken, undertrained: from random init the rank-1 loss
  goes 9.0 -> 2.1 at step 30 -> 0.26 at step 40 -> 0.07 at step 70 (L4; L8 similar). The debug run used 10 steps.
  Fix: separate `--das_steps` (default 100).
- **C2 (Adam lr=2, no norm constraint) lands at |v|=460-480**, vs |C1|=28-32 and |resid|=51-55 at the city
  token, and is nearly orthogonal to C1 (cos 0.09-0.12). It is an overwrite of the residual, not an edit.
  Its large KL (country binKL 1.4, generic 0.4) reflects optimizer settings, not task difficulty.
  Fix: keep it as "naive C2" (and record |v|), add **C2n** (norm capped at |C1| of the same layer) and
  **C2nkl** (capped + generic KL penalty) as the careful gradient classes.
- **Leakage was not measured in Phase 4**, though Phase 2 found it to be the main collateral damage. Added:
  third-state test cities (2 other states, 6 cities, fs1) -> `third_to_tgt`, `third_kept`.

Re-debug with fixes (still n=6, one pair, L4, 50 steps; direction only):
| class | flip fs1 / ho_fs | 3rd-state -> target | 3rd-state kept | binKL country | KL generic | abs(v) |
|---|---|---|---|---|---|---|
| C0 paste | 6/6 / 5/6 | 6/6 | 0/6 | 0.15 | 0.13 | 45 |
| C1 mean-diff | 3/6 / 3/6 | 0/6 | 4/6 | 0.25 | 0.015 | 28 |
| C2 (uncapped) | 6/6 / 6/6 | 6/6 | 0/6 | 1.40 | 0.41 | 481 |
| C2n (capped at abs(C1)) | 5/6 / 5/6 | 6/6 | 0/6 | 0.17 | 0.011 | 28 |
| C3 DAS rank-1 | 3/6 / 3/6 | 0/6 | 6/6 | 0.02 | 0.002 | — |
**Candidate insight:** gradient training on "output the target capital" finds an *answer* direction
("say Sacramento") that fires on any city, even when norm-capped. Mean-diff and DAS find the *state*
variable and do not leak. A grader that includes third-state leakage would separate these; to be tested at scale.

### Phase 3 changes (before its first full run)
- Same C2 overshoot issue: added `gradn_city` / `hop2n_city` (norm capped at |md_city|).
- **Overfitting control:** `hop2_city` trains a keep-source-state penalty on STATE_Q (dev cities) and was
  evaluated on STATE_Q (test cities), so success could mean "memorized that prompt". Added `STATE_Q_HO`
  (different wording, never trained on) and `hop2_success_ho`.
- Debug (1 pair, L8, 20 steps, n=6): hop2n_city flips 6/6 and keeps the source state 6/6 on STATE_Q
  *and* 6/6 on STATE_Q_HO, at abs(v)=31.6 (= abs(md_city)). md_city: HOP2 1/6. perp_city (md with the
  STATE_Q direction projected out) flips 0/6. md_final at L23 flips the capital but also the state answer
  (the vector is applied at the final token of STATE_Q too). Full run launched.

### Plan changes
1. Phase 4 full run: 12 pairs (was 6), grad layers 3,4,5,8,15, DAS 100 steps, C2n/C2nkl, leakage.
2. The grader should include third-state leakage (pending Phase 4 at scale). This is the knob that seems
   to separate answer-direction hacks (C2, C0) from the state variable (C1, DAS).
3. Still open from the review: the env's `optimize_vector` tool makes C2 one call away. If leakage is
   penalized, uncapped/answer-direction C2 fails anyway; decide after Phase 4.
4. Order: Phase 4 + Phase 3 (running concurrently on the A10G) -> Phase 5 (3B, gemma-2-2b, 7B) -> set tiers
   + grader in edithunt/env -> Claude calibration (ANT_KEY now in .env) -> FINDINGS.md. Deadline Oct 8.
