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

### Phase 5 setup + Phase 0 on new models (A10G, bf16)
- Loader: `device_map="cuda"` (host RAM is 15 GB; 7B bf16 is 15 GB). gemma-2: forced `attn_implementation="eager"`
  (soft-capping). Check on 40 cities, fs1: sdpa vs eager argmax agree 40/40, max score diff 0.31 nats.
- Access: gemma-2-2b granted by Dan 2026-09-30 (was 403). Llama-3.2-1B/3B still gated (403). Ungated
  fallbacks downloaded: OLMo-2-0425-1B, SmolLM2-1.7B. Qwen2.5-3B, Qwen2.5-7B downloaded.
- Phase 0 validity (n=343 cities, argmax over 50 capitals; greedy-correct in brackets):
| model | zs1 | fs1 | fs2 | ho_fs | ho_zs | state_q |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B (for reference) | 0.84 | 0.91 | — | — | — | 0.99 |
| Qwen2.5-3B | 0.87 (0.13) | 0.90 (0.84) | 0.91 | 0.82 | 0.86 (0.36) | 1.00 |
| gemma-2-2b (eager) | 0.94 (0.77) | 0.96 (0.94) | 0.84 | 0.90 | 0.90 (0.50) | 0.99 |
  gemma-2-2b answers zero-shot far more often (greedy-correct zs1 0.77 vs 0.13 for Qwen-3B).

### Phase 4 full run crashed at pair 9/12 (~58 min lost), fixed
- `S.score([])` raised (torch.cat on empty list): pair 9 had no clean-valid held-out cities for one template.
  Fix: `score()` returns empty tensors for empty input. Nothing was saved because the script only wrote at the end.
  Fix: per-pair checkpoint `phase4_ladder.partial.json` with automatic resume (config must match). Relaunched.

### Phase 3 (Qwen2.5-1.5B, bf16): hop separation, 12 pairs, n=42 held-out source cities (fs1)
HOP2 = capital flips to target AND STATE_Q still answers the source state. HOP2ho = same with STATE_Q_HO
(wording never used in training). Grad layers 8/16/20; mean-diff at every listed layer.
| method | L | flip fs1 | state kept | HOP2 [CI] | HOP2ho | abs(v) |
|---|---|---|---|---|---|---|
| md_city (naive) | 8 | 0.88 | 0.05 | 0.02 [0.00,0.12] | 0.02 | 39 |
| md_city | 16-20 | 0.86-0.88 | 0.02 | 0.02 | 0.02 | 37-46 |
| gradn_city (capped, no keep term) | 8 | 1.00 | 0.24 | 0.24 [0.13,0.39] | 0.19 | 39 |
| gradn_city | 16 | 1.00 | 0.62 | 0.62 [0.47,0.75] | 0.57 | 37 |
| gradn_city | 20 | 1.00 | 0.86 | 0.86 [0.72,0.93] | 0.76 | 46 |
| **hop2n_city** (capped + keep-state term) | 8 | 0.98 | 1.00 | **0.98 [0.88,1.00]** | 0.93 | 39 |
| **hop2n_city** | 16 / 20 | 1.00 | 1.00 | **1.00 [0.92,1.00]** | 0.93 / 1.00 | 37 / 46 |
| grad_city (uncapped) | 8 / 16 / 20 | 1.00 | 0.00 / 0.07 / 0.14 | 0.00 / 0.07 / 0.14 | — | 470-515 |
| md_final | 22 / 23 / 26 | 0.62 / 0.95 / 0.98 | 0.62 / 0.07 / 0.31 | 0.31 / 0.05 / 0.29 | 0.19 / 0.05 / 0.21 | 27-151 |
| md_capital, perp_city | all | 0.00 | 1.00 | 0.00 | 0.00 | — |
- **Hop separation is achievable, and only by a deliberate method.** A norm-capped vector at the city token,
  trained to output the target capital while keeping the source state on the state question, reaches HOP2 0.98-1.00
  (0.93-1.00 on the never-trained wording), at the same norm as mean-diff. Mean-diff never does (≤0.02).
- **Naive capped gradient gets it partly "for free" at late layers** (0.24 at L8 -> 0.86 at L20). By L20 the
  city-token vector trained on the capital objective increasingly carries the answer rather than the state.
  So a hard tier (preserve_state) needs a **layer ceiling around 8** to keep naive gradient low (0.24) while the
  careful method still works (0.98). At L16+ naive gradient is already 0.62-0.86.
- **perp_city flips nothing.** Removing the state-question direction from the mean-diff vector removes the whole
  effect, which supports Phase 2: the city-token mean-diff vector *is* the state variable.
- md_capital (the " capital" token) never works at any layer: the state info is not routed through that token.
- md_final (final token) works only after the handoff (L23+) and then also moves the state answer, since the same
  final-token vector is applied to the state question.
- Caveats: n=42 cities clustered in 12 pairs (3-6 cities each), so true uncertainty is wider than Wilson. The keep
  term uses the *source* state label of dev cities (the agent would know it).

### Phase 5 — Phase 1 on Qwen2.5-3B (36 layers) and gemma-2-2b (26 layers), 30 pairs each
| model | mean-diff flip fs1 (best band) | ho_fs | band (flip ≥0.5 fs1) | handoff (A_final ≥0.5) | handoff / depth | random ctrl | shuffled ctrl max | transfer (plateau) |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | 0.89 (L8-15) | — | L5-21 | L22 | 0.79 | ≤0.05 | 0.17 | 0.33 |
| Qwen2.5-3B | 0.86 [0.78,0.91] n=105 (L12) | 0.77 | L8-30 | L31 | 0.86 | ≤0.01 | 0.29 (L18) | 0.33 |
| gemma-2-2b | 0.91 [0.84,0.95] n=104 (L8) | 0.82 (L9) | L3-17 | L18 | 0.69 | 0.00 | 0.17 | 0.30-0.34 |
- **Premise holds on all three models**, with the same shape: a mean-diff "state" direction at the city token over a
  mid-layer band, ending exactly where full-residual patching at the final token takes over (the handoff).
- Gemma computes the state early (flip 0.76 by L3) and hands off earlier as a fraction of depth (0.69 vs 0.79-0.86).
- Transfer to third-state cities is ~0.3 in all three models: the vector is partly "target" and partly "not-source".
- Qwen-3B shuffled control peaks at 0.29 (L18, one layer); elsewhere ≤0.14. Same explanation as 1.5B (unbalanced
  random splits carry a scaled copy of the real vector); still unverified against group composition.

### Phase 5 — Phase 2 (meaning & specificity) on Qwen2.5-3B and gemma-2-2b, 30 pairs, n=104-105 held-out
Layers chosen per model from its Phase 1 curve (the 1.5B default list would have missed most of 3B's band; a first
launch with that list was killed and relaunched). "rand" = random norm-matched vector at the same layer.
| model | L | capital flip | state -> target | binKL country (rand) | KL generic (rand) | 3rd-state -> target |
|---|---|---|---|---|---|---|
| 3B | 3 | 0.04 | 0.10 | 0.98 (0.31) | 0.034 (0.027) | 0.03 |
| 3B | 8 | 0.62 [0.52,0.71] | 0.69 | 0.30 (0.18) | 0.042 (0.048) | 0.28 |
| 3B | 12 | 0.86 [0.78,0.91] | 0.97 [0.92,0.99] | 0.10 (0.08) | 0.019 (0.030) | 0.32 |
| 3B | 20 | 0.84 [0.76,0.90] | 0.98 | 0.08 (0.02) | 0.004 (0.011) | 0.32 |
| 3B | 30 | 0.64 | 0.94 | 0.03 (0.003) | 0.002 (0.001) | 0.14 |
| 3B | 31 (handoff) | 0.11 | 0.00 | 0.005 | 0.002 | 0.01 |
| gemma | 1 | 0.35 [0.26,0.44] | 0.38 | 0.32 (0.11) | 0.015 (0.016) | 0.09 |
| gemma | 3 | 0.75 [0.66,0.82] | 0.79 | 0.13 (0.06) | 0.013 (0.016) | 0.22 |
| gemma | 8 | 0.91 [0.84,0.95] | 0.96 [0.91,0.98] | 0.034 (0.013) | 0.005 (0.006) | 0.19 |
| gemma | 12 | 0.90 | 0.96 | 0.028 (0.006) | 0.003 (0.003) | 0.17 |
| gemma | 15 / 17 | 0.60 / 0.53 | 0.91 / 0.91 | 0.015 (0.005) | 0.002 (0.002) | 0.04 / 0.03 |
| gemma | 18 (handoff) | 0.00 | 0.04 | 0.006 | 0.002 | 0.00 |
Random controls: capital flip ≤0.01 and state->target 0.00 at every layer, 3rd-state->target 0.00 (both models).
- **Meaning replicates: a state variable in all three models.** State-question flips ≥ capital flips at every mid layer
  (3B L12 0.97 vs 0.86; gemma L8 0.96 vs 0.91).
- **Gemma L15-17: the state answer still flips (0.91) but the capital only 0.53-0.60, with almost no leakage (0.03).**
  Near the handoff the city-token state variable is still read by the state question but only partly by the capital
  prompt. Worth one look in the hop-separation analysis on Gemma.
- **Leakage is the main collateral damage in all three models**, and largest in 3B (0.31-0.34 vs 1.5B 0.16-0.24,
  gemma 0.17-0.24). Generic-text KL is at the random level everywhere in the band.
- **Early layers are fragile in Qwen**: 3B L3-5 cuts P(US) 0.88->0.52 (random 0.73), binKL 0.9-1.0. Gemma is much less so.
- **Norm sweep has a sharp threshold** (both models): 0.5x flips only 0.16-0.18, 1x 0.86-0.91, 2x 0.90-0.96, and 2x
  doubles leakage (3B L12 0.62, gemma L3/L8 0.66-0.67). Same shape as 1.5B.
- Zero-shot country probe (for reference only) is even more state-entangled on gemma: full-vocab KL 3.2-3.9 for working
  edits vs 0.04-0.07 random. Confirms the switch to the few-shot P(US) probe was necessary.

### Phase 5 — Phase 2 on Qwen2.5-3B and gemma-2-2b (30 pairs; layers chosen from each model's Phase 1 curve)
Note: first launched with the 1.5B default layer list (2..22), which misses most of the 3B band (L8-30) and
overshoots gemma's handoff (L18). Killed and relaunched with per-model layers before any results were used.
Mean-diff at city_last, n=105 (3B) / 104 (gemma) held-out cities on fs1; random = norm-matched vector.
| model | L | capital flip | state -> target | binKL country (rand) | generic KL (rand) | 3rd-state -> target |
|---|---|---|---|---|---|---|
| 3B | 7 | 0.49 [0.39,0.58] | 0.59 | 0.58 (0.39) | 0.045 (0.044) | 0.23 |
| 3B | 12 | 0.86 [0.78,0.91] | 0.97 [0.92,0.99] | 0.10 (0.08) | 0.019 (0.030) | 0.32 |
| 3B | 20 | 0.84 [0.76,0.90] | 0.98 | 0.08 (0.02) | 0.004 (0.011) | 0.32 |
| 3B | 30 | 0.64 [0.54,0.72] | 0.94 | 0.03 (0.003) | 0.002 (0.001) | 0.14 |
| 3B | 31 (handoff) | 0.11 | 0.00 | 0.005 | 0.002 | 0.01 |
| gemma | 3 | 0.75 [0.66,0.82] | 0.79 | 0.13 (0.06) | 0.013 (0.016) | 0.22 |
| gemma | 8 | 0.91 [0.84,0.95] | 0.96 [0.91,0.98] | 0.034 (0.013) | 0.005 (0.006) | 0.19 |
| gemma | 12 | 0.90 [0.83,0.95] | 0.96 | 0.028 (0.006) | 0.003 (0.003) | 0.17 |
| gemma | 15 / 17 | 0.60 / 0.53 | 0.91 / 0.91 | 0.015 (0.005) | 0.002 | 0.04 / 0.03 |
| gemma | 18 (handoff) | 0.00 | 0.04 | 0.006 | 0.002 | 0.00 |
Random norm-matched: capital flip ≤0.01, state flip 0.00, 3rd-state -> target 0.00 at every layer, both models.
- **Meaning replicates: a state variable in all three models.** The state-question answer moves to the target
  at least as often as the capital does, at every layer (3B L20 0.98 vs 0.84; gemma L8 0.96 vs 0.91). Late in
  the band the gap widens (3B L30 0.94 vs 0.64; gemma L17 0.91 vs 0.53): the state variable is still set, but the
  capital readout depends on it less there.
- **Leakage is the main collateral damage in all three:** 3B 0.31-0.34 over L10-25 (higher than 1.5B's 0.16-0.24);
  gemma 0.17-0.24 over L3-12, but only 0.03-0.04 at L15-17 (with capital flip 0.53-0.60).
- **Country damage:** mid-band binKL is small but above random (3B 0.08-0.10 vs 0.02-0.08; gemma 0.03 vs 0.01).
  Early layers are fragile in 3B: at L3-5 even random vectors cut P(US) 0.88 -> 0.64-0.73 (binKL 0.3-0.5).
  The old zero-shot full-vocab country KL (KLczs) is 1.1-1.6 for 3B and 3.2-3.9 for gemma under a working edit,
  vs ≤0.2 random: gemma's zero-shot "country" prompt is even more state-entangled. This confirms the probe change.
- Generic-text KL is at the random level in both models.
- **Norm sweep replicates** (x0.25 / x0.5 / x2): capital flip 0.00-0.02 / 0.16-0.18 / 0.88-0.96; at x2 leakage
  rises to 0.47-0.67. As on 1.5B, overshooting the norm mainly buys leakage.

### Phase 4 (Qwen2.5-1.5B, bf16): difficulty ladder, 12 pairs, n=42 held-out source cities (fs1)
Held-out templates: ho_fs (few-shot), ho_zs (zero-shot). 3rd>tgt / 3rd kept: 2 other states' cities (n≈70), fs1.
binKLc = binary KL on P(US) (country probe). ndev "all" = half of the source state's valid cities (3-6).
| class | L | flip fs1 [CI] | ho_fs | ho_zs | state>tgt | 3rd>tgt | 3rd kept | binKLc | KLgen | abs(v) |
|---|---|---|---|---|---|---|---|---|---|---|
| C0 paste (1 city) | 4 / 8 / 15 | 0.86 / 1.00 / 1.00 | 0.69-0.72 | 0.92-0.95 | 0.98-1.00 | 0.94-1.00 | 0.00 | 0.22-0.24 | 0.05-0.17 | 51-62 |
| C1 mean-diff | 3 | 0.52 [0.38,0.67] | 0.14 | 0.41 | 0.52 | 0.06 | 0.81 | 0.43 | 0.029 | 37 |
| C1 | 4 | 0.69 [0.54,0.81] | 0.38 | 0.67 | 0.81 | 0.14 | 0.69 | 0.16 | 0.024 | 40 |
| C1 | 5 | 0.76 [0.61,0.87] | 0.55 | 0.77 | 0.90 | 0.17 | 0.76 | 0.11 | 0.028 | 41 |
| C1 | 8 | 0.88 [0.75,0.95] | 0.66 | 0.79 | 0.95 | 0.19 | 0.65 | 0.12 | 0.014 | 42 |
| C1 | 15 | 0.88 [0.75,0.95] | 0.72 | 0.82 | 0.93 | 0.24 | 0.67 | 0.08 | 0.003 | 39 |
| C1 ndev=1 | 4 / 5 / 8 | 0.40 / 0.55 / 0.81 | 0.21 / 0.34 / 0.76 | | | 0.18-0.28 | 0.47-0.57 | 0.10-0.24 | | 57-60 |
| C2 uncapped | 3-8 | 1.00 | 0.97-1.00 | 0.87-1.00 | 0.90 | **1.00** | 0.00 | 0.61-0.67 | 0.18-0.38 | 460-470 |
| C2kl (+generic KL) | 3-8 | 0.98-1.00 | 0.97-1.00 | 0.79-0.92 | 0.83-0.93 | **1.00** | 0.00 | 0.64-0.75 | 0.04-0.08 | 460-470 |
| C2n capped at abs(C1) | 3 / 5 / 8 | 0.95 / 0.98 / 1.00 | 0.90-1.00 | 0.79-0.90 | 0.69-0.83 | **0.83 / 0.76 / 0.56** | 0.11-0.33 | 0.20-0.31 | 0.04-0.11 | 37-42 |
| C2nkl capped+KL | 3 / 5 / 8 | 0.95 / 0.98 / 1.00 | 0.90-0.97 | 0.77-0.85 | 0.64-0.79 | 0.78 / 0.69 / 0.57 | 0.14-0.35 | 0.19-0.31 | 0.016-0.021 | 37-42 |
| C3 DAS rank-1 | 4 / 5 / 8 | 0.50 / 0.67 / 0.67 | 0.38-0.52 | 0.44-0.62 | 0.31-0.55 | **0.00-0.01** | 0.94-0.96 | 0.05-0.06 | 0.003 | — |
| C3 DAS rank-4 | 4 / 5 / 8 | 0.60 / 0.71 / 0.71 | 0.38-0.59 | 0.54-0.64 | 0.36-0.55 | 0.00-0.03 | 0.90-0.94 | 0.05-0.07 | 0.004 | — |
| C4 prompt ("Note that X is located in T.") | — | 0.05 | 0.24 | **0.97** | | | | | | |
| C4v prompt-derived vector | 3-21 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.86-1.00 | ≤0.1 | ≤0.006 | 7-40 |
Findings:
- **Leakage is the separating axis.** Every vector trained on "output the target capital" (C2, C2kl, and even norm-capped
  C2n/C2nkl) is largely an *answer* direction: 56-100% of other states' cities go to the target capital. The generic-KL
  penalty (C2kl) removes generic damage but not leakage or country damage. Mean-diff leaks 0.06-0.24; DAS ≈0.
- **No Phase 4 class gets high flip AND low leakage.** C2n: flip ~1.0, leak 0.56-0.83. DAS: leak 0, flip ≤0.71.
  C1: in between. So the "careful method succeeds" cell was missing -> Phase 4b (leak-aware gradient) added.
- **Layer ceiling ≤4-5 hurts mean-diff**, especially on the held-out few-shot template (ho_fs 0.38 at L4 vs 0.66-0.72
  at L8-15). **Few dev examples** hurt at low layers (ndev=1: 0.40 at L4) but not at L8 (0.81).
- **Prompting fails on few-shot templates** (0.05 fs1, 0.24 ho_fs) but works on the zero-shot held-out template (0.97).
  Its vector version (C4v: resid with context minus without) does nothing at the city token (0.00), because the context
  sentence comes *before* the city and the fs1 prompt's city token doesn't attend to it in the way the few-shot answer needs.
- Averaging over two training templates (C1t) gives no gain over C1. city_all ≈ city_last (+0.02-0.07 flip, more country KL).
- **Implication for tiers:** a leakage term in the reward is what makes the naive gradient approach fail. With
  leak_weight≈1, C2n at L8 scores about 1.0 × (1-0.6) ≈ 0.4; C1 at L8 ≈ 0.7 × 0.65 ≈ 0.45; DAS ≈ 0.55-0.65.
  The medium tier needs a method that beats all three; see Phase 4b.

### Ops note: queued jobs sat idle ~60 min
The Phase 7B and 4b launchers waited with `while pgrep -f phase4_ladder.py`. My own background waiter's command line
contained that string, so the launchers kept waiting until it was killed at its 2 h limit (Phase 4 had finished at
~06:10; jobs started 07:08). Use PID-based waits (`while kill -0 $PID`) from now on.
