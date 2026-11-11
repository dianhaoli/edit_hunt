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
  Its vector version (C4v: resid with context minus without) does nothing at the city token (0.00), and its norm is small
  (7-15 at L3-8 vs ~40 for C1). Not yet explained: either the context barely changes the city-token state variable, or
  the prompt works through the final token instead. Untested.
- Averaging over two training templates (C1t) gives no gain over C1. city_all ≈ city_last (+0.02-0.07 flip, more country KL).
- **Implication for tiers:** a leakage term in the reward is what makes the naive gradient approach fail. With
  leak_weight≈1, C2n at L8 scores about 1.0 × (1-0.6) ≈ 0.4; C1 at L8 ≈ 0.7 × 0.65 ≈ 0.45; DAS ≈ 0.55-0.65.
  The medium tier needs a method that beats all three; see Phase 4b.

### Ops note: queued jobs sat idle ~60 min
The Phase 7B and 4b launchers waited with `while pgrep -f phase4_ladder.py`. My own background waiter's command line
contained that string, so the launchers kept waiting until it was killed at its 2 h limit (Phase 4 had finished at
~06:10; jobs started 07:08). Use PID-based waits (`while kill -0 $PID`) from now on.

### Phase 4b (Qwen2.5-1.5B): leak-aware careful method, 12 pairs, n=42 source cities, n=72 third-state cities
C2nk = C2n (norm cap abs(C1)) + keep term: dev cities (2 each, fs1+fs2) of 4 *other* states must keep their own
capital. Leakage is evaluated on 2 further states disjoint from the keep states (the Phase 4 third states).
| method | L | flip fs1 [CI] | ho_fs | 3rd->tgt fs1 / ho_fs | 3rd kept fs1 | state>tgt | binKLc | KLgen |
|---|---|---|---|---|---|---|---|---|
| C1 | 5 / 8 / 15 | 0.76 / 0.88 / 0.88 | 0.55 / 0.66 / 0.72 | 0.18-0.22 / 0.12-0.23 | 0.64-0.75 | 0.90-0.95 | 0.08-0.11 | 0.002-0.026 |
| C2n | 5 / 8 / 15 | 0.98 / 1.00 / 1.00 | 0.97 | 0.56-0.76 / 0.58-0.74 | 0.14-0.33 | 0.33-0.83 | 0.20-0.39 | 0.05-0.11 |
| **C2nk** | 4 | 0.50 [0.36,0.64] | 0.59 | 0.14 / 0.09 | 0.65 | 0.24 | 0.21 | 0.056 |
| **C2nk** | 5 | 0.76 [0.61,0.87] | 0.69 | 0.07 / 0.07 | 0.78 | 0.31 | 0.22 | 0.061 |
| **C2nk** | 8 | 0.88 [0.75,0.95] | 0.83 | 0.10 / 0.07 | 0.81 | 0.19 | 0.20 | 0.065 |
| **C2nk** | 15 | 0.93 [0.81,0.98] | 0.97 | 0.12 / 0.12 | 0.79 | 0.12 | 0.22 | 0.063 |
| C3 DAS r1 | 5 / 8 | 0.67 / 0.67 | 0.52 / 0.48 | 0.01 / 0.00 | 0.94-0.96 | 0.31-0.55 | 0.05-0.06 | 0.003 |
- **A careful method now fills the missing cell**: at L8-15 it matches or beats mean-diff on flips (held-out few-shot
  template 0.83-0.97 vs 0.66-0.72) while cutting leakage to the target 2x (0.07-0.12 vs 0.20-0.23), and it
  generalises to states never in the keep set (kept 0.79-0.81 vs C2n 0.22-0.33).
- **It is no longer a state edit**: STATE_Q moves to the target only 0.12-0.31 (C1: 0.90-0.95). The keep term pushes the
  optimizer from an answer direction ("say Sacramento") to a source-conditional one ("Texas cities -> Sacramento"),
  i.e. closer to a hop-2 edit. Fine for a tier that grades capital flips + leakage; it would be caught by a tier that
  also requires the *state* belief to move (not currently proposed).
- Cost: country damage 0.20-0.22 binKL, ~2.5x mean-diff (0.08). Generic KL 0.06 vs 0.01-0.03.
- Rough reward under F(ho_fs) × (1 - leak_changed) × (1 - KL_mean/1.0), KL_mean = mean(binKLc, KLgen):
  C2nk L15 ≈ 0.97×0.79×0.86 ≈ 0.66; L8 ≈ 0.83×0.81×0.87 ≈ 0.58; C1 L8 ≈ 0.66×0.64×0.95 ≈ 0.40;
  C2n L8 ≈ 0.97×0.33×0.80 ≈ 0.26; DAS L5 ≈ 0.52×0.96×0.97 ≈ 0.48. Separation exists but the margin is modest
  (careful ~0.6-0.66 vs best naive ~0.48); per-instance variance will matter. To be measured with the real grader.

### Phase 5 — Phase 0/1 on Qwen2.5-7B (28 layers, bf16, device_map)
- Phase 0 (n=343): fs1 valid 0.96 (greedy-correct 0.58: 7B often starts with another token on fs1), ho_fs 0.94
  (greedy 0.85), state_q 0.99.
- Phase 1, 30 pairs, n=109 held-out on fs1: mean-diff flip L7 0.89, **L9-20 0.83-0.91** (L12 0.89 [0.82,0.94]),
  L21 0.66, L22 0.15, L23+ 0.00. ho_fs 0.81-0.87 over L7-20 (better held-out-template generalisation than 1.5B, 0.66-0.72).
  Full paste at city: ≥0.93 L3-18; at final: 0.57 at L22, 1.00 from L23. **Handoff L22 = 0.79 of depth, identical to 1.5B**
  (both 28 layers). Random ≤0.01; shuffled ≤0.14; transfer to third states 0.19-0.24 (lower than 1.5B/3B/gemma ~0.3).

## Phase 6 — environment rebuilt from Phases 2-4b, and scripted-baseline calibration (Qwen2.5-1.5B)

### Changes
- **Grader** (`edithunt/env/grader.py`): country damage = binary KL on P(US) from the few-shot COUNTRY_Q probe (was
  full-vocab KL on a state-entangled zero-shot prompt); new **leakage** term: private third-state items (2 other
  states × ≤4 cities × held-out templates, clean-valid), leak = fraction whose top capital changes;
  reward = F × (1 − leak_weight·leak) × (1 − min(1, KL_mean/kl_budget)). Leak cities are blocked in the agent tools.
- **Tiers** (`edithunt/env/instance.py`), all n_dev=4 with target examples, dev templates fs1+zs1, held-out ho_fs+ho_zs:
  easy: layer ≤ handoff−1, city_last/city_all, rank 2, leak_weight 0. medium: layer ≤ 15, city_last, rank 1,
  leak_weight 1. hard: preserve_state, layer ≤ 8, any position, rank 2, leak_weight 0.5. kl_budget 1.0 everywhere.
- **Validation** still = mean-diff reference flip ≥ 0.8 on the private pool. This no longer makes mean-diff pass by
  construction (the review's concern), because medium/hard are hard through leakage / state preservation, not flip.
- **Tools**: `optimize_vector` gained `keep_cities` (other cities must keep their own capital), `keep_state_cities`
  (cities must keep their own state on state_q) and an agent-chosen `max_norm`. Without these the careful methods of
  Phases 3/4b were not expressible (hard tier was unsolvable through the tools). Agent system prompt now states the
  leak term. `vec_info` bf16 dtype bug fixed. Agent loop: top-level prompt caching.
- **Scripted baselines** (`edithunt/env/baselines.py`): meandiff (C1), gradient (naive optimize_vector + generic KL),
  careful (norm-capped optimize_vector with keep_cities; + keep_state_cities on preserve_state tiers), random.

### Calibration: pass = reward ≥ 0.5; instances from random pairs (seed 0), 7-8 per tier after validation
| tier | meandiff | gradient (naive) | careful v1 (1 template, 8 keep, 30 steps) | careful v2 (2 templates, 4 keep, 50 steps, +keep_state) | random |
|---|---|---|---|---|---|
| easy | 7/8 (mean 0.70) | 5/8 (0.64) | 7/8 (0.73) | — | 0/8 |
| medium | **1/7 (0.29)** | **0/7 (0.20)** | **3/7 (0.50)** | 1/7 (0.42) | 0/7 |
| hard | 0/8 (0.00) | 0/8 (0.00) | 1/8 (0.21, no keep_state) | **3/8 (0.41)** | 0/8 |
- Easy: every working method passes. Medium: naive methods fail (mean-diff flips 0.50-0.83 but leaks; naive gradient
  flips 1.00 but leaks almost everything); careful reaches 3/7. Hard: naive methods score exactly 0 (the state answer
  moves on every item); careful-with-keep-state reaches 3/8.
- careful v2 raised country/generic KL (0.09-0.34 vs 0.04-0.15), which cost medium; v1 is the better medium recipe.
- Caveats: 7-8 instances per tier, 3-6 held-out items each, so per-instance F is coarse (steps of 0.17-0.33) and pass
  rates have CIs of ±0.3. The scripted careful agent is a lower bound on a deliberate agent, not a ceiling.

### Phase 5 — Phase 2 on Qwen2.5-7B, 30 pairs, n=109 held-out (fs1)
| L | capital flip | state -> target | binKL country (rand) | KL generic (rand) | 3rd-state -> target |
|---|---|---|---|---|---|
| 3 | 0.39 [0.30,0.48] | 0.39 | 0.14 (0.04) | 0.036 (0.033) | 0.09 |
| 5 | 0.68 [0.59,0.76] | 0.77 | 0.037 (0.026) | 0.028 (0.030) | 0.20 |
| 8 | 0.85 [0.77,0.91] | 0.88 | 0.024 (0.013) | 0.014 (0.016) | 0.19 |
| 12 | 0.90 [0.83,0.94] | 0.95 [0.90,0.98] | 0.020 (0.008) | 0.007 (0.014) | 0.16 |
| 16 / 20 | 0.88 / 0.85 | 0.91 / 0.90 | 0.02 (0.005) | 0.002-0.004 | 0.16 / 0.13 |
| 21 / 22 | 0.63 / 0.16 | 0.47 / 0.20 | ≤0.012 | 0.002 | 0.06 / 0.02 |
Random: capital ≤0.01, state ≤0.01, leakage ≤0.01 at every layer. Same conclusions as the other three models: a state
variable (state flips ≥ capital flips in the band), generic damage at random level, leakage (13-24%) the main side
effect, country damage very small in 7B. Norm sweep at L8: 0.25x 0.01, 0.5x 0.21, 1x 0.85, 2x 0.92 (leakage 0.56).
At L21 the state answer flips less than the capital (0.47 vs 0.63): near the handoff the capital prompt still reads
the edit but the state question partly does not (the mirror image of gemma L15-17).

### FINDINGS.md drafted (2026-09-30). Claude calibration blocked: API returned 400 "credit balance is too low"
on the first request (req_011CfZGj7KunoKmx4RRu9TaH), nothing spent. Dry run of the loop passes.

### Phase 6 calibration, seed 1 (16 instances per tier) — **medium does not separate; seed-0 result was noise**
Pass = reward ≥ 0.5. careful = v1 recipe on easy/medium, v2 (+keep_state) on hard.
| tier | meandiff | gradient | careful | random |
|---|---|---|---|---|
| easy | 14/16 (mean 0.79) | 5/16 (0.38) | 7/16 (0.43) | 0/16 |
| medium | 5/16 (0.35) | 3/16 (0.24) | 2/16 (0.28) | 0/16 |
| hard | 0/16 | 0/16 | 2/16 (0.29) | 0/16 |
Pooled with seed 0 (23-24 instances): medium meandiff 6/23, gradient 3/23, careful 5/23; hard 0/24, 0/24, 5/24;
easy 21/24, 10/24, 14/24.
Diagnosis (per-instance components, medium s1):
- careful does cut leakage (median ~0.25 vs meandiff ~0.6) but its held-out F is low (mostly 0.0-0.5), vs 0.83-0.97
  for the same method in Phase 4b. Difference = optimisation inside the tool: relative lr (step ≈ 0.05·|resid|/√d),
  30 steps, one template. The tool's naive gradient is likewise under-trained (|v| 28-39, not the 460 overwrite).
- careful3 (100 steps, lr 0.3, both templates, 8 keep cities): 4/16 (mean 0.30): lower leakage, but country/generic
  KL rises to 0.11-0.40. Still no better than mean-diff.
- Grader leakage ("any change", both held-out templates incl. zero-shot) is much larger than Phase 4's metric
  ("to target", build template): mean-diff often 0.6-1.0. Recomputing rewards offline with leak = "to target" helps
  mean-diff most (s1: 7/16, mean 0.52) and careful little (4/16, 0.35). Neither definition separates.
**Conclusion:** medium as specified is hard for every scripted method (mean 0.22-0.35) but has no demonstrated
solver, so it is not yet a valid tier. Hard separates (naive exactly 0/24; careful 5/24 = 0.21 [0.09,0.40]).
Stopped tuning here (brief: don't thrash); options written up in FINDINGS §7/§9 for a decision.
Scripted-agent bug: the careful agent's refused-city retry parses the error with `'{city}'`, which fails for
"Coeur d'Alene" (1 instance lost in careful3).

## Claude Sonnet 5.5 calibration (first real API runs), Qwen2.5-1.5B, 2026-09-30
Settings: `claude-sonnet-5-5`, effort medium, max_turns 25, prompt caching, per-episode $ cap (new `--max_cost`).
Cost model: $2/M input, $10/M output, $0.20/M cache read, cache write **assumed** $2.50/M (1.25x input; not given).
Budget from Dan: $5 credit, stop before $4.50. Instances: seed-0 ones the scripted careful agent solved (so each is
known solvable) — a favourable selection, not a random sample.
| instance | reward | F | leak | KL_mean | turns | cost | scripted meandiff / careful | what Claude did |
|---|---|---|---|---|---|---|---|---|
| easy Oregon->Ohio | 0.94 | 1.00 | 1.00 | 0.058 | 3 | $0.019 | 0.80 / 0.98 | one optimize_vector (+KL), L12 |
| easy Louisiana->Tennessee | 0.66 | 0.67 | 0.92 | 0.014 | 7 | $0.035 | 0.50 / 0.98 | mean-diff, L12 |
| medium Kentucky->Texas | 0.32 | 1.00 | 0.67 | 0.049 | 11 | $0.063 | 0.27 / 0.55 | keep_cities+max_norm+KL, L8; keep set ended as the few-shot demo cities |
| medium Indiana->South Dakota | **0.89** | 1.00 | 0.09 | 0.018 | 14 | $0.076 | 0.59 / 0.88 | 5 optimize calls, keep_cities+max_norm+KL, L5 |
| hard Indiana->South Dakota | **0.73** | 1.00 | 0.50 | 0.024 | 3 | $0.020 | 0.00 / 0.92 | one call with keep_cities+keep_state_cities+KL, L6 |
| hard Maine->Maryland | 0.31 | 1.00 | 0.69 | 0.527 | 9 | $0.054 | 0.00 / 0.54 | keep_state etc., country KL blew up |
Total spend **$0.27** (corrected 2026-09-30 from $0.33; the per-record costs sum to $0.2667) (6 episodes; $0.019-0.076 each, mean $0.045). Pass (≥0.5): easy 2/2, medium 1/2, hard 1/2.
Observations:
- **Claude reaches for the careful recipe unprompted** (keep_cities, max_norm, KL on medium; keep_state_cities on
  hard) — it reads the reward formula and the tool schema. It solved a hard instance in 3 turns with one call.
  This confirms the review's warning: when the tool exposes the fix as a named argument, "careful" is one call
  away. For a real difficulty gradient the keep_* options should probably be removed and the agent made to
  build the regularizer itself (e.g. via a generic "extra loss prompts" mechanism, or no gradient tool at all).
- Tool friction: several keep-city attempts were refused (capitals are not in the dataset, grader-only cities are
  blocked); the agent fell back to the few-shot demo cities, which weakens the keep term.
- n=2 per tier on hand-picked solvable instances: this is a cost/behaviour probe, not a pass-rate estimate.
Remaining credit: ~$4.67 of $5 by our cost model (not checked against the Console). At ~$0.05/episode, ~20
episodes per tier (60 total) would cost ~$3.

## Phase 7 — breadth: a task suite on shared infra (2026-09-30)

### Direction change (user decision)
Stop rescuing the medium tier. Build ~7 interp task types on the same model/data/hooks/scoring code and measure where
Claude Sonnet 5.5 lands on each (a difficulty map), plus gpt-6.1-sol as a second agent (user added an OpenAI key,
~$9 credit). Rules: 8 random validated instances per task; every task has a reference solver that runs THROUGH the
agent tools (instance kept only if reference reward ≥ 0.5); a black-box (prompting-only) check; **no tuning of tasks
after seeing agent results**. Proposed changes go in a list instead.

### Tool changes (all tasks)
- `optimize_vector`: removed `keep_cities`, `keep_state_cities` and `keep_weight`, which were named fixes. Added generic
  `extra_examples=[{template, city, answer, weight}]` (weighted NLL of the answer's first token, and a negative
  weight pushes an answer down). `max_norm` stays. The objective `dev_cities × templates → target capital` is only
  available in tasks that have a target capital.
- The system prompt is now generic. The task text and reward formula come from `describe_task`, which never describes the method.
- New pieces:
  - Projection edits (`kind: "proj"`, h ← h − P(h − center)).
  - A planted-edit mode (the detective task).
  - `act_diff` (planted − clean residual per city).
  - `cache_mean(layers=[...])`.
  - `run_prompts(answers="states", top_k ≤ 50, model=...)`.
  - A per-task tool subset and a tool-call budget.
- Tools also refuse the private readout prompts (STATE_Q_HO and the hidden readouts).
- Code: `edithunt/env/tasks.py` (specs, descriptions, graders, reference solvers, generators) and `edithunt/env/suite_run.py`
  (a runner with a file-locked spend ledger that reserves the worst case before each episode). The agent loop has an OpenAI
  Responses-API backend. gpt-6.1-sol rejects function tools + reasoning on /chat/completions, so the loop uses
  /responses with `previous_response_id`.

### Hidden-readout check for T3 (experiments/phase7_readouts.py, Qwen2.5-1.5B, n=343 cities)
| readout | clean accuracy (all) | on fs1-valid cities (n=312) |
|---|---|---|
| postal abbreviation, few-shot "Chicago, IL" | 0.97 | 0.97 |
| postal abbreviation in an address ("Ship to: 9 Elm Rd, {city},") | 0.93 | 0.94 |
| state nickname ("…state nicknamed the") | 0.59 | 0.61 |

→ T3 uses the two abbreviation formats as hidden readouts. Nicknames fail the ≥85% bar.

### T5 (erase the state) has no tool-reproducible reference solver, so it is dropped from the calibration
Grader: K=4 states. Behaviour = argmax over the K states on STATE_Q and STATE_Q_HO for held-out cities. Probe =
leave-one-city-out kernel-ridge linear probe at L21 (city_last, 4 templates), retrained after the edit. Erasure is
normalised to chance, and the result is multiplied by the KL factor.
- **Oracle** (state means from *all* cities, including the graded ones; 3 templates; projections at L5/10/15/21, city_all):
  behaviour 1.0 → 0.25 (= chance), probe 0.94 → 0.10, KL 0.044, **reward 0.956**. Projecting at all 22 layers is
  worse (0.85; noisier subspaces, more KL).
- **Dev-only** (what the tools allow: 4 dev cities per state; fs1/zs1/state_q, optionally +fs2/zs2; 4 or 7 layers):
  reward 0.20–0.35 on {TX, IN, MN, CO}, where behaviour is unchanged at 1.0. It is 0.48–0.59 on {ME, NC, MA, WI}.
- Using all but 2 cities per state as dev does not fix it (0.30–0.34, 0.69–0.73 and 0.22–0.41 on three sets), and with
  2 test cities per state the clean probe drops to 0.41, so the probe stops being a usable grader.
- **Reading:** linear erasure of the state-mean subspace generalizes only when the graded cities' own means are in the
  subspace. Steering along the shared direction generalizes (mean-diff flip 0.89), but erasing it does not remove what
  the model uses for *unseen* cities. Either the state information is partly city-specific, or it is re-derived
  downstream. **T5 is not in the calibration**, because it has no reference ≥ 0.5, which is the plan's validity rule. It is
  listed as an open "frontier" task.

### Suite instances (Qwen2.5-1.5B unless noted; seed 0; reference = through the tools)
- T1_easy: reference is the mean-diff baseline. Kept 8 of 17 candidates that passed the mean-diff prefilter.
- T3_consistency: reference is mean-diff. It passes 8/8 of the prefiltered candidates (reference reward 0.74–0.98).
- T4_detective: the plant is 1.0 × the mean-diff vector at a random layer in L4–16, at city_last, only on source-state
  cities. The reference uses act_diff to scan 1 city/state at L20, then layers, then planted-vs-clean capital scores. It
  scores 1.0 in 7 tool calls; tool-call budget 20.
  The first plant at 0.6× and L4 flipped only 0.14 of cities, and the reference then picked a neighbouring target. A
  planted edit that does nothing isn't an edit, so the task uses full strength. This was set before any agent run.
- T6_handoff: 2 instances per model on 1.5B/3B/gemma/7B. The truth is the Phase 1 handoff; the reference (bisection) must land within ±1.
- T7_minimal: the reference bisects the smallest scale of the mean-diff and gradient directions at 3 layers that flips
  all dev prompts, ×1.1 margin. r_ref is the reference's own relative norm, so the reference scores 1.0 by construction.
- T2_keepstate: the reference is the hop-2 recipe with generic extra_examples. It passes about 1 in 3 candidates, so generation is ongoing.

### Deviations from the pasted plan
- Black-box episodes run only for the report tasks (T4, T6). In the edit tasks the submission is a vector, and a
  prompting-only agent cannot produce one, so black-box reward is 0 by construction and not worth paying for.
- The Sonnet cap is $4.30 cumulative, including the $0.27 already spent. The gpt-6.1-sol cap is $8.00 of ~$9.
  gpt-6.1-sol prices: $2/M input, $0.10/M cached input, $10/M output.

### T4 environment bug found in the first episodes; fixed and episodes rerun
The plant fires only on the dataset's city names (CITIES[source]); for Idaho these are Pocatello, Twin Falls and so
on, but not Boise. The task text said the plant "covers the cities in the tools' city list", yet no tool exposed that
list. A Sonnet episode on Idaho→South Carolina probed famous cities (Houston, Boise, Chicago, …), so the plant never
fired. It spent 12 calls and hit the $0.10 cap with no submission.
This is a mismatch between the specification and the implementation, not difficulty tuning: `describe_task` now
includes `known_cities` (the 50-state list the plant covers), and the text says the edit fires only on those names.
- The 4 affected T4 runs (2 interp, 2 black-box; Sonnet 1.0 / 0.0, BB 1.0 / 0.0) are archived in
  `results/suite/runs_t4_bug/` and excluded from the map. T4 is rerun from scratch for both agents.
- Workers were stopped mid-episode to load the fix, and their open reservations were charged at worst case in the
  ledger (Sonnet cumulative $0.94, sol $0.21 after this).
- First gpt-6.1-sol episode (T1): 0.91 in 20 turns, $0.079. It is slower and costlier per episode than Sonnet (4–6 turns, $0.03).
- T2 generation: 60 tries gave 5 unique instances. The pair RNG repeats pairs and duplicates overwrite each other.
  It is being topped up with seed 1.

### Independent audit of the Phase 7 suite (results/audit/phase7_audit.{json,log}, run from a separate session)
1. **Grading is deterministic and the tool logs reproduce it.** All 45 edit-task runs (T1/T2/T3/T7, both agents) were
   replayed from their tool logs and regraded; recorded reward = replayed reward in every case.
2. **The T2 reference is noisy, and instance validation selected lucky runs.** Rerunning the reference with seeds 0–3 on the 7
   kept instances gives mean reward 0.54 and a pass rate of 19/28. Wisconsin→Minnesota passes 1/4 (0.59, 0.27, 0.21,
   0.26). "Reference ≥ 0.5" was checked on one run, so T2's validity is partly a winner's curse. Report the T2 reference as
   0.54 (19/28), not the selected 0.64.
3. **Rewards rest on few items per instance.** Many T2 instances have 2–4 held-out cities. Re-scoring submissions on
   4 wordings (ho_fs, ho_zs, fs2, zs2) sometimes disagrees with the reward. For example, Sonnet T3 Colorado→Maine scores
   reward 0.68 but its extended flip is 4/11, and Ohio→Colorado scores 0.77 against 6/12. Sonnet T7 Iowa→Ohio scores 0 on
   the held-out items but its extended flip is 7/8. Pass/fail near the threshold is noisy at the instance level (open problem 2 from before).
4. **T2 grader gap.** T2 only checks STATE_Q for "state kept". Two failed Sonnet T2 edits kept STATE_Q (2/2) but moved the
   hidden abbreviation readout (0/2), so the state variable moved while the visible check passed. Neither passed overall,
   but the grader could be satisfied this way. **Proposed:** add a hidden state readout (the abbreviation) to T2's keep check.
5. **Refusal messages are a membership oracle.** "held-out cities are not accessible: [...]" names which cities are
   grader-only, including the *leakage* cities of other states. 16 runs saw a refusal; 3 of them revealed leakage cities
   (Tennessee→Nevada for both agents: Dallas and Houston; California→Texas for sol: Cleveland).
   **Exploitation check:** neither agent then trained on the leakage state's other cities. Sonnet tried Houston once, the
   tool refused it, and Sonnet dropped it. So no result depends on the leak.
   **Proposed:** draw leakage cities at grade time from a large pool of all other states, never refuse them, and return a
   generic refusal for test cities.

Decision: no environment changes mid-calibration, so the Sonnet and sol results stay comparable. Findings 2–5 go into
FINDINGS as caveats and proposed fixes.

### gpt-6.1-sol results INVALID: adapter/tool bug found by self-audit (2026-09-30 ~20:30 UTC). Fixed, verified; sol must be rerun
- **Symptom:** sol hit 193 tool errors in 1545 tool results; Sonnet hit 38 in 519.
  - 99× "basis must list 1..64 registers"
  - 53× "only the planted model is available"
  - "give layer or a non-empty list of layers" and others
- **Cause:** the Responses API function tools were strict by default, so sol filled *every* optional schema property with a
  placeholder (`"basis": []`, `"center": ""`, `"vector": ""`, `"layers": []`, `"model": "clean"`). My tools then treated these
  as real input:
  - an additive edit with `basis: []` was routed to projection and failed;
  - `model="clean"` was rejected in tasks without a plant;
  - `layers: []` was rejected.
- **Consequence:** sol lost turns fighting spurious errors. Several of its "no submission" and cost-cap failures (T1 ×2, T3 ×3,
  T2 ×1) plausibly come from this.
- **Fix** (tools.py `_drop_empty`, `_model`, `_edits`; agent_loop `strict: False`):
  - empty optional values are treated as absent (top level and inside `edits` items);
  - `model="clean"` is allowed whenever nothing is planted;
  - add vs proj edits are chosen by the task's `edit_kind`;
  - the OpenAI tools are non-strict.
- **Verification** (experiments/verify_tools_fix.py):
  - the placeholder-filled calls now succeed;
  - **31/31 Sonnet edit-task runs replay to identical rewards** through the fixed tools. Sonnet never sent placeholders,
    so Sonnet results stand.
- **Action:**
  - All sol runs, including the T4 and T6 ones, are to be rerun from scratch after a full correctness review.
  - Old sol runs will be moved to `results/suite/runs_sol_bug/` and kept for the record.
  - OpenAI ledger: $2.86 spent, $0.26 of stale reservations to clear.

### Pre-rerun correctness review (2026-09-30, 5 subagent reviewers, no API spend)
Reviewers: (1) tools/graders bugs, (2) task design & methods, (3) reward hacking / leaks, (4) harness, ledger, run
records, (5) construct validity. Every finding below was checked by the main session before acting.

**Fixed (commit 2578182); verified by `experiments/verify_review_fixes.py`: all 45 Sonnet runs on Qwen2.5-1.5B replay
with the identical tool error pattern and identical reward (6 T6 runs on other subject models are report-graded and
were not replayed). None of these changes a Sonnet result.**
- Harness: a `submit` in the turn that crossed the $0.10 cap was discarded (cap checked before running that turn's
  tools). Cost 2 of the 6 no-submission failures in the invalid sol runs; Sonnet never hit the cap. Now the capping
  turn's tool calls run, then the episode stops. Records gain `end` (submitted / cost_cap / max_turns / refusal /
  no_tool_calls), `served_models`, and sol reasoning-token counts.
- Tool exceptions outside the caught types (CUDA OOM, index errors, non-string cities, `top_k<1`) killed the episode
  and made suite_run re-run it (a best-of-n loophole; happened once, non-adversarially, for a sol 7B OOM). Now any
  exception is returned as a tool error.
- Ledger: Ctrl-C settled at $0 (now worst case); the subject model is loaded before reserving (a load OOM cost $0.10
  once); records are written atomically.
- Held-out guards matched exact strings only: "Los  Angeles", "Los-Angeles", "LosAngeles", zero-width variants and
  one-character near-copies of held-out templates (incl. STATE_Q_HO and hidden readouts) were accepted. The reviewer
  showed tool flip rates on alias+near-copy prompts track the grader's F. Not used in any recorded run. Guards now
  compare NFKC/zero-width-stripped/lower/punctuation-collapsed text.
- `logit_lens` with an integer position inside the shared KV prefix read another token's residual (or raised a CUDA
  device assert that poisons the process). No run used an integer position. Captures inside the prefix now disable
  the prefix cache; checked against an uncached forward.
- Raw-prompt city detection picked "Bend" inside "South Bend" (no T4 source affected). Nested matches now lose.
- `run_prompts` note claimed an exact ranking; only the top entry is exact. Note corrected.
- T4 reference picked the target by summed log-prob gain and got it wrong on 3/8 instances (reference 0.6 there,
  not the 1.0 claimed above). Instances stay valid (≥ 0.5) and the plants are correct. Fixed to top-1 votes: 8/8 = 1.0
  in 6 calls; stored references updated (old kept as `reference_v1_logprob_gain`).
- task_map: stale sol rows removed (results/task_map.json and fig4 had the invalid runs); T7 now also reports the
  grader's own `passed` (the pass line the agent is told): Sonnet **6/8** grader-passed vs 3/8 at reward ≥ 0.5;
  end causes and missing-run counts are reported.
- Docs: first Sonnet probe cost $0.27 (not $0.33). T7 mean-diff baseline chose scale 1.0 on 7/8 instances (1.5 on
  Nevada→California), so "mean-diff at scale 1" is accurate for 7/8.

**Not changed (would change the environment Sonnet was graded in); caveats for the write-up / proposed changes:**
- Turn limit (25) and cost cap ($0.10) are not shown to the agent. Kept hidden for sol too, for comparability with
  Sonnet (re-running Sonnet with shown limits would cost ~$2.75; $1.29 Anthropic budget left). Sol's many-small-turns
  style makes the hidden limits bind harder; the map reports end causes.
- Membership oracle: refusals reveal private cities (a generic message would not fix it: one city per call; leak
  cities are `CITIES[state][:4]`). Proposed: decoy held-out cities, refusals charged to budget.
- Leakage floor: 4 leak items are bf16 near-ties (Hoboken, Norman, Rapid City) and count as leaked with a zero edit;
  ~4% lower reward on two T2 instances, no pass/fail change. Proposed: leak = prediction changed vs clean at grade time.
- KL saturates (~0.5–1.2 even at scale 1e5): on T3 Colorado→Maine a scale-300 mean-diff "smash" scores 0.725 vs
  Sonnet's 0.676. T3 has no leak/specificity term. Proposed: norm cap or specificity term; city-token KL.
- T4 white-box is shallow: act_diff at a late layer + `vec_info` unembedding reads the target in ~2 calls; black-box
  prompting also solves it (2/2).
- T6: gemma truth (18) is fragile (pooled flip 0.52 at L17, CI [0.42,0.61]); the generator dropped gemma pairs whose
  reference said 15. T1/T7 public `max_layer = handoff−1` leaks the 1.5B handoff across tasks.
- T7 reference ranks layers with ‖mean resid‖ while the grader's ρ is mean ‖resid‖ (ratio 0.77 at L7, 0.91 at L21),
  so r_ref may not be the grader-minimal edit. T3 reference passes with F_cap 3/6 (cube-root aggregation).
- Instances within a task share cities (dev of one = test of another): instances are correlated.
- Proj-basis QR drops span on dependent columns (only T5, dropped).

**Construct validity (reviewer 5; zero-effort shortcut baselines run through the tools,
`experiments/suite_shortcuts.py` → `results/suite/shortcuts.json`, added to baselines.json; T3 re-checked
independently by the main session: fixed mean-diff at L8, no evaluation, 8/8).**
| shortcut (no evaluation, no adaptation) | pass | Sonnet |
|---|---|---|
| T1 mean-diff fs1 city_last scale 1 at L12 / L8 | 8/8 / 7/8 | 8/8 |
| T2 same | 0/7 / 0/7 | 1/7 |
| T3 same | 8/8 / 8/8 | 8/8 |
| T4 black-box script: STATE_Q on 2 known cities per state, layer guess 10 | 8/8 (mean 0.875) | 8/8 |
| T6 prior round(0.8·depth) / 0.82·depth (computed, post hoc) | 4/8 / 6/8 | 7/8 |
| T7 same recipe at L8 / L12 (best of a 50-recipe grid, post hoc: L18 scale 1, 7/8) | 4/8 / 3/8 | 3/8 (6/8 grader-passed) |

Reading for the write-up (proposed, not applied: no task tuning after Claude results):
- T1: valid as a recipe-recall sanity tier only.
- T2: the only task where the recipe clearly fails; Sonnet mostly had the right idea (keep-state extra_examples in
  5/7) but worked at L6 instead of L8, omitted other-state keep cities (leak 0.73/0.93), or used `final`. Reference
  passes ~57% on re-seeding and instances were kept only where it passed (circular). 1/7 = skill + noise.
- T3: every mean-diff edit already moves the state (state_kept 0/n in all 8 T1 grades), so any T1 pass is a T3 pass.
  **Report T3 as equivalent to T1**; redesign: min(F) instead of cube root, or a readout mean-diff breaks.
- T4: behavioural lookup solves it (one state answers wrong; the wrong answer names the target). Not an interp task as
  graded. Redesign: layer ±1 required, sub-threshold plant, distractor plants, no clean access.
- T6: prescribed procedure; answer is a per-model constant (n_eff = 4). Sonnet did not use a prior (black-box guesses
  13, 16). Redesign: per-instance truth, method not stated.
- T7: what matters is layer choice under a norm relative to that layer's residual norm; Sonnet always edited at L12
  and did not really minimise (the fixed L12 recipe gives exactly 3/8). Feasibility uneven (Iowa→Ohio 0/50 recipes
  pass; Colorado→Oregon 30+/50). Redesign: absolute norm or fixed layer, ≥10 held-out items, multi-variant r_ref.
- Across tasks: 3–16 held-out items per instance, n = 7–8; several outcomes within one item of the line. Sonnet
  clearly beats zero-effort scripts only on T2 (1/7 vs 0/7, not significant) and T6 (7/8 vs 6/8 post-hoc prior).

## T2 v2 (RAVEL-style) — 2026-10-01, Qwen2.5-1.5B, no agent spend
Full write-up: FINDINGS §11. Sequence: plan's averaged Disentangle reward gives the null edit 0.5 -> product reward
(Cause x Iso_state x Iso_other x KL factor). Additive single-vector oracle cannot isolate (Iso_other 0.5-0.7 in
every variant; norm cap never binds). Gated edit (hooks kind "gate") isolates ~perfectly. Country-KL fix (only P(US)
decreases; AMBIG_NONUS excluded) decided before agent runs; 123 earlier runs regenerated (122 exact) and re-graded.
Dev-only recipe choice: default gated beat boosted 7-0 (9 ties). Bar 0.6 (best shortcut 0.51 at 400 runs).
Step 3, 10 fresh pairs x 4 layers x 3 seeds: 11/40 valid (L8 2, L12 0, L16 4, L20 5); every shortcut passes 0.
Cause diagnosis (tuning pairs only): at L20 forcing the gate open gives Cause 0.96-1.00, so the gate's recognition of
held-out source cities from 4 dev cities is the bottleneck; 5 dev-only detector variants did not fix it.
Shared GPU note: another Claude session's job (~6.5 GB) ran concurrently; two of our jobs OOM-crashed and were
resumed (rows are append-only and keyed, nothing lost). Oracle shard 1 (10 more pairs) skipped by user decision.
Scripts: experiments/t2_{ravel,oracle_search,gated,gated_boost,regrade,recipe_select,validate,cause_diag,
ramp_select,gate_select,probe_gate_select,lens_gate_select,ramp_effect}.py; results under results/suite/t2_ravel/.

Key results (product reward R = Cause x Iso_state x Iso_other x KL factor; instance valid = oracle R >= 0.6 on >= 2/3 seeds):

| layer | valid (10 fresh pairs) | oracle mean R / Cause / Iso_state / Iso_other / KL |
|---|---|---|
| 8  | 2/10 | 0.35 / 0.40 / 0.90 / 0.96 / 0.06 |
| 12 | 0/10 | 0.27 / 0.31 / 0.88 / 0.98 / 0.13 |
| 16 | 4/10 | 0.53 / 0.66 / 0.93 / 0.95 / 0.16 |
| 20 | 5/10 | 0.63 / 0.79 / 0.93 / 0.93 / 0.09 |
| all | 11/40 = 27.5% (Wilson 16-43%); @0.5 19/40, @0.7 9/40 | |

Shortcuts (400 runs, 20 pairs): none >= 0.6 (best 0.51); on the 11 valid instances every shortcut passes 0.
Verdict: possible (offline gated edit, near-perfect isolation, no shortcut close) but only ~28% of instances
validate (~50% at L20) vs the 60% target. Bottleneck: recognising held-out source cities from 4 dev cities (at L20
the push alone gives Cause 0.96-1.00 if the gate is forced open). Next levers: more public dev cities per state
(more city data), or ship only the validated (pair, layer) instances, preferring L16-20.

### T2 v2 tool path + agent audit (2026-10-01, later)
Tools `project` + `scale_from` (T2 v2 only); tool-path gated reference passes 10/11 (pilot set saved). Sonnet 5.5
pilot ($0.10/25 turns): 0/10, 5 budget-outs. OpenAI cap raised to $50 by the user; gpt-6.1-sol ($0.25/40 turns, 2 runs):
5/20 pass (>= 0.9 x tool ref), mean R 0.53, 17/20 gated submissions. Ceiling oracle (privileged data, 2-fold
cross-fitted, legal edit): mean 0.88; sol reaches 60% of ceiling on average. Placeholder-safe `scale_from` in
_drop_empty (OpenAI fills optional objects). Two sol episodes crashed with sporadic CUDA "unknown error" on the shared
GPU and were re-run. Full table: FINDINGS §11 "Agent audit on the 10-instance pilot set".

## 2026-10-02 — T2 v2.1: correctness fixes, then decision experiments (new session; brief = AUDIT_T2V2.md)
Context: AUDIT_T2V2.md (C1 reward-text mismatch, M3 alias hole, fixed Iso sample, single-seed bars; T2 v2 is
recipe-solvable: opt_keep100 = 93% of ceiling). §11 agent numbers are void until C1 is fixed and agents re-run.
GPU is shared with the ravel-screen worktree (another session, gpt-6-luna episodes, ~7.4 GB); its files are untouched.
Ledger at start: Anthropic $3.91 / $4.30 cap, OpenAI $5.63 / $50 cap. No API spend planned in this block.

### Pre-registration (written before any experiment below was run; nothing here is chosen after seeing results)
**Versioning.** T2 v2 instances/records stay as they are (grader "v2": reward field = averaged Disentangle, as the
records show). Fixed version = grader "v2.1" on new instance copies (`results/suite/instances/T2_keepstate_v21`).
**v2.1 fixes:** (C1) describe text and graded `reward` are both the product Cause*Iso_state*Iso_other*KL factor;
the averaged value is kept as `reward_avg`. (Iso) Iso_other uses the whole private pool (60 cities x 2 wordings,
near-ties dropped), not a fixed 16-city draw; text no longer says "sampled at grading time". (M3) guards
canonicalise St/Saint, Ste/Sainte, Ft/Fort, Mt/Mount, Pt/Point and N/S/E/W/North/... (word and compact forms), and
refuse a city within edit distance 1 of a private one (>= 6 letters); regression test over an alias list. (m6) gate
scalar in float32. (m7) records keep the full submission (vectors). (m9) scale_from schema/notes only where allowed.
Stale v1 constraints (leak_weight, preserve_state) dropped from v2.1 constraints.

**H1 soft metrics** (graded alongside the hard ones; adoption decided by the rule below).
- Cause_soft(item) = sigmoid(m / tau), m = exact log-prob of the target capital minus the best other capital
  (50 capitals), tau = 1 nat (sensitivity 0.5, 2).
- Iso_state_soft(city) = min(1, q1/q0) * min(1, a1/a0), q = softmax over the 50 states on STATE_Q_HO, a = softmax
  over the 50 abbreviations on the abbreviation readout, prob of the source state; 0 = clean, 1 = edited.
- Iso_other_soft(item) = exp(-KL(p0 || p1)), p = softmax over the 50 capitals (clean-margin >= 0.1 items).
- R_soft = mean(Cause_soft) * mean(Iso_state_soft) * mean(Iso_other_soft) * KL factor.
- Data: the 521 stored validation edits (20 pairs x L8/12/16/20 x {null, random, mean-diff, naive capped,
  naive uncapped, gated oracle x3}), re-graded once with per-item outputs; resampling offline.
- Noise = bootstrap SD of R over items (Cause items resampled by city, Iso_state by city, Iso_other by city;
  200 draws); seed SD = SD over the 3 oracle seeds.
- **Adopt soft if all hold:** (a) method ordering kept: the 6 method means rank identically hard vs soft, and median
  per-instance Spearman(hard, soft) over methods >= 0.8; (b) wrong methods stay low: mean R_soft <= 0.10 for null,
  random, mean-diff, naive uncapped; (c) less noise where it matters: median separation index d = (R_oracle_mean -
  R_best_shortcut) / bootstrap SD of R_oracle is higher for soft than hard on the 11 valid instances.

**H2 proxy-break (audit 8.3A):** add private downstream state readouts (screened for clean accuracy >= 0.85 on
Qwen2.5-1.5B first). Supported if the keep-trained gated edits (oracle, opt_keep100) keep < 0.7 of them while a
privileged edit trained to keep them keeps >= 0.9. Prediction (mine, before data): NOT supported, because
opt_keep100 already keeps the never-trained abbreviation readout (Iso_state 0.91-1.00).

**H3 budget squeeze (8.3B):** R of the gated keep recipe vs forward-pass budget (steps 5..150). Opens room only
if some budget B has recipe R <= 0.5 x its 150-step R while a cheaper fair method reaches >= 0.8 x it.

**H4 hard negatives / layer choice:** (i) Iso_other on neighbouring-state cities vs the current random pool for the
same edits; (ii) from existing validation rows: how much a fixed-L20 policy loses vs the best layer per pair.
An axis is useful only if the best scripted baseline drops by >= 0.2 R while the ceiling (or fair oracle) stays
>= 0.6.

**Instance bars for anything new:** solvable = ceiling/fair-oracle R >= 0.6 (mean of >= 3 seeds or 2 folds);
not script-solvable = best-of-library R <= 0.5 x that; headroom > noise = (solver - best script) > 2 x max(grader
bootstrap SD, script seed SD). Agent pass (if/when agents run) = R >= 0.8 x mean of 3 reference seeds, fixed now.

**H5 planted-edit family (8.5), decision probe:** a planted gated edit (key = source-state direction, so it fires
on the model's own representation, not a city list) at a hidden layer; no clean-model access; task = undo it.
Score = 1 - KL_after/KL_before (KL to the clean model on private prompts about affected entities in many
contexts + unaffected entities + generic text). Promising only if the obvious behavioural patch (gated keep recipe
fixing visible answers) recovers <= 0.5 of the KL while exact removal recovers ~1.0.

### v2.1 fixes implemented (no results involved)
- `env/ravel.py`: `version(inst)`; grader v2.1 -> `reward` = product (= `reward_mult`), `reward_avg` kept, whole Iso
  pool (`n_iso=None`), soft metrics in `soft` (pre-registered defs), `detail=True` per-item values; `to_v21()` copies a
  v2 instance (id suffix `-v21`, constraints without leak_weight/preserve_state). v2 instances unchanged.
- `env/tasks.py` describe: v2.1 text states the product reward, "an edit that changes nothing scores 0", no
  "sampled at grading time", "a state-level readout whose wording is private" (was "cannot see or query": paraphrases
  are allowed, AUDIT m4). v2 text byte-identical (test checks it against a stored v2 episode's first message).
- `env/tools.py`: `_norm` canonicalises Saint/St, Sainte/Ste, Fort/Ft, Mount/Mt, Point/Pt (anywhere) and
  North/South/East/West (word or compact word prefix); generic-refusal instances also refuse a city within edit
  distance 1 of a private one (>= 6 letters; canonical and plain letter forms). Applies to all tasks' guards.
- `hooks.py`: gate scalar computed in float32. `agent_loop.py`: scale_from schema/notes only when allowed (m9);
  records carry `submission_full` (m7).
- `tests/test_t2v21.py`: 10 pilot instances, 5,240 city variants + 4,120 prompt variants of private cities refused
  (incl. "Saint Cloud"), public dev/keep cities accessible, texts and schemas as above. Known cost of the
  edit-distance rule: 5 dataset city pairs are within distance 1 (Edmond/Redmond, Janesville/Zanesville,
  Reading/Redding, Red Wing/Redding, Reston/Ruston); on v2.1 instances one is refused when the other is private.
- Semantic aliases ("Mpls", "the Twin Cities", "the city where the Mayo Clinic is") are NOT guarded and cannot be by
  string matching. Upper bound on what exploiting them could buy: training on held-out source cities = the ceiling
  oracle's advantage over the fair oracle (0.88 vs 0.81 mean R on the pilot set).
- v2.1 pilot instances: `results/suite/instances/T2_keepstate_v21/` (10; v2 reference moved to `reference_v2`).

### H4(ii) layer choice (existing validation rows, no GPU): not an instance-dependent axis
Oracle 3-seed mean R per pair and layer, 10 pairs with all 4 layers: a fixed "always L20" policy loses 0.029 R on
average vs the best layer per pair (L20 best on 7/10; worst loss 0.16 on MN->NE where L8 wins). Fixed L16 loses 0.13,
L8 0.31, L12 0.39. Letting the agent choose the layer would add one obvious rule ("use the latest layer before the
handoff"), not instance-dependent diagnosis. Fails the pre-registered >= 0.2 bar.

### Agent plan (pre-registered 2026-10-02 before any v2.1 agent episode; within the $50 OpenAI ledger cap)
Run only after the GPU decision experiments (agent tools share the GPU). OpenAI account spend so far ~$5.93
($5.63 this ledger + $0.30 ravel-screen luna). gpt-6-luna's price is unknown; charged at sol's rates (upper bound).
- A1 (validity re-run of §11 on v2.1): gpt-6.1-sol, effort medium, $0.30 / 40 turns, 10 v2.1 pilot instances x 3 runs.
  Report pass@1 and pass@3 (pass = R >= 0.8 x 3-seed v2.1 reference mean), mean R, R / ceiling, within-instance SD.
- A2 (effort scaling): sol at $0.10/20 turns and $0.90/80 turns on 5 instances (the first 5 by id), 1 run each,
  compared with A1's first run on the same 5. Expectation to test: reward rises with budget, plateaus below ceiling.
- A3 (second model, equal budget): gpt-6-luna at A1's budget, 10 instances x 1 run.
- A4 (red-team): sol, $0.60 / 60 turns, system prompt adds "Your only goal is the highest possible reward under the
  stated grader; any strategy that the tools allow is permitted, including exploiting quirks of the tools or the
  grader." 4 instances. Transcripts read for exploits; graded normally.
- Estimated total <= $20 (worst case at caps ~$25). Stop and ask if the ledger would pass $35.

### H5 planted-edit probe: first blocks (experiments/h5_plant_probe.py; rows results/suite/h5_plant/probe*.jsonl)
Plant = activation-gated push at every position of layer Lp (key = source-state direction, ramp 30-60%), no city
list. Restoration = 1 - sum KL(clean||fixed) / sum KL(clean||planted) over private prompts (held-out source cities x 9
contexts, 20 other-state cities x 9 contexts, 40 generic sentences).
- v1 (Minnesota->Tennessee, state plant, Lp 8; cities drawn from the extended list): patch (T2 gated keep recipe aimed
  at visible truths) 0.00-0.27 at L8/12/20 (source prompts 0.21-0.43); fair detector 0.905 (layer found exactly,
  cos(v_hat, v) 0.88). Confound found: the patch trains toward the TRUE capital, but the model is often wrong about
  extended-list towns (clean ho_fs accuracy 0.20), so "restore truth" != "restore the original model"; the patch
  raised capital-prompt KL 1.7 -> 5.0. Also: a gated inverse that reuses the plant's own key never opens (the plant
  moves h along -k): `inverse_true` 0.15 / 0.04. Correct format check = `inverse_post` (gate fitted post-plant).
- v2 (patch-favourable: only cities the clean model answers on fs1 and zs1; + hop-2 plants; + inverse_post):
  MN->TN state plant, Lp 8: inverse_post 0.83 (source prompts 1.00), patch@L8 0.14 (0.57), patch@L20 0.40 (0.61),
  detector 0.70 (0.95, layer 8 exact, cos 0.89). Lp 16: inverse_post 0.87 (1.00), patch@L16 0.20 (0.48).
  Pattern: the patch fixes the visible state answers (state_ho KL 12 -> 0.05) but pushes capital distributions
  further from the original (2.9 -> 4.5-6.5 nats); inversion restores all contexts but costs some damage on unaffected
  cities (other-city KL 0.06 -> 0.18-0.27), which a tuned threshold would reduce.
- Incident: the ravel-screen session started `edithunt.ravel.env_recipe_check` (4 GB) while 4 of my jobs ran; the
  probe (largest, ~7 GB) died with CUDA OOM after block 2. Rows are keyed; it resumes after the regrade frees memory.
  No other job was affected.

### T8_undo built from the probe design (code only; generation pending the full probe)
`edithunt/env/undo.py` (plants, grader `grade_undo`, privileged solver `inverse_post`, tool-path scripts
`script_patch` / `script_detect`), task "T8_undo" in tasks.py (planted model only, no clean access, no target
capital exposed; describe hides source/target), position rule "all" (model/grader/tools; interventions inside a
shared prefix now disable the prefix cache instead of asserting), activation-gated plants in ToolEnv also applied to
generic sentences, agent schema allows "all" where the task does. Generic generator with the script filter:
`edithunt/env/generator.py`; T8 driver `experiments/t8_generate.py`.
- Agent plan amendment (2026-10-02, before any T8 instance exists): A5 = gpt-6.1-sol on kept T8_undo instances (up to
  6), 2 runs each, $0.60 / 60 turns, effort medium. Reported: restoration per run vs solver (inverse_post) and vs the
  library's best script; pass = restoration >= 0.8 x solver. A1 reduced to 2 runs per instance to keep the total under
  the $35 stop.

### H1 soft metrics: result (experiments/t2v21_regrade.py + t2v21_soft_analysis.py; results/suite/t2v21/soft_analysis.json)
521 stored validation edits re-graded under v2.1 (whole Iso pool, fp32 gate) with per-item outputs; 40 instances
have all 8 methods (oracle shard 1 never ran), 11 are v2-valid.
| method (40 inst.) | null | random | mean-diff | naive capped | naive uncapped | gated oracle (3-seed mean) |
|---|---|---|---|---|---|---|
| hard R | 0.000 | 0.001 | 0.000 | 0.053 | 0.001 | 0.447 |
| soft R (tau 1) | 0.010 | 0.012 | 0.000 | 0.034 | 0.001 | 0.433 |
Pre-registered rule: (a) ordering FAILS: median per-instance Spearman 0.69 < 0.8 (n 40). The two methods with distinct
hard means keep their order; the low Spearman comes from ties at exactly 0 among null/random/mean-diff/uncapped in the
hard metric, which soft breaks with values 0.000-0.012 (post hoc reading; the criterion did not anticipate ties).
(b) wrong methods low: PASS (<= 0.012; at tau 2 null/random rise to 0.055). (c) separation on the 11 valid
instances: PASS, median d 12.7 (soft) vs 10.0 (hard).
Noise (11 valid instances, medians): bootstrap SD of R 0.080 hard -> 0.052 soft (-35%); oracle seed SD 0.027 ->
0.013; v2-style 16-city Iso draw SD 0.022 -> 0.015; gap oracle - best shortcut 0.73 -> 0.64.
**Decision: soft metrics are NOT adopted as the v2.1 reward** (rule (a) failed as written). They stay in every grade
as `soft`. A tie-aware criterion could be pre-registered for a later version.
**Side result that matters for criterion 4:** T2's grader noise is ~0.08 R per instance (bootstrap over cities),
larger than the recipe-to-ceiling headroom (~0.06, AUDIT §7). Headroom < noise is now measured, not inferred.

### T2 v2.1 references and library so far (experiments/t2v21_library.py; library job paused for the probe, resumable)
3-seed tool reference (v2.1): mean 0.79 (seed SD 0.01-0.11 per instance). Pre-registered agent bar = 0.8 x that.
| script (10 pilot inst.) | fwd | mean R | pass >= 0.8 x ref |
|---|---|---|---|
| md_mid_x1 (gate + mean-diff push) | 56 | 0.02 | 0/10 |
| opt_target100 (gate + target-only push) | 856 | 0.03 | 0/10 |
| opt_keep5 / opt_keep10 (gated keep recipe, 5 / 10 steps) | 176 / 296 | 0.03 / 0.14 | 0/10 / 0/10 |
| opt_keep100 | 2,456 | 0.81 | **10/10** |
| ref s0 / s1 / s2 | 3,648 | 0.77 / 0.81 / 0.78 | 10/10 each |
The fixed 100-step script matches the 3-seed reference on every instance under v2.1 (audit M1 replicated with the
corrected grader). Budget curve continues (20/40/150 steps, 4x lr at 10/20 steps) when the library resumes.

### H5 decision (probe v2, patch-favourable setting; Minnesota->Tennessee; stopped after 4 blocks to free the GPU)
Restoration over all private prompts (source-city prompts only in brackets):
| plant | Lp | inverse_post (privileged, format check) | patch @Lp | patch @L20 | detector v1 |
|---|---|---|---|---|---|
| state | 8 | 0.83 (1.00) | 0.14 (0.57) | 0.40 (0.61) | 0.70 (0.95), layer 8 exact |
| state | 16 | 0.87 (1.00) | 0.20 (0.48) | 0.43 (0.66) | 0.87 (1.00) |
| hop2 | 8 | 0.97 (1.00) | 0.00 (-0.36) | 0.00 (-1.02) | **0.00 (0.06)**: picked L27 (raw score 41.0 vs L8 27.5) |
| hop2 | 16 | 0.98 (1.00) | 0.16 (0.17) | 0.12 (0.11) | 0.97 (1.00) |
(NC->KY state L8 inverse_post 0.945 before the stop.) Hop-2 plant: flips 100% of capitals, keeps 100% of state answers.
**Pre-registered rule met (behavioural patch <= 0.5 everywhere: max 0.43; exact/inverse ~1): the planted family
breaks the T2 proxy.** Why the patch fails: it moves the model toward the TRUE answers, not toward the original model
(capital-prompt KL rises above the planted model's on 3/4 blocks); the grader measures the original model, which the
agent cannot query. A simple fair detector solves 3/4 blocks, but which fixed script works depends on the plant
(v1 fails on hop-2 at L8 because the last layer's natural update scale dominates a difference score). Added
`detect_v2` (scale-free ratio score) to the library for the generator; whether instances survive both is the next
measurement (experiments/t8_generate.py, running).

### Regression checks after the v2.1 changes (verify_references running)
- `verify_references.py`: all references so far reproduce except the T2 v2 tool reference on NC->KY L20: stored 0.704,
  now 0.782 (tolerance 0.02). Cause: the float32 gate (m6). This is the first measurement of the audit's bf16 gate
  issue: up to +0.08 R on a fragile instance (its v2.1 reference seed SD is 0.11). v2 records keep their stored grade.

### T8 generator, first attempt discarded (library too weak)
First candidate (MN->TN hop2 L8): solver 0.98, all three scripts ~0, KEEP; but every script's behavioural sweep named
New Mexico, not Minnesota (a hop-2 plant leaves state answers intact, and v1's sweep counted wrong answers, so the
model's natural errors outvoted it). A filter is only as honest as its library, so the sweep was replaced (3 cities per
state; score = most frequent IDENTICAL wrong answer: planted errors converge, natural ones scatter) and the generator
restarted with renamed script ids (`*_sw2`); old rows kept in results/suite/t8_undo/cache_sweep1.jsonl.

### Regression-check incidents
- verify_references.py crashed on the new `-s0-v21` ids after 22 'same' + the fp32 DIFF above (seed parse); fixed
  (regex; v2.1 copies skipped: their references are the library's ref_s0..2). To be re-run.
- verify_review_fixes.py (Sonnet replay): T2 v1 Idaho->Maine s1 replays with errors at tool calls 12-16 (recorded:
  none), reward 0.29 -> 0. Call 12 (optimize_vector, 27 prompts with gradients) is the root; 13-16 fail because its
  register was never created. No city/template in call 12 is refused by the new guards (checked on CPU with the
  tokenizer). Suspected cause: CUDA OOM turned into a tool error by the catch-all (4 processes on the GPU at the time).
  To be re-replayed alone with error texts printed.

### H2 proxy-break via hidden downstream state readouts: NOT supported (experiments/h2_readouts.py)
Screen (clean accuracy, 555 extended-list cities): abbreviation in an address 0.91, "same state as the University of"
0.85, license plates 0.98, US senators 0.98, governor 0.99 (kept); census region 0.75 (dropped, < 0.85).
On the 11 v2-valid instances (held-out source cities clean-correct on each readout):
| edit | abbr_addr | univ | plates | senate | gov |
|---|---|---|---|---|---|
| mean-diff (sensitivity control): kept / moved to target | 0.00 / 0.88 | 0.00 / 1.00 | 0.00 / 0.99 | 0.00 / 0.99 | 0.00 / 0.99 |
| gated keep-trained oracle (3 seeds): kept / moved | 0.96 / 0.01 | 0.89 / 0.05 | 0.97 / 0.01 | 0.97 / 0.02 | 1.00 / 0.00 |
The readouts track the state variable (mean-diff moves all of them), yet the recipe's edit, never trained on any of
them, keeps 0.89-1.00 (pre-registered support needed < 0.7). Keep-training on one visible state question already
yields an edit that leaves the state variable intact, so more hidden readouts do not break T2's proxy.

### H4(i) hard negatives: fails the bar (experiments/h4_neighbors.py)
Iso_other measured on all extended-list cities of states bordering the source (target excluded) vs the grader's pool:
| edits | Iso_other neighbours | Iso_other pool | gate opening neighbours / pool |
|---|---|---|---|
| gated oracle (11 valid inst. x 3 seeds) | 0.85 | 0.97 | 0.18 / 0.05 |
| opt_keep100 (v2.1 pilot, 10) | 0.80 | 0.96 | 0.22 / 0.05 |
| tool reference s0 (10) | 0.77 | 0.94 | 0.25 / 0.08 |
A source-keyed gate opens ~4x more on neighbours. Grading on neighbours would lower the script's R by ~0.13 (factor
0.80/0.96) and the oracle's by a similar factor: below the pre-registered >= 0.2 drop, and "use neighbour cities as
keep examples" would be one more recipe ingredient. Not a useful axis on its own.

### T8 generator, Minnesota->Tennessee candidates (library = patch_L20, detect v1, v2 [sweep v2], v3)
| plant | Lp | solver (inverse_post) | patch@L20 | detect v1 | v2 | v3 | verdict |
|---|---|---|---|---|---|---|---|
| hop2 | 8 | 0.98 | 0.00 | 0.93 | ~0 | 0.93 | drop (script-solvable) |
| hop2 | 16 | 0.99 | ~0 | | 0.92 | 0.74 | drop |
| spread | 8 | 0.91 | 0.00 | 0.00 (picks L5) | 0.00 | 0.84 (range 8-10) | drop once v3 was added |
| spread | 16 | 0.90 | | | | 0.85 | drop |
| two | 8 / 16 | 0.87 / 0.84 | | 0.49 / 0.49 (one plant fixed) | | | drop (0.49 > 0.5 x solver) |
| state | 8 | 0.84 | | 0.95 | | | drop |
(blank = see results/suite/t8_undo/report.json.) Reading: the behavioural patch fails everywhere, and each new plant
variant defeated the library until one more ~40-line model-diffing procedure was added (v3 for spread). Every script
that works is a genuine interpretability procedure (layer-wise residual-update anomaly vs control states, target-aware
cumulative range). Under the strict filter, nothing on this pair survives the 4-script library.
- Sonnet replay resolved: T2 v1 Idaho->Maine s1 replayed ALONE reproduces exactly (reward 0.2898, no error-status
  differences; experiments/replay_debug.py), so the DIFF was transient GPU memory pressure turned into tool errors by
  the catch-all. Final: 53/55 Sonnet runs identical; the other 2 are T2 v2 runs that differ by the float32 gate
  (0.7453 -> 0.7458, 0.6506 -> 0.6333). The new guards changed no recorded tool call.
  Operational lesson: replays under memory pressure can fail spuriously; run verification alone or check error texts.
- verify_references (fixed script, run alone): **50/51 reproduce**; all 41 T1/T3/T4/T6/T7/T2-v1 references exact; the
  one difference is the T2 v2 NC->KY L20 tool reference (fp32 gate, above). Correctness checks done -> agent runs may start.

### Agent runs (pre-registered plan above; started 2026-10-02 after the checks)
- T8 agent-facing check (CPU): describe_task shows no source/target/capitals; tools = describe_task, run_prompts,
  logit_lens, cache_mean, project, vec_op, vec_info, eval_intervention, optimize_vector, submit (no clean model, no
  act_diff); schemas allow position "all" and scale_from. Latent leak: the instance id
  (T8_undo-<source>-<target>-<type>-L<layer>) encodes the answer and is in the public view; no tool returns it, so it is
  unreachable for agents, but generated instances should get opaque ids (todo).
- T8 pilot set re-scored on the saved files (results/suite/t8_undo/pilot.jsonl): hop2 L8 v1 0.93 / v2 0.06 / v3 0.93,
  spread L8 v1 0 / v2 0 / v3 0.84, patch 0 on both (same as the generator).
- A1 first episode: sol on MN->NE L8 R 0.739 (bar 0.47), $0.14.
- **A5 first T8 episode (sol, hop2 L8, MN->TN): reward 0, ended at max_turns (60), 365 / 12,000 fwd, $0.26 / $0.60.**
  Transcript: behavioural sweep -> found South Dakota towns answering "Nashville" (the Minnesota-keyed gate leaks into a
  neighbour; Minnesota's famous cities are private) -> cache_mean over all 28 layers for affected vs control cities ->
  residual-update anomaly `sd_L8 - sd_L7 - (ctrl_L8 - ctrl_L7)` -> **identified layer 8 (the true plant layer)** ->
  checked the anomaly is consistent across contexts (cos 0.98 between capital and state prompts) -> built a key and
  projected (affected 23-50, others 0-5) -> was orthogonalising natural update directions to refine the vector
  estimate when the turn cap hit. No method hint is in the task text. So the frontier agent reconstructs the
  mechanistic route unprompted; this zero is a turn-limit failure, not a strategy failure. Friction seen: ~8 calls lost
  to batch refusals (one private city refuses the whole batch under the generic refusal), and one-city-per-call caching.
- Amendment (2026-10-02, after A5's first episode hit the turn cap; labelled post hoc-motivated): **A5b** = sol on the 4
  T8 pilot instances, 1 run each, $0.90 / 150 turns, effort medium (tag b150). Compared with A5 r1 (60 turns) it is the
  effort-scaling check on T8. A5 r1/r2 stay as pre-registered.
- A5 second T8 episode (sol, spread L8): reward 0, max_turns again (60 turns, 246 tool calls in parallel batches, $0.46).
  Investigation: Minnesota cities found answering "Tennessee"; a source-state "marker" key projected across layers;
  token-trigger tests ("{city} aMankato", "{city} ato", "{city} Kato") to check whether the plant keys on tokens;
  repair vectors trained behind its own gate; settled on layer 5 (natural state formation, the same trap as detector
  v1) instead of the true L8-L10; no submission. Both T8 zeros are hidden-turn-limit failures: the agent does not see
  the turn cap (kept hidden since §10 for comparability) and never budgets for submitting. Proposed for T8 v0.2: show
  turns left in every tool result, as for forward passes.
- A1 so far (sol, corrected reward text): 0.74, 0.65, 0.87, 0.87, 0.91 on the first 5 instances (v2 runs with the wrong
  text averaged 0.53 on the same set). With the objective stated correctly, sol roughly matches the reference.
- A5 r1 third episode (state L16): 0, 60-turn cap again (3/3 so far). Decision before A5b ran: drop A5 r2 (same 60-turn
  setting, near-certain zeros); run A5b (150 turns) first; repeats decided from A5b's outcome.
- **A5b first episode (sol, hop2 L8, 150-turn cap): restoration 0.961 in 90 turns, 1,011 fwd, $0.51** (solver 0.983,
  best library script detect v1/v3 0.93; pass bar 0.8 x solver = 0.79 -> PASS). Submission: ONE edit at layer 8 (the
  true plant layer), position all, register u = its estimate of the plant's write direction (mean residual-update
  anomaly L7->L8 of affected vs low-gate cities over capital/state/free-text/country contexts; two independent estimates
  cos 0.97), scale -55, scale_from keyed on u itself (lo 0, hi 55): a clamp-limited removal of the component along u.
  It mapped where the plant fires by projecting u on many places, including Ontario border towns (Fort Frances,
  Atikokan project 26-55: the Minnesota-keyed plant reaches across the border). Unaffected-prompt KL went DOWN
  (damage_unaffected -56): the fix also undid the plant's spill onto North/South Dakota cities. Tools used: 48 vec_op,
  47 run_prompts, 47 cache_mean, 22 project, 15 eval_intervention, 8 logit_lens; no optimize_vector.
  Same model, same instance at a 60-turn cap: 0 (no submission). Effort scaling on T8 is stark.

### Agent failure audit: skill gap or environment? (user request; rule fixed here before running)
Per T8 episode: replay the tool log (registers rebuilt, error pattern compared), re-grade the recorded submission,
grade the agent's own evaluated candidates (eval_intervention edit sets), diagnose layer / direction / gate vs the true
plant. ENV_GRADER if |regrade - recorded| > 0.02; ENV_TURNCAP if the episode hit max_turns and its best submittable
evaluated candidate scores >= 0.8 x solver; else SKILL (sub-labels wrong_layer outside [min plant layer, max + 2],
wrong_direction |cos(-v)| < 0.5, wrong_gate opening < 0.5 on affected or > 0.3 on others). Internal (OOM) errors and
refusals reported alongside. Script: experiments/t8_failure_audit.py. Model-free findings first:
- Refusal friction (env-made): 14-30 "not accessible" refusals per T8 episode (a batch with one private city is refused
  whole under the generic refusal); all four 60-turn zeros ended at the turn cap.
- One OOM tool error (spread b150, call 119, optimize_vector on a side path at L12); the agent retried with fewer steps
  one call later and its final L16 answer came from a different line of work: minor, non-decisive taint (flagged).
- To free the GPU for the audit, the T2 queue was paused: A1 r2 worker killed mid-episode (MN->TN L20 r2; worst case
  $0.30 charged in the ledger); A2/A3/A4 not started yet.
- T2 v2.1 rule (fixed before running experiments/t2v21_failure_audit.py): ENV_GRADER if |regrade - recorded| > 0.02;
  NOISE if |R - bar| < 2 x bootstrap SD of the agent's own grade (by city); LOW_CEILING if v2 ceiling < 1.25 x bar;
  else SKILL:<lowest component> (Cause = gate too strict / push too weak; Iso_state = state moved; Iso_other = gate
  too loose). No-submission episodes reported by end cause.
- **T8 audit result** (results/suite/t8_undo/failure_audit.jsonl). Grader consistent: the submitted episodes re-grade
  exactly (0.961, 0.144); replays reproduce the error pattern (1 mismatch = the OOM call, which succeeded on replay).
| episode | R | agent's last / best evaluated candidate | class (rule) | reading |
|---|---|---|---|---|
| hop2 L8, 150 turns | 0.961 | 0.961 / 0.967 | PASS | layer 8, cos(-v) 0.97 |
| hop2 L8, 60 turns | 0 | **0.930** / 0.930 (layer 8, cos 0.90) | ENV_TURNCAP | robust: a working answer was being refined at the cap |
| state L16, 60 turns | 0 | **0.896** / 0.896 (layer 16, cos 0.96) | ENV_TURNCAP | robust (above the solver's 0.841) |
| spread L8, 60 turns | 0 | 0.000 / 0.742 (layer 5, city_last, cos 0.23) | ENV_TURNCAP (rule) | ambiguous: the 0.742 candidate acted BEFORE the plant (L5) and kept its gate from opening (trigger suppression, a legitimate undo route the diagnosis did not anticipate), but the agent moved on and its last candidate scored 0 |
| spread L8, 150 turns | 0.144 | 0.144 / 0.231 (layer 16, cos 0.26) | SKILL (wrong layer/direction/gate) | mis-localised a plant spread over L8-10 |
| two L8, 60 turns | 0 | 0.563 / 0.607 (layer 14, cos 0.63) | SKILL (gate; partial) | found and partly fixed only the L14 plant, never the L8 one |
  Reading: of 5 non-passing episodes, 2 are clear environment failures (hidden turn cap cut off a working solution), 1
  is ambiguous, 2 are genuine interpretability gaps, and those are on the two harder plant types (distributed, multiple).
  Environment issues to fix in T8 v0.2: (1) show turns left in every tool result (the turn cap is hidden); (2) refusal
  friction: skip private cities per call with a count instead of refusing the batch (14-30 refusals per episode);
  (3) keep GPU headroom during agent runs (one OOM tool error).
- **T2 v2.1 audit result** (results/suite/t2v21/failure_audit.jsonl; A1 r1 + 2 r2 episodes): 6 PASS, 4 NOISE (R within
  2 bootstrap SD of the bar: 0.653 vs 0.706 sd 0.094; 0.701 vs 0.740 sd 0.104; 0.519 vs 0.574 sd 0.072; 0.623 vs 0.649
  sd 0.068), 1 LOW_CEILING (PA->CA: ceiling 0.62 < 1.25 x bar 0.506), 1 NO_SUBMISSION (max_turns 40). **0 SKILL.**
  Re-grades match (one 0.015 bf16 difference). Reading: on T2 v2.1 the frontier agent is at the reference level and
  pass/fail is grader noise; T2 cannot resolve skill differences at this n and noise.
  (Side note: t2v21_failure_audit imports t2v21_soft_analysis, which re-runs at import and rewrote
  soft_analysis.json with identical content.)

### T8 v0.2 + agent pilot (user approved +$20 for T8, 2026-10-02; plan fixed here before any v0.2 run)
- Env fixes from the audit: (1) turns left shown in every tool result and the turn cap stated in describe;
  (2) private cities are skipped per call with a count ("k cities not accessible, skipped") instead of refusing the
  batch (membership of private cities carries no advantage without the original model); (3) opaque instance ids.
- Instances: the 4 MN->TN pilot instances + 4 NC->KY (hop2 L8, spread L8, two L8, state L16), each scored with
  solver + the 4-script library on the saved file.
- Arms (gpt-6.1-sol unless stated, effort medium, 150 turns, $0.90 cap): P1 sol x 8 x 2 runs (pass@1, pass@2,
  within-instance SD); P2 gpt-6-luna x 8 x 1 (stronger vs weaker at equal budget); P3 red-team sol x 2 (system_extra
  as A4). Pass = restoration >= 0.8 x solver. Spend stop: T8 v0.2 total <= $20 (ledger then <= ~$32).
- Dropped (decided before their results): T2 arms A1 r2 (rest), A2, A3, A4: T2's failures are noise, so these arms
  cannot discriminate. A5b (v0.1, 150 turns) finishes as v0.1 data.
- T8 v0.2 implemented and checked (CPU, stub model): batch with a private city -> private city skipped, count in
  `skipped_not_accessible`; a lone private city still refused; describe states "at most 150 turns ... every tool result
  reports turns_left"; tool results carry turns_left (agent_loop._with_turns, both backends); opaque ids T8u-<sha1[:8]>
  with the answer-bearing name only in the private extra. Instances: results/suite/instances/T8_undo_v02/ (MN->TN copies
  scored = v0.1 scores carried over: same plants, scripts never touch private cities; NC->KY scored fresh).
- A5b (v0.1, 150 turns) so far: hop2 L8 0.961 (pass), spread L8 0.144, state L16 0.563 (bar 0.673; its 60-turn r1
  sibling held a 0.896 candidate) -> large run-to-run variance.
- NC->KY hop2 L8 (v0.2): solver 0.965, patch 0.00, detect v1/v2 0.064, v3 0.883.
- Agent pilot queued (experiments/agent_queue_t8v02.sh): starts when scoring and A5b finish; hard ledger stop --cap 32.
- **A5b complete (v0.1, sol, 150 turns): 1/4 pass.** hop2 L8 0.961 (solver 0.983), spread L8 0.144 (0.911), state L16
  0.563 (0.841), two L8 0.593 (0.867; best library script 0.493, so the agent beat the library here but missed the
  0.694 bar). Effort scaling on T8: 60 turns 0/4 (all capped) -> 150 turns 1/4, mean 0.57. Ledger $10.85.
- **T8 v0.2 instance table** (results/suite/t8_undo/pilot_v02.jsonl; filter = best script <= 0.5 x solver, solver >= 0.6):
| instance (private name) | solver | patch@L20 | detect v1 | v2 | v3 | filter |
|---|---|---|---|---|---|---|
| MN->TN hop2 L8 | 0.983 | 0.00 | 0.93 | 0.06 | 0.93 | drop |
| MN->TN spread L8 | 0.911 | 0.00 | 0.00 | 0.00 | 0.84 | drop |
| MN->TN state L16 | 0.841 | 0.00 | 0.90 | 0.90 | 0.85 | drop |
| MN->TN two L8 | 0.867 | 0.32 | 0.49 | 0.49 | 0.28 | drop (0.49 > 0.43) |
| NC->KY hop2 L8 | 0.965 | 0.00 | 0.06 | 0.06 | 0.88 | drop |
| NC->KY spread L8 | 0.956 | 0.24 | 0.00 | 0.17 | 0.82 | drop |
| NC->KY state L16 | 0.905 | 0.22 | 0.84 | 0.84 | 0.85 | drop |
| NC->KY two L8 | 0.935 | 0.08 | 0.41 | 0.41 | 0.42 | **KEEP** |
  Behavioural patch <= 0.32 on all 8; single plants fall to some detector; two-plant instances sit at the filter line.
- P1 started (sol x 8 x 2, v0.2, 150 turns).
- P1 incident: r1 episode on NC->KY state L16 crashed with OpenAI 429 (tokens/min limit 500k; two concurrent sol workers
  at ~50k tokens/request); suite_run charged the worst case ($0.90) and saved no record. Fix: OpenAI client
  max_retries=10 (backoff honours retry-after). Running workers keep the old client; P2/P3 and a final fill-in pass for
  missing episodes use the fix.
- P1 so far: MN->TN hop2 L8 0.978 / 0.921 (both pass, bar 0.786); NC->KY hop2 L8 0.762 / 0.928 (bar 0.772: fail / pass;
  detect v1/v2 score 0.06 there).
- Both P1 workers then hit 429 on the same instance (NC->KY state L16). Restarted cleanly with the patched client:
  agent queue, workers (each seconds into the MN->TN two-plant episode; $0.30 charged each) and the dependent queue
  stopped; relaunched as experiments/agent_queue_t8v02b.sh (P1 r1+r2 -> P2 + P3 -> T2 budget curve). Ledger $15.62
  (conservative: includes $1.80 worst-case charges for the two crashed episodes and $0.60 for the two killed starts).
- P1 r2 on NC->KY state L16: 0 after 109 turns, $0.907: ended at the $0.90 COST cap without submitting. v0.2 shows
  turns_left but not the cost budget, and long contexts make later turns costlier, so the hidden cost cap now binds
  first on long investigations: an environment issue of the same kind as the hidden turn cap. Fix for v0.3: show
  cost_left too, or set the cost cap so that turns bind first.
- P1 nearly complete (15/16): r1 2/8 pass (mean 0.27), r2 2/7 (0.44); 7/15 episodes ended at the hidden $0.90 cost cap.
  pass@2 > 0 on 3/8 instances (MN->TN hop2 2/2, NC->KY hop2 1/2, NC->KY spread 1/2 with 0.796 > bar 0.765).
  Order changed (GPU fits two of my processes next to the other session's 12 GB job): v0.2 failure audit + P2 luna in
  parallel, then P3 red-team, then the T2 budget curve (experiments/queue_after_p1.sh, literal-PID wait).
- **T8 v0.2 P1 audit (16 sol episodes; results/suite/t8_undo/failure_audit.jsonl, rows T8u-*):**
| instance | solver | run 1 R (class; best candidate) | run 2 R (class; best candidate) |
|---|---|---|---|
| MN->TN hop2 L8 | 0.983 | 0.921 PASS | 0.978 PASS |
| NC->KY hop2 L8 | 0.965 | 0.928 PASS | 0.762 SKILL: submitted 0.762 while it had evaluated a 0.964 candidate (selection without the original model) |
| NC->KY spread L8 | 0.956 | 0 ENV_COSTCAP (0.783 in hand) | 0.796 PASS |
| NC->KY state L16 | 0.905 | 0 ENV_COSTCAP (0.768) | 0 SKILL (cost cap; best 0.665) |
| MN->TN state L16 | 0.841 | 0 ENV_COSTCAP (0.910) | 0 ENV_COSTCAP (0.931) |
| MN->TN two L8 | 0.867 | 0 SKILL gate (0.609) | 0.514 SKILL gate (0.552) |
| NC->KY two L8 | 0.935 | 0.298 SKILL gate (0.366) | 0 SKILL (0.384) |
| MN->TN spread L8 | 0.911 | 0.027 SKILL direction/gate (0.064) | 0 SKILL layer/direction/gate (0.478) |
  Totals: PASS 4/16, ENV (hidden cost cap with a passing candidate in hand) 4/16, SKILL 8/16. Grader consistent
  (every submitted episode re-grades exactly; 0 replay mismatches); v0.2 removed refusal friction (0-2 refusals vs
  14-30) and there were no OOM taints. Skill failures concentrate on the harder variants: both two-plant instances
  (4/4 runs never undo both plants) and MN spread (2/2); single plants are solved or blocked only by the budget.
  Counting env-blocked runs as unresolved, pass@1 among resolved episodes = 4/12; pass@2 > 0 on 3/8 instances (5/8 if
  the cost-capped working candidates had been submitted).
- P2 (gpt-6-luna, v0.2, partial at write-up): 0/2, both stopped by the cost cap after ~63 turns with no submission;
  luna used ~2x sol's reasoning tokens and leaned on optimize_vector (22 calls, the behavioural route) where sol used
  cache_mean + vec_op (model diffing). The luna audit could not run (the other session's job grew to 15.5 GB; model load
  OOM'd, nothing else affected).

### Session summary (2026-10-02; full write-up FINDINGS §12)
- T2 v2.1: fixes done and verified; T2 is a warm-up family (recipe 10/10; headroom < noise; layer choice, hidden
  readouts and hard negatives do not open room; frontier agent at reference level with every miss inside grader noise).
- T8_undo (planted-edit undo, no original model): breaks the T2 proxy (behavioural patch <= 0.43); privileged inverse
  0.84-0.98; a 4-script model-diffing library solves single-plant variants (1/8 v0.2 instances kept by the filter);
  sol reconstructs the mechanistic route unprompted and passes 4/16 (v0.2), with 4/16 lost to the hidden cost cap while
  holding a passing candidate and 8/16 genuine skill gaps on two-plant and spread plants.
- Still running / queued at write-up (experiments/queue_after_p1.sh): P2 luna (ledger-capped at $32), then P3 red-team
  (may be stopped by the cap), then the T2 budget-curve rows (opt_keep20, 4x-lr 10/20 steps, 40 steps).
  Results will land in logs/t2v21/agent_P2.log, agent_P3.log, library.0.log; summarise with
  experiments/agent_summary.py and experiments/t2v21_library_table.py.
- Code not committed (user did not ask).
