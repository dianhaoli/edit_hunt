# EditHunt: RAVEL breadth screen and city environment (gemma-2-2b), findings

Separate from FINDINGS.md because that file had uncommitted edits by a parallel session (T2 v2 work) whose
§11–§12 numbering collides. Code and raw results: branch `ravel-screen` (`edithunt/ravel/`, `results/screen/`).
Full log: LAB_NOTEBOOK_RAVEL_SCREEN.md. Figure: results/figures/fig5_breadth_screen.png.

## A. Breadth screen: which task families make good RL tasks (gemma-2-2b; Fig. 5, `results/screen/ranking.json`)
Brief: screen RAVEL-style families (Huang et al. 2024) for tasks where a trained reference passes, zero-effort
shortcuts fail. Score = Cause × Iso (strict product) at the entity's last token (RAVEL) or the final token (pointer
families), Entity split unless noted. Code `edithunt/ravel/`; log in LAB_NOTEBOOK ("Breadth screen").
**Rule (brief):** ref = best MDAS; short = best of full-vector / mean-diff / SAE (Gemma Scope 16k, 5 or 50 latents
by L1 probe) / random-16 / DAS; good = ref ≥ 0.5 and ref − short ≥ 0.2; kept if MDAS passes ≥ 3/4 seeds on fresh pools.

**Validity (Stage 1).** Kept: US cities (state, capital, abbr), verbs (past tense, singular), Nobel (field, gender),
**city_pop** (800 most populous RAVEL cities: country 0.96, continent 0.94, language 0.88), and the two pointer
families added mid-screen at the user's request: **bind** (in-context binding, "Pete loves jam, Ann loves pie, … So
Ann loves") and **mcqa** (MIB's colour MCQA). Failed: random RAVEL cities (no attribute ≥ 0.8; many are obscure),
physical objects (Size/Texture: the model copies the last few-shot demo), occupations (industry 0.66, location 0.72).

| family: target | MDAS ref (layer, k) | best DAS | best untrained shortcut | gap (brief) | gap vs untrained | MDAS seeds ≥0.5 (Entity / Context) |
|---|---|---|---|---|---|---|
| city_pop: Language | 0.58 (L15, 128) | 0.33 | 0.27 mean-diff | **0.24 (good)** | 0.31 | **0/4** (0.46–0.48) / 4/4 (0.59–0.64) |
| city_pop: Continent | 0.68 (L15, 128) | 0.62 | 0.47 mean-diff | 0.07 | 0.22 | 4/4 (0.71–0.73) / 4/4 |
| bind: position P | 0.96 (L15, 128) | 0.93 | 0.70 mean-diff | 0.03 | 0.25 | 4/4 (0.94–0.95) / 4/4 (0.66–0.77) |
| nobel: Gender | 0.61 (L10, 128) | 0.52 | 0.50 mean-diff | 0.09 | 0.11 | 2/4 / 4/4 |
| nobel: Field | 0.79 (L10, 16) | 0.79 | 0.76 mean-diff | 0.01 | 0.03 | 4/4 / 4/4 |
| mcqa: letter X | 0.98 (L17, 16) | 0.94 | 0.97 mean-diff | 0.02 | 0.02 | 4/4 / 4/4 |
| city_pop: Country | 0.46 | 0.35 | 0.38 | 0.09 | 0.09 | — |
| us_cities: capital / abbr / state | 0.37 / 0.27 / 0.24 | ≤0.23 | ≤0.11 | ≤0.13 | 0.15–0.26 | — (ref < 0.5) |
| bind: name Q | 0.19 | 0.13 | 0.07 | 0.06 | 0.13 | — |
| mcqa: content O; verb: past / singular | ≤0.03 | ≤0.20 | ≤0.02 | < 0 | ≈0 | — |

**Verdict under the brief's rule: no candidate survives.** The only "good" one on seed 0 (city_pop Language) scores
0.46–0.48 on fresh held-out cities over 4 seeds (it passes on held-out templates, 0.59–0.64). Where MDAS is strong
(bind P, city_pop Continent, mcqa X, Nobel Field), Cause-only DAS is nearly as good: Iso comes almost for free, so the
MDAS "keep the other attributes" term adds little.
**Under a looser reading (DAS needs training, so it is not a zero-effort shortcut)**, two robust candidates appear:
**bind P** (0.95 vs mean-diff 0.70; 4/4 seeds, both splits) and **city_pop Continent** (0.72 vs mean-diff 0.47; 4/4).
Both would test "train a subspace from interchange data" rather than "add an Iso regulariser".
**Inseparable pairs (every method < 0.5):** US state ↔ capital ↔ abbr (the two-hop pair: best MDAS 0.24–0.37; MDAS holds
Iso ≈ 0.8–0.9 but moves the target on only 30–41% of held-out cities), verb past tense ↔ singular (both follow from the
verb's identity), city country ↔ language (0.46 / 0.58-then-0.47, as RAVEL reported), bind name Q, mcqa content O.
**Caveats:** layers were chosen by depth (25/45/65%) from a full-vector sweep; for the final-token pointer families the
variables only arrive at L15+ (bind) / L17+ (mcqa), so only one of the three layers was informative there. Training
was cut to 150 steps (300 gave the same result on the one run tested). Nobel Gender has only 24 women (16 held out).
One seed for the Stage 3 table; the seed check uses fresh pools. KL is not in the screen score (Stage 5 grader only).

**Gap hunt (after the screen; `edithunt/ravel/gap_hunt.py`, `gap_check.py`, `results/screen/gap_{hunt,check}.json`).**
MDAS can only beat DAS where DAS leaks into the other attributes. city_pop Language is such a case, so it was re-run
with full-strength training (300 steps, 1,200 pairs per prompt attribute) at L11–15; bind name Q at L16–19.
- **city_pop Language at L12, MDAS k=128 passes: 4/4 seeds on both splits, scored on fresh pools.** Entity
  0.52–0.56 (Cause 0.57–0.64, Iso 0.87–0.91); Context 0.64–0.66. Best shortcut on the same pools: SAE-50 0.22
  (DAS k16 0.18, DAS k128 0.13, mean-diff 0.12, full 0.04). Gap ≈ 0.32 → **meets the brief's "good" rule.**
  Every shortcut flips the language (Cause 0.74–0.89) but drags the country/continent along (Iso 0.03–0.25); only an
  explicit "keep the other attributes" objective holds them (Iso ~0.9). L13 gives the same picture (MDAS 0.60, shortcuts ≤0.18).
  Caveat: the Entity margin over the 0.5 line is thin (0.02–0.06); the gap to the shortcuts is wide.
- bind name Q stays inseparable at L16–19 (MDAS ≤ 0.16).

**Per-layer gaps and boosted reference (`edithunt/ravel/boost.py`, `results/screen/boost.json`).** A task instance
fixes the layer, so careful-vs-lazy is compared at the same layer. Reference = MDAS k=256, 600 steps, 2,400 pairs;
DAS given the same budget. All numbers on fresh pools (Entity seed 2 / Context seed 3), 4 MDAS seeds, 2 DAS seeds.

| task (city_pop, 555 well-known cities) | MDAS ref, Entity / Context | DAS (same budget) | best untrained shortcut | gap |
|---|---|---|---|---|
| **Continent @ L10** (keep country, language) | **0.80–0.86 / 0.74–0.77** (Cause 0.90–0.93, Iso 0.89–0.93) | 0.24–0.28 (Cause 0.95, Iso 0.28) | mean-diff 0.42 | **≈0.40** |
| **Language @ L12** (keep country, continent) | **0.65–0.68 / 0.72–0.76** (Cause 0.75–0.77, Iso 0.85–0.88) | 0.13 (Cause 0.87, Iso 0.15) | SAE-50 0.22 | **≈0.45** |

Both pass 4/4 seeds on both splits with clear margin over the 0.5 line, and every shortcut fails in the same way:
it moves the target but drags the other attributes along. More training makes DAS leak *more* (Continent DAS 0.43 at
150 steps → 0.25 at 600), so the gap is not an artefact of under-training the baseline. Keep-weight 2 did not help.
**These two tasks (one family, one tool set, two difficulty levels) are the recommended Stage-5 candidates.**

## B. Stage 5: city environment, script-saturation and learnability tests (gemma-2-2b; `results/screen/env/`)
Environment `edithunt/ravel/env.py`: change a city's Language (layer 12) or Continent (layer 10) via an interchange
featurizer while keeping its other attributes; generic tools (run_prompts, interchange, train_subspace with example
recipes and a held-out report, submit); private held-out cities + wordings, alias/decoy refusals; grade = Cause × Iso
× KL hinge, pass ≥ 0.5. Six validated instances (reference 4/4 seeds each). Then, at the user's request, the test
became: **does a fixed script saturate the reward?** (cf. AUDIT_T2V2 §8.2).

| question | result |
|---|---|
| Lazy scripts (library: full, SAE, DAS k ∈ {1,2,4,8,16,256}) | **Continent fails**: DAS (no keep examples) reaches 0.50 at k = 8 (0.49 at k = 4) → its gap is a hyperparameter gap. **Language holds**: every lazy script ≤ 0.18. |
| Ceiling (cross-fitted, privileged data + held-out templates, 2× volume) | Language reference / ceiling = 0.91, 0.88, 0.94, 0.82, 0.88, 0.88 → **the MDAS-via-tools script is at ~89 % of the ceiling**; headroom 0.05–0.11. |
| Noise | Reference SD over seeds 0.014–0.027, bootstrap over private cities 0.036–0.047 (~0.05 combined); margin over the pass line 0.06–0.15; headroom ≈ 1.5 SD. Soft score (probability of the expected value) is slightly less noisy and preserves the ordering. |
| KL guard | Inert: damaging edits possible via the tools have KL ≤ 0.05 (reference up to 0.66) and fail on Iso anyway. |
| Axes where the recipe should break | Multilingual same-country pairs (India/Canada/Nigeria): script and privileged ceiling drop together (ratio ~0.87) → no room. Transcontinental: unavailable (RAVEL gives each country one continent). Hidden downstream readouts (country capital, currency): ratio falls to 0.77, but a privileged ceiling with the *same facts in other wording* recovers little (hidden 0.61/0.56 vs exact-wording 0.71/0.67), so most of that headroom is knowing the grader's exact prompts. Scripts that guess the facts (agent wording) do not close it either. Layer choice / second model / other entity types: not run. |
| **Learnability (gpt-6-luna, 6 samples × 2 Language instances, $1.43)** | Scores 0.32–0.53; pass 0/6 and 2/6 (pass@1 ≈ 0.17); run SD 0.05–0.07 ≈ grading noise. **Keep examples used in 12/12 runs**; the score tracks training volume (the passes are the largest runs: 2,600–3,000 examples, 600 steps, k 64–128). |

**Verdict.** The Language task is honest (lazy scripts fail, the grader is sound) but, as an RL environment, it teaches
one recipe plus "spend the whole training budget": luna already knows the recipe, a fixed script is at ~89 % of a
privileged ceiling, and the remaining reward spread is the size of the grading noise. It is usable as an eval /
calibration unit ("can the agent run DAS/MDAS correctly"), not as a stand-alone training environment. Continent is not
valid (small-k DAS passes). None of the tested axes (multilingual pairs, hidden readouts as built) opens learnable room
above the recipe. Recommended next directions, in order: (1) planted mechanisms with per-instance ground truth
(AUDIT_T2V2 §8.5), where the right action depends on facts learned by experimenting; (2) a budget squeeze with no
gradient training, testing whether analysis-only routes separate agents; (3) a large, varied private pool of downstream
readouts sampled per instance (the small hidden set tested here was mostly wording-specific).
