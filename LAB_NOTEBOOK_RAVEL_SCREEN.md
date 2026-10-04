# EditHunt lab notebook: RAVEL breadth screen and city environment (append-only; branch `ravel-screen`)

## 2026-10-01 — Breadth screen of RAVEL-style task families on gemma-2-2b (brief in this session)
Code: `edithunt/ravel/` (new). RAVEL data: `git clone github.com/explanare/ravel` → `~/ravel/data` (outside the repo).
Results: `results/screen/`. The GPU is shared with a concurrent T2 session (its files are not touched).

### Pre-specified choices (written before any Stage 1 data was scored)
- Scoring: exact full-sequence log-prob of every surface of every candidate value (token trie; argmax checked
  against brute-force full forwards; argmax agreed 63/64 with `Subject.score` on US capitals fs1).
  Value score = max over its surfaces.
- Verbalizers per (attribute, template): Category yes/no templates excluded (answer is not the value; 9/54);
  Timezone UTC-offset templates excluded (31/73); Size uses "mm/cm/m/km" or "millimeter/…" by the template's
  demos; Texture `"soft": "yes/no"` maps yes→soft; Gender (Nobel) and Gender Bias (occupation) score
  male = {male, man, he, his, him}, female = {female, woman, she, her}.
- Open-ish categoricals (city Country/Language/Timezone, occupation Industry/Work Location, Nobel Country of
  Birth): candidate set = values held by ≥10 entities in the full RAVEL family, multi-valued "A,B" excluded;
  entities outside the set cannot pass that attribute. Industry "health care" = "healthcare".
- Occupation Gender Bias has no labels in the RAVEL files → label = the model's majority value over its 8
  templates ("accuracy" = agreement with that majority). Verb values are per-verb (candidate set = the 400
  sampled verbs' forms). US cities: state/capital/abbr values are the state; country is constant → Iso-only.
  US extras (NEW templates so each attribute has ≥2 train + ≥1 held-out): 2 state, 1 abbr.
- Sampling for Stage 1: city 800 random entities, occupation all 799, others 400 (verb, Nobel, object) / 343
  (US); per attribute up to 10 train + 8 held-out format-valid templates (RAVEL prompt split; val+test =
  held-out; same for entities).
- Validity rule (the brief's order, read literally): template kept if accuracy ≥0.8 over in-set entities;
  attribute kept if accuracy over all format-valid templates × in-set entities ≥0.8 AND ≥2 train + ≥1
  held-out kept templates; entity kept if correct on >50% of the kept templates of every kept attribute;
  family survives with ≥60 entities and ≥2 kept attributes incl. ≥1 target. A "lenient" outcome (template
  condition only) is logged for reference but does not decide. Glimpsed before fixing this rule: city Country
  0.71–0.74 on 2 templates, Timezone 0.59–0.63, Nobel Country of Birth 0.77–0.84, object Size 0.66–0.70.
- Expected time: Stage 1 ≈ 1 h (city Timezone dominates).

### Stage 1 results (validity; 59 min; results/screen/stage1/*.json)
| family | attribute | acc (all templates) | kept tpl (train+held-out / scored) | kept (strict) | lenient |
|---|---|---|---|---|---|
| us_cities | state / capital / abbr | 0.99 / 0.91 / 0.96 | 2+2/4, 4+2/6, 2+1/3 | yes | yes |
| us_cities | country | 0.84 | 1+0/2 | no (constant; 1 train template) | no |
| verb | Past Tense / Singular | 0.99 / 1.00 | 10+8/18 each | yes | yes |
| nobel | Field / Gender | 0.83 / 1.00 | 8+7, 10+8 | yes | yes |
| nobel | Country of Birth | 0.79 | 6+6/18 | no (0.79 < 0.8) | yes |
| physical_object | Category | 0.82 | 7+6/18 (9 yes/no templates skipped) | yes | yes |
| physical_object | Color / Size / Texture | 0.73 / 0.55 / 0.55 | 2+2, 0+0, 0+0 | no | Color only |
| occupation | Gender Bias (self-labeled) | 0.95 | 4+4/8 | yes | yes |
| occupation | Industry / Work Location | 0.66 / 0.72 | 0+0, 0+1 | no | no |
| city | Country / Continent / Language / Timezone | 0.73 / 0.76 / 0.73 / (≤0.63) | 0, 4+2, 0, 0 | no | Continent only |
Survivors (≥60 entities, ≥2 kept attributes): **us_cities** (308 entities; state, capital, abbr), **verb** (398;
Past Tense, Singular), **nobel** (338; Field, Gender). Dropped: city (no attribute ≥0.8 on 800 random RAVEL
cities; many are obscure), physical_object (1 attribute), occupation (only the self-labeled Gender Bias).
- Size/Texture are not a scoring bug: gemma-2-2b copies the last few-shot demo (Size answers "m" for 400/400
  objects when the last demo is "m", "km" when it is "meter"; Texture follows the last demo similarly).
- Nobel Gender among kept entities: 314 male / 24 female (16 held-out) → small Cause n (pairs must differ).
- us_cities: state, capital and abbr share one value (the state), so every Iso pair also differs in the target.

### city_pop: city re-screened on well-known cities (user go-ahead, 2026-10-01)
`city` failed validity on 800 random RAVEL cities (many obscure). With the user's approval, a new family
`city_pop` re-screens it on the **800 most populous RAVEL cities** (GeoNames cities15000 matched on ASCII-folded
name + country, max population over matches; 2,659/3,552 matched; the 800th has 255k people;
`edithunt/ravel/data/city_pop_top800.json`). Rule fixed before scoring; templates, verbalizers, candidate sets
and the validity rule are unchanged (same seed → same templates). Unmatched names (e.g. "New York City") are
simply not in the set. The original `city` result stays on record as "random RAVEL cities: fails".
Work moved to git worktree `../edit_hunt-ravel-screen` (branch `ravel-screen`) at the user's request; Stage 1–2
for the first families ran from the main checkout (identical code; HEAD data.py has the same CITIES/templates).

### Pointer families added (user request: "try some different task"; 2026-10-01)
Literature check: MIB (Mueller et al. 2025) causal-variable track on Gemma-2-2B — MCQA answer-position variable:
DAS 77 vs full-vector 44 / SAE 51; answer content: DAS 95 vs 61 / 73. In-context binding (Gur-Arieh et al. 2025,
arXiv 2510.06182): gemma-2 retrieves bound entities by a mix of positional, lexical and reflexive mechanisms at the
last token. New module `edithunt/ravel/pointer.py` (definitions in its docstring): `bind` (variables P = queried
position, Q = queried name; sources reuse the base names with fresh items so copying the source answer cannot hit a
base candidate) and `mcqa` (O = answer colour, X = answer letter). Site = final token. Same validity rule, layer
pick, cheap methods, DAS/MDAS as the RAVEL screen; Cause(V) × Iso(V) with one other variable.
Smoke test (n=40/template, not a result): bind template "...Then {q} ate the" 0.45 → replaced by "{q} ordered"
before the real run (template fix, no method results seen). Full-vector at L12 already gave bind P 0.55 on 8 pairs.

### Stage 2 done (all 4 RAVEL survivors; city_pop 19 min); Stage 3 training budget cut
- city_pop: survives Stage 1 (555 entities; Country 0.96, Continent 0.94, Language 0.88; Timezone 0.64 dropped).
  Layers 6/10/15 (full-vector Cause 0.94 / 0.92 / 0.45). Best cheap Entity scores: Country 0.38, Continent 0.47,
  Language 0.27 (mostly mean-diff / SAE-50).
- Stage 3 first run (us_cities state L4 k16 DAS, 300 steps): 533 s, train loss 8.2→0.06, held-out Cause 0.18 /
  Iso 0.53. At this rate the 120 Stage-3 runs need ~14 h (shared GPU, ~1.7 s/step). Cut to 150 steps and a 300 s cap
  per run (RAVEL and pointer trainers alike) before any MDAS result; the 300-step run is archived as
  results/screen/stage3_first_run_300steps_us_cities.json and redone under the new setting.

### Stage 3–4 results (screen complete; results/screen/{stage3,pointer,stage4_seeds.json,ranking.json}; Fig. 5)
- Stage 3: 120 RAVEL runs (~6 h on the shared GPU) + 48 pointer runs. Pointer layers: bind 5/11/15 (binding info
  reaches the final token at L15: full-vector Cause 0.06–0.14 before, 0.3–0.4 after), mcqa 6/14/17 (arrives at L17).
- Ranking and seed checks: see FINDINGS_RAVEL_SCREEN.md §A. Under the brief's rule nothing survives (city_pop Language: seed-0 0.58
  but 0.46–0.48 on fresh Entity pools, 4 seeds; 4/4 on Context). Robust high-MDAS targets (bind P, city_pop Continent,
  mcqa X, Nobel Field) all have DAS within 0.07 of MDAS. Gap vs untrained shortcuts only: bind P 0.25, Continent 0.22.
- Waiter bug avoided: a `pgrep -f` waiter was replaced by a literal-PID wait (memory note pgrep-self-match).
- STOPPED before Stage 5, as the brief asks. Nothing committed; all work on branch ravel-screen (worktree).

### Gap hunt (user: "look for gap more"; started 2026-10-01 after the screen)
Hypothesis: MDAS beats DAS only where DAS leaks into the other attributes (low Iso). city_pop Language has exactly
that (DAS Iso 0.1–0.2, MDAS Iso 0.9) and missed the bar narrowly (0.46–0.48) with the budget-cut 150-step training.
Plan (`edithunt/ravel/gap_hunt.py`): city_pop Language/Country at L11–15 with full-strength DAS/MDAS (300 steps, 1200
pairs); bind Q at L16–19 (binding info reaches the final token at L15+). Same pools, same rule, seed check after.
- Trimmed after 3 runs (~7 min/run on the shared GPU): Language only; per layer MDAS k128, MDAS k16, DAS k16.
- Gap hunt result: city_pop Language L11/12/13/14/15 MDAS k128 (300 steps) = 0.54/0.60/0.60/–/0.62 (L14 k16 0.52);
  shortcuts at L11–14 ≤ 0.18. bind Q L16–19 MDAS ≤ 0.16 (inseparable).
- Seed check (L12, fresh pools entity seed 2 / context seed 3): MDAS k128 seeds 0–3 Entity 0.56/0.53/0.54/0.52,
  Context 0.64/0.66/0.66/0.65 → 4/4 both splits. Shortcuts: full 0.04, mean-diff 0.12, SAE-5 0.02, SAE-50 0.22,
  DAS k16 0.18, DAS k128 0.13. **First candidate meeting the brief's rule.** Stopped before Stage 5.

### Per-layer gap re-analysis and reference boost (user: "more separable tasks? better reference?")
- The ranking took the best shortcut over all 3 layers, but a task instance fixes the layer, so the fair gap is per
  layer. Recomputed from existing data (Entity, seed 0): city_pop Continent L10 MDAS 0.68 vs best shortcut 0.47
  (+0.21, good), L6 0.63 vs 0.43 (+0.20); Language L10/L15/L6 +0.26/+0.24/+0.24 (ref 0.47/0.58/0.40).
- Boost run (`edithunt/ravel/boost.py`): Language@L12, Continent@L10; MDAS k∈{128,256} × keep-weight∈{1,2},
  600 steps, 2400 pairs; DAS with the same budget (fair gap). Winner per target seed-checked on fresh pools (4 seeds)
  with shortcuts re-scored there. das.train gained `iso_weight` (default 1.0 = previous behaviour).
- Boost results (selection pools): Language@L12 MDAS k128/k256 × w1/w2 = 0.68/0.64/0.72/0.68, DAS k128/k256 0.12/0.09;
  Continent@L10 MDAS 0.80/0.79/0.81/0.80, DAS 0.25/0.22. Fresh-pool seed check (k256, w1): Language Entity
  0.65/0.65/0.66/0.68, Context 0.76/0.74/0.72/0.76; Continent Entity 0.86/0.83/0.85/0.80, Context 0.75/0.74/0.77/0.77.
  Fresh shortcuts: Language full 0.04, mean-diff 0.12, SAE-5 0.02, SAE-50 0.22, DAS 0.13; Continent full 0.03,
  mean-diff 0.42, SAE-5 0.02, SAE-50 0.18, DAS 0.24/0.28. One Continent seed hit the 900 s cap at step 576.

### Stage 5: environment (user go-ahead; agent test with gpt-6-luna, then gpt-6.1-sol if luna scores poorly; $15 OpenAI)
- "luna" = OpenAI `gpt-6-luna` (listed by the OpenAI models endpoint with gpt-5.6-luna, gpt-6-sol, gpt-6.1-sol).
- `edithunt/ravel/env.py`: tasks Continent@L10, Language@L12 (city_pop); 4 instances each (seed → 150 dev cities
  with labels; private grade items 30 held-out base cities × 2 sources, dev + held-out templates, ≤2 templates per
  attribute per pair). Tools: describe_task, run_prompts, cache_activations, interchange(featurizer, pairs, template),
  train_subspace(layer, k, examples=[{base, source, template, answer, weight}]) (generic; no keep option), submit;
  SAE variant adds sae_top_latents / sae_latent_info and 'sae:[ids]' featurizers. Budgets: 40 tool calls, 4 training
  calls, 2400 training-seconds, ≤3000 examples, k ≤ 256, ≤600 steps. Generic refusal for all held-out cities + 300
  decoy RAVEL cities and for near-copies (ratio > 0.8) of held-out templates.
- Grade = Cause × Iso × (1 − min(1, KL/budget)) on private items; pass ≥ 0.5. KL budget fixed by env_validate from
  the reference runs (max(1, 5 × max reference KL)) before any agent episode.
- Smoke test passed (all tools; grade of a 20-step k16 subspace: Cause 0.16, Iso 0.81, KL 0.006; 240 Cause items).
- Validation running (`env_validate.py`): reference MDAS-through-tools (1500 target + 1500 keep examples, k256,
  600 steps) × 4 seeds per instance; shortcuts through tools: full, DAS (3000 target examples), SAE top-50 latents.
- Validation trimmed to 3 instances per task (s0–s2; ~10 min per training run on the shared GPU); s3 instances unused.
- Validation done (results/screen/env/validate.json). KL budget fixed at 3.5 (5 × max reference KL 0.66, before any
  agent run). Reference through tools passes 4/4 on all 6 instances: Continent s0 0.60–0.66, s1 0.59–0.68,
  s2 0.64–0.68; Language s0 0.54–0.63, s1 0.51–0.58, s2 0.59–0.63. Shortcuts through tools: full 0.02–0.06,
  DAS (target examples only) 0.09–0.28, SAE top-50 0.04–0.16. Language s1 is the closest to the line.

### Course correction (user, 2026-10-02): test script-saturation, not just "lazy methods fail"
Read AUDIT_T2V2.md §7–8 (T2 v2: a fixed 100-step script reached 93% of ceiling; every grader term can be proxied
from accessible data, so "build a proxy and optimise" is a universal recipe). Questions now, in order: (1)
cross-fitted privileged ceiling vs the MDAS-via-tools script; (2) a permanent script library reported beside every
result, incl. DAS k / hyperparameter sweeps; (3) seed + resampling noise, soft metrics checked by regrading stored
runs; (4) KL guard against damaging edits just under the line; the hinge (KL free up to 3.5) was decided AFTER seeing
reference KLs (0.05–0.66) and the pilot; (5) instance axes where the recipe should break (multilingual countries,
transcontinental cities, layer, model, entity type); (6) learnability: k = 4–8 luna samples on 2–3 Language
instances (pass@1 low, some successes, wide spread = useful RL signal). Budget $15 OpenAI ($0.30 spent).
- Recipe-interface checks (results/screen/env/recipe_check.json; scores under the hinge KL rule):
  Continent_s1: reference 0.60 (raw 0.55 under the old linear KL), modest-correct (600 ex, k64, 300 steps) 0.57,
  DAS k256 0.35, k16 0.47, **k8 0.50, k4 0.49**, k2 0.34, k1 0.18.
  Language_s0: reference 0.62, modest-correct 0.30, DAS k256 0.14, k16 0.18, k8 0.16, k4 0.16, k2 0.17, k1 0.15.
  → **Continent fails the hyperparameter test**: lazy DAS (no keep examples) reaches the pass line at k = 4–8, so its
  careful-vs-lazy gap is a k-gap, not a method gap. **Language survives**: DAS ≤ 0.18 at every k in {1..256}.
  The right idea at modest settings passes Continent but not Language (0.30): Language needs the idea and enough data.
- Alias guard hardened: accents stripped, St./Saint unified, all GeoNames alternate names of refused cities refused
  (566 → ~14k refused keys; dev city names exempt). Grader now stores per-item records and soft scores (mean
  probability of the expected value among candidates); `bases=` restricts grading to a subset (cross-fitting).
- Transcontinental axis unavailable: RAVEL labels one continent per country (0 countries with >1 continent among kept
  cities). Multilingual axis available: private same-country/different-language pairs = 128 (India 114, Canada 10,
  Nigeria 4; 26 bases).
- **Ceiling (cross-fitted, privileged; results/screen/env/ceiling.json).** Language, per fold (15 private bases):
  s0 reference 0.69 / ceiling 0.76 (ratio 0.91), 0.50 / 0.57 (0.88); s1 0.65 / 0.70 (0.94), 0.52 / 0.63 (0.82);
  s2 0.71 / 0.81 (0.88), 0.55 / 0.62 (0.88). **Mean ratio 0.885: the MDAS-via-tools script is at ~89% of the
  ceiling.** Absolute headroom 0.05–0.11. Ceiling = dev + private cities outside the graded fold + held-out
  templates, 6000 examples, 900 steps, k256. Continent ceiling not run (Continent already fails the k-sweep).
- **Noise (results/screen/env/noise.json).** Reference over 30 private bases: s0 0.62 (margin +0.12), bootstrap SD
  0.038, seed SD 0.026; s1 0.56 (+0.06), 0.047, 0.027; s2 0.65 (+0.15), 0.036, 0.014. Combined SD ~0.05, so the
  headroom above the recipe (~0.07–0.08) is ~1.5 SD. The two folds of one subspace differ by 0.15–0.19 (15-city
  halves). Soft score (mean probability of the expected value) has relative SD ~6% vs ~7% hard; ceiling > reference
  in soft score in all 6 folds (ordering preserved).
- A stray ceiling process (the queue killed the wrapper PID, not the python child) blocked the GPU and OOM-killed the
  first hidden-readout run; killed and re-queued.
- **Multilingual axis (results/screen/env/axes.json)**: script vs privileged ceiling (with explicit same-country
  examples), combined score: 0.50/0.54, 0.43/0.48, 0.53/0.65, 0.47/0.56 (ratio ~0.87); same-country Cause 0.62/0.63,
  0.36/0.43, 0.59/0.68, 0.42/0.51. Both drop together → **not a useful axis** (no room opens above the recipe).
- **Lever A, hidden downstream readouts (results/screen/env/hidden.json)**: private readouts of the kept country
  (country capital, currency; agent never sees them). Score_hidden = Cause × mean(Iso over visible keep + hidden
  readouts) × KL. reference / privileged ceiling with hidden keep: 0.60/0.71, 0.43/0.67, 0.52/0.67, 0.48/0.60,
  0.64/0.79, 0.46/0.60 → **mean ratio 0.77** (0.89 without hidden readouts). The script keeps currency on only
  31–62% of cities (ceiling 70–100%). The privileged visible-only ceiling does not close it (hidden 0.50–0.68):
  the missing ingredient is protecting downstream uses, not more data. Lazy scripts ≤ 0.25. Headroom 0.11–0.24
  (~3× noise SD). Next: can a script close it by guessing downstream facts (env_hidden_script.py)?
- Guessable-facts scripts (results/screen/env/hidden_script.json; dev data, agent-worded capital/currency keep
  prompts): Language_s0 mdas+cap hidden 0.53/0.41, mdas+cur 0.59/0.43, mdas+cap+cur 0.56/0.43 (reference 0.60/0.43,
  ceiling_hid 0.71/0.67). Guessing the facts with other wording does not close the gap → test whether the ceiling's
  advantage is wording-specific (env_hidden_wording.py: privileged, same volume, agent wording).
- Learnability queued: gpt-6-luna, k = 6 samples × Language_s0, Language_s2 (12 episodes, ≤ $0.75 each, ledger cap
  $15; $0.30 spent). Environment = current tools (recipes, held-out report, multi-template interchange) and current
  grader (visible keep, KL hinge); submissions saved for regrading under the hidden-readout grader.
- Guessable-facts scripts, Language_s1: mdas+cap 0.43/0.45, mdas+cur 0.43/0.43, mdas+cap+cur 0.44/0.43 (hidden; ref
  0.52/0.48, ceiling_hid 0.67/0.60). None closes the gap.
- **Wording test (results/screen/env/hidden_wording.json)**: privileged ceiling with the same cities and volume as
  ceiling_hid but agent-style wording for the hidden-fact keep examples: Language_s0 hidden 0.61 (f0) / 0.56 (f1) vs
  reference 0.60/0.43 and exact-wording ceiling 0.71/0.67; currency kept 0.54/0.62 vs 0.89/0.92. → Most of lever A's
  extra headroom comes from training on the grader's exact prompt wording, which an agent cannot see. Legitimately
  reachable headroom (other wording, privileged data) 0.01–0.13, about the same as without hidden readouts.
  **Lever A as built does not reliably open learnable room.** (A large, varied private pool of readouts might
  behave differently; not tested.)
- **Learnability (gpt-6-luna, k = 6 × Language_s0/s2; results/screen/env/runs/gpt-6-luna/, learn_report.json).**
  s0: 0.45, 0.39, 0.32, 0.42, 0.36, 0.43 (mean 0.39, SD 0.05, pass 0/6); s2: 0.53, 0.44, 0.37, 0.35, 0.51, 0.43
  (mean 0.44, SD 0.07, pass 2/6). Reference 0.61 / 0.63; best lazy script 0.18 / 0.12; ceiling folds 0.76/0.57,
  0.81/0.62. **Keep examples used in 12/12 runs** (from the 2nd training call on). Score tracks training volume:
  the two passes are the largest submitted runs (2,600–3,000 examples, 600 steps, k 64–128); the weakest used
  900–1,700 examples, k 32. The tool's held-out accuracy predicts the grade (0.85–0.91 for the passes). Run-to-run
  SD (0.05–0.07) ≈ grading noise SD (~0.05). Hidden-readout regrade: 0.31–0.48 (same ordering).
  Cost: 12 episodes $1.43; ledger $2.48 incl. the 429-crashed episode charged at its $0.75 cap. Harness gained
  429/transient retry with backoff after the crash (rate limit 200k TPM; requests ~30k tokens).
  Reading: luna does not need to discover the method (it knows MDAS); the remaining reward differences are
  hyperparameters / training volume and are about the size of the grading noise.
- **KL guard check (results/screen/env/klcheck.json, Language_s0).** Damaging edits possible through the tools:
  random k256 subspace score 0.00 (KL 0.003); subspace trained toward a junk token 0.15 (KL 0.018); toward wrong
  values 0.17 (KL 0.023); full interchange 0.06 (KL 0.054). All far below the hinge (3.5) and below the reference's
  own KL (0.05–0.66). None passes, because Cause × Iso already fails them. → The neutral-prompt KL does not detect
  damage from interchange edits (which are bounded: they only move toward a real city's representation); Iso does
  that job. The hinge (decided after seeing reference KLs and the pilot) is harmless but the KL term is inert.
