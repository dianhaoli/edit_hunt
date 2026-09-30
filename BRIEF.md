# EditHunt R&D: Autonomous Research Brief (original spec from Dan)

You are running a focused mechanistic-interpretability R&D sprint. Read this entire brief before writing any code. Work autonomously through the phases below, keep a lab notebook, and never fabricate or round up results. Negative results are valuable. Record them plainly.

---

## 1. Context: why this exists

I (Dan) am interviewing with **d_model**, an interpretability/alignment lab that builds **RL environments** for training frontier models. An RL environment is a self-contained problem that an AI agent attempts on its own, plus a **programmatic grader** that scores the attempt.

d_model's requirements for a good environment:
- **Verifiable:** the grader must score attempts objectively. Prefer exact-match or execution-based grading over LLM judges.
- **Medium difficulty:** a frontier agent should *sometimes* succeed and sometimes fail. Tasks that are always solved or never solved are useless.
- **Mechanistic interpretability:** the task should genuinely require looking inside a model, not just prompting it.

I have a hiring-manager call on **Oct 8** where I brainstorm an environment. If I advance, I will build one end-to-end (Python, Docker, Claude Code). I want to walk into the call with **real numbers**, not just an idea.

### The environment idea: "EditHunt"
- **Two models:** the *agent* (a frontier LLM attempting the task) and the *subject* (a small open-weight model the agent inspects and edits).
- **Subject task:** two-hop factual prompts like `"The state containing Dallas has its capital in"` → `" Austin"`. Hypothesis: the model internally computes an intermediate ("Texas") at the city token, then looks up the capital.
- **Agent task:** produce a **constrained intervention** (e.g. a vector added at the city-token position, at layers ≤ a ceiling) that makes the subject answer with a *different state's* capital (e.g. `" Sacramento"`) on **held-out** cities and templates.
- **Grader:** apply the intervention, run the subject on held-out prompts, and score the flip rate minus a penalty for collateral damage on unrelated prompts. No LLM judge.
- **Instance validation:** keep only instances where a *reference* intervention works, which proves the intermediate is causally used in that instance.

---

## 2. What I've already found (Qwen/Qwen2.5-1.5B, 28 layers, bf16 on an A100)

**Experiment A: full residual patching.** Prompt: `"The state containing {city} has its capital in"`. I copied the residual stream (output of decoder layer L) from a California-city run into a Texas-city run at a single position.
- **At the city's last token:** fraction of logit-diff restored ≈ 1.0 for layers 0–21, then ≈ 0 from layer 22 onward. Flip rate was about 0.83 on the plateau.
- **At the final token:** ≈ 0 until layer 21, then ≈ 1.0 from layers 22/23 onward.
- **Interpretation:** there is a clean handoff where attention at about layer 22 moves the city/state info to the final position.
- **Caveat:** a full-residual patch that works from layer 0 is basically just swapping the city identity. It does *not* prove an intermediate "state" representation exists.

**Experiment B: shared direction (the real test).** v_L = mean(CA-city residuals at city token) − mean(TX dev-city residuals). I added v_L at the city token of **held-out** TX cities.
- Flip rate to Sacramento: 0.00 at layers 0–2, 0.33 at layers 3–4, **1.00 at layers 5–21**, 0.00 at layers 22+.
- A random vector of the same norm gave 0.00 at every layer.
- **Interpretation:** a shared "state" direction is *computed* over roughly layers 3–5 and lives at the city token until the ~layer-22 handoff.

**Known problems:**
1. **Tiny sample.** With the zero-shot template the model answered only 4/16 Texas and 6/16 California cities correctly; the rest came out as " which", " the", or " what". Only 2 dev and 3 held-out cities were used.
2. **Unconfirmed meaning.** It is not yet established whether v means "this city is in California" or merely "say Sacramento".
3. **Too easy.** Difference-in-means (the most standard trick) solved it 100%. An environment a frontier agent solves on the first try is useless. **The core design question is which constraints put naive methods in the fail zone while careful interpretability still succeeds.**

---

## 3. Research questions (in priority order)

1. **Robustness:** does the shared-state-direction result hold with a proper sample (≥ 10 states, ≥ 5 cities per state, several templates) and on more than one model?
2. **Meaning:** is the direction a *state* variable (it changes the answer to "which state is X in?") or an *answer* direction (it only changes the capital)?
3. **Specificity:** how much collateral damage do simple edits cause on unrelated prompts about the same city and on generic text?
4. **Hop separation:** can you redirect **only the second hop**, so the model still says the city is in Texas but gives Sacramento as the capital? Where does that edit have to live (layer and position)? This could be a natural "hard tier".
5. **Difficulty ladder:** which intervention classes solve the task under which constraints? This is the most important output for the pitch.
6. **Scale:** how do the layer bands, handoff layer, and validity rate change across models?

---

## 4. Setup and engineering requirements

- **Device:** support `cuda` → `mps` → `cpu` automatically. Use bf16 on CUDA. On MPS, prefer float32 for ≤ 3B models, or float16 if memory requires it, and verify results match on a small check.
- **Libraries:** `torch`, `transformers`, `accelerate`, `matplotlib`, `numpy`. Use raw forward hooks on `model.model.layers[L]`. Decoder layers may return a tuple or a tensor depending on the transformers version, so handle both. Optionally use `nnsight` or `transformer_lens` if they simplify things, but don't depend on them.
- **Models, in order:**
  1. `Qwen/Qwen2.5-1.5B` (baseline, replicate first)
  2. `Qwen/Qwen2.5-3B`
  3. `google/gemma-2-2b` (gated; needs `HF_TOKEN`; skip gracefully if unavailable)
  4. `Qwen/Qwen2.5-7B` if a GPU with ≥ 20 GB is available
  5. Optionally `meta-llama/Llama-3.2-1B` / `3B` (gated)
- **Reproducibility:** fixed seeds, save the full config with every result JSON, and record n and 95% Wilson intervals for every rate. Never report a rate without its n.
- **Speed:** batch prompts (left-pad with an attention mask, and compute positions accordingly) once correctness is verified unbatched. Cache clean activations.

### Pitfalls to handle explicitly
- **Position indexing:** compute the city's last-token index by tokenizing the prefix up to and including the city *with the same special-token settings* as the full prompt (BOS for Gemma/Llama). Assert that the decoded token at that index ends the city name.
- **Multi-token answers:** score by the full-sequence log-prob of each of the 50 capital strings and take the argmax (preferred), and verify it agrees with greedy decoding on validity.
- **Dataset hygiene:** exclude capital cities themselves, cities whose name contains a state name, and ambiguous names that exist in several states.
- **Validity filter:** an instance (city, template) is valid only if the unpatched model's top capital is correct. Report the validity rate per model and template.
- **Few-shot templates** are needed because zero-shot fails on many cities. Few-shot example states must never appear as source or target states in the same run.

---

## 5. Phases

After each phase, append to `LAB_NOTEBOOK.md` and commit. If a phase's result **breaks the premise**, stop scaling up, write it up clearly, and move to the diagnostic step described in that phase.

- **Phase 0: Dataset + validity.** 50 states/capitals, ~8–12 unambiguous cities per state (≥ 15 states with ≥ 6 cities), ≥ 4 templates (2 zero-shot, 2 few-shot) plus held-out template(s) never used to build interventions. Run the validity filter per model.
- **Phase 1: Replicate and scale A and B** on Qwen2.5-1.5B with the few-shot template, across ≥ 10 (source, target) pairs, disjoint dev/test cities, plus a held-out template. Output per-layer flip rate with CIs and the handoff layer. Measure cross-pair transfer (does a Texas→California vector move Florida cities toward Sacramento?). **Premise check:** if held-out flip < 50% at every layer for most pairs, diagnose (template, position, all-city-tokens) and try the next model before concluding.
- **Phase 2: Meaning and specificity.** State-belief question (`"Q: Which US state is {city} in?\nA: The state of"`), unrelated city facts (country; top-token-unchanged rate and KL), generic text KL on ~50 sentences at a matched position, other-state leakage.
- **Phase 3: Hop separation (candidate hard tier).** Output the target capital while the state-belief answer stays the original state. Search layer × position (final token before handoff, "capital" token, city token at late layers) and intervention types (mean-diff at those positions, DAS-style rank-1 swaps). Report whether achievable, where, how surgical.
- **Phase 4: Difficulty ladder.** Classes: C0 full residual paste (reference; disallowed), C1 mean-diff (naive), C2 gradient-optimized vector (± KL penalty), C3 rank-1/rank-k DAS subspace swap, C4 black-box prompting, C5 optional SAE feature clamping (Gemma Scope). Constraints: layer ceiling ∈ {5, 10, 15, handoff−1}; position ∈ {city last, all city tokens}; KL threshold; number of dev examples ∈ {0, 2, 5, 20}; whether target-state examples are provided. **Key output:** a table of which settings make C1/C2/C4 fail while a deliberate strategy still succeeds → the medium tier.
- **Phase 5: Scale comparison.** Rerun Phases 1–2 (and key Phase 4 cells) on Qwen2.5-3B, Gemma-2-2B, 7B. Tabulate validity, layer band, handoff as fraction of depth, specificity.
- **Phase 6: Environment prototype.** Instance generator with reference validation; grader outside the agent sandbox scoring held-out flip × (1 − specificity penalty) and verifying constraints; agent tool API with forward-pass budget; JSON submission; if an API key is set, a minimal Claude agent loop on 10–20 instances per tier reporting pass rate.

## 6. Rules of engagement
- Do not overclaim. State n and CIs everywhere.
- Run the obvious controls before trusting any positive result (random vectors of matched norm, same-state sources, shuffled labels).
- If something looks too clean, look for a bug (off-by-one position, leaked held-out city, answer string in the prompt).
- Keep runs cheap: smallest model and small n to debug, then scale.
- Stop and write up if blocked. Don't thrash.
- Deadline: Oct 8 call needs pitch-ready numbers.

## 7. Final deliverable: `FINDINGS.md` (≤ 2 pages)
1. Premise verdict (models, layer bands, n, CIs)
2. Meaning and specificity
3. Difficulty ladder table (naive-fails / careful-succeeds settings = the medium tier to pitch)
4. Hop separation
5. Scale
6. Reward-hacking risks observed and how constraints block them
7. Recommended environment spec (instance format, grader formula, difficulty knobs, expected pass-rate band)
8. Three plots worth showing on the call (PNGs)
9. Open problems / next steps
