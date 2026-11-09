# What I found, and how it maps to CTGT (Research Intern: Interpretability & Reliability)

Sources for my results: `edit_hunt` (FINDINGS.md, FINDINGS_RAVEL_SCREEN.md) and `ideating-rl-tests` (docs/STATUS_2026-10-04.md,
docs/results/*). All numbers below are held-out, copied from those write-ups. CTGT facts come from search snippets of
their pages (ctgt.ai, YC, HN could not be fetched directly), so check them against the live pages before quoting.

---

## Part 1. What CTGT actually does

**Lineage.** The posting says "born out of Stanford research". Every other source points to UC San Diego (HDSI) instead:
CEO Cyril Gorlla lists Mikhail Belkin as his advisor and has published on EigenPro / NTK / **Recursive Feature Machines
(RFM)**. "Founding research stems from feature learning in neural networks" almost certainly refers to the Belkin-lab RFM /
AGOP line of work. Check which is right before you write either institution.

**Feature extraction and steering (open weights).**
- The closest public method is *Toward universal steering and monitoring of AI models* (Beaglehole, Radhakrishnan,
  Boix-Adserà, Belkin; arXiv 2502.03708, Science 2026). It is from Belkin's lab, but nothing shows a CTGT co-author.
  - Method: at each block, fit an RFM that predicts a concept from activations. The top eigenvectors of the
    **AGOP** (average gradient outer product) become concept vectors.
  - Those vectors are used both to *steer* (add them) and to *monitor* (detect hallucination or toxicity).
  - Cost: under 500 examples and under a minute on one GPU. That fits "no new artifact per behavior".
- CTGT's own paper: *A Feature-Level Approach to Mitigating Bias and Censorship in DeepSeek-R1* (Gorlla & Tuttle).
  - It finds a late-layer direction `f_censor` that separates censored from answered outputs.
  - At inference it applies `h ← h − α·proj_{f_censor}(h)`.
  - Full answers went from 32% to 96–100% on 100 sensitive queries, with a claimed negligible effect on reasoning.
  - This is directional ablation, the same family as refusal-direction work.
- The postings name **control vectors** (steering) and **activation probes** (monitoring).

**Policy Engine / Mentat (works for closed models too).**
- Mentat is an OpenAI-compatible endpoint that combines feature-level intervention (open weights) with
  **graph-based policy verification**. The verification part needs no weights, which is how they cover closed APIs.
- Policies are compiled into a graph that decides which rules apply to a given output. It resolves conflicts between
  rules and writes an audit trail.
- HaluEval claim (self-reported): GPT-OSS-120B goes from 92.7% to 96.5% with CTGT, against base scores of 95.9% (Gemini 3
  Pro) and 95.1% (Claude 4.5 Opus). The margins are small, there is a refusal confound, and their pages give
  inconsistent numbers.
- They also open-sourced **Concordance**, which traces why agent pipelines give nondeterministic outputs.

**What the intern does (from the posting).** Find what a model represents, intervene on it, and **bound how wrong the
system can be**. Make it work repeatably across model families. Evaluations should bound the error, not average it.
The bar is "a calculation that has to be right, not a demo that usually works".

---

## Part 2. My findings, mapped to that

### A. EditHunt: steering one feature without collateral damage (strongest match)

**1. I found a linear feature and steered it at runtime, in 4 models from 2 families.**
- The edit: a difference-of-means vector (target state minus source state), added at the city's last token.
- It flips held-out two-hop answers ("The state containing Dallas has its capital in" → *Sacramento*) at a rate of
  0.86–0.91. Random vectors of the same norm flip ≤0.05. The models: Qwen2.5-1.5B/3B/7B and gemma-2-2b.
- It works over a mid-layer band and stops at a sharp **handoff layer**, at 0.69–0.86 of depth. After that, the
  information has moved to the final token.
- *CTGT link:* this is literally "extract a feature and steer it at runtime with a control vector", done "repeatably
  across model families". The handoff result says *where* an intervention can work at all.

**2. I showed it is a state variable, not an answer direction.**
- The same vector flips a *different* question ("Which state is X in?") at 0.95–0.98.
- Projecting the state direction out of the vector removes the whole effect.
- Generic text is unchanged: KL stays at the random-vector level.
- *CTGT link:* "a better way to find what a model represents". You can test whether a direction *means* the concept
  rather than just moves one output.

**3. Optimizing for the output finds a shortcut direction.**
- A vector trained by gradient to "output Sacramento" flips 0.97–1.00, but it also **leaks 100%**: it makes *every*
  city answer Sacramento. Its norm is 15× the mean-diff vector.
- Capping its norm still leaves 56–83% leakage.
- Only adding explicit "keep" terms fixes it: flip 0.83–0.97, leakage 0.07–0.12.
- **Leakage is the axis that separates methods; average success does not.**
- *CTGT link:* this is their reliability problem in miniature. A steering vector tuned for a behavior can learn "always
  do the behavior". Mean-diff steering also has a side effect on other inputs: other states' cities leak 13–34%.

**4. A gated edit (probe plus steer) is the existence proof for specificity (T2 v2, RAVEL-style).**
- A single fixed additive vector cannot isolate the change. That is structural: it moves every city.
- The fix was `x ← x + clamp((k·x − b)/w, 0, 1)·v`. A linear **probe** `k` decides *whether* to fire, and a
  **control vector** `v` decides *what* to do.
- Result: isolation went to ≥0.94 (other states' cities untouched), and no shortcut came close. Out of 400 shortcut
  runs, the best scored 0.51, against an oracle up to 0.91.
- The bottleneck is the **probe's generalization from limited data**. With a detector fit on 4 dev cities, only
  ~28% of instances validate (50% at late layers). Forcing the gate open gives Cause 0.96–1.00, so the steering vector
  itself works.
- Moving the threshold trades isolation for coverage about one for one.
- *CTGT link:* this is the closest thing I have to their product.
  - Activation probe plus control vector equals a conditional runtime policy.
  - The failure mode I measured is exactly their "calibration under limited ground truth": a monitor fit on a few
    examples misses the tail.

**5. Training a subspace with "keep" objectives beats both lazy methods and SAEs (RAVEL screen, gemma-2-2b).**
- The tasks: change a city's Language (or Continent) while keeping its country and continent (or language).
- Multi-attribute DAS (MDAS) with keep terms scores 0.65–0.86, passing 4/4 seeds.
- DAS on its own leaks *more* the longer it trains (0.43 → 0.25). SAE-latent edits reach 0.22, mean-diff 0.12–0.42.
- Some pairs are inseparable by every method (state ↔ capital ↔ abbreviation; country ↔ language).
- *CTGT link:* SAE features are not automatically the right handles for steering. Supervised, targeted extraction
  with disentangling objectives can do better, which is close to their supervised, RFM-style philosophy.

### B. ideating-rl-tests: measuring whether interventions and claims are real

**6. LatentKnockout: "flipped" is not "removed" (SAE latent ablation, gemma-2-2b).**
- For 30 of 43 concept groups, ablating ≤5 SAE latents breaks the answer.
- But on held-out prompts the right answer leads by only a median **1.75 logits**, and 45–86% of the "flipped" answers
  are still in the top 3.
- Requiring a ≥0.5 logit loss makes only 49% of pairs feasible with the study's reference search.
- **Famous concepts (Texas, California, soccer, English) cannot be knocked out with ≤5 latents, while less famous ones
  can.** This is plausibly feature splitting or redundancy that grows with frequency.
- A simple gradient×activation ranking finds the right latents as well as a careful search does.
- *CTGT link:* this is "bound error rather than average it". A censorship-removal or ablation claim based on top-1
  flips overstates removal. The famous-concept redundancy is a concrete reliability risk for feature-ablation products
  such as their R1 paper.

**7. ShiftHunt: removing a spurious cue from a probe with SAE latents (SHIFT, Marks et al.).**
- The task: a linear probe on gemma-2-2b secretly relies on a surface cue. The agent must name the cue and remove the
  reliance with ≤20 latent ablations.
- The grader is exact (58/58 regrades identical) and every cheap script scores 0.
- Pass rates: Opus 5.5 75% and gpt-6.1-sol 70% on the hard tier; Haiku and luna 0%.
- The misses are long-tail cases: the cue also travels through context latents.
- *CTGT link:* they monitor with **activation probes**, and a probe that keys on a spurious cue is the main way such a
  monitor fails. This is a working method for auditing a probe and repairing it.

**8. FeatureMatch: SAE latent labels mostly reflect writing style.**
- A script with no hypothesis gets 87% of a reference's accuracy just by feeding styled texts through the model.
- Only 7% of candidate latents kept their label on everyday-style text; the rest fired on encyclopedia formatting.
- *CTGT link:* explanations of "what a feature represents" are fragile out of distribution. You have to validate them
  by intervening, not by reading them.

**9. Model diffing: a behavioral claim grader that measured wording, not content (stopped by a pre-registered kill).**
- An auditor's correct claims scored 0.28–0.49 in its own words, against 0.72–0.74 in the answer key's wording.
- Re-grade SD was 0.155 (target ≤0.07), and refutation 0.84 (target ≥0.90).
- Two causes:
  - Option log-prob effects saturate when the base model already prefers the claim.
  - Fine-tunes flatten the output distribution, so any on-topic wrong option gains.
- *CTGT link:* checking whether a model's claims are true is the core of hallucination evals like HaluEval and of
  their verification layer. I have concrete failure modes for likelihood-based verification: wording dependence,
  ceiling effects, and too few effective tests (about 1–2 per claim).

**10. ShiftHunt v3: an exact tool set made planted mechanisms scriptable.**
- Planted modules combined by OR or AND were recovered by scripts that never named the cue. A script given the cue
  names did no better than one without (33% vs 35%).
- *CTGT link:* when you claim to have found a mechanism, test it against a no-understanding baseline. Otherwise you
  are measuring the tools, not the understanding.

### C. Method habits that match "a calculation that has to be right"
- Wilson CIs everywhere. Held-out entities **and** held-out wordings. Private evaluation sets.
- Shortcut and null baselines: zero edit, random, mean-diff, naive gradient, SAE, black-box.
- Pre-registered gates and kill criteria, with independent skeptic audits. Several projects were stopped as written-up
  negative results instead of being dressed up.
- Cross-fitted privileged "ceiling" oracles to measure the remaining headroom. Re-grade and seed noise reported next to
  each result. Fixes to grader artifacts (for example, the KL term penalizing a *correct* change in P(US)), with every
  earlier run re-graded after each fix.

---

## Part 3. What to submit, and how to frame it

**The artifact.** I would lead with EditHunt §A3–A4: *steer one internal variable without collateral damage*.
- One-sentence version: "In 4 open models I found the linear 'which state' variable a model uses for two-hop
  questions. I showed that output-optimized steering vectors learn an 'always say X' shortcut that leaks onto 100% of
  other inputs. A probe-gated control vector restores ≥0.94 isolation, and the remaining failure is the gate's
  calibration from few examples, which I quantified."
- Why it matters for them: it is their probe-plus-control-vector product in miniature, measured by its tails
  (leakage) rather than its average.

**Second point, if there is room:** "flipped ≠ removed" (§B6). The margin-based removal metric and the
famous-concept redundancy speak directly to bounding how wrong a feature-ablation system is.

**A small extension that would speak their language (about 1–2 days on the existing code):**
1. Add **RFM/AGOP concept vectors** (2502.03708) as one more extraction method in the EditHunt ladder. Compare them
   with mean-diff, DAS and gradient on flip rate **and** leakage, plus calibration of the AGOP vector used as a
   monitor.
2. Or make the T2 gate a calibrated monitor. Report the gate's false-open and false-closed rates with conformal or
   Wilson bounds as a function of the number of dev examples (4 → 18). This turns the "probe generalization" bottleneck
   into an error bound.

**Questions worth asking them:**
- How do you measure side effects of a steering or ablation intervention on inputs outside the policy's scope?
  (That is my leakage axis.)
- Does R1 censorship removal hold under a margin-based metric, and on famous or high-frequency topics?
- For closed models, how is the graph verifier's error bounded when the ground truth itself can't be checked?
