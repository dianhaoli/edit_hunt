# How agents succeed on T8 (undo a planted edit without the original model)

Scope: 13 analysed episodes (12 gpt-6.1-sol, plus luna through the contrasts), 6 paired contrasts, and the full T8 run population (`results/suite/runs/*/T8_undo*/`, `t8_undo/failure_audit.jsonl`, `pilot*.jsonl`). The PASS bar is 0.8 × solver. In every transcript the assistant turns are tool calls only, with no prose and with reasoning removed. Every "quote" below is therefore a tool argument or a tool output, and any statement about the agent's intent is inferred from what it called.

An important caveat about the audit: `failure_audit.jsonl` regrades only the **last 6 distinct candidates** of each episode (`experiments/t8_failure_audit.py` ~l.119). So "first passing candidate" and "best candidate" are lower bounds on what the agent actually had.

---

## 0. Population context

| Set | Runs | Submitted | PASS | Ended at cap | Cap hit while holding a passing candidate (audit) |
|---|---|---|---|---|---|
| v0.1 sol (r1: hidden 60-turn cap) | 4 | 0 | 0 | 4 (turns) | 3: hop2 0.930, state 0.896, spread 0.742 (bar 0.729) |
| v0.1 sol (b150) | 4 | 4 | 1 (hop2 0.961) | 0 | – |
| v0.2 sol (r1/r2) | 16 | 8 | 4 | 8 (cost $0.90) | 4: ae523eee.r1 0.783, 675b9fb4.r1 0.768, d18af9bd.r1 0.910, d18af9bd.r2 0.931 |
| v0.2 luna | 8 | 4 | 0 | 4 (cost) | none graded |

On v0.2 sol, an oracle that submitted the best evaluated candidate would pass 9 of 16 runs (the 4 actual passes, the 4 cost-capped runs, and 651dcd13.r2 at 0.964). The agent was credited with 4. **More than half of the competence the agent showed is lost at submission time.**

---

## 1. Episodes analysed

R = reward; Best = best audited candidate (a lower bound, see above); bar = 0.8 × solver.

| Episode | Category | Instance kind | R / Best / Solver (bar) | Winning or best edit | Route | First passing candidate (turn, fwd, $) |
|---|---|---|---|---|---|---|
| v0.1 `hop2-L8.b150` | clean_pass | hop2 MN→TN L8, 1 gate | **0.961** / 0.967 / 0.983 (0.786) | `ufinal`, L8, all, −55, self-key lo0/hi55 | L7→L8 update anomaly (affected − control), pooled over 25 cities × 4 templates | Earliest graded: T39 (~376 fwd, ~$0.2). Ungraded calls with the same behaviour at T26–T33 |
| v0.2 `57572593.r1` | clean_pass | same | **0.921** / 0.959 / 0.983 | `undo60 = 0.6·undoopt + 0.4·(−55v)`, self-key v lo2/hi57 | same anomaly, then optimize_vector (hurt) | T10 (196 fwd), ungraded but cos 0.986 with vsub0 (0.955). Earliest graded: T52 |
| v0.2 `57572593.r2` | clean_pass | same | **0.978** / 0.982 / 0.983 | `restore = norm(0.6k3 + 0.4k8)`, L8, all, −55, self-key lo0.3/hi55.3 | anomaly + within-city high/low-context contrasts | **T10, 231 fwd, ≈$0.07** (graded 0.9645) |
| v0.2 `651dcd13.r1` | clean_pass | hop2 NC→KY L8 | **0.928** / 0.975 / 0.965 (0.772) | `(optB+optC)/2`, self-key `−normalize(optB)` lo1/hi43 | L8−L7 deltas, pooled `undoBig`, then optimize (hurt) | T18 (224 fwd), ungraded. Earliest graded: T87 (0.975) |
| v0.2 `ae523eee.r2` | pass (thin) | **spread** NC→KY L8/9/10 | **0.796** / 0.813 / 0.956 (0.765) | `restore8`, L8, city_last, 0.95, key `keyDL` lo−8/hi20 | L23 detour, then NC−KY distance and j8 persistence → L8, then optimize | T69 (7.0k fwd) graded; earlier ones ungraded |
| v0.2 `651dcd13.r2` | passing cand. not credited | hop2 NC→KY | 0.762 / **0.964** / 0.965 | best: `pc4`, L8, all, −46, self-key lo2/hi48. Submitted: `learnundo` (optimized) | double-diff PCA of L8−L7 jumps | T35 graded (585 fwd); pc1 at T27 ungraded |
| v0.2 `ae523eee.r1` | not credited (cost cap) | spread NC→KY | 0 / 0.783 / 0.956 | `undo8` (optimized), L8, all, 0.7, key nk8 | behavioural plus diff-built gate | T67 (0.766) |
| v0.2 `675b9fb4.r1` | not credited (cost cap) | state NC→KY L16 | 0 / 0.768 / 0.905 (0.724) | `plantavg·1.124`, L16, all, key fit22 | L16 delta-of-deltas plus hand-built regression gate | T44 graded (653 fwd) |
| v0.2 `d18af9bd.r1` | not credited (cost cap) | state MN→TN L16 | 0 / 0.910 / 0.841 (0.673) | `plant2`, L16, all, −1, key kg | inc16 spike, Gram-Schmidt key | T71 graded (≈$0.37) |
| v0.2 `d18af9bd.r2` | not credited (cost cap) | state MN→TN L16 | 0 / **0.931** / 0.841 | `−32·pc`, L16, all, key fitkey2 lo0/hi32 | inc16 spike, PCA, CG ridge gate | T76 graded; T61 ungraded |
| v0.1 `state-L16.b150` | best_partial | state MN→TN L16 | 0.563 / 0.563 / 0.841 | `u`, L16, all, −32, key gateM | right layer and direction (cos 0.98) but gate fit only on Dakota spill cities | none |
| v0.1 `two-L8.b150` | best_partial | two: MN→TN L8 + CA→PA L14 | 0.593 / – / 0.867 (0.694) | `v3`, L14 only, −1.2, key gate6 | only the L14 plant was found | none |
| v0.2 `6c7213a5.r2` | best_partial | two (same) | 0.514 / 0.552 / 0.867 | `refine2`, L14 only | only the L14 plant was found, then over-optimized | none |

---

## 2. The winning recipe

Every clean pass uses the same five steps. The hop2 L8 instances give the cleanest examples.

**Step 1. Find symptoms behaviourally, using spill-over cities.** The Minnesota and North Carolina test cities are private, and a batch that contains one is refused or silently skipped. The gate leaks onto neighbouring states, so sweeping run_prompts across one or two cities per state finds the target answer on accessible cities.
- `57572593.r2` (T3): `"city":"Fargo","next_tokens":[[" Nashville",0.8488]` and Sioux Falls `["Nashville",-0.01]`
- `hop2.b150` msg 22: `"city":"Sioux Falls","next_tokens":[[" Nashville",0.9945]`
- `651dcd13.r1` msg 18: `"city":"Cary","next_tokens":[[" Frank",0.998],...`

The successful runs never needed to name the source state. That works on hop2 because the gate adds the same vector v wherever it fires, so a vector estimated on leaked cities is still v (cos 0.93–0.97).

**Step 2. Locate the layer by model diffing without the original model.** For every layer, take the per-layer residual update of the affected group minus that of the control group, `(aff_L − aff_{L−1}) − (ctl_L − ctl_{L−1})`, and look for the one-layer spike.
- `57572593.r2`: `"expr":"dak_L8 - dak_L7 - ctl_L8 + ctl_L7"`, then vec_info `"dak_L7":-0.02,"dak_L8":0.611`
- `57572593.r1`: jump8 norm 38.2 vs jump7 13.4
- state L16 runs: `inc16` norm 21.38 vs inc15 4.39 and inc17 5.07 (`d18af9bd.r1`)
- `651dcd13.r1` msg 24: `"register":"dr","norm":46.0666 ... "register":"dm","norm":11.9552`. The excess of about 40 matches |v| = 40.8.

**Step 3. Take the plant vector from the same diff, then clean it.** The anomaly vector is the plant estimate. Averaging it over templates and over larger affected sets raises cos(−v) from about 0.92 to 0.97. Agreement between independent estimates was the agent's model-free quality check, e.g. `57572593.r1` `cos(jstate,jump8)` = 0.985 and `hop2.b150` cos(vc, vbest) = 0.968. The best variants cancel city identity with within-city context contrasts (`57572593.r2` k3/k8) or with pooled controls (`651dcd13.r1` `undoBig`, R 0.975, above the solver).

**Step 4. Undo with a self-keyed clamp gate, scaled from the saturation plateau.** The core edit is `{register: u, layer: L, position: "all", scale: −s, scale_from: {key: u, lo: ~0–2, hi: ~s}}`, which is approximately `−(u·h − lo)·u` and amounts to projection removal along the plant's output direction. Projecting h onto u at layer L shows a bimodal pattern: about 0 on controls, and a plateau near |v| on saturated cities. That plateau sets s.
- `57572593.r2` msg 20 (verified): `"Fargo":35.478,"Grand Forks":53.922,"Minot":53.848,"Dickinson":14.545,"Sioux Falls":53.76...`
- The first edit of that form, call 29 at T10 (verified): `{"register":"key","layer":8,"position":"all","scale":-55,"scale_from":{"key":"key","lo":1,"hi":56}}`, graded **0.9645**.

The key has cos only 0.07–0.14 with the true gate key. It works because it reads the plant's own write, which is present in the post-plant residual. Audit: gate_affected 0.94–0.999, gate_others 0.10–0.22.

**Step 5. Validate on the planted model.** The checks were:
- (a) affected cities flip to a plausible capital, e.g. `651dcd13.r1` msg 36: `{"city":"Cary","clean":"Frankfort","edited":"Raleigh"} ... "kl_generic":0.0027`
- (b) controls are unchanged and kl_generic is about 0.001
- (c) for hop2, state_q answers are unchanged (changed_rate 0.0)
- (d) the projection contrast is clean, and independent estimates agree

Here "clean" in eval_intervention means the *planted* model. kl_on_these_prompts measures distance from the planted model, not restoration. These proxies separate a broken edit from a working one. They **do not rank** working candidates. In two runs they ranked candidates backwards:
- `57572593.r1`: the optimized blend looked best by the proxies and scored 0.921, while vsub looked worst and scored 0.959.
- `651dcd13.r2`: `learnundo` looked "perfect" and scored 0.762, while pc4 scored 0.964.

No clean pass used optimize_vector to produce its winning direction. In the three runs that applied it as a final polish, it lowered true R every time: 0.959→0.921, 0.975→0.928, 0.964→0.762. The cause is that its targets are real-world answers rather than the original model's answers. In `651dcd13.r2` it trained `('fs1','Statesboro','Atlanta')`, while the base model's Georgia answer tends to be Augusta. That rotated the vector away from −v (cos 0.931→0.844) and made the norm 16% too large.

---

## 3. What separates success from failure (ranked by causal weight)

1. **Submitting at all. (Environment, with a skill component.)** This is the largest single factor. Nine of 16 v0.2 sol runs and 4 of 4 v0.1 r1 runs ended at a cap, and seven of those held a passing candidate when they stopped. Caps are hidden: in v0.1 the turn cap, in v0.2 the $0.90 cap, while turns_left is shown. `d18af9bd.r2` ended with `"budget_left":10699,"turns_left":48` after holding 0.931 since T76.
   - The v0.1 contrast shows this decided the outcome. The 150-turn pass had *not* submitted by T60 either, and held 0.945 at T40.
   - The skill part is that no run banks its best-so-far candidate.
   - The NC spread contrast shows the pass itself was luck: r2's submit turn was the turn that crossed the cap ($0.9081). r2 stopped because its *forward-pass* budget ran out, not because it knew about the cost cap.
2. **Selecting the candidate without the original model. (Skill.)** Over-refinement after a pass can turn a pass into a fail (`651dcd13.r2`, a 0.20 loss). It also costs margin in most runs. Submitted R is below best R in 8 of 8 submitted v0.2 runs.
   - The successful r1 on 651dcd13 used gentle anchored optimization (lr 0.002, kl_weight 2, unchanged-city anchors), averaged two runs, and compared them on a damage proxy (state_q KL 0.12). r2 used lr 0.01, kl_weight 1 and max_norm 55, and never compared its candidates head to head.
3. **Doing model diffing before optimizing. (Skill.)** Luna (0/8) chose layers by how well optimize_vector worked on a single city (Grand Island, layer 10, last token only). It found the jump at L8 (`43.230`) only at T58. Sol found it at T5. The behavioural library patch scores 0.0 on this instance, so the optimize-first route probably cannot pass.
4. **Getting the source/target pair and the population right. (Skill, made harder by the environment.)** The failures that "fix the wrong thing" are of this kind.
   - `d5fb4ef0.r1` (MN spread, 0.027) fitted the Dakota side effects at L12 and never queried an accessible Minnesota city. The Minnesota state-question KL stayed at 8.42→8.42. The environment made this harder: Minneapolis and Duluth, the first two listed cities, are private.
   - `state-L16.b150` had the right layer and direction but gated on Dakota cities, giving R_src 0.498.
   - Both two-plant runs never looked for a second plant.
   - On hop2, by contrast, the leaked vector *is* the plant vector, so getting the population wrong does no harm.
5. **Speed of detection. (Mostly variance or luck.)** Batch-error handling (sol split the list into batches of 16, luna truncated it, which dropped Fargo and Sioux Falls) and lucky control choices (Great Falls turned out to be affected) move the first-pass turn by 10–24 turns. This mainly matters through factor 1.

---

## 4. Efficiency

**What a pass actually needs.** `57572593.r2` reached 0.9645 at T10 with 231 forward passes, roughly $0.05–0.07. It submitted 0.978 at T46 for $0.36 and 1,168 forward passes. The other hop2 passes show the same pattern:
- `hop2.b150`: the working edit form existed by T26–T39, and T40–T89 added +0.02.
- `651dcd13.r1`: working by T18, then 80 more turns that left R about the same or lower.

**Forward passes are never the constraint.** Usage was 1–25% of 12,000, except for optimize_vector-heavy runs. Dollars go on **re-reading context**: 4.1–4.5M cache-read tokens in the capped runs. Cost per turn grows with the length of the transcript.

**Where the money goes:**
- **Hand-rolled linear algebra one vec_op per turn.** Gram-Schmidt keys `fit1..fit39` (`675b9fb4.r1`, about 60 of 103 turns), CG ridge regression (`d18af9bd.r2` about 45 turns at 3 turns per CG step, `6c7213a5.r2` about 57 turns), and `q0..q48` interpolation keys (`d18af9bd.r1`). These cost no forward passes, but each step re-sends the whole context. This is the mechanism behind the cost caps: `675b9fb4.r1` used 1,248 forward passes and $0.91.
- **optimize_vector runs.** About 1.2–2.4k forward passes each, and they usually lowered R.
- **Wrong-layer detours.** L23 readout in `ae523eee.r2` (~25 turns); late final-position diffs in v0.1 runs.
- **Wrong leads and refusal friction.** Georgia/Dahlonega in `ae523eee.r1` (15 turns); about 8–15 turns of "not accessible" re-splitting in v0.1.
- **Wide multi-call turns with 16-city outputs.** These inflate context growth: about 4.1k vs 2.6k characters per turn in the NC spread pair.

**Estimate.** A disciplined run needs roughly 10–20 turns, under 500 forward passes and $0.10–0.15 to reach a passing hop2 or state edit. That is about 15% of the $0.90 cap. The other 85% buys at most +0.01–0.03 R, and quite often a negative change.

---

## 5. Why the recipe does not transfer to spread and two-plant instances

- **Spread (gates at L8/9/10, each carrying ⅓ of the shift):**
  - There is no single-layer spike. Diffs at the final position point to the L23 readout instead, which is the same trap detect_v2 falls into (it picks L23).
  - An L8-only undo can cancel at most the first third directly. The best runs got about 0.78–0.81 (bar 0.765), with cos(−v) only 0.27–0.51, so they relied on optimized behaviour rather than geometry.
  - Getting the source state right is essential. `d5fb4ef0.r1` scored 0.027 after fixing Dakota side effects. v0.1 `spread.r1` reached 0.742 (bar 0.729) with a behaviourally optimized L5 city_last vector (layer_ok false, cos 0.23), which shows the margin can come from behaviour rather than mechanism.
  - Library detect_v3, which diffs over a layer range, gets 0.820. The best agent candidate is about the same or slightly below.
- **Two-plant (MN→TN L8 + CA→PA or WI→UT L14):**
  - In 4 of 4 sol runs, and in the library too, the plant whose gate leaks widely was found, and the plant whose salient cities are private was missed. Neither agent searched again after the first plant was fixed, although the task says "at one or more layers".
  - Clues were visible and ignored:
    - `two.b150` msg 56: `"city":"Fargo","next_tokens":[[" Nashville",0.1664]]`. Fargo was then used *as a control*.
    - Accessible MN/WI cities (St. Cloud, Mankato, Kenosha, Eau Claire, ...) were never queried.
  - Best partials:
    - MN/CA instance: 0.593 (b150; L14 direction cos 0.977, the best of any run), 0.609 (r1 candidate), and 0.552 / 0.514 (r2).
    - NC/WI instance: 0.366 / 0.298. The leaky L8 NC gate is harder, and its plant carries a smaller share of the KL, about 0.44 of the ceiling.
  - Even a perfect single-plant fix caps restoration_src at roughly the plant's city share (about 0.68 or 0.44), so a single-plant fix cannot pass.
- **State L16 (single plant, cost-capped):** the recipe *does* transfer. 0.91–0.93 beats the solver at 0.841, but no run submitted. The v0.1 b150 state run failed only because its gate was fit to the spill population.

---

## 6. Agent vs scripted library

| Instance | Best library (pilot*.jsonl) | Agent best submitted / best candidate |
|---|---|---|
| hop2 MN→TN L8 | detect_v3 0.932, v1 0.930 | 0.978 / 0.982 |
| hop2 NC→KY L8 | detect_v3 0.883 (v1/v2 0.064 at L27) | 0.928 / 0.975 (> solver 0.965) |
| state MN→TN L16 | detect_v1/v2 0.904 | 0 credited / 0.931 (> solver) |
| state NC→KY L16 | detect_v3 0.850 | 0 / 0.768 (below library) |
| spread NC→KY | detect_v3 0.820 | 0.796 / 0.813 (≈ library) |
| two MN/CA | detect_v1 0.493 (L14 only) | 0.593 / 0.609 |
| two NC/WI | detect_v1 0.41 | 0.298 / 0.384 (below library) |

The agent adds value in four ways:
1. **Robust localization.** It finds L8 on NC→KY hop2, where detect_v1/v2 pick L27.
2. **Cleaner direction estimates.** Within-city context contrasts and pooled controls give cos(−v) of 0.96–0.97.
3. **Self-sizing projection-removal gates** that also remove spill (damage_unaffected about −56).
4. **Edits that beat the privileged solver** on 3 instances.

It falls behind where hand-built gate regression is too tight (675b9fb4: gate_affected ≤ 0.58 vs the library's plant-style ramp), and on multi-plant instances. Libraries take about 300 forward passes and zero judgement. The agent's advantage only reaches the score when it submits.

---

## 7. Implications

**For T8 v0.3 (environment):**
- **Show `cost_left`, or make turns the binding cap.** This alone would likely have doubled the v0.2 sol pass count (4 → about 8–9 of 16).
- **Consider auto-crediting the best evaluated candidate or the last submittable one when a cap is hit, or allow provisional `submit` that can be overwritten.** This separates "can solve" from "manages a hidden budget".
- **Make the audit grade all distinct candidates, not the last 6.** The current turn-to-first-pass numbers are only upper bounds.
- **Drop or reweight script-solvable instances.** hop2 MN→TN is already marked "drop (script-solvable)". Discrimination lives in spread, two-plant and leaky-gate instances.
- **Report the private-city structure honestly.** Silent skips and whole-batch refusals (v0.1) cost turns and hide the source state. Either return per-city refusals or say which states have accessible cities.

**For prompts and tools:**
- **Prompt:**
  - "Submit a provisional answer as soon as one candidate restores affected cities with low generic KL; refine afterwards."
  - "Your proxies measure distance from the planted model. optimize_vector toward real-world answers can move away from the original model."
  - "There may be several plants. After fixing one, sweep the accessible cities of other states again."
  - "When a state's listed cities are private, probe its other listed cities."
- **Tools:**
  - **A batched linear-algebra op** (least squares, Gram-Schmidt or PCA over a register list in one call). This removes the dominant cost sink.
  - **A per-layer anomaly scan helper** (affected vs control update norms across all layers in one call).
  - **A compact output mode for 16-city results.**

---

## Verification notes

I spot-checked three claims against the raw files. All three held, and nothing had to be corrected:

1. **`57572593.r2`: the first gated candidate is tool_log[29] at T10, the projection plateau is about 54, and the submission is −55 at L8/all.**
   - tool_log[29] is `eval_intervention` with edits `{"register":"key","layer":8,"position":"all","scale":-55,"scale_from":{"key":"key","lo":1,"hi":56}}`, `used: 231`.
   - msg 20 contains `"Fargo":35.478,"Grand Forks":53.922,"Minot":53.848,"Dickinson":14.545,"Sioux Falls":53.76`. This also confirms the skeptic's correction that the analyst's quote had skipped Dickinson.
   - submission_edits: layer 8, all, scale −55.0, scale_from present.
   - Run end: 46 turns, submitted, $0.3619, R 0.9776.
2. **`651dcd13.r2` submitted early, under the bar, with turns left.** The final tool output is `{"status":"submitted","n_edits":1,"budget_left":9595,"turns_left":72}`. Run: 78 turns, $0.7681, R 0.7623 (bar 0.7724). Submitted scale 1.0 at L8/all with a key vector. Audit: best 0.9643, class SKILL:other.
3. **Cost-cap population claims.** `d18af9bd.r2`: end=cost_cap, 102 turns, $0.9104, R 0, audit best 0.9308. Across all T8 runs, 8 of 16 v0.2 sol runs ended at cost_cap; the four with audited best above the bar are ae523eee.r1, 675b9fb4.r1 and d18af9bd.r1/r2. Luna was 0/8. The v0.1 r1 runs are all `max_turns` at 60.

During the population check I also found that v0.1 `spread-L8.r1` had a passing candidate (0.742 against a bar of 0.729: `undo5`, L5, city_last, layer_ok false). It is included in sections 0 and 5. No analysed anatomy covered it.

Remaining caveats:
- Turn and forward-pass figures for "first passing candidate" are bounded by the 6-candidate audit.
- Per-turn dollar figures are estimates, because only episode totals are logged.
- All motive claims are inferred from tool calls.

Sources:
- /home/ec2-user/edit_hunt/results/suite/runs/gpt-6.1-sol/T8_undo{,_v02}/
- /home/ec2-user/edit_hunt/results/suite/runs/gpt-6-luna/T8_undo_v02/
- /home/ec2-user/edit_hunt/results/suite/t8_undo/failure_audit.jsonl
- /home/ec2-user/edit_hunt/results/suite/t8_undo/pilot.jsonl
- /home/ec2-user/edit_hunt/results/suite/t8_undo/pilot_v02.jsonl
- /home/ec2-user/edit_hunt/LAB_NOTEBOOK.md (~940–1120)

Helper dumps are in /tmp/claude-1000/-home-ec2-user/85aea9e9-ef02-4e0e-9015-c2daed111312/scratchpad/.