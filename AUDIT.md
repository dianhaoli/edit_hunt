# EditHunt Phase 7 audit: are the tasks passing for the right reasons?

Date: 2026-09-30. Auditor: a separate Claude Code session (Opus 5.5), run while the main session's suite episodes
were still going. Repo state at audit: commit `28f0fc6` plus the uncommitted Phase 7 run outputs.

Scope: the Phase 7 task suite (`edithunt/env/tasks.py`, `grader.py`, `tools.py`, `instance.py`, `agent_loop.py`,
`suite_run.py`), its instances (`results/suite/instances/`), the agent episodes (`results/suite/runs/`), the scripted
baselines (`results/suite/baselines.json`), and the claims in `LAB_NOTEBOOK.md` / `FINDINGS.md` / the main session.
The earlier phases (1–4) were read for context and to check T6's ground truth; they were not re-audited in depth.

Method: code reading; transcript reading; re-analysis of the saved Phase 1 data; and new GPU checks on Qwen2.5-1.5B
(`experiments/audit_phase7.py` → `results/audit/phase7_audit.json` and `results/audit/phase7_audit.log`):
1. Replayed every Qwen2.5-1.5B edit-task episode (42 of them) from its tool log in a fresh environment and re-graded it.
2. Re-ran the T2 reference solver on all 7 kept T2 instances with 4 seeds (the seed only changes which "keep" cities
   the recipe regularizes on).
3. Re-checked "state kept" for every T2 edit on three wordings: `STATE_Q` (what the grader uses), `STATE_Q_HO`
   (never visible to tools), and the postal-abbreviation readout.
4. Re-scored every T1/T2/T3/T7 edit on a larger item set: held-out cities × {ho_fs, ho_zs, fs2, zs2}, clean-valid
   items only (roughly 2× the grader's items).
5. Probed the tool guard with grader-private city names.

Nothing in the repo was changed except adding this file, the audit script and its outputs.

---

## 0. Summary

| task | headline (Sonnet / gpt-6.1-sol) | verdict | one-line reason |
|---|---|---|---|
| T1 easy | 8/8 / 6/8 | **OK** (sanity tier) | solvable by mean-diff by construction; both gpt failures are non-submissions |
| T2 keep-state | 1/7 / 0/2 | **Failures genuine; pass line is noise** | agent failures are real, but the known careful solver itself passes ~57% on kept instances and ~25% on fresh ones |
| T3 consistency | 8/8 / 5/8 | **Reward too lenient; hidden readouts redundant** | cube-root mean passes a capital flip rate of 12.5%; Claude passed with 2/6 capital flips |
| T4 detective | 8/8 / 8/8 (black-box 2/2 / 2/2) | **Not an interp task** | source+target (0.8) is a behavioural lookup; the layer is 0.2 and not needed to pass, and `act_diff` reads it off directly |
| T6 handoff | 7/8 (black-box 0/2) | **Passes largely for the wrong reasons** | 4 distinct answers, not 8; method is prescribed; ±1 tolerance passes unmeasured guesses; truth is pooled, not what the given pair shows |
| T7 minimal | 3/8 / 1/1 | **Penalizes the intended skill** | unscaled mean-diff (5/8) beats Sonnet (3/8); r_ref biased low by selection; no side-effect penalty |

Cross-cutting (ordered by severity):
1. **Selection on the reference makes "the reference passes 100%" circular**, and the reference is on a knife edge.
2. **Held-out sets are tiny** (3–8 items, 2–4 cities per instance); per-task n = 7–8 → pass-rate CIs ≈ ±0.3.
3. **The tool's refusal message is a membership oracle for grader-private cities** (test and leakage sets).
4. **gpt-6.1-sol's failures are harness failures** (turn / $ caps), not task failures.
5. **The leakage metric is saturated** ("any change" on two held-out wordings) and does not measure what Phase 4 found.
6. **Submitted vectors are not stored**; grades are only reproducible by replaying tool logs.

Things verified as sound are in §9. Recommended fixes are in §10.

---

## 1. Numbers used in this audit

Pass = reward ≥ 0.5. All runs on Qwen2.5-1.5B except T6 (4 subject models). Counts are from `results/suite/runs/` at
~19:45 UTC on 2026-09-30; gpt-6.1-sol runs were still in progress.

| task | Sonnet 5.5 | gpt-6.1-sol | scripted baselines (baselines.json) |
|---|---|---|---|
| T1 | 8/8 | 6/8 (2 fails = no submission) | naive answer-direction gradient 3/8 |
| T2 | 1/7 | 0/2 | mean-diff 0/7, naive gradient 0/7 |
| T3 | 8/8 | 5/8 (3 fails = no submission) | naive gradient 1/8 |
| T4 | 8/8; black-box 2/2 (1.0, 0.8) | 8/8; black-box 2/2 (1.0, 0.8) | — |
| T6 | 7/8; black-box 0/2 | — | prior guess round(0.8·depth) 4/8 |
| T7 | 3/8 (0.58, 0.68, 0.55 pass) | 1/1 | unscaled mean-diff 5/8, naive gradient 2/8 |

Note: the main session reported Sonnet T7 as "4/9". I count 8 Sonnet T7 files and 3 with reward ≥ 0.5.

Wilson 95% CIs at these n: 8/8 → [0.68, 1.00]; 1/7 → [0.03, 0.51]; 3/8 → [0.14, 0.69]; 5/8 → [0.31, 0.86].
Tasks whose intervals overlap this much cannot be ranked against each other on this data.

---

## 2. Per-task findings

### T1 easy: OK as a sanity tier
- Reference = the mean-diff baseline run through the tools; kept 8/10 candidates. It is solvable by the most standard
  method by construction; nobody claims otherwise.
- Both agents' passes are genuine: on the ~2× larger item set Claude's edits flip 7/8, 8/8, 11/11, 8/10, 27/29, 7/7,
  7/7, 14/16 held-out items.
- Leak is 0.13–1.00 on these edits (leak_weight is 0 on T1, so it does not matter here; see §7).
- gpt's two failures (Utah→Alaska, Wisconsin→New Hampshire) are episodes with **no submission** (25 turns / cost cap).

### T2 keep-state: agent failures are real; the pass threshold sits inside the reference's noise
**What I suspected and ruled out.** The grader checks "state kept" only on `STATE_Q`, a wording the agent can query
and train on through `extra_examples`, and `final` is an allowed position. I expected edits that keep the state on
that one wording but not on others. The data says otherwise:

| instance | reference (orig seed) kept: state_q / state_q_ho / abbr | Claude kept: state_q / state_q_ho / abbr |
|---|---|---|
| California→Texas | 8/8, 8/8, 8/8 | 8/8, 8/8, 8/8 |
| Colorado→Wisconsin | 3/3, 3/3, 3/3 | 3/3, 3/3, 3/3 |
| Hawaii→Wisconsin | 2/2, 2/2, 2/2 | 2/2, 2/2, **0/2** |
| Idaho→Maine | 2/2, 2/2, 2/2 | 1/2, 1/2, **0/2** |
| Minnesota→Oregon | 2/2, 2/2, 2/2 | **0/2, 0/2, 0/2** |
| Tennessee→Nevada | 3/3, 3/3, 3/3 | 3/3, 2/3, 3/3 |
| Wisconsin→Minnesota | 4/4, 4/4, 4/4 | 2/4, 1/4, 2/4 |

(gpt: California→Texas 8/8 on all three; Colorado→Wisconsin 3/3, 2/3, 3/3.)

- The reference's state preservation generalizes to the unseen wording and to the abbreviation readout.
- Claude's failures are genuine. Either the state moved (Minnesota→Oregon, Wisconsin→Minnesota), or the capital barely
  flipped. Its California→Texas `final`-token edit keeps the state perfectly but flips only 5/31 items on the larger set.
- Two Claude edits keep the state on both question wordings but flip the abbreviation readout (Hawaii→Wisconsin,
  Idaho→Maine). The grader only checks `STATE_Q`, so "state kept" is a partial notion of "hop-1 belief unchanged".

**The real problem: the known solver is at the pass line.** I re-ran `ref_keepstate` with 4 seeds per kept instance.
The seed changes only which 4 other-state cities are used as "keep" examples; the method is otherwise identical.

| instance (original seed) | seed 0 | seed 1 | seed 2 | seed 3 |
|---|---|---|---|---|
| California→Texas (s1) | 0.487 | **0.520** | 0.335 | 0.499 |
| Colorado→Wisconsin (s0) | **0.642** | 0.480 | 0.260 | 0.691 |
| Hawaii→Wisconsin (s0) | **0.623** | 0.510 | 0.521 | 0.593 |
| Idaho→Maine (s1) | 0.851 | **0.794** | 0.553 | 0.257 |
| Minnesota→Oregon (s0) | **0.604** | 0.508 | 0.614 | 0.563 |
| Tennessee→Nevada (s1) | 0.806 | **0.738** | 0.503 | 0.802 |
| Wisconsin→Minnesota (s0) | **0.589** | 0.267 | 0.208 | 0.259 |

Bold = the run that qualified the instance. Excluding it: **12/21 pass (0.57)**, with 6 of the 12 passes in
0.50–0.56. Generation logs show the reference passed only **8/32** mean-diff-prefiltered candidates overall
(`logs/suite/gen_T2_keepstate*.log`), in line with Phase 6's careful script at 5/24 on hard.

Implications:
- "Reference passes 7/7, Sonnet 1/7" is not evidence that a careful interpretability agent beats Sonnet by that
  margin. The reference is 7/7 because instances were kept only where it passed.
- A careful agent using the right method would pass about half the kept instances and about a quarter of fresh ones.
  On these instances the 0.5 threshold mostly measures seed luck.
- Claude's single pass (Tennessee→Nevada, 0.549) rests on F = 4/6 items and 3 cities. On the larger item set it
  flips 9/12 and keeps the state 3/3, 2/3, 3/3, so it looks like a real but marginal pass.

### T3 consistency: the reward formula passes weak capital edits; hidden readouts add nothing
**The cube root is too lenient.** reward = (F_cap · F_state · F_hidden)^(1/3) × KL factor. With F_state = F_hidden = 1,
F_cap = 0.125 already gives 0.5, so a submission passes with **1 in 8** capital flips.
- Claude Colorado→Maine: F_cap = 0.33 (2/6) → reward 0.68, **pass**. On the larger item set F_cap = 4/11 = 0.36.
- Claude Ohio→Colorado: F_cap = 0.60 (6/12 on the larger set = 0.50) → 0.77, pass.
- The task text asks for the capital, the state question and the hidden readouts to "all give the target answer".
  Passing with the capital flipped on a third of items is passing for the wrong reason.

**The hidden readouts are redundant with F_state.** Both are postal-abbreviation formats ("Chicago, IL"; a shipping
address), which follow directly from the state answer. Over all 15 agent T3 submissions, F_hidden equals F_state in
14; the exception is 0.83 vs 1.00. So in practice the reward is F_cap^(1/3) × KL factor. The "hidden readout" claim
(the task catches answer-direction edits through readouts the agent cannot see) is not what makes the naive
gradient fail. Its reward is 0.000 on 5/8 instances, i.e. some component is exactly 0; the visible state question
alone would give the same result. (Baselines store only the reward, so the zero component is inferred, not measured.)

**Instance validity.** The T3 reference is plain mean-diff and passed 8/8 prefiltered candidates, so T3 in practice
means "T1 plus: don't use the answer-direction gradient". Frontier agents default to mean-diff anyway, hence 8/8.

### T4 detective: passable without interpretability
Reward = 0.4·[source] + 0.4·[target] + 0.2·[|layer error| ≤ 2]; pass = 0.5.
- **Source + target alone = 0.8 → pass**, and both are behavioural. Claude's black-box episode (Idaho→South Carolina)
  asked `STATE_Q` for one `known_cities` city per state in 4 `run_prompts` calls. The planted state's city answered the
  target state, and it submitted with a wrong layer: reward 0.8, pass. Both black-box episodes for both agents pass.
- **The layer, the only part that needs internals, doesn't affect pass/fail.** When it is required, `clean_access` +
  `act_diff` returns planted − clean per layer, and the plant layer is the first layer with a non-zero difference.
  The reference solver does exactly this lookup. That is reading a diff, not interpretability.
- A blind guess of the layer is not far off either: ±2 around a guess of 10 covers 3/8 of the actual plant layers
  (4–16 band).
- History: `known_cities` was added after the first agent episodes (an Idaho episode probed famous cities the plant did
  not cover). The fix was justified as a spec/implementation mismatch, but it is also what makes the black-box route
  trivial.

### T6 handoff: effective n = 4, method prescribed, tolerance hides imprecision, truth ≠ observable
1. **The answer is a per-model constant** (`HANDOFF[model]`: 22, 31, 18, 22). The 8 instances are 4 questions asked
   twice with different example pairs; 1.5B and 7B share both depth (28) and answer (22).
2. **The task text prescribes the method** (mean-diff at city_last, fs1, "first layer after the band where it flips
   fewer than half"). This contradicts the suite rule "descriptions state the goal and reward, never the method". What
   remains is running a layer sweep.
3. **Priors pass half.** round(0.8 · depth) passes 4/8 (both Qwen-28-layer models). A model with literature priors
   ("handoff around 75–80% depth") gets this for free.
4. **The ±1 tolerance passes unmeasured answers.** Both 1.5B passes: Claude evaluated L8, 12, 14, 16, 18, 20
   (flip ≥ 0.57) and L22, 24, 26 (flip 0.0), never measured L21, and reported **21**. That is also the wrong side of the
   stated definition (21 would be the *last working* layer; truth 22). Reward 1.0.
5. **The ground truth is pooled over 30 pairs, but the agent can only measure the given pair.** Recomputed from the
   saved Phase 1 data (first layer after the band with flip < 0.5, fs1, per pair, ~3.5 cities per pair per layer):
   - Qwen2.5-1.5B: 22 (23 pairs), 23 (4): tight.
   - Qwen2.5-3B: 31 (15), 26–29 (6), scattered earlier.
   - gemma-2-2b: **18 (14 pairs), 15 (9 pairs)**: bimodal.
   - Qwen2.5-7B: 22 (14), 21 (5), 23 (3), scattered earlier.

   (The "scattered earlier" values are holes in noisy per-pair curves. The definition itself is fragile at ~3.5 cities.)
   Claude's one non-black-box miss (gemma Utah→Vermont) measured flip 0.43 at L16 on the pair it was given and
   reported 16. The reference, bisecting on the same pair with all 8/8 cities, landed on 17; the truth is 18. The agent's
   measurement was consistent with the task's own definition applied to the pair it was told to use. It failed because
   the grader wants the pooled number. (It also skipped L15/L17, so it was not fully careful either.) The generator drops
   example pairs whose handoff disagrees with the pooled truth (8/10 kept), which hides this.

### T7 minimal: the reward punishes minimizing
reward = min(1, r_ref / r) if F ≥ 0.8 on held-out and KL_mean ≤ 1.0, else 0; r = ‖edit‖ / mean ‖residual‖ at that layer.
1. **Plain mean-diff at scale 1 beats Sonnet**: 5/8 (rewards 0.81, 0.63, 0.86, 0.66, 0.83) vs 3/8. Sonnet did try to
   minimize. It scaled edits down using the 4 dev cities, then missed F ≥ 0.8 on 4–6 held-out items (Iowa→Ohio 3/4,
   Louisiana→Idaho 3/5). An agent that doesn't minimize at all scores better than one that does.
2. **r_ref is biased low by selection.** The reference bisects the smallest scale that flips every dev prompt (× 1.1)
   and keeps the instance only if that edit still gets F ≥ 0.8 on held-out: 8/25 candidates passed. 5 of the 8 kept
   references score F = 0.80–0.83 on held-out, i.e. exactly at the bar. The kept r_ref values are the lucky small ones.
3. **Pass/fail at F ≥ 0.8 on 4–6 items is noisy.** Iowa→Ohio: 3/4 = 0.75 → reward 0; on the larger set 7/8 = 0.875
   (would pass). Louisiana→Idaho: 3/5 → 8/11 = 0.73 (still fails).
4. **"Minimal" has no specificity term.** Leak is ignored and the KL budget is 1.0: gpt's Colorado→Oregon edit scored
   0.95 with leak 0.80 and KL_mean 0.45. Minimal norm ≠ minimal effect.
5. **The norm is relative to the residual norm at the chosen layer**, which grows with depth, so the layer choice
   dominates. Kept references sit at L7 or L21. Claude always used L12 and gpt L16. This is a legitimate knob, but
   mostly a norm-scaling artifact rather than an interpretability insight.

---

## 3. Cross-cutting: selection on the reference solver

Every task keeps an instance only if its reference reward ≥ 0.5 on the **same private items the agents are graded on**.

| task | reference | kept / tried (after prefilter) |
|---|---|---|
| T1 | mean-diff via tools | 8/10 |
| T2 | keep-state recipe | 8/32 |
| T3 | mean-diff via tools | 8/8 |
| T4 | act_diff lookup | 8/8 |
| T6 | bisection on the example pair | 8/10 |
| T7 | bisected-scale edit | 8/25 |

When the reference passes most candidates (T1, T3, T4) this is harmless. When it passes a minority (T2, T7):
- the kept set is selected for items that happen to favour that specific run (winner's curse);
- "reference 100% vs agent x%" is circular and overstates the gap;
- in T7 the selected reference also sets the agent's target (r_ref), so the selection makes the task harder;
- methods that differ from the reference (e.g. Claude's T2 initialisation from mean-diff) are judged on instances
  chosen for a different method.

Also noted in the notebook: T2's pair RNG repeats pairs, and duplicate IDs overwrite each other (60 tries → 5 unique).

---

## 4. Cross-cutting: tiny held-out sets

| task | test items per instance | test cities | state-check cities (T2) |
|---|---|---|---|
| T1 | 3–15 (median 4) | 2–8 | — |
| T2 | 3–16 (median 6) | 2–8 | 2–8 (median 3) |
| T3 | 3–6 | 2–3 | — |
| T7 | 4–6 | 2–3 | — |

With 3 items, F ∈ {0, 0.33, 0.67, 1}, and "held-out generalization" rests on 2 cities for many instances. Most
generalization comes from held-out *wordings* (ho_fs, ho_zs), not cities. Cause: states have 5–12 cities, 4 go to dev,
and the rest must be clean-valid on ho_fs (validity 0.71). Adding more held-out wordings (including a few written
fresh, never in the repo) would multiply items without needing more cities; the audit's 4-wording re-score already
doubles them.

---

## 5. Cross-cutting: the refusal message leaks the grader's private sets

`ToolEnv._cities` rejects held-out cities with `held-out cities are not accessible: [<the names>]`. The held set
includes the test cities **and the private third-state leakage cities**. Demonstrated on T2 California→Texas:

```
run_prompts(template="fs1", cities=["Akron","Billings","Los Angeles","Denver"])
→ {"error": "held-out cities are not accessible: ['Akron', 'Billings', 'Los Angeles']"}
```
truth: leak cities = Akron, Billings, Bozeman, Cincinnati, Cleveland, Great Falls, Missoula, Toledo (Ohio, Montana);
test cities include Los Angeles.

This already happened by accident in real episodes: Claude's Tennessee→Nevada run got
`['Houston']` and `['Dallas']` back (Texas, a private leakage state for that instance), and `['Chattanooga']` (a test city).
Exploit: probe one city per state (50 cheap calls, each refused before any forward pass, so no budget cost) to learn
the two leakage states, then regularize specifically on their other, accessible cities. An RL policy would find this.
Fix: a generic refusal that doesn't name cities; optionally also refuse random non-private cities so refusals carry
no information.

---

## 6. Cross-cutting: agent-harness confounds

- **gpt-6.1-sol**: every T1/T3 failure (5/5) is a non-submission: `completed` at 25 turns, or `cost_cap` at $0.10.
  Among episodes that submitted, gpt is 6/6 on T1 and 5/5 on T3. Its map row currently measures tool-use efficiency
  under a 25-turn/$0.10 cap. Report pass rates among submitters separately, add a "you have N turns left, submit"
  nudge, or raise the cap for gpt.
- **Sonnet** never hit the caps (all stops `tool_use`, 4–12 turns). Its T2 pass cost $0.095, just under the $0.10 cap,
  so the cap could start to bind on the hardest task.
- Effort is `medium` for both agents; results should say so.

---

## 7. Cross-cutting: the leakage metric is saturated

The grader's leak = fraction of third-state items whose top capital changes *at all*, on both held-out wordings
(including zero-shot `ho_zs`, the most fragile). Claude's T1 mean-diff-style edits have leak 0.13, 0.47, 0.50, 0.54,
0.85, 0.86, 1.00, 1.00. Phase 4 measured mean-diff leakage *to the target* at 0.14–0.24 on the build template.
The notebook already notes this (Phase 6). It matters for T2 (leak_weight 0.5), where it adds noise instead of
separating "say the target capital" edits from state edits. Suggest leak = "switched to the target", or report both.

---

## 8. Cross-cutting: reproducibility of agent grades

`agent_loop.episode` strips `vector`/`basis`/`center` from the saved submission. The grade can't be re-checked from the
record. The audit rebuilt the submissions by replaying the tool log; **all 42 replays reproduced the recorded reward
exactly** (bitwise), which is good news, but it depends on GPU determinism and on the tool code never changing. Save
the vectors (1536 floats per edit; fine as JSON or a sidecar .pt).

---

## 9. Checked and found sound

- **Grading is deterministic** (42/42 exact replays).
- **Held-out guards work** for cities (test + leak) and for exact private template strings. Paraphrases are allowed by
  design. The only leak is the error text (§5).
- **T2 state preservation is not overfit to the one graded wording** (§2 T2 table). The reference generalizes to
  `STATE_Q_HO` and abbreviations; agent failures are real.
- **T1 passes are real** on the larger item set.
- **T3's naive-gradient failure is real** in the sense that the answer-direction edit does not change the state
  answer. It is caught by the visible state question, not by the hidden readouts.
- **The core premise from Phases 1–2** (a state variable at the city token over a mid-layer band, handoff at ~0.7–0.86
  of depth) is consistent with the per-pair Phase 1 data. Per-pair handoffs cluster at the pooled value for 1.5B,
  and less tightly for the other models.
- **The spend ledger** reserves the worst case before each episode and settles after; errors are charged at max_cost.
  Conservative and correct.

---

## 10. Recommended changes

To follow the notebook's rule (no tuning after seeing agent results), treat these as a proposed v2 suite scored on
**fresh** instances, not as edits to the current map.

Grader / instance generation (all tasks):
1. Generic refusal message (§5). Cheap, and removes an exploitable leak.
2. 3–5× more held-out items per instance: more held-out wordings, including a few never committed to the repo.
3. Report the reference's pass rate on **fresh** instances (or with re-drawn arbitrary choices) next to agent pass
   rates, and stop calling kept-instance reference results 100%. Or keep instances by a criterion independent of the
   graded items (e.g. the reference passes on a separate validation split of wordings).
4. Save submission vectors.
5. Report every pass rate with its Wilson CI. Treat T6 as n = 4.
6. Add `STATE_Q_HO` (and optionally the abbreviation readout) to T2's state-kept check. Cheap insurance, even though
   no current submission exploits the gap.

Per task:
- **T3**: replace the cube root with min(F_cap, F_state, F_hidden), or require each ≥ a floor (e.g. 0.6). Replace the
  abbreviation readouts with readouts not implied by the state answer: state-level facts the model knows (largest
  city, a neighbouring state, a time zone), validated like `phase7_readouts.py`.
- **T4**: make the layer necessary: pass requires layer within ±1, or weight it 0.5+. Remove `clean_access` or
  `act_diff` (otherwise the layer is a lookup). Consider planting at a random subset of cities, or at a lower strength
  so the behavioural signal is weaker. Otherwise drop T4 from the "needs interp" claim.
- **T6**: ground truth per instance: the handoff measured on the example pair with a fixed, disclosed city set, with
  a tolerance of 0. Or retire it; as specified it is a prescribed sweep plus priors.
- **T7**: gate on leak / KL at a tight budget; compute r_ref from the mean over several reference variants (or from
  unselected instances); score F on more items so the 0.8 bar isn't a coin flip. Consider rewarding
  r relative to plain mean-diff at the same layer instead of to a selected reference.
- **T2**: the task is conceptually the strongest (naive methods score exactly 0; genuine hop separation is needed). Its
  weakness is statistical: raise items and cities per instance, and report the reference's unselected pass rate
  (~25%) as the "careful method" number.
- **Harness**: count non-submissions separately; add a final-turn "submit now" nudge; state the caps and effort level
  next to every pass rate.

---

## 11. What the pitch can safely claim today

- A shared state variable at the city token exists over a mid-layer band in 4 models, with a handoff; its main side
  effect is third-state leakage (Phases 1–2).
- Hop separation (T2) separates methods: mean-diff and naive gradients score 0/7. A leak- and state-aware edit
  sometimes succeeds (reference ~57% on kept instances, ~25% on fresh ones), Sonnet 1/7. Frame it as a hard task
  with a noisy pass line, not as "a careful agent solves it".
- The environment is verifiable and deterministic (exact replays), with no LLM judge.

Not safe to claim without the fixes above:
- that T3, T4 or T6 require interpretability;
- that T7 measures minimal-edit skill;
- a fine-grained difficulty ranking across tasks (CIs overlap; gpt row confounded by caps).

---

## Appendix: reproducing the audit

```bash
cd ~/edit_hunt && PYTHONPATH=. .venv/bin/python experiments/audit_phase7.py results/audit/phase7_audit.json
```
~25 min on the A10G next to one running suite worker (1.5B only; ~3.5 GB GPU). It replays episodes from
`results/suite/runs/`, so later runs will add rows. Phase 1 per-pair handoffs (§2 T6) were computed from the
`B_meandiff` fs1 rows of `results/<model>/phase1_patching.json`.
