# T8 transcript analysis (32 episodes)

Part 1: failures. Part 2: successes. Generated 2026-11-11 from results/suite/runs/*/T8_undo*/ (read-only).

## Why agents fail T8_undo: a report across all 32 episodes (critic-corrected)

### Summary

**Were transcripts tracked?** Partly. Every tool call and every tool result was saved, but the agents' reasoning was not. Across the 32 episodes there are 7,680 assistant items, and every one is a `function_call`; there is no assistant text at all. The models did reason: each episode used 3.1k to 28k reasoning tokens (`usage.reasoning_tokens`). But `edithunt/env/agent_loop.py` line 328 drops those items on purpose: `log.append({"role": "assistant", "content": [it for it in items if it.get("type") != "reasoning"]})`. So every "why" below is inferred from tool arguments and results, not read from the agent's own words.

**Outcomes.** 5 of 32 episodes pass (16%).
- gpt-6.1-sol: 5/24 (v0.1: 1/8; v0.2: 4/16).
- gpt-6-luna: 0/8.
- No `state` episode (0/8) and no `two` episode (0/8) passed.

**Main finding.** Half of all episodes (16/32) ended at a cap the agent could not see, with no submission:
- All 4 v0.1 `.r1` runs hit the hidden 60-turn cap.
- 8 sol v0.2 runs and 4 luna runs hit the hidden $0.90 cost cap.

Primary causes break down as follows:
1. **Environment budget artefacts are the primary cause in 8 episodes.** In 7 of them, a sol candidate scoring above the pass bar had already been evaluated. In 4 of those 7 the agent's latest candidate also passed (robust). In the other 8 capped episodes, skill was the primary cause, and the cap only turned a failing partial score into 0.
2. **Skill gaps are the primary cause in the remaining 19 failures:**
   - The second plant was never found in any of the 8 two-plant episodes (6/6 sol runs labelled INCOMPLETE_TWO_PLANT).
   - The source state was never correctly identified in 5 of 8 luna runs and 1 sol run. In all 5 of those luna runs the agent never queried a single accessible source-state city.
   - Gate modelling errors (2).
   - Downstream behavioural patching (2).
   - Wrong layer (1).
   - Choosing a worse candidate when there is no restoration signal (1).

**Environment and skill interact.** The source state's first-listed `known_cities` are private, and v0.2 skips them silently. This is what kept the missed plant in both two-plant instances invisible: no run ever queried an accessible city of that state.

---

### 1. Episode table (final primary cause after verification)

- R = graded reward.
- Bar = 0.8 × solver.
- "Best" = best evaluated candidate as regraded by `failure_audit.jsonl`; "last" = last evaluated candidate.
- No audit regrade exists for luna, MN state b150 or MN two b150.

| # | Episode | Model | Ver | Kind | R / solver (bar) | End | Final primary cause | One-line reason |
|---|---|---|---|---|---|---|---|---|
| 1 | MN-TN-hop2-L8.r1 | sol | v0.1 | hop2 | 0 / 0.983 (0.786); best = last 0.930 | max_turns 60 | ENV_TURNCAP_HIDDEN | Found L8 and a gated direction. Evaluated a 0.930 candidate in the final turn and was cut off before submitting |
| 2 | MN-TN-hop2-L8.b150 | sol | v0.1 | hop2 | **0.961** / 0.983 | submitted t90 | PASS | L8, position all, scale_from keyed on its own direction, cos(-v) 0.966 |
| 3 | MN-TN-spread-L8.r1 | sol | v0.1 | spread | 0 / 0.911 (0.729); best 0.742, last 0 | max_turns 60 | ENV_TURNCAP_HIDDEN (fragile) | An L5 city_last repair (call 155) scored 0.742, just above the bar. Later candidates scored 0.04-0.67. The last (`cvalue`, call 244) scored 0. Then the hidden cap |
| 4 | MN-TN-spread-L8.b150 | sol | v0.1 | spread | 0.144 / 0.911; best 0.231 | submitted t69 | WRONG_LAYER (gate co-primary) | Gated answer repair at L16. The hand-built gate opened only 0.22 on graded cities |
| 5 | MN-TN-state-L16.r1 | sol | v0.1 | state | 0 / 0.841 (0.673); best = last 0.896 | max_turns 60 | ENV_TURNCAP_HIDDEN | Passing L16 candidates at turns 48-49, then 11 turns of Gram-Schmidt and no submit |
| 6 | MN-TN-state-L16.b150 | sol | v0.1 | state | 0.563 / 0.841 | submitted t129 | WRONG_GATE_OR_POSITION | Direction and scale right (cos 0.98). The gate key was fitted exactly to a few Dakota cities: cos -0.16 with the true key |
| 7 | MN-TN-two-L8.r1 | sol | v0.1 | two | 0 / 0.867 (0.694); best 0.607, last 0.563 | max_turns 60 | INCOMPLETE_TWO_PLANT | Fixed only CA→PA at L14. Never looked for MN→TN at L8 |
| 8 | MN-TN-two-L8.b150 | sol | v0.1 | two | 0.593 / 0.867 | submitted t91 | INCOMPLETE_TWO_PLANT | Same as #7. Saw Fargo→Nashville at msg 56, then put Fargo into the "unaffected" baseline (msg 59) |
| 9 | T8u-57572593.r1 | sol | v0.2 | hop2 | **0.921** / 0.983 | submitted t72 | PASS | Mild selection loss: vsub (0.959) looked worse on argmax than undo60 |
| 10 | T8u-57572593.r2 | sol | v0.2 | hop2 | **0.978** / 0.983 | submitted t46 | PASS | First gated candidate (turn 10) already scored 0.9645 |
| 11 | T8u-651dcd13.r1 | sol | v0.2 | hop2 | **0.928** / 0.965 | submitted t99 | PASS | Preferred the optimize_vector blend over undoBig (0.975) on in-sample fs1 accuracy |
| 12 | T8u-651dcd13.r2 | sol | v0.2 | hop2 | 0.762 / 0.965 (0.772); best 0.964 | submitted t78 | SELECTION_ERROR | Replaced near-solver pc2-pc6 (0.961-0.964) with learnundo, trained on real-world capital labels. A later learnundo variant scored 0.775, at the bar within noise |
| 13 | T8u-675b9fb4.r1 | sol | v0.2 | state | 0 / 0.905 (0.724); best 0.768, last 0.714 | cost_cap t103 | ENV_COSTCAP_HIDDEN (fragile) | Passing candidates at calls 159/222 (0.760/0.768). Its last candidate (0.714) was below the bar |
| 14 | T8u-675b9fb4.r2 | sol | v0.2 | state | 0 / 0.905; best 0.665 | cost_cap t109 | WRONG_GATE_OR_POSITION | Linear, unsaturated gate (lo 0, hi 30) fitted by about 75 turns of hand-run CG. Never held a passing candidate |
| 15 | T8u-6c7213a5.r1 | sol | v0.2 | two | 0 / 0.867 (0.694); best 0.609 | cost_cap t108 | INCOMPLETE_TWO_PLANT | L14 only, plus a leaky gate. The cap turned about 0.61 into 0 |
| 16 | T8u-6c7213a5.r2 | sol | v0.2 | two | 0.514 / 0.867; best 0.552 | submitted t121 | INCOMPLETE_TWO_PLANT | L14 only (damage_unaffected +69). Grade decomposition: a perfect gate would still reach only about 0.60 |
| 17 | T8u-929ffed7.r1 | sol | v0.2 | two | 0.298 / 0.935 (0.748); best 0.366 | submitted t95 | INCOMPLETE_TWO_PLANT | L8 NC only. Its only Wisconsin probes were Milwaukee and Green Bay, both private and silently skipped. 10 of the 18 graded source cities are in Wisconsin |
| 18 | T8u-929ffed7.r2 | sol | v0.2 | two | 0 / 0.935; best 0.384 | cost_cap t111 | INCOMPLETE_TWO_PLANT | Same as #17 (only Milwaukee probed). Cut off mid-CG |
| 19 | T8u-ae523eee.r1 | sol | v0.2 | spread | 0 / 0.956 (0.765); best 0.783, last 0.740 | cost_cap t83 | ENV_COSTCAP_HIDDEN (fragile) | undo8 passed at turn 72. The agent was refining towards a 0.740 blend when the cap hit |
| 20 | T8u-ae523eee.r2 | sol | v0.2 | spread | **0.796** / 0.956 | submitted t93 ($0.908) | PASS (lucky) | L8 city_last only. Margin 0.031 against a grade boot_sd of 0.064. Crossed the cost cap on the submit turn; the submit still ran (see §3.7) |
| 21 | T8u-d18af9bd.r1 | sol | v0.2 | state | 0 / 0.841 (0.673); best 0.910, last 0.860 | cost_cap t123 | ENV_COSTCAP_HIDDEN | Five candidates scored 0.86-0.91. 51 serial Gram-Schmidt steps. Cap hit with turns_left 27 |
| 22 | T8u-d18af9bd.r2 | sol | v0.2 | state | 0 / 0.841; best 0.931, last 0.928 | cost_cap t102 | ENV_COSTCAP_HIDDEN | Six candidates scored 0.917-0.931. Cap hit during optimize_vector polishing, turns_left 48 |
| 23 | T8u-d5fb4ef0.r1 | sol | v0.2 | spread | 0.027 / 0.911; best 0.064 | submitted t57 | OTHER: source misidentified | Read the MN gate's spill onto the Dakotas as a Dakota→Carolina plant. Trained capital fixes at L12 |
| 24 | T8u-d5fb4ef0.r2 | sol | v0.2 | spread | 0 / 0.911; best 0.478 | cost_cap t81 | WRONG_METHOD_BEHAVIOURAL | Stacked optimize-to-answer patches downstream (L12-L20). Its L12 candidates, inside the audit's layer range, scored worse than its L20 one |
| 25 | T8u-57572593.luna | luna | v0.2 | hop2 | 0 / 0.983 | cost_cap t64 | INEFFICIENT_EXPLORATION | 25 turns chasing Grand Island, 30 turns of behavioural L10/11 work. Found L8 six turns before the cap |
| 26 | T8u-651dcd13.luna | luna | v0.2 | hop2 | 0 / 0.965 | cost_cap t63 | ENV_COSTCAP_HIDDEN (proximate; pass unlikely) | Only identified NC at turn ~49. Its city_last behavioural fix8final was never graded |
| 27 | T8u-675b9fb4.luna | luna | v0.2 | state | 0 / 0.905 | cost_cap t61 | WRONG_DIRECTION (source misidentified) | Locked onto Bentonville (AR) leaking toward Kentucky; all edits at L20/L27 for Arkansas. Its only NC probe was Charlotte, which is private |
| 28 | T8u-6c7213a5.luna | luna | v0.2 | two | 0 / 0.867 | submitted t45 | INEFFICIENT_EXPLORATION | "Bad" set = cities with wrong zero-shot capitals; L5 diff-of-means. Never found CA or MN. Its edit slightly raised KL (789.8→790.1) |
| 29 | T8u-929ffed7.luna | luna | v0.2 | two | 0 / 0.935 | submitted t47 | INEFFICIENT_EXPLORATION | Fixated on Homer→Ohio (p=0.19, base-model noise). Never queried any NC or WI city. Its edit was a no-op on graded prompts (kl_after == kl_before exactly) |
| 30 | T8u-ae523eee.luna | luna | v0.2 | spread | 0 / 0.956 | cost_cap t64 | WRONG_METHOD_BEHAVIOURAL | Per-city answer patches at L12 city_last with non-selective gates. Never tried L8-10 or position all |
| 31 | T8u-d18af9bd.luna | luna | v0.2 | state | 0 / 0.841 | submitted t45 | OTHER: source misidentified | Correct layer (L16), but took the Dakotas as the source; its only MN probe was Minneapolis (private). Forced Bismarck/Pierre: KL 282→461, damage_unaffected +118 |
| 32 | T8u-d5fb4ef0.luna | luna | v0.2 | spread | 0 / 0.911 | submitted t39 | INEFFICIENT_EXPLORATION | Abandoned the state sweep at Kansas, 6 states before MN (only MN probe: Minneapolis, private). Fixed base-model capital errors at L26 |

### 2. Failure taxonomy

#### Overall (32 episodes)

| Final primary cause | sol v0.1 (8) | sol v0.2 (16) | luna v0.2 (8) | Total |
|---|---|---|---|---|
| PASS | 1 | 4 | 0 | **5** |
| ENV_TURNCAP_HIDDEN | 3 | – | – | **3** |
| ENV_COSTCAP_HIDDEN | – | 4 | 1 | **5** |
| INCOMPLETE_TWO_PLANT | 2 | 4 | 0 | **6** |
| Source state misidentified (OTHER / WRONG_DIRECTION) | 0 | 1 | 2 | **3** |
| INEFFICIENT_EXPLORATION (never found the plant) | 0 | 0 | 4 | **4** |
| WRONG_METHOD_BEHAVIOURAL | 0 | 1 | 1 | **2** |
| WRONG_GATE_OR_POSITION | 1 | 1 | 0 | **2** |
| WRONG_LAYER | 1 | 0 | 0 | **1** |
| SELECTION_ERROR | 0 | 1 | 0 | **1** |

**Episodes ending at a hidden cap with no submission:** 16 of 32 (all 4 v0.1 r1; sol v0.2 8/16; luna 4/8). 8 of these have ENV as the primary cause. In the other 8 (#7, #14, #15, #18, #24, #25, #27, #30) a skill failure was primary.

**Overlapping labels.** "Source misidentified", "inefficient exploration" and "behavioural method" overlap. Six luna episodes carry INEFFICIENT_EXPLORATION, OTHER or WRONG_DIRECTION (#25, #27, #28, #29, #31, #32). Five of them (#27, #28, #29, #31, #32) never queried any accessible city of the source state. Those five probed only private cities (Minneapolis, Charlotte), which v0.2 silently skips, or none at all (#29).

#### By instance kind (all models)

| Kind | n | PASS | Env cap primary (passing candidate in hand?) | Skill primary |
|---|---|---|---|---|
| hop2 L8 | 8 | 4 (sol 4/6) | 2: #1 (yes, robust); luna #26 (no) | 2: #12 selection, #25 luna exploration |
| spread L8 | 8 | 1 (#20, NC) | 2: #3, #19 (yes, both fragile) | 5: #4, #23, #24, #32 (MN), #30 (NC luna). MN spread is 0/5 overall, and its only passing candidate (#3) was fragile |
| state L16 | 8 | 0 | 4: #5, #13, #21, #22 (yes; robust: #5, #21, #22) | 4: #6, #14 gate; #27, #31 luna source misidentified |
| two L8 | 8 | 0 | 0 | 8: 6 sol INCOMPLETE_TWO_PLANT, 2 luna exploration |

The state L16 row shows the clearest environment damage:
- Pilot detectors solve these instances (detect_v1 0.84-0.90; detect_v3 0.85).
- Sol built passing candidates in 4 of its 6 state runs, and lost every one to a hidden cap. MN state b150 has no audit row, so it is unknown whether it ever held one.

### 3. Recurring behavioural patterns, with transcript evidence

#### 3.1 How agents locate the plant: spillover, then a layer-delta scan

Every sol success found the plant the same way:
1. Notice wrong capitals on cities next to the source state, where the plant's gate leaks.
2. Compare residual increments (L_k − L_(k−1)) for those cities against controls.
3. Pick the layer where the increment spikes.

Evidence:
- MN-TN-hop2-L8.r1, msg 14: `"city":"Sioux Falls","next_tokens":[[" Nashville",0.7543]`
- T8u-57572593.r1, msg 8: `{"register":"jump8","norm":38.2249` (jump7 13.4, jump9 21.1)
- T8u-d18af9bd.r1, msg 16: `{"register":"inc16","norm":21.3783` (neighbouring layers about 4-5)
- MN-TN-state-L16.r1, msg 32: value `21.644844`

For a single-layer plant the scan is reliable: every one of the 12 sol hop2 and state runs reached the planted layer (6 hop2 at L8, 6 state at L16).

It has two failure modes:
- **The norm scan is not target-aware.** On MN spread it lands on L5, where the model naturally forms the state identity. Pilot detect_v1 and detect_v2 also pick L5 there; only detect_v3, which projects onto the target direction, solves it (0.841).
  - Evidence from MN-TN-spread-L8.r1: `vec_op({"name":"jump5",...})` gave norm 30.556.
  - Negating that increment pushed Minnesota cities elsewhere rather than restoring them (call #104): `{"city":"St. Cloud","clean":"Tennessee","edited":"Georgia"},{"city":"Mankato","clean":"Tennessee","edited":"Illinois"}`.
  - The agent kept working at L5 anyway.
- **Spillover shows where the gate leaks, not where the source is.** The successful sol runs worked from Dakota cities and never named Minnesota. T8u-57572593.r2 never probed any of the five accessible Minnesota cities (St. Cloud, Mankato, Eden Prairie, Moorhead, Bemidji); its only MN calls used Minneapolis and Duluth, both private. The same signal misled other runs:
  - T8u-d5fb4ef0.r1 read the leak as a Dakota→Carolina plant and wrote extra_examples such as `{"template":"fs1","city":"Fargo","answer":"Bismarck"}` (msg 37), even though its own output showed `" TN"` 0.36 for Minot (msg 92).
  - The luna runs on 675b9fb4 (Bentonville) and d18af9bd (the Dakotas) made the same mistake.

#### 3.2 Private source cities hide the source state (and the second plant)

Every graded test city is private. Several of them appear in `known_cities`, listed first for their state:
- MN: Minneapolis, Duluth.
- NC: Charlotte, Asheville, Chapel Hill, Winston-Salem, Outer Banks (5 of the 8 listed).
- WI: Milwaukee, Green Bay, Oshkosh, Racine, Appleton.

In v0.2 these cities are dropped and only a bare count comes back:
- T8u-d18af9bd.luna, msg 6: its one-city-per-state sweep batch returned `"skipped_not_accessible":2`, one of which was Minneapolis, its only Minnesota city.
- T8u-6c7213a5.r2, msg 160: `"skipped_not_accessible":1`, again Minneapolis. The agent never tried St. Cloud, Mankato or Bemidji.

Measured over all 32 runs, these episodes never ran a prompt on an accessible source-state city:
- Sol: MN hop2 r1, MN state b150, MN two r1 and b150, 6c7213a5 r1 and r2, 57572593 r2.
- Luna: 675b9fb4, 6c7213a5, 929ffed7 (no NC city at all), d18af9bd, d5fb4ef0.

In both two-plant instances, the missed plant's state was probed only through private cities, or not at all:
- MN in MN-two and 6c7213a5: only Minneapolis, in all five runs.
- WI in 929ffed7: Milwaukee and Green Bay (r1), Milwaukee (r2), nothing (luna).

No run ever queried an accessible Minnesota city in a MN two-plant instance, or an accessible Wisconsin city (Kenosha, Eau Claire, La Crosse) in 929ffed7.

v0.1 had the opposite problem. A whole batch was refused if it held a single private city, which produced 14-30 refusals per episode, e.g. `{"error":"not accessible","budget_left":11968}` (MN-TN-hop2-L8.b150).

Accessible source cities existed in every instance, so this is partly an agent coverage gap. But the environment's ordering and silent skipping make the gap very easy to fall into, and it plausibly contributes to the two-plant failures.

#### 3.3 Two-plant instances: no run ever found both plants (0/8)

Every sol run fixed whichever plant dominated its first contrast and never looked for another, even though the task text says the edit is added "at one or more layers" and max_rank is 2.
- MN-TN-two-L8.b150, msg 56: `"city":"Fargo","next_tokens":[[" Nashville",0.1664]],"top_capitals":[["Nashville",-1.79]]`. This is the L8 plant leaking. At msg 59 the agent put Fargo into its `"name":"low"` (unaffected) baseline.
- MN-TN-two-L8.r1, msg 20: `"city":"Tucson","next_tokens":[[" Pennsylvania",0.5888],[" Arizona",0.4047]`. This set the whole run on the CA→PA plant at L14.
- T8u-929ffed7.r1, msg 8: `"city":"Greensboro","next_tokens":[[" Kentucky",0.9984]`. It found NC→KY at L8, and its only Wisconsin probes hit private cities (§3.2).

There is also a structural ceiling:
- On T8u-6c7213a5.r2, the grade decomposition gives about 0.60 even with zero off-target leakage.
- The best single-plant scores are 0.607, 0.609 and 0.384, against bars of 0.694 and 0.748.
- The pilot scripts are single-plant too and also miss the second plant (detect_v1 0.49 and 0.42; detected_state California / North Carolina).

Even a perfect single-plant fix fails these instances.

#### 3.4 Spread instances: one layer at best

- **NC spread** passes only when the agent lands near L8. No run ever edited L9 or L10 (T8u-ae523eee.r2 submission diag: cos(-v) 0.51).
- **MN spread** fails because the agent anchors on L5 (natural state formation) or patches downstream.
  - T8u-d5fb4ef0.r2 msg 72 shows d8, d12 and d16 all flipping 4/4 cities. That is the "state of Minnesota minus state of Tennessee" difference forcing the answer, and it says nothing about where the plant is.
  - No sol run tried the target-aware cumulative anomaly over L8-10 that detect_v3 uses (0.841).

#### 3.5 Why luna fails (0/8)

- **It rarely localises the plant by model diffing.** It averaged 7.5 `cache_mean` calls per episode, against 51.3 for sol. It averaged 8.75 `optimize_vector` calls, against 3.5 for sol (luna range 2-22).
- **It often never looks at the source state.** 5 of 8 luna runs never queried an accessible source-state city (§3.2).
- **It mistakes base-model noise or spillover for the plant.**
  - T8u-929ffed7.luna, msg 14: `"city":"Homer","next_tokens":[[" Ohio",0.1868],[" Alaska",0.1284]]`. This is a one-off with p=0.19; the agent never checked how confident the answer was.
  - T8u-d5fb4ef0.luna, msg 27: `cache_mean {"name":"sus","cities":["Savannah","Sioux Falls","Boston","Concord","Morgantown","Rehoboth Beach"]`
  - T8u-675b9fb4.luna, msg 12: `"city":"Bentonville" ... "top_states":[["Arkansas",-3.17],["Kentucky",-3.29]`
- **It defines the objective as "make answers correct".** The tool error `"this task has no target capital: use extra_examples to define the objective"` triggers this; it appears in 7 of 8 luna runs and in no sol run.
- **It tried to reach a clean model.** In two runs (57572593 and ae523eee) luna called `run_prompts` with `"model":"clean"` and got back `only the planted model is available in this task` (e.g. ae523eee.luna msg 43→44). The tool schema invites this: `agent_loop.py` line 38 still advertises `model` as an enum `["planted","clean"]` in T8.
- **It costs more per turn.** About $0.0128 per turn overall ($0.0144 in its cost-capped runs), against sol v0.2's $0.0085 ($0.0089 in capped runs). Reasoning tokens average about 20k per luna episode against about 10.5k for sol v0.2: about 1.9× per episode and about 3× per turn. Luna is also priced at sol's rates as an upper bound (LAB_NOTEBOOK line 789). Four luna runs hit the hidden cap at 61-64 turns.
- **It stops early when it does submit.** Four luna runs submitted with 103-111 turns left, and none of those edits restored anything:
  - #29 was an exact no-op.
  - #28 and #31 slightly or strongly increased KL against the original model.

Even with no budget limits, I see no luna candidate that would plausibly pass. The one possible exception is 651dcd13 (fix8final, ungraded, city_last only), and that is unlikely.

#### 3.6 Choosing a candidate without the original model

The `eval_intervention` field `clean` refers to the planted model, and every KL figure the agent sees is measured against the planted model. So agents choose candidates by argmax "correct capital", which systematically favours behaviourally tuned vectors.
- T8u-651dcd13.r2 trained on `{"template":"fs1","city":"Statesboro","answer":"Atlanta"}` (first at msg 111, again at 113), although Savannah's clean capital in this model is Augusta. The agent saw the drift (msg 120, `"value":0.842194`, cos to pc1) and submitted anyway: 0.762, against pc4's 0.964.
- T8u-57572593.r1, msg 132: the best geometric candidate looked worse on argmax, `{"city":"St. Cloud","clean":"Nashville","edited":"Harrisburg"}`. Its R was 0.959; the submitted one scored 0.921.

Across the audit rows, optimize_vector refinements lowered R relative to the plain contrastive direction in 651d r1, 651d r2, 5757 r1 and ae52 r1. In 5757 r1, undoopt alone scored 0.821, the lowest of its 6 candidates.

#### 3.7 Budget management: serial hand-written linear algebra plus caps the agent cannot see

The sol cost-capped runs spent 40-75 turns fitting gate keys with one-step `vec_op` Gram-Schmidt or conjugate-gradient iterations:
- MN-TN-hop2-L8.r1 msgs 51-79: `"expr":"normalize(proj_out(proj_out(proj_out(proj_out(proj_out(proj_out(n6_L8-n6_L7,normal),u1),u2),u3),u4),u5))"`
- `{"error":"expression too long"` appears in 8 sol runs: MN two r1; 675b r2; 6c72 r1/r2; 929f r2; d18a r1/r2 (some of these more than once).

Each turn re-sends the whole context (about 4M cache-read tokens in a capped run), so later turns cost more.

The prompt mentions only turns: "at most 150 turns (assistant responses, each may contain several tool calls); every tool result reports turns_left. Submit before you run out: no submission scores 0." (v0.2 describe). Nothing in the prompt mentions cost. v0.1 runs also had a cost cap ($0.60 for r1, $0.90 for b150), but it never bound.

How the cap works (`agent_loop.py` lines 329-356): cost is checked after each response. The response that crosses the cap still has its tool calls executed, then the episode ends with no warning. This is why #20's submit counted at $0.908, while #13, #19, #21 and #22 ended mid-refinement. Cut-off points:
- T8u-675b9fb4.r1: `{"value":-0.159141,"note":"scalar result; nothing stored","budget_left":10752,"turns_left":47}`
- T8u-ae523eee.r1: `"kl_generic":0.0012,"budget_left":3969,"turns_left":67`

`submit` ends the episode, so the agent cannot bank a provisional answer. From the agent's point of view, continuing to refine was rational.

### 4. Environment artefacts vs genuine skill gaps

**Clear environment artefacts** (a passing candidate was held, and the latest candidate also passed): #1, #5, #21, #22. That is 4 sol episodes.

**Fragile environment artefacts** (a passing candidate existed, but the agent had moved to failing ones):
- #3: 0.742, against a last candidate of 0.
- #13: 0.768, against a last candidate of 0.714.
- #19: 0.783, against a last 0.740 blend.

That is 3 more.

**Environment effect on reward but not on pass/fail:** #7 (turn cap), #14, #15, #18 and #24 (cost cap). In each, a best candidate of 0.38-0.67 was turned into 0. The luna cost-capped runs (#25, #27, #30, and #26 where ENV is primary) belong here too. The luna caps may be partly an accounting artefact, since luna is priced at sol's rates.

**Environment friction that helped cause skill failures:**
- Private source cities listed first in `known_cities` and dropped silently in v0.2. This hid the source state in 5 luna runs and the missed plant's state in all two-plant runs (§3.2).
- v0.1 batch refusals (14-31 `not accessible` errors per episode).
- The tool schema advertises `model: "clean"`, which errors in T8 (2 luna runs).
- Tool limits: 16 cities, 16 extra_examples, vec_op expression length, no least-squares primitive.
- One CUDA OOM in #4 that cost 1,404 forward passes.

**Grader observation (not a bug, but it hides information):** restoration is floored at 0 (`edithunt/env/undo.py` line 116: `max(0.0, 1 - ...)`). A no-op edit (#29), an edit that makes things slightly worse (#28) and one that makes things much worse (#31: KL 282→461) all score 0. The grader itself was consistent: every submitted episode regrades exactly, with 0 replay mismatches in v0.2.

**Genuine skill gaps:**
- Two-plant discovery (0/8), compounded by the private-city friction.
- Target-aware localisation of a multi-layer plant (MN spread 0/5).
- Gate modelling: a linear unsaturated gate (#14); an overfitted interpolation (#6).
- Selection by argmax proxies (#12, with smaller losses in passes).
- Source-state identification (5/8 luna, 1 sol).
- No stopping rule: refining a passing answer until a cap hits.

**Estimated pass rates with environment fixes** (visible cost and turn budgets, or auto-submit of the best or last evaluated edit):

| Population | Now | With env fixes (robust to optimistic) |
|---|---|---|
| sol overall (24) | 5 (21%) | 9-12 (38-50%) |
| sol v0.2 (16) | 4 (25%) | 6-8 (38-50%) |
| sol single-plant v0.2 (12) | 4 (33%) | 6-8 (50-67%) |
| sol two-plant (6) | 0 | 0 |
| luna (8) | 0 | ~0 (one ungraded, likely failing) |

These are upper-bound-style estimates:
- The optimistic end assumes the agent would have submitted its best evaluated candidate. That is not its last candidate in #3, #13 or #19.
- The sample is small: 2 runs per instance, with large spread between runs (651d 0.928 vs 0.762; MN state 0.896 in hand vs a submitted 0.563).

### 5. Recommendations

#### T8 v0.3 environment fixes

1. **Make the cost budget visible.** Show `cost_left` in describe and in every tool result, or size the cap so that turns bind first: about $1.3-1.6 for sol and about $2.2 for luna at 150 turns. Price luna at its real rate.
2. **Add a fallback at the cap.** Either give one final submit-only turn when the cap triggers, or add `submit(provisional=true)` to record a checkpoint, and grade the last checkpoint when the cap hits.
3. **Log reasoning.** Request reasoning summaries and stop dropping `type=="reasoning"` items (agent_loop.py line 328), or log them to a separate field. Without this, "why it failed" can never be read directly.
4. **Fix `known_cities`.** Remove private cities from it, or list at least 3 accessible cities per state first. Do not name the skipped cities: that would leak the source state, because the test cities are source-state cities.
5. **Add a batched linear-algebra tool**: a ridge or least-squares fit of a gate key over many cached vectors, and multiple expressions per vec_op. This removes the 40-75-turn hand-CG pattern that drives cost.
6. **Fix the tool documentation and schema:**
   - Say that `clean` in eval outputs means the planted model.
   - Remove `"clean"` from the `model` enum in T8.
   - Explain why optimize_vector errors when no extra_examples are given.
   - Say that `kl_weight` applies only to 4 generic sentences.
   - Explain gate saturation in `scale_from`; linear lo 0 / hi 30 gates under-fire.
7. **Leave GPU headroom** so tool calls do not OOM.
8. **Consider a signed restoration score** in the audit (keep the floored reward for scoring), so that harmful submissions can be told apart from no-ops.

#### Making T8 a better benchmark

- **Raise the ceiling on single-plant instances.** detect_v3 solves 6 of 8 instances with about 320 forward passes. Single-plant instances mostly test whether the agent rediscovers a known script within budget; keep them as warm-ups.
- **Check that two-plant instances are discoverable.** No agent or script ever surfaced the second plant, and no agent ever queried an accessible city of that state. Run a script that sweeps the accessible cities of every state to confirm the second plant can be found.
- **Make plant count part of the scoring.** Either say "1-2 plants" explicitly, or give partial credit per plant.
- **Score discovery separately from repair.** Ask the agent to report source state, target state and layer(s), and score these as auxiliary outputs, so failures can be attributed without transcripts.
- **Report noise properly.** Report R/solver and pass@k with more runs. Solver boot_sd reaches 0.114 on MN state, and #20 passed by 0.031 against a grade boot_sd of 0.064. Raise n per instance or enlarge the eval sets.
- **Extend the audit rule.** Add the labels MISSED_PLANT, SOURCE_MISIDENTIFIED and NO_SUBMIT_UNDER_HIDDEN_CAP. Split ENV into "robust" (the latest candidate passes) and "fragile". Add a coverage check ("queried an accessible source-state city: yes/no").

### 6. Where this report disagrees with failure_audit.jsonl and LAB_NOTEBOOK.md

**Disagreements with failure_audit.jsonl:**
- **Two-plant runs** (MN two r1; T8u-6c7213a5 r1/r2; T8u-929ffed7 r1). The audit says `SKILL:wrong_gate`, and `SKILL:other` for 929f r2. My verdict is INCOMPLETE_TWO_PLANT. The audit's diagnostics compare against one plant and have no missed-plant label. The gate defects are real but secondary: the 6c72 r2 decomposition caps a perfect-gate L14 fix at about 0.60, and LAB_NOTEBOOK itself says "4/4 runs never undo both plants".
- **T8u-651dcd13.r2:** `SKILL:other` → SELECTION_ERROR. Four near-solver candidates (pc2-pc6, 0.961-0.964) were evaluated and not submitted.
- **T8u-675b9fb4.r2:** `SKILL:other` → WRONG_GATE_OR_POSITION (linear unsaturated gate). I also overturned the analyst's INEFFICIENT_EXPLORATION. The analyst attributed an optimize_vector route to detect_v3, which actually uses vec_op plus a saturating ramp.
- **T8u-d5fb4ef0.r1:** `SKILL:wrong_direction,wrong_gate` → source state misidentified. The wrong direction and gate both follow from reading the Dakota spill as a separate plant.
- **T8u-d5fb4ef0.r2:** `wrong_layer,...` → WRONG_METHOD_BEHAVIOURAL. Its L12 candidates are inside the audit's own layer_ok range but scored worse (0.22-0.28) than the out-of-range L20 candidate (0.478).
- **MN spread r1 (v0.1):** the audit says ENV_TURNCAP; the analyst said WRONG_LAYER. I keep ENV_TURNCAP but mark it fragile, as the notebook does ("ambiguous"; "its last candidate scored 0").
- **T8u-675b9fb4.r1 and T8u-ae523eee.r1** (ENV_COSTCAP): I agree, but in both runs the agent's newest candidate was below the bar (0.714, 0.740). Removing the cap alone would not guarantee a pass.
- **No audit rows exist** for MN state b150, MN two b150 or any luna episode, so those causes rest on the per-episode verdicts only.

**Disagreements with LAB_NOTEBOOK.md:**
- *"Single plants are solved or blocked only by the budget"* (v0.2 audit reading, line 1096) is overstated. 651d r2 (selection) and 675b r2 (gate) are single-plant skill failures. MN spread is single-source and failed on skill in 3 of 5 sol runs; the other 2 were cut off by caps.
- *Luna "leaned on optimize_vector (22 calls)"* (line 1100). This was written when only 2 luna runs existed ("partial at write-up"), and the 22 is from 57572593 alone. Across all 8 luna runs, optimize_vector counts are 2-22 (mean 8.75). The consistent differences from sol are much less model diffing (cache_mean 7.5 vs 51 per episode) and never probing accessible source-state cities. The notebook's "~2x sol's reasoning tokens" holds per episode (about 1.9×).
- *"Effort scaling on T8 is stark"* (line 986, based on hop2 r1 vs b150). More turns did not help on MN state (b150 submitted 0.563, while r1 had held 0.896) or on MN spread (0.144).
- *"Trigger suppression, a legitimate undo route"* for the L5 0.742 candidate (line 1012) is a plausible hypothesis, not a verified mechanism. That candidate was an answer-target optimize_vector repair with cos(-v) 0.23.
- *"pass@2 > 0 … 5/8 if the cost-capped working candidates had been submitted"* holds only if the agent would have submitted those specific candidates. In 675b r1 and ae52 r1 it was not on track to.
- *Decoys that analysts described as "natural model errors"* (Dakota→Carolina, Bentonville→Kentucky). Several things point to gate spillover instead:
  - Cross-instance evidence: the same city shifts toward Kentucky only on NC→KY instances.
  - The lab notebook's own spill notes.
  - The Tennessee/TN tokens in the agents' outputs.

  The decoys came largely from the plant itself.

**Uncertainty.**
- All beliefs and motives are inferred from tool calls; no reasoning was logged.
- The counterfactuals ("would have submitted X") are judgements.
- Luna's candidates are ungraded because its audit OOM'd.
- The design has n=2 runs per instance for sol and n=1 for luna, with large variance between runs, and several pass/fail margins are inside bootstrap noise.

Data sources:
- `/home/ec2-user/edit_hunt/results/suite/t8_undo/failure_audit.jsonl`
- `/home/ec2-user/edit_hunt/results/suite/t8_undo/pilot_v02.jsonl`
- `/home/ec2-user/edit_hunt/results/suite/instances/T8_undo_v02/*.private.json`
- `/home/ec2-user/edit_hunt/LAB_NOTEBOOK.md` (lines 789, 975-1105)
- `/home/ec2-user/edit_hunt/edithunt/env/agent_loop.py` (lines 38, 208-216, 328-356)
- `/home/ec2-user/edit_hunt/edithunt/env/undo.py` (line 116)
- `/home/ec2-user/edit_hunt/edithunt/env/tools.py` (lines 340-345)
- The run JSONs under `/home/ec2-user/edit_hunt/results/suite/runs/{gpt-6.1-sol,gpt-6-luna}/T8_undo*/`

Tool-mix counts, message-type counts, cost per turn and source-city coverage were all recomputed directly from those run files.

---

### Critic changes

**Checks run:**
- All 32 episodes are in the table.
- The taxonomy counts match the verified per-episode list: 5/3/5/6/3/4/2/2/1/1, columns summing to 8/16/8.
- These numbers were recomputed and match: 7,680 function_call items; reasoning tokens 3.1k-28k; cache_mean means 7.5 vs 51.3; optimize_vector means 8.75 vs 3.5; detect_v3 6/8.
- Spot-checked quotes against transcripts, all confirmed: MN two b150 msg 56/59 (Fargo, "low"); 929f.luna msg 14 (Homer); 651d r2 Statesboro (first at msg 111) and msg 120 `0.842194`; d5fb r1 msgs 37 and 92; turns_left at cap for 675b r1 (47), d18a r1 (27), d18a r2 (48).

**Fixes:**
1. **Main finding.** The draft said the caps "ended 8 episodes with no submission" and called the two groups "roughly equal". In fact 16/32 ended at a hidden cap with no submission (all 4 v0.1 r1, 8 sol v0.2, 4 luna). ENV is primary in 8 of them; skill is primary in 19 failures overall.
2. **#3 (MN spread r1).** Its last candidate was `cvalue` at call 244 with R=0, not the 0.670 variant; the notebook (line 1012) agrees. I also dropped the unverified word "ungated".
3. **Layer scan.** "13/13 hop2 and state sol runs" → 12/12 (6 + 6).
4. **Luna cost and tokens.** The draft's $0.014 vs $0.0098 per turn and "2-3× reasoning tokens" were wrong. Recomputed: luna $0.0128 per turn ($0.0144 capped) vs sol v0.2 $0.0085 ($0.0089 capped); about 1.9× reasoning tokens per episode, about 3× per turn.
5. **Luna label overlap.** "Five of the six luna INEFFICIENT_EXPLORATION/OTHER" was miscounted. The six episodes span three labels, and 5 of them never queried an accessible source-state city (verified directly).
6. **d18af9bd.luna msg 6.** The skipped count of 2 includes Minneapolis but is not only Minneapolis.
7. **`only the planted model is available`.** This is a tool error returned to a `model:"clean"` call, in 2 luna runs, not 1. I added that the T8 schema still advertises a `clean` enum (agent_loop.py line 38).
8. **New angle: private cities and two-plant failures.** In both two-plant instances the missed plant's state was probed only through private, silently skipped cities, or not at all, by every run. I added per-episode source-city coverage. This reframes part of the 0/8 as environment friction.
9. **New angle: grader floor.** The 0 floor (undo.py line 116) hides harmful luna submissions (#28, #31) behind the same 0 as a no-op (#29).
10. **New angle: cap mechanics.** The cost cap is checked after each response, the crossing response's tool calls still execute, and there is no warning (explains #20). v0.1 also had a non-binding cost cap.
11. **Notebook disagreement on luna's "22 calls".** Softened: the notebook was written with 2 luna runs.
12. **Smaller fixes:**
    - "expression too long" run list completed.
    - 651d r2 passing candidates corrected (pc2-pc6, 0.961-0.964; the 0.775 is at the bar within noise).
    - By-kind table relabelled with episode numbers.
    - "MN spread failed on skill 3/3 sol" corrected to 3 of 5 sol runs.
    - Luna "stops early" claim restated from the grade KL numbers.
---

## How agents succeed on T8 (undo a planted edit without the original model)

Scope: 13 analysed episodes (12 gpt-6.1-sol, plus luna through the contrasts), 6 paired contrasts, and the full T8 run population (`results/suite/runs/*/T8_undo*/`, `t8_undo/failure_audit.jsonl`, `pilot*.jsonl`). The PASS bar is 0.8 × solver. In every transcript the assistant turns are tool calls only, with no prose and with reasoning removed. Every "quote" below is therefore a tool argument or a tool output, and any statement about the agent's intent is inferred from what it called.

An important caveat about the audit: `failure_audit.jsonl` regrades only the **last 6 distinct candidates** of each episode (`experiments/t8_failure_audit.py` ~l.119). So "first passing candidate" and "best candidate" are lower bounds on what the agent actually had.

---

### 0. Population context

| Set | Runs | Submitted | PASS | Ended at cap | Cap hit while holding a passing candidate (audit) |
|---|---|---|---|---|---|
| v0.1 sol (r1: hidden 60-turn cap) | 4 | 0 | 0 | 4 (turns) | 3: hop2 0.930, state 0.896, spread 0.742 (bar 0.729) |
| v0.1 sol (b150) | 4 | 4 | 1 (hop2 0.961) | 0 | – |
| v0.2 sol (r1/r2) | 16 | 8 | 4 | 8 (cost $0.90) | 4: ae523eee.r1 0.783, 675b9fb4.r1 0.768, d18af9bd.r1 0.910, d18af9bd.r2 0.931 |
| v0.2 luna | 8 | 4 | 0 | 4 (cost) | none graded |

On v0.2 sol, an oracle that submitted the best evaluated candidate would pass 9 of 16 runs (the 4 actual passes, the 4 cost-capped runs, and 651dcd13.r2 at 0.964). The agent was credited with 4. **More than half of the competence the agent showed is lost at submission time.**

---

### 1. Episodes analysed

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

### 2. The winning recipe

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

### 3. What separates success from failure (ranked by causal weight)

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

### 4. Efficiency

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

### 5. Why the recipe does not transfer to spread and two-plant instances

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

### 6. Agent vs scripted library

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

### 7. Implications

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

### Verification notes

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