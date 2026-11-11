"""Instance generator with a built-in script filter (AUDIT_T2V2 §8.4; bars pre-registered in LAB_NOTEBOOK 2026-10-02).

A family supplies candidate instances, a solver (evidence that the instance is solvable; run for several seeds), a
library of scripted baselines (fixed recipes a policy could memorise) and a grader-noise estimate. A candidate is kept
iff all three hold:
  solvable             mean solver R >= solve_bar                                   (0.6)
  not script-solvable  best library R <= script_frac x mean solver R                (0.5)
  headroom > noise     mean solver R - best library R > 2 x max(noise SD, solver seed SD)
Every result is cached per (instance id, method, seed) in <out>/cache.jsonl, so adding a script to the library later
re-filters every candidate without re-running solvers; the report lists each candidate's numbers and drop reason.
The library is meant to grow: add each new trick found (by audits or by trained policies) and regenerate.
"""
from __future__ import annotations

import json
import statistics as st
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

SOLVE_BAR, SCRIPT_FRAC, NOISE_K = 0.6, 0.5, 2.0


@dataclass
class Family:
    name: str
    candidates: Callable[[], Iterable]                     # -> instances (objects with .id)
    solve: Callable[[object, int], dict]                   # (inst, seed) -> {"R": float, ...}
    library: dict[str, Callable[[object], dict]]           # name -> (inst) -> {"R": float, ...}
    noise: Callable[[object, dict], float] | None = None   # (inst, solver result) -> grader noise SD of R
    seeds: tuple = (0, 1, 2)
    bars: dict = field(default_factory=lambda: {"solve_bar": SOLVE_BAR, "script_frac": SCRIPT_FRAC, "noise_k": NOISE_K})


class Cache:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = {}
        if path.exists():
            for line in path.read_text().splitlines():
                r = json.loads(line)
                self.rows[(r["id"], r["method"], r.get("seed", 0))] = r

    def get(self, iid, method, seed, fn):
        k = (iid, method, seed)
        if k not in self.rows:
            r = {"id": iid, "method": method, "seed": seed, **fn()}
            with self.path.open("a") as f:
                f.write(json.dumps(r, default=float) + "\n")
            self.rows[k] = r
        return self.rows[k]


def judge(solver: list[dict], scripts: dict[str, dict], noise_sd: float | None, bars: dict) -> dict:
    Rs = [r["R"] for r in solver]
    sR = st.mean(Rs)
    seed_sd = st.pstdev(Rs) if len(Rs) > 1 else 0.0
    best_name = max(scripts, key=lambda m: scripts[m]["R"]) if scripts else None
    best = scripts[best_name]["R"] if scripts else 0.0
    nz = max(noise_sd or 0.0, seed_sd)
    checks = {"solvable": sR >= bars["solve_bar"],
              "not_script_solvable": best <= bars["script_frac"] * sR,
              "headroom_gt_noise": (sR - best) > bars["noise_k"] * nz}
    return {"solver_R": sR, "solver_seed_sd": seed_sd, "noise_sd": noise_sd, "best_script": best_name,
            "best_script_R": best, "headroom": sR - best, **checks, "keep": all(checks.values()),
            "drop_reason": [k for k, v in checks.items() if not v]}


def screen(fam: Family, out: Path, max_keep: int | None = None, verbose: bool = True) -> dict:
    cache = Cache(out / "cache.jsonl")
    report, kept = [], []
    for inst in fam.candidates():
        solver = [cache.get(inst.id, "solver", s, lambda s=s: fam.solve(inst, s)) for s in fam.seeds]
        if st.mean(r["R"] for r in solver) < fam.bars["solve_bar"]:  # unsolvable: skip the (costly) library
            j = judge(solver, {}, None, fam.bars) | {"library_skipped": True}
        else:
            scripts = {m: cache.get(inst.id, m, 0, lambda fn=fn: fn(inst)) for m, fn in fam.library.items()}
            nsd = fam.noise(inst, solver[0]) if fam.noise else None
            j = judge(solver, scripts, nsd, fam.bars) | {"scripts": {m: r["R"] for m, r in scripts.items()}}
        report.append({"id": inst.id, **j})
        if verbose:
            print(f"{fam.name} {inst.id}: solver {j['solver_R']:.2f} best script {j['best_script']} "
                  f"{j['best_script_R']:.2f} -> {'KEEP' if j['keep'] else 'drop ' + ','.join(j['drop_reason'])}", flush=True)
        if j["keep"]:
            kept.append(inst)
            if max_keep and len(kept) >= max_keep:
                break
    summ = {"family": fam.name, "bars": fam.bars, "library": sorted(fam.library), "n_candidates": len(report),
            "n_kept": len(kept), "report": report}
    (out / "report.json").write_text(json.dumps(summ, indent=1, default=float))
    return {"kept": kept, **summ}
