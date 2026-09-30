"""Run agent episodes over the task suite with a hard spend cap per provider.

  python -m edithunt.env.suite_run --model claude-sonnet-5-5 --tasks T1_easy,T3_consistency --n 8 \
         [--blackbox_tasks T4_detective,T6_handoff --n_blackbox 2] [--worker 0 --workers 3] [--models Qwen/Qwen2.5-1.5B]

Spend ledger: results/suite/ledger.json (file-locked; shared by parallel workers). Before an episode, the worker
reserves max_cost + margin (an episode can overshoot max_cost by at most one turn); it stops when
spent + reservations + that amount would exceed the provider cap. Episodes are interleaved across tasks
(instance 1 of every task, then instance 2, ...), so a budget stop leaves every task with a similar n.
Existing output files are skipped (resume).
"""
from __future__ import annotations

import fcntl
import json
import time
from contextlib import contextmanager
from pathlib import Path

from ..common import ROOT
from .agent_loop import episode, make_client, provider_of
from .instance import load_instance

LEDGER = ROOT / "results" / "suite" / "ledger.json"
CAPS = {"anthropic": 4.30, "openai": 8.00}  # cumulative $ incl. earlier runs (user budget $5 / ~$9)
MARGIN = 0.03


def _prior_anthropic() -> float:
    """Spend of the first Sonnet 5.5 episodes (results/agent_runs, before the suite)."""
    tot = 0.0
    for p in (ROOT / "results" / "agent_runs").glob("*/*.json"):
        if "dryrun" not in p.name:
            tot += json.loads(p.read_text()).get("cost_usd", 0.0)
    return tot


@contextmanager
def ledger():
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER.with_suffix(".lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        d = json.loads(LEDGER.read_text()) if LEDGER.exists() else {
            "anthropic": {"spent": round(_prior_anthropic(), 4), "reserved": 0.0, "episodes": []},
            "openai": {"spent": 0.0, "reserved": 0.0, "episodes": []}}
        yield d
        LEDGER.write_text(json.dumps(d, indent=1))
        fcntl.flock(lk, fcntl.LOCK_UN)


def reserve(provider: str, amount: float, cap: float) -> bool:
    with ledger() as d:
        L = d[provider]
        if L["spent"] + L["reserved"] + amount > cap:
            return False
        L["reserved"] = round(L["reserved"] + amount, 4)
        return True


def settle(provider: str, amount: float, cost: float, tag: str):
    with ledger() as d:
        L = d[provider]
        L["reserved"] = round(max(0.0, L["reserved"] - amount), 4)
        L["spent"] = round(L["spent"] + cost, 4)
        L["episodes"].append({"id": tag, "cost": cost, "t": time.strftime("%Y-%m-%d %H:%M:%S")})


def plan(tasks: list[str], n: int, bb_tasks: list[str], n_bb: int, models: list[str] | None) -> list[tuple]:
    per = {}
    for t in tasks:
        ps = sorted((ROOT / "results" / "suite" / "instances" / t).glob("*.private.json"))
        if models:
            ps = [p for p in ps if json.loads(p.read_text())["model"] in models]
        per[t] = [(t, p, False) for p in ps[:n]]
        if t in bb_tasks:
            per[t] = [(t, p, True) for p in ps[:n_bb]] + per[t]
    out, i = [], 0
    while any(i < len(v) for v in per.values()):
        out += [v[i] for v in per.values() if i < len(v)]
        i += 1
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--tasks", default="T1_easy,T2_keepstate,T3_consistency,T4_detective,T6_handoff,T7_minimal")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--blackbox_tasks", default="T4_detective,T6_handoff")
    ap.add_argument("--n_blackbox", type=int, default=2)
    ap.add_argument("--models", default="", help="only instances on these subject models (comma list)")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max_turns", type=int, default=25)
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--max_cost", type=float, default=0.10)
    ap.add_argument("--cap", type=float, default=None, help="override the provider's cumulative cap ($)")
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()
    prov = provider_of(a.model)
    cap = a.cap if a.cap is not None else CAPS[prov]
    items = plan(a.tasks.split(","), a.n, [t for t in a.blackbox_tasks.split(",") if t], a.n_blackbox,
                 [m for m in a.models.split(",") if m] or None)[a.worker::a.workers]
    client = make_client(a.model)
    S = None
    out_root = ROOT / "results" / "suite" / "runs" / a.model
    for task, p, bb in items:
        inst = load_instance(p)
        out = out_root / task / f"{inst.id}{'.bb' if bb else ''}.json"
        if out.exists():
            continue
        amount = a.max_cost + MARGIN
        if not reserve(prov, amount, cap):
            print(f"budget stop before {inst.id} (cap ${cap})", flush=True)
            break
        cost = 0.0
        try:
            if S is None or S.name != inst.model:
                from ..model import Subject
                S = None
                import torch; torch.cuda.empty_cache()
                S = Subject(inst.model)
            rec = episode(S, inst, a.model, client, bb, a.max_turns, 16000, a.effort or None, a.max_cost, a.bs)
            cost = rec["cost_usd"]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(rec, indent=1, default=str))
            g = rec["grade"]
            print(f"{task:15s} {inst.id:55s} {'BB ' if bb else ''}reward {g['reward']:.3f} turns {rec['turns']} "
                  f"calls {rec['tool_calls']} ${cost:.4f} stop {rec['stop_reason']}", flush=True)
        except Exception as e:  # keep going; the ledger still gets settled
            print(f"ERROR {inst.id}: {type(e).__name__}: {e}", flush=True)
            cost = a.max_cost  # unknown partial spend: charge the worst case
        finally:
            settle(prov, amount, cost, f"{a.model}/{inst.id}{'.bb' if bb else ''}")
    with ledger() as d:
        print(f"{prov} spent ${d[prov]['spent']:.4f} (cap ${cap})", flush=True)


if __name__ == "__main__":
    main()
