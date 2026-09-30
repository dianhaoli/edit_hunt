"""Re-run every task's reference solver through the current tools (Qwen2.5-1.5B instances) and compare with the
reward stored at generation time. Guards against tool changes silently breaking the references."""
import sys; sys.path.insert(0, __import__("os").path.dirname(__file__) + "/..")
import glob
from edithunt.env.tasks import run_reference
from edithunt.env.instance import load_instance
from edithunt.model import Subject

S = Subject("Qwen/Qwen2.5-1.5B")
n = bad = 0
for p in sorted(glob.glob("results/suite/instances/*/*.private.json")):
    inst = load_instance(p)
    if inst.model != S.name or inst.suite_task not in ("T1_easy", "T2_keepstate", "T3_consistency", "T4_detective", "T6_handoff", "T7_minimal"):
        continue
    seed = int(inst.id.rsplit("-s", 1)[1]) if inst.suite_task != "T4_detective" else 0
    r = run_reference(S, inst, seed=seed)  # generation passed the instance seed (only ref_keepstate uses it)
    new, old = r["grade"]["reward"], inst.extra["reference"]["reward"]
    ok = r["error"] is None and abs(new - old) < 1e-6
    n += 1; bad += not ok
    print(("same " if ok else "DIFF ") + inst.id, round(old, 4), round(new, 4), r["error"] or "", flush=True)
print(f"references: {n - bad}/{n} reproduce the stored reward")
