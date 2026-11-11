"""Regression tests for the T2 v2.1 fixes (AUDIT_T2V2 C1, M3, m9). No GPU: guards and texts only.
  PYTHONPATH=. .venv/bin/python tests/test_t2v21.py"""
import glob
import json
from types import SimpleNamespace

from edithunt.data import CITIES, CITIES_EXT, STATES
from edithunt.env.agent_loop import tools_for
from edithunt.env.instance import load_instance
from edithunt.env.ravel import to_v21
from edithunt.env.tasks import describe, make_env
from edithunt.env.tools import ToolEnv, ToolError, _letters, _near, _norm

STUB = SimpleNamespace(n_layers=28, d_model=1536, name="Qwen/Qwen2.5-1.5B")


def variants(c: str) -> list[str]:
    """Spellings an agent could use for city c (aliases, compact forms, punctuation, case, zero-width)."""
    out = {c, c.upper(), c.lower(), c.replace(" ", ""), c.replace(" ", "-"), c.replace(".", ""), "​".join(c)}
    for a, b in (("St. ", "Saint "), ("St. ", "St "), ("St. ", "Saint"), ("Fort ", "Ft. "), ("Fort ", "Ft "),
                 ("Mount ", "Mt. "), ("North ", "N. "), ("North ", "N "), ("South ", "S. "), ("East ", "E. "),
                 ("West ", "W. "), ("Point ", "Pt. ")):
        if c.startswith(a):
            out |= {b + c[len(a):], (b + c[len(a):]).replace(" ", "")}
        if c.startswith(b.strip() + " ") or c.startswith(b):
            pass
    if c.startswith("Saint "):
        out |= {"St. " + c[6:], "St " + c[6:]}
    if c.startswith("Ft. "):
        out |= {"Fort " + c[4:]}
    return sorted(out)


def refused_city(env, c) -> bool:
    try:
        env._cities([c])
        return False
    except ToolError:
        return True


def refused_text(env, s) -> bool:
    try:
        env._check_text(s)
        return False
    except ToolError:
        return True


def main():
    paths = sorted(glob.glob("results/suite/instances/T2_keepstate_v2/*.private.json"))
    assert paths, "no pilot instances"
    n_var = n_txt = 0
    for p in paths:
        inst = to_v21(load_instance(p))
        env = ToolEnv(STUB, inst, tools=["describe_task"])
        held = list(inst.test_cities) + list(inst.extra["private_cities"])
        # 1. every alias/compact/typo variant of every private city is refused (cities= and raw prompts)
        for c in held:
            for v in variants(c):
                assert refused_city(env, v), (inst.id, c, v)
                n_var += 1
            for v in variants(c):
                if "​" in v:
                    continue
                assert refused_text(env, f"Q: What is the capital of the state that {v} is in?\nA:"), (inst.id, c, v)
                n_txt += 1
            if len(_letters(c)) >= 6:  # one-letter typo
                t = c[:-1] + ("x" if c[-1] != "x" else "y")
                assert refused_city(env, t), (inst.id, c, t)
        # the audit's concrete case
        if "St. Cloud" in held:
            assert refused_city(env, "Saint Cloud") and refused_text(env, "I live in Saint Cloud, MN.")
        # 2. public cities the task hands out stay accessible
        for c in list(inst.dev_source) + list(inst.dev_target) + [x for cs in inst.extra["keep_cities"].values() for x in cs]:
            assert not refused_city(env, c), (inst.id, "public city refused", c)
        # 3. C1: the v2.1 text states the product reward; v2 text unchanged
        d = describe(SimpleNamespace(inst=inst, S=STUB, regs={}, budget=1, call_budget=None, templates={}))
        assert "reward = Cause * Iso_state * Iso_other" in d["reward"] and "Disentangle" not in d["reward"], d["reward"]
        assert "sampled at grading time" not in d["reward"]
        assert "leak_weight" not in d["constraints"] and "preserve_state" not in d["constraints"]
    # v2 text must equal what the v2 agents saw (first user message of a stored v2 episode)
    rec = sorted(glob.glob("results/suite/runs/*/T2_keepstate_v2/*.json"))[0]
    r = json.loads(open(rec).read())
    inst = load_instance(f"results/suite/instances/T2_keepstate_v2/{r['instance']}.private.json")
    d = describe(SimpleNamespace(inst=inst, S=STUB, regs={}, budget=inst.constraints["fwd_budget"], call_budget=None,
                                 templates={}))
    assert d["reward"] in r["messages"][0]["content"].replace('\\"', '"'), "v2 reward text changed"
    # 4. accessible-city false positives across the whole extended list: which public cities collide with others
    names = sorted({c for cs in CITIES_EXT.values() for c in cs})
    coll = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]
            if len(_letters(b)) >= 6 and _near(_letters(a), _letters(b))]
    print("edit-distance-1 collisions among dataset cities (would refuse each other if one is private):", coll)
    # 5. m9: scale_from only in schemas of tasks that allow it
    t1 = load_instance(sorted(glob.glob("results/suite/instances/T1_easy/*.private.json"))[0])
    for inst, allow in ((t1, False), (to_v21(load_instance(paths[0])), True)):
        env = SimpleNamespace(inst=inst, tools={"eval_intervention", "optimize_vector", "submit"})
        s = json.dumps(tools_for(env))
        assert ("scale_from" in s) == allow, (inst.id, allow)
    print(f"ok: {len(paths)} instances, {n_var} city variants and {n_txt} prompt variants refused; texts and schemas ok")
    print("examples:", _norm("Saint Cloud"), "|", _norm("St. Cloud"), "|", _norm("NorthPlatte"), "|", _norm("Ft. Worth"))


if __name__ == "__main__":
    main()
