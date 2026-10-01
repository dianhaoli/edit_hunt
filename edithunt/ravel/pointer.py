"""Pointer-style task families (added 2026-10-01 at the user's request, after the RAVEL screen): variables are read at
the FINAL token, and the expected output under an intervention comes from a small causal model of the prompt.

bind  (in-context entity binding; Feng & Steinhardt 2023, Gur-Arieh et al. 2025 "Mixing Mechanisms"):
      "Pete loves jam, Ann loves pie, and Tom loves tea. So Ann loves" -> " pie".
      Variables: P = position of the queried pair (0..2), Q = the queried name.
      Source prompts reuse the base's 3 names in another order with 3 FRESH items, so copying the source's answer
      token cannot hit a base candidate.
      Cause(P): P_s != P_b and the source's name sits elsewhere in the base -> expect base item at P_s.
      Cause(Q): Q_s != Q_b and pos_base(Q_s) != P_s                     -> expect base item bound to Q_s.
      Iso(P): P_s == P_b, Q_s != Q_b -> expect the base answer.   Iso(Q): Q_s == Q_b, P_s != P_b -> base answer.
mcqa  (MIB's MCQA: Mueller et al. 2025): "The box is brown. What color is the box?\nA. red\nB. brown ..." -> " B".
      Variables: O = answer content (colour), X = answer position (letter).
      Source: another object; options = a permutation of the base's 4 colours.
      Cause(O): O_s != O_b, X_s != pos_base(O_s) -> expect letter pos_base(O_s).
      Cause(X): X_s != X_b, pos_base(O_s) != X_s -> expect letter X_s.
      Iso(O): O_s == O_b, X_s != X_b -> base letter.   Iso(X): X_s == X_b, O_s != O_b -> base letter.
Score = Cause(V) x Iso(V) (one other variable). Entity split = held-out names+items / objects; Context split =
held-out templates. Methods, layer choice and thresholds are the same as stages 1-3.
Usage: python -m edithunt.ravel.pointer bind mcqa
"""
from __future__ import annotations

import json
import random
import sys
import time

import torch

from .core import Item, Iv, Model, iv_add, iv_interchange, iv_set
from .das import Subspace
from .evalkit import SCREEN
from .sae import SAE, iv_sae, probe_select
from .stage2 import pick_layers

OUT = SCREEN / "pointer"
NAMES = ("Pete Ann Tom Mary John Lisa Mark Sarah Paul Emma Jack Kate Mike Anna David Laura James Linda Ben Alice "
         "Sam Nina Carl Rose Fred Jane Bob Eve Dan Amy Greg Lucy Ivan Olga Hugo Iris Leo Mia Noah Zoe").split()
ITEMS = ("jam pie tea cake soup rice bread milk coffee juice pizza pasta salad cheese honey fish beans corn eggs toast "
         "wine beer chips nuts candy fruit meat sugar cookies butter apples grapes lemons pears plums peaches beef "
         "chicken pork tuna").split()
OBJS = ("box car hat shirt ball cup door chair lamp bag book pen bike phone kite sock shoe mug bowl vase rug bench "
        "fence boat key clock tent scarf bottle plate").split()
COLS = "red blue green yellow black white brown pink purple orange gray".split()
LET = "ABCD"

BIND_T = [("{n0} loves {i0}, {n1} loves {i1}, and {n2} loves {i2}. So {q} loves", "train"),
          ("{n0} ordered {i0}. {n1} ordered {i1}. {n2} ordered {i2}. {q} ordered", "train"),
          ("At the market, {n0} bought {i0}, {n1} bought {i1} and {n2} bought {i2}. {q} bought", "train"),
          ("Lunch orders: {n0} - {i0}; {n1} - {i1}; {n2} - {i2}. Order for {q}:", "train"),
          ("{n0} brought {i0}, {n1} brought {i1}, and {n2} brought {i2} to the party. {q} brought", "test"),
          ("For dinner {n0} chose {i0}, {n1} chose {i1}, and {n2} chose {i2}. The dish {q} chose was", "test")]
MCQA_T = [("The {o} is {c}. What color is the {o}?\nA. {a}\nB. {b}\nC. {cc}\nD. {d}\nAnswer:", "train"),
          ("Question: A {o} is {c}. Which color is the {o}?\nA. {a}\nB. {b}\nC. {cc}\nD. {d}\nAnswer:", "train"),
          ("My {o} is {c}.\nQ: What is the color of my {o}?\nA. {a}\nB. {b}\nC. {cc}\nD. {d}\nA:", "train"),
          ("There is a {c} {o}. What color is the {o}?\nA. {a}\nB. {b}\nC. {cc}\nD. {d}\nAnswer:", "test"),
          ("Fact: the {o} is {c}.\nWhat colour does the {o} have?\nA. {a}\nB. {b}\nC. {cc}\nD. {d}\nCorrect option:", "test")]


def ent_pools(fam: str):
    """entity split: (train pool, test pool)."""
    if fam == "bind":
        return (NAMES[:24], ITEMS[:24]), (NAMES[24:], ITEMS[24:])
    return (OBJS[:20],), (OBJS[20:],)


# ------------------------------------------------------------------ generation
def gen_base(fam, rng, pools, tpl):
    if fam == "bind":
        names, items = rng.sample(pools[0], 3), rng.sample(pools[1], 3)
        p = rng.randrange(3)
        return dict(tpl=tpl, names=names, items=items, P=p, Q=names[p])
    o = rng.choice(pools[0])
    opts = rng.sample(COLS, 4)
    x = rng.randrange(4)
    return dict(tpl=tpl, obj=o, opts=opts, O=opts[x], X=x)


def text(fam, it):
    if fam == "bind":
        return it["tpl"].format(n0=it["names"][0], n1=it["names"][1], n2=it["names"][2],
                                i0=it["items"][0], i1=it["items"][1], i2=it["items"][2], q=it["Q"])
    a, b, c, d = it["opts"]
    return it["tpl"].format(o=it["obj"], c=it["O"], a=a, b=b, cc=c, d=d)


def cands(fam, it):  # surfaces (single tokens with a leading space)
    return list(it["items"]) if fam == "bind" else list(LET)


def answer(fam, it):
    return it["items"][it["P"]] if fam == "bind" else LET[it["X"]]


def gen_source(fam, rng, base, pools, tpl):
    if fam == "bind":
        names = base["names"][:]
        rng.shuffle(names)
        items = rng.sample([i for i in pools[1] if i not in base["items"]], 3)
        p = rng.randrange(3)
        return dict(tpl=tpl, names=names, items=items, P=p, Q=names[p])
    o = rng.choice([x for x in pools[0] if x != base["obj"]])
    opts = base["opts"][:]
    rng.shuffle(opts)
    x = rng.randrange(4)
    return dict(tpl=tpl, obj=o, opts=opts, O=opts[x], X=x)


VARS = {"bind": ("P", "Q"), "mcqa": ("O", "X")}


def expected(fam, kind, var, b, s):
    """expected answer for a pair of type (kind, var), or None if the pair does not qualify."""
    if fam == "bind":
        pos_b = b["names"].index(s["Q"])
        if kind == "cause" and var == "P" and s["P"] != b["P"] and pos_b != s["P"]:
            return b["items"][s["P"]]
        if kind == "cause" and var == "Q" and s["Q"] != b["Q"] and pos_b != s["P"]:
            return b["items"][pos_b]
        if kind == "iso" and var == "P" and s["P"] == b["P"] and s["Q"] != b["Q"]:
            return answer(fam, b)
        if kind == "iso" and var == "Q" and s["Q"] == b["Q"] and s["P"] != b["P"]:
            return answer(fam, b)
        return None
    pos_b = b["opts"].index(s["O"])
    if kind == "cause" and var == "O" and s["O"] != b["O"] and s["X"] != pos_b:
        return LET[pos_b]
    if kind == "cause" and var == "X" and s["X"] != b["X"] and pos_b != s["X"]:
        return LET[s["X"]]
    if kind == "iso" and var == "O" and s["O"] == b["O"] and s["X"] != b["X"]:
        return answer(fam, b)
    if kind == "iso" and var == "X" and s["X"] == b["X"] and s["O"] != b["O"]:
        return answer(fam, b)
    return None


# ------------------------------------------------------------------ model helpers
def enc(M: Model, fam, it) -> Item:
    ids = M._ids(text(fam, it))
    return Item(ids, len(ids) - 1)


def tok1(M: Model, w: str) -> int:
    ids = M.cand_ids(" ", (w,))[0]
    assert len(ids) == 1, w
    return ids[0]


@torch.no_grad()
def predict(M: Model, fam, its, iv: Iv | None = None, bs=64):
    """argmax over each prompt's own candidates (all single tokens) -> list of surfaces."""
    out = []
    for s in range(0, len(its), bs):
        ch = its[s:s + bs]
        fn = iv.chunk(s, s + len(ch)) if iv is not None else None
        logits, _, _, _ = M.forward([enc(M, fam, it) for it in ch], fn, iv.layer if iv else None)
        for r, it in enumerate(ch):
            cs = cands(fam, it)
            ids = torch.tensor([tok1(M, c) for c in cs], device=M.device)
            out.append(cs[int(logits[r, ids].argmax())])
    return out


def capture(M, fam, its, layers):
    return M.capture([enc(M, fam, it) for it in its], layers)


# ------------------------------------------------------------------ pools
def tpls(split_t):
    return lambda fam: [t for t, sp in (BIND_T if fam == "bind" else MCQA_T) if sp == split_t]


def make_pairs(M, fam, split, n, seed, kept_t):
    """pairs per (kind, var): list of (base, source, expected). Both prompts must be answered right by the clean model."""
    e_sp, t_sp = {"entity": ("test", "train"), "context": ("train", "test"), "train": ("train", "train")}[split]
    pools = ent_pools(fam)[0 if e_sp == "train" else 1]
    T = [t for t in kept_t if t[1] == t_sp]
    rng = random.Random(seed)
    want = [(k, v) for k in ("cause", "iso") for v in VARS[fam]]
    pairs = {f"{k}_{v}": [] for k, v in want}
    tries = 0
    while any(len(pairs[f"{k}_{v}"]) < n for k, v in want) and tries < 200:
        tries += 1
        cand = []
        for _ in range(256):
            b = gen_base(fam, rng, pools, rng.choice(T)[0])
            s = gen_source(fam, rng, b, pools, rng.choice(T)[0])
            for k, v in want:
                if len(pairs[f"{k}_{v}"]) < n:
                    e = expected(fam, k, v, b, s)
                    if e is not None:
                        cand.append((f"{k}_{v}", b, s, e))
                        break
        if not cand:
            continue
        flat = [c[1] for c in cand] + [c[2] for c in cand]
        pr = predict(M, fam, flat)
        for j, (key, b, s, e) in enumerate(cand):
            if pr[j] == answer(fam, b) and pr[len(cand) + j] == answer(fam, s) and len(pairs[key]) < n:
                pairs[key].append((b, s, e))
    return pairs


def run_pairs(M, fam, pairs, layer, make_iv, src_cache):
    """accuracy per pair type under the intervention."""
    res = {}
    for key, ps in pairs.items():
        if not ps:
            continue
        src = src_cache[key]
        iv = make_iv(key, ps, src)
        pr = predict(M, fam, [b for b, _, _ in ps], iv)
        res[key] = sum(p == e for p, (_, _, e) in zip(pr, ps)) / len(ps)
    return res


def score_rows(fam, acc):
    out = {}
    for v in VARS[fam]:
        c, i = acc.get(f"cause_{v}", float("nan")), acc.get(f"iso_{v}", float("nan"))
        out[v] = dict(cause=c, iso=i, score=c * i, disentangle=(c + i) / 2)
    return out


# ------------------------------------------------------------------ generic subspace fitting (also Stage 5's tool)
def fit_subspace(M, fam, layer, k, examples, src, seed=0, steps=150, bs=32, lr=2e-3, max_seconds=300):
    """examples: list of (base item dict, label surface); src: [N,d] source residuals aligned with examples.
    Interchange-intervention training of a k-dim orthonormal subspace (CE on the label token)."""
    t0 = time.time()
    rng = random.Random(seed)
    S = Subspace(M.d, k, seed).to(M.device)
    opt = torch.optim.Adam(S.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1 - s / steps)
    hist = []
    for step in range(steps):
        if time.time() - t0 > max_seconds:
            break
        idx = rng.sample(range(len(examples)), min(bs, len(examples)))
        items = [enc(M, fam, examples[i][0]) for i in idx]
        labs = torch.tensor([tok1(M, examples[i][1]) for i in idx], device=M.device)
        U, srows = S.U(), src[idx].to(M.device)
        logits, _, _, _ = M.forward(items, lambda x: x.float() + (srows - x.float()) @ U @ U.T, layer, grad=True)
        loss = torch.nn.functional.cross_entropy(logits, labs)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        hist.append(loss.item())
    info = dict(steps=len(hist), loss_first=sum(hist[:10]) / max(1, len(hist[:10])),
                loss_last=sum(hist[-10:]) / max(1, len(hist[-10:])), seconds=time.time() - t0)
    return S.U().detach().float().cpu(), info


# ------------------------------------------------------------------ driver
def validity(M, fam, n=200, seed=0):
    rng = random.Random(seed)
    res = {}
    for t, sp in (BIND_T if fam == "bind" else MCQA_T):
        its = [gen_base(fam, rng, ent_pools(fam)[0 if i % 2 else 1], t) for i in range(n)]
        pr = predict(M, fam, its)
        res[t] = (sum(p == answer(fam, it) for p, it in zip(pr, its)) / n, sp)
    kept = [(t, sp) for t, (a, sp) in res.items() if a >= 0.8]
    allacc = sum(a for a, _ in res.values()) / len(res)
    ok = allacc >= 0.8 and sum(sp == "train" for _, sp in kept) >= 2 and sum(sp == "test" for _, sp in kept) >= 1
    return dict(tpl_acc=res, acc_all=allacc, kept=kept, survives=ok)


def run(M: Model, fam: str, n_eval=200, n_train=600):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{fam}.json"
    R = json.loads(path.read_text()) if path.exists() else {"family": fam, "vars": VARS[fam]}
    save = lambda: path.write_text(json.dumps(R, indent=1))
    log = lambda s: print(f"  {fam} {s} ({time.time() - t0:.0f}s)", flush=True)
    if "validity" not in R:
        R["validity"] = validity(M, fam); save()
        log(f"validity acc_all={R['validity']['acc_all']:.2f} survives={R['validity']['survives']} "
            f"{[round(a, 2) for a, _ in R['validity']['tpl_acc'].values()]}")
    if not R["validity"]["survives"]:
        return R
    kept_t = [tuple(x) for x in R["validity"]["kept"]]
    if "sweep" not in R:
        sp = make_pairs(M, fam, "entity", 32, 99, kept_t)
        layers = list(range(M.n_layers))
        srcs = {k: capture(M, fam, [s for _, s, _ in v], layers) for k, v in sp.items()}
        R["sweep"] = {}
        for L in layers:
            acc = run_pairs(M, fam, sp, L, lambda k, ps, src: iv_set(L, src), {k: srcs[k][L] for k in sp})
            R["sweep"][str(L)] = sum(acc.get(f"cause_{v}", 0) for v in VARS[fam]) / len(VARS[fam])
        R["layers"] = pick_layers(R["sweep"], M.n_layers); save()
        log(f"sweep {[round(R['sweep'][str(L)], 2) for L in layers]} -> {R['layers']}")
    layers = R["layers"]
    pairs = {sp: make_pairs(M, fam, sp, n_eval, {"entity": 0, "context": 1}[sp], kept_t) for sp in ("entity", "context")}
    R["pair_counts"] = {sp: {k: len(v) for k, v in p.items()} for sp, p in pairs.items()}
    trp = make_pairs(M, fam, "train", n_train, 1000, kept_t)
    tr_items = [b for v in trp.values() for b, _, _ in v]
    R.setdefault("stage2", {}); R.setdefault("stage3", {})
    for L in layers:
        srcs = {sp: {k: capture(M, fam, [s for _, s, _ in v], [L])[L] for k, v in p.items()} for sp, p in pairs.items()}
        if not all(str(L) in R["stage2"].get(sp, {}) for sp in pairs):
            reps = capture(M, fam, tr_items, [L])[L]
            sae = SAE(L)
            g = torch.Generator().manual_seed(L)
            Urand = torch.linalg.qr(torch.randn(M.d, 16, generator=g))[0]
            with torch.no_grad():
                z = torch.cat([sae.enc(reps[i:i + 512].cuda()) for i in range(0, len(reps), 512)])
            lat = {v: probe_select(z, [str(it[v]) for it in tr_items])[0] for v in VARS[fam]}
            means = {}
            for v in VARS[fam]:
                vals = sorted({str(it[v]) for it in tr_items})
                means[v] = {x: reps[torch.tensor([str(it[v]) == x for it in tr_items])].mean(0) for x in vals}
            for sp, p in pairs.items():
                if str(L) in R["stage2"].get(sp, {}):
                    continue
                res = {}
                res["full"] = score_rows(fam, run_pairs(M, fam, p, L, lambda k, ps, src: iv_set(L, src), srcs[sp]))
                res["random16"] = score_rows(fam, run_pairs(M, fam, p, L,
                                                            lambda k, ps, src: iv_interchange(L, Urand, src), srcs[sp]))
                for v in VARS[fam]:
                    zero = torch.zeros(M.d)
                    mv = lambda k, ps, src: iv_add(L, torch.stack([means[v].get(str(s[v]), zero) - means[v].get(str(b[v]), zero)
                                                                   for b, s, _ in ps]))
                    res.setdefault("meandiff", {})[v] = score_rows(fam, run_pairs(M, fam, p, L, mv, srcs[sp]))[v]
                    for kk in (5, 50):
                        sv = lambda k, ps, src: iv_sae(L, sae, lat[v][kk], src)
                        res.setdefault(f"sae{kk}", {})[v] = score_rows(fam, run_pairs(M, fam, p, L, sv, srcs[sp]))[v]
                R["stage2"].setdefault(sp, {})[str(L)] = res; save()
                log(f"L{L} {sp} stage2: " + "; ".join(
                    f"{v}: best {max((res[m][v]['score'], m) for m in res)[1]} "
                    f"{max(res[m][v]['score'] for m in res):.2f}" for v in VARS[fam]))
            del sae, z
            torch.cuda.empty_cache()
        tr_src = {k: capture(M, fam, [s for _, s, _ in v], [L])[L] for k, v in trp.items()}
        for v in VARS[fam]:
            other = [w for w in VARS[fam] if w != v][0]
            for kk in (16, 128):
                for m in ("das", "mdas"):
                    key = f"{v}|L{L}|k{kk}|{m}|s0"
                    if key in R["stage3"]:
                        continue
                    ex, srows = [], []
                    keys = [f"cause_{v}"] + ([f"iso_{v}"] if m == "mdas" else [])
                    for kname in keys:
                        for j, (b, s, e) in enumerate(trp[kname]):
                            ex.append((b, e)); srows.append(tr_src[kname][j])
                    U, info = fit_subspace(M, fam, L, kk, ex, torch.stack(srows), seed=0)
                    res = {sp: score_rows(fam, run_pairs(M, fam, p, L, lambda k, ps, src: iv_interchange(L, U, src),
                                                         srcs[sp]))[v] for sp, p in pairs.items()}
                    R["stage3"][key] = dict(info=info, **res); save()
                    log(f"{key}: entity C {res['entity']['cause']:.2f} I {res['entity']['iso']:.2f} "
                        f"score {res['entity']['score']:.2f} | context {res['context']['score']:.2f} "
                        f"loss {info['loss_first']:.2f}->{info['loss_last']:.2f}")
    return R


if __name__ == "__main__":
    M = Model()
    for f in sys.argv[1:]:
        run(M, f)


# ------------------------------------------------------------------ Stage 4 helpers
SHORT_P = ("full", "meandiff", "sae5", "sae50", "random16")


def table_rows(fam: str) -> list[dict]:
    """ranking rows in stage4's format (Entity split)."""
    path = OUT / f"{fam}.json"
    if not path.exists():
        return []
    R = json.loads(path.read_text())
    if not R["validity"]["survives"] or not R.get("stage3"):
        return []
    rows = []
    for v in VARS[fam]:
        best = {}
        for L, res in R["stage2"]["entity"].items():
            for m in SHORT_P:
                r = res[m][v]
                if r["score"] == r["score"] and r["score"] > best.get(m, {"score": -1})["score"]:
                    best[m] = dict(r, layer=int(L))
        for key, run in R["stage3"].items():
            tv, L, k, m, _ = key.split("|")
            r = run["entity"]
            if tv == v and r["score"] == r["score"] and r["score"] > best.get(m, {"score": -1})["score"]:
                best[m] = dict(r, layer=int(L[1:]), k=int(k[1:]))
        if "mdas" not in best:
            continue
        sm = max((m for m in best if m != "mdas"), key=lambda m: best[m]["score"])
        ref, short = best["mdas"]["score"], best[sm]["score"]
        rows.append(dict(family=fam, target=v, ref=ref, ref_cfg=dict(layer=best["mdas"]["layer"], k=best["mdas"]["k"]),
                         ref_cause=best["mdas"]["cause"], ref_iso=best["mdas"]["iso"],
                         ref_disentangle=best["mdas"]["disentangle"], short=short, short_method=sm,
                         short_cause=best[sm]["cause"], short_iso=best[sm]["iso"], gap=ref - short,
                         good=ref >= 0.5 and ref - short >= 0.2, inseparable=max(b["score"] for b in best.values()) < 0.5,
                         best_by_method={m: dict(score=b["score"], cause=b["cause"], iso=b["iso"], layer=b["layer"],
                                                 k=b.get("k")) for m, b in best.items()}))
    return rows


def seed_runs(M, fam, var, L, k, n_eval=200, n_train=600, seeds=range(4)):
    """MDAS at (L, k) with 4 seeds, scored on FRESH pairs (entity seed 2, context seed 3)."""
    R = json.loads((OUT / f"{fam}.json").read_text())
    kept_t = [tuple(x) for x in R["validity"]["kept"]]
    pairs = {sp: make_pairs(M, fam, sp, n_eval, {"entity": 2, "context": 3}[sp], kept_t) for sp in ("entity", "context")}
    srcs = {sp: {kk: capture(M, fam, [s for _, s, _ in vv], [L])[L] for kk, vv in p.items()} for sp, p in pairs.items()}
    out = []
    for seed in seeds:
        trp = make_pairs(M, fam, "train", n_train, 1000 + seed, kept_t)
        ex, srows = [], []
        for kname in (f"cause_{var}", f"iso_{var}"):
            src = capture(M, fam, [s for _, s, _ in trp[kname]], [L])[L]
            for j, (b, s, e) in enumerate(trp[kname]):
                ex.append((b, e)); srows.append(src[j])
        U, info = fit_subspace(M, fam, L, k, ex, torch.stack(srows), seed=seed)
        res = {sp: score_rows(fam, run_pairs(M, fam, p, L, lambda kk, ps, src: iv_interchange(L, U, src), srcs[sp]))[var]
               for sp, p in pairs.items()}
        out.append(dict(seed=seed, info=info, **res))
        print(f"  {fam}|{var} L{L} k{k} seed {seed}: entity {res['entity']['score']:.2f} context {res['context']['score']:.2f}",
              flush=True)
    return out
