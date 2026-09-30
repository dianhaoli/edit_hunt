"""Phase 1: replicate & scale Exp A (full-residual patch at city-last vs final)
and Exp B (mean-difference state direction at city-last), with controls
(random norm-matched, shuffled labels) and cross-pair transfer.
Vectors are built from the `--build` template only; other templates are held out."""
import sys; sys.path.insert(0, __import__("os").path.dirname(__file__) + "/..")
import time

import torch

from edithunt.common import base_args, save, seed_all, valid_table
from edithunt.data import CAPITALS, CITIES, TEMPLATES
from edithunt.instances import eligible_states, pick_pairs, split
from edithunt.metrics import fmt, rate
from edithunt.model import Subject
from edithunt.runner import Job, run_jobs

ap = base_args(__doc__)
ap.add_argument("--build", default="fs1")
ap.add_argument("--eval_templates", default="fs1,fs2,ho_fs,zs1,ho_zs")
ap.add_argument("--n_pairs", type=int, default=12)
ap.add_argument("--layer_stride", type=int, default=1)
args = ap.parse_args()
seed_all(args.seed)
S = Subject(args.model)
V = valid_table(args.model)
caps = list(CAPITALS.values())
evalT = args.eval_templates.split(",")
layers = list(range(0, S.n_layers, args.layer_stride))
states = eligible_states(V, args.build)
pairs = pick_pairs(states, args.n_pairs, args.seed)
print("eligible states", len(states), "pairs", pairs, flush=True)

# cache residuals for all build-valid cities of eligible states
allc = [c for s in states for c in CITIES[s] if V[args.build][c]]
encB = {c: S.encode(TEMPLATES[args.build], c) for c in allc}
t0 = time.time()
Rc = S.resid([encB[c] for c in allc], layers, "city_last", bs=args.bs)
Rf = S.resid([encB[c] for c in allc], layers, "final", bs=args.bs)
ix = {c: i for i, c in enumerate(allc)}
print(f"cached residuals [{time.time()-t0:.0f}s]", flush=True)


def mean(cs, L):
    return Rc[L][[ix[c] for c in cs]].mean(0)


g = torch.Generator().manual_seed(args.seed)
rows = []  # flat records
for pi, (src, tgt) in enumerate(pairs):
    dev_s, test_s = split(V, src, args.seed, args.build)
    dev_t, test_t = split(V, tgt, args.seed, args.build)
    sc_, tc_ = CAPITALS[src], CAPITALS[tgt]
    jobs = []
    for L in layers:
        v = mean(dev_t, L) - mean(dev_s, L)
        rnd = torch.randn(S.d_model, generator=g); rnd = rnd / rnd.norm() * v.norm()
        pool = dev_s + dev_t
        perm = torch.randperm(len(pool), generator=g).tolist()
        ga, gb = [pool[i] for i in perm[:len(dev_t)]], [pool[i] for i in perm[len(dev_t):]]
        shuf = mean(ga, L) - mean(gb, L)
        for i, x in enumerate(test_s):
            y = test_t[i % len(test_t)]
            e = encB[x]
            jobs.append(Job(e, [(L, "set", [e.city_last], Rc[L][ix[y]])], dict(exp="A_city", L=L, T=args.build, city=x)))
            jobs.append(Job(e, [(L, "set", [e.final], Rf[L][ix[y]])], dict(exp="A_final", L=L, T=args.build, city=x)))
            for tn in evalT:
                if not V[tn][x]:
                    continue
                et = encB[x] if tn == args.build else S.encode(TEMPLATES[tn], x)
                jobs.append(Job(et, [(L, "add", [et.city_last], v)], dict(exp="B_meandiff", L=L, T=tn, city=x)))
                if tn in (args.build, "ho_fs"):
                    jobs.append(Job(et, [(L, "add", [et.city_last], rnd)], dict(exp="ctrl_random", L=L, T=tn, city=x)))
                    jobs.append(Job(et, [(L, "add", [et.city_last], shuf)], dict(exp="ctrl_shuffled", L=L, T=tn, city=x)))
        # cross-pair transfer: third-state cities
        for s3 in [s for s in states if s not in (src, tgt)][pi % 3::6][:3]:
            for x in [c for c in CITIES[s3] if V[args.build][c]][:4]:
                e = encB[x]
                jobs.append(Job(e, [(L, "add", [e.city_last], v)], dict(exp="transfer", L=L, T=args.build, city=x, third=s3)))
    # clean baselines for restoration
    base_jobs = [Job(encB[x]) for x in test_s] + [Job(encB[y]) for y in test_t]
    bsc = run_jobs(S, base_jobs, caps, exact=[sc_, tc_], bs=args.bs)
    ld = {j.enc.city: (bsc[k, caps.index(tc_)] - bsc[k, caps.index(sc_)]).item() for k, j in enumerate(base_jobs)}
    scores = run_jobs(S, jobs, caps, exact=[sc_, tc_], bs=args.bs)
    for j, sc in zip(jobs, scores):
        pred = caps[sc.argmax()]
        m = j.meta
        r = dict(m, pair=f"{src}->{tgt}", pred=pred, flip=pred == tc_,
                 ld=(sc[caps.index(tc_)] - sc[caps.index(sc_)]).item())
        if m["exp"].startswith("A_"):
            x = m["city"]; y = test_t[test_s.index(x) % len(test_t)]
            r["restore"] = (r["ld"] - ld[x]) / (ld[y] - ld[x])
        if m["exp"] == "transfer":
            r["own_cap"] = CAPITALS[m["third"]]; r["kept_own"] = pred == r["own_cap"]
        rows.append(r)
    print(f"pair {pi} {src}->{tgt} dev/test src {len(dev_s)}/{len(test_s)} tgt {len(dev_t)}/{len(test_t)} jobs {len(jobs)} [{time.time()-t0:.0f}s]", flush=True)

# aggregate
agg = {}
keys = sorted({(r["exp"], r["T"]) for r in rows})
for exp, tn in keys:
    agg[f"{exp}|{tn}"] = {}
    for L in layers:
        rr = [r for r in rows if r["exp"] == exp and r["T"] == tn and r["L"] == L]
        d = {"flip": rate([r["flip"] for r in rr])}
        if exp.startswith("A_"):
            d["restore_mean"] = float(sum(r["restore"] for r in rr) / len(rr))
        agg[f"{exp}|{tn}"][L] = d
for k, v in agg.items():
    print(k, " ".join(f"{L}:{v[L]['flip']['rate']:.2f}" for L in layers))
save(args.model, "phase1_patching", vars(args) | {"pairs": pairs, "layers": layers}, {"agg": agg, "rows": rows})
