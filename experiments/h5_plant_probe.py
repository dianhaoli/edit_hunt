"""H5 decision probe (pre-registered in LAB_NOTEBOOK 2026-10-02): does a planted-edit 'undo' task break the
'proxy + optimise' recipe that solves T2?

Plant (privileged designer): at layer Lp, EVERY position: h += clamp((k.h - b)/w, 0, 1) * v, with k = unit(mean
source-state cities - mean other cities) at Lp, ramp 30-60% between the two group means, v = scale * (mean target
cities - mean source cities) ("state plant"). It fires on the model's own representation (no city list), like a
rank-one MLP edit. The agent would only see the planted model.

Score: restoration = 1 - sum KL(clean || fixed) / sum KL(clean || planted), clipped at 0, over private prompts:
P1 source-state cities (not used by any fix) x 9 contexts, P2 other-state cities x 9 contexts, P3 generic text.
Full next-token distribution at the final token.

Fixes compared:
  null                nothing                                                     (0 by definition)
  exact               plant removed                                               (1 by definition)
  inverse_true        gated edit at Lp, all positions, true k/b/w, -v (never opens: the plant moves h along -k)
  inverse_post        true -v behind a gate fit on post-plant activations       (what the edit format can express)
  patch_L{x}          T2 recipe for restoration at layer x, city_last: gate from planted activations (8 accessible
                      source cities vs 8 other cities), push trained 150 steps on visible truths (true capital fs1/zs1,
                      true state on STATE_Q, true abbreviation on a visible abbreviation prompt, other cities keep
                      their capital); x in {Lp, Lp+4, 20}
  patch_L{x}_all      same, but the patch is applied at all positions (gate decides)
  detect              fair detector: localise Lp by the layer where source-city residual updates deviate most from
                      control states' (planted model only), estimate v from that update, gate from post-plant
                      projections, apply the inverse at all positions
  PYTHONPATH=. .venv/bin/python experiments/h5_plant_probe.py"""
import json
import math
import random
import time

import torch

from edithunt.common import ROOT
from edithunt.data import (ABBR, CAPITALS, CITIES, CITIES_EXT, CITY2STATE_EXT, COUNTRY_Q, GENERIC_SENTENCES, READOUTS,
                           STATE_Q, STATE_Q_HO, STATES, TEMPLATES, study_states)
from edithunt.hooks import Intervention
from edithunt.metrics import kl
from edithunt.model import Subject

OUT = ROOT / "results" / "suite" / "h5_plant"
OUT.mkdir(parents=True, exist_ok=True)
S = Subject("Qwen/Qwen2.5-1.5B")
S.use_prefix_cache = False  # the plant must also act inside few-shot prefixes
FIT_T = ["fs1", "zs1"]
VIS_ABBR = "Austin, TX\nBoston, MA\nDenver, CO\n{city},"  # a visible abbreviation prompt (not the grader's)
CTX = {  # evaluation contexts (private to the grader)
    "cap_ho_fs": TEMPLATES["ho_fs"], "cap_ho_zs": TEMPLATES["ho_zs"], "state_ho": STATE_Q_HO,
    "abbr": READOUTS["abbr"][0], "country": COUNTRY_Q,
    "free_known": "{city} is best known for its",
    "free_team": "I grew up in {city}, so naturally my favourite football team is the",
    "free_weather": "In January, the weather in {city} is usually",
    "free_drive": "If you drive two hours from {city}, you can reach",
}


class Plant:
    def __init__(self, L, k, b, w, v):
        self.L, self.k, self.b, self.w, self.v, self.on = L, k.to(S.device), float(b), float(w), v.to(S.device), True
        self.fix = None  # optional all-position gated fix applied after the plant at layer fix[0]
        self.handles = [S.layers[L].register_forward_hook(self.hook)]

    def gate(self, h, k, b, w):
        return ((h.float() @ k - b) / w).clamp(0, 1)

    def hook(self, mod, inp, out):
        if not self.on:
            return None
        h = out[0] if isinstance(out, tuple) else out
        h = h + self.gate(h, self.k, self.b, self.w).to(h.dtype).unsqueeze(-1) * self.v.to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    def remove(self):
        for hd in self.handles:
            hd.remove()


class AllPosFix:
    """Gated edit at every position of layer L (registered after the plant, so it sees post-plant h)."""

    def __init__(self, L, k, b, w, v):
        self.k, self.b, self.w, self.v = k.to(S.device).float(), float(b), float(w), v.to(S.device)
        self.hd = S.layers[L].register_forward_hook(self.hook)

    def hook(self, mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        g = ((h.float() @ self.k - self.b) / self.w).clamp(0, 1)
        h = h + g.to(h.dtype).unsqueeze(-1) * self.v.to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    def remove(self):
        self.hd.remove()


def enc(tpl, cities):
    out = []
    for c in cities:
        try:
            out.append(S.encode(tpl, c))
        except AssertionError:
            pass
    return out


def acts(encs, layers):
    return S.resid(encs, layers, "city_last", bs=16)


def first_tok(a):
    return S.cand_tokens([a])[0][0]


def train_gated_push(L, groups, E, Ek, steps=150, lr=0.05):
    """Gate key from E (fire) vs Ek (stay closed) at layer L, ramp 30-60%; push v trained on groups [(encs, ids)]
    at city_last. Returns (k, b, w, v)."""
    Rs, Ro = acts(E, [L])[L].float(), acts(Ek, [L])[L].float()
    k = Rs.mean(0) - Ro.mean(0); k = k / k.norm()
    ms, mo = (Rs @ k).mean().item(), (Ro @ k).mean().item()
    b, w = mo + 0.3 * (ms - mo), 0.3 * (ms - mo)
    rn = Rs.norm(dim=-1).mean().item()
    v = torch.zeros(S.d_model, device=S.device, requires_grad=True)
    opt = torch.optim.Adam([v], lr=lr * rn / math.sqrt(S.d_model))
    kd, bd, wd = k.to(S.device), torch.tensor(b, device=S.device), torch.tensor(w, device=S.device)
    for _ in range(steps):
        opt.zero_grad()
        loss = 0.0
        for es, ids in groups:
            pos = [[e.city_last] for e in es]
            lp, *_ = S.forward(es, [Intervention(L, pos, "gate", (kd, bd, wd, v))], grad=True)
            loss = loss - lp[torch.arange(len(es)), torch.tensor(ids, device=S.device)].mean()
        loss.backward(); opt.step()
    return k, b, w, v.detach()


def truth_groups(src, oth, capital_of=None):
    """Visible-truth objective: capital (fs1/zs1), state (STATE_Q), abbreviation (a visible abbreviation prompt) of
    src cities; other cities keep their own capital. capital_of overrides the capital target for src (plant design)."""
    E = enc(TEMPLATES["fs1"], src) + enc(TEMPLATES["zs1"], src)
    cap = capital_of or (lambda c: CAPITALS[CITY2STATE_EXT[c]])
    groups = [(E, [first_tok(cap(e.city)) for e in E])]
    Es = enc(STATE_Q, src); groups.append((Es, [first_tok(CITY2STATE_EXT[e.city]) for e in Es]))
    Ea = enc(VIS_ABBR, src); groups.append((Ea, [first_tok(ABBR[CITY2STATE_EXT[e.city]]) for e in Ea]))
    Ek = enc(TEMPLATES["fs1"], oth) + enc(TEMPLATES["zs1"], oth)
    groups.append((Ek, [first_tok(CAPITALS[CITY2STATE_EXT[e.city]]) for e in Ek]))
    return groups, E, Ek


def train_patch(L, src, oth, steps=150):
    """T2 gated keep recipe, objective = restore visible truths (the agent knows the right answers)."""
    groups, E, Ek = truth_groups(src, oth)
    return train_gated_push(L, groups, E, Ek, steps)


def detect(src, ctrl_states, oth, n_layers):
    """Fair detector on the planted model only. Layer = argmax over L of the source-vs-others difference in the
    residual UPDATE (h_L - h_{L-1}) minus the same quantity for control states; v = that update difference."""
    Ls = list(range(n_layers))
    def upd_mean(cs):
        A = acts(enc(TEMPLATES["fs1"], cs) + enc(TEMPLATES["zs1"], cs), Ls)
        return {L: (A[L] - A[L - 1]).mean(0) for L in Ls[1:]}, A
    Us, As = upd_mean(src)
    Uo, Ao = upd_mean(oth)
    Uc = [upd_mean(cs)[0] for cs in ctrl_states]
    sc = {L: (Us[L] - Uo[L]).norm().item() - sum((u[L] - Uo[L]).norm().item() for u in Uc) / len(Uc) for L in Ls[1:]}
    Lh = max(sc, key=sc.get)
    vh = Us[Lh] - Uo[Lh] - sum(u[Lh] - Uo[Lh] for u in Uc) / len(Uc)
    k = As[Lh].mean(0) - Ao[Lh].mean(0); k = k / k.norm()
    ms, mo = (As[Lh] @ k).mean().item(), (Ao[Lh] @ k).mean().item()
    return Lh, sc, k, mo + 0.3 * (ms - mo), 0.3 * (ms - mo), -vh


def logp_set(encs, ivs_fn=None):
    out = []
    for i in range(0, len(encs), 16):
        ch = encs[i:i + 16]
        out.append(S.logp_final(ch, ivs_fn(ch) if ivs_fn else (), bs=16))
    return torch.cat(out)


def main():
    """v2 (2026-10-02, after v1 block 1): only cities the clean model answers correctly on fs1 and zs1 (the patch's
    'truth' then equals the clean model's answer on its training cities: the patch-favourable setting); plant types
    'state' (mean-diff push) and 'hop2' (capital-only push, trained by the designer with keep-state/keep-others);
    fixes null / exact / inverse_post / patch_L{Lp} / patch_L20 / detect. Rows -> probe2.jsonl (v1 rows: probe.jsonl)."""
    from edithunt.env.ravel import validity
    pairs = [("Minnesota", "Tennessee"), ("North Carolina", "Kentucky"), ("Wisconsin", "Mississippi"),
             ("Pennsylvania", "California")]
    out = OUT / "probe2.jsonl"
    done = {(r["pair"], r["Lp"], r["plant"], r["fix"]) for r in map(json.loads, out.read_text().splitlines())} \
        if out.exists() else set()
    f = out.open("a")
    others_all = [s for s in study_states(min_cities=5)]
    allc = sorted({c for s_ in STATES for c in CITIES_EXT[s_]})
    V = validity(S, allc)
    ok = lambda c: V[c]["fs1"] and V[c]["zs1"]
    for (src_s, tgt_s) in pairs:
        rng = random.Random(f"h5-{src_s}-{tgt_s}")
        scs = [c for c in CITIES_EXT[src_s] if ok(c)]; rng.shuffle(scs)
        vis_src, eval_src = scs[:6], scs[6:]
        oth_states = [s_ for s_ in others_all if s_ not in (src_s, tgt_s)]
        rng.shuffle(oth_states)
        okc = lambda s_: [c for c in CITIES_EXT[s_] if ok(c)]
        fit_oth = [c for s_ in oth_states[:12] for c in okc(s_)[:2]]           # designer's "others"
        vis_oth = [okc(s_)[2] for s_ in oth_states[12:20] if len(okc(s_)) > 2]  # fixes' keep cities
        ctrl = [okc(s_)[3:8] for s_ in oth_states[20:23]]                       # detector control states
        eval_oth = [okc(s_)[-1] for s_ in oth_states[:20]]
        evals = {f"src/{k}": enc(t, eval_src) for k, t in CTX.items()} | {f"oth/{k}": enc(t, eval_oth) for k, t in CTX.items()}
        evals["generic"] = [S.encode_text(t) for t in GENERIC_SENTENCES[10:50]]
        allenc = [e for v_ in evals.values() for e in v_]
        nsrc = sum(len(evals[f"src/{c}"]) for c in CTX)
        bounds = {}; o = 0
        for kk, v_ in evals.items():
            bounds[kk] = (o, o + len(v_)); o += len(v_)
        clean = logp_set(allenc)
        for plant_type in ("state", "hop2"):
            for Lp in (8, 16):
                if all((f"{src_s}->{tgt_s}", Lp, plant_type, fx) in done for fx in ("null", "detect")):
                    continue
                Efs = enc(TEMPLATES["fs1"], scs) + enc(TEMPLATES["zs1"], scs)
                Efo = enc(TEMPLATES["fs1"], fit_oth) + enc(TEMPLATES["zs1"], fit_oth)
                if plant_type == "state":
                    tcs = [c for c in CITIES_EXT[tgt_s] if ok(c)]
                    Eft = enc(TEMPLATES["fs1"], tcs) + enc(TEMPLATES["zs1"], tcs)
                    Rs, Rt, Ro = acts(Efs, [Lp])[Lp], acts(Eft, [Lp])[Lp], acts(Efo, [Lp])[Lp]
                    k = Rs.mean(0) - Ro.mean(0); k = k / k.norm()
                    ms, mo = (Rs @ k).mean().item(), (Ro @ k).mean().item()
                    pk, pb, pw, v = k, mo + 0.3 * (ms - mo), 0.3 * (ms - mo), Rt.mean(0) - Rs.mean(0)
                else:  # hop2: source cities answer the target capital, keep their state/abbreviation; others keep
                    groups, E, Ek = truth_groups(scs, fit_oth[:12], capital_of=lambda c: CAPITALS[tgt_s])
                    pk, pb, pw, v = train_gated_push(Lp, groups, E, Ek, steps=150)
                    v = v.cpu()
                P = Plant(Lp, pk, pb, pw, v)
                planted = logp_set(allenc)
                kl_b = kl(clean, planted)
                cap = [CAPITALS[x] for x in STATES]
                e1 = evals["src/cap_ho_fs"]
                sc1, _ = S.score(e1, cap, bs=16)
                P.on = False; sc0, _ = S.score(e1, cap, bs=16); P.on = True
                st_e = evals["src/state_ho"]
                ss1, _ = S.score(st_e, STATES, bs=16)
                info = {"pair": f"{src_s}->{tgt_s}", "Lp": Lp, "plant": plant_type,
                        "plant_flip_ho_fs": (sc1.argmax(1) == cap.index(CAPITALS[tgt_s])).float().mean().item(),
                        "clean_acc_ho_fs": (sc0.argmax(1) == cap.index(CAPITALS[src_s])).float().mean().item(),
                        "plant_state_kept": (ss1.argmax(1) == STATES.index(src_s)).float().mean().item(),
                        "v_norm": float(v.norm()), "n_eval_src": len(eval_src),
                        "kl_before_by_set": {kk: kl_b[a:b].mean().item() for kk, (a, b) in bounds.items()}}
                print(json.dumps(info), flush=True)

                def score(name, fixed, extra=None):
                    if (info["pair"], Lp, plant_type, name) in done:
                        return
                    kl_a = kl(clean, fixed)
                    r = {**info, "fix": name, "restoration": max(0.0, 1 - kl_a.sum().item() / kl_b.sum().item()),
                         "restoration_src": 1 - kl_a[:nsrc].sum().item() / kl_b[:nsrc].sum().item(),
                         "kl_after_by_set": {kk: kl_a[a:b].mean().item() for kk, (a, b) in bounds.items()}, **(extra or {})}
                    f.write(json.dumps(r) + "\n"); f.flush()
                    print(plant_type, Lp, name, "restoration %.3f (src %.3f)" % (r["restoration"], r["restoration_src"]), flush=True)

                score("null", planted)
                P.on = False; score("exact", logp_set(allenc)); P.on = True
                Ps_, Po_ = acts(Efs, [Lp])[Lp], acts(Efo, [Lp])[Lp]  # post-plant activations
                kp = Ps_.mean(0) - Po_.mean(0); kp = kp / kp.norm()
                mps, mpo = (Ps_ @ kp).mean().item(), (Po_ @ kp).mean().item()
                fx = AllPosFix(Lp, kp, mpo + 0.3 * (mps - mpo), 0.3 * (mps - mpo), -v)
                score("inverse_post", logp_set(allenc)); fx.remove()
                for Lx in sorted({Lp, 20}):
                    t0 = time.time()
                    kx, bx, wx, vx = train_patch(Lx, vis_src, vis_oth)
                    ivs = lambda ch: [Intervention(Lx, [[e.city_last] if e.city_last >= 0 else [] for e in ch], "gate",
                                                   (kx, torch.tensor(bx), torch.tensor(wx), vx))]
                    score(f"patch_L{Lx}", logp_set(allenc, ivs), {"secs": time.time() - t0})
                Lh, sc, kh, bh, wh, vh = detect(vis_src, ctrl, vis_oth, S.n_layers)
                fx = AllPosFix(Lh, kh, bh, wh, vh)
                score("detect", logp_set(allenc), {"detected_layer": Lh, "cos_v": torch.nn.functional.cosine_similarity(
                    -vh, v.to(vh.device, vh.dtype), dim=0).item(), "layer_scores": {L: round(x, 2) for L, x in sc.items()}})
                fx.remove()
                P.remove()


if __name__ == "__main__":
    main()
