"""Stage 5: RL environment for the two screened tasks (city_pop, gemma-2-2b).

Task: at a GIVEN layer, at the city's last token, find a featurizer (an intervention subspace) such that interchanging
it from a source city makes the base city's TARGET attribute take the source's value while its other attributes
(KEEP) stay. Instances: target Continent @ L10 or Language @ L12; per instance a seed picks 150 dev cities (with labels,
visible) and the private grading items (30 base cities x 2 sources, held-out cities; dev and held-out templates).

Tools are primitives; nothing names Iso / keep. Grade (private items only):
  Cause = share of target-attribute prompts answering the source's value (pairs that differ in the target)
  Iso   = mean over keep attributes of the share of their prompts still answering the base's value
          (pairs differing in the target and in that attribute)
  KL    = mean next-token KL(clean || edited) on neutral prompts mentioning the base city;
          factor = 1 - min(1, max(0, KL - KL_BUDGET) / KL_BUDGET)  (a damage guard: free up to the budget)
  score = Cause x Iso x factor; pass = score >= PASS.
Refusals: every held-out city and a decoy set of other RAVEL cities get the same generic refusal; templates that
near-copy a held-out template are refused the same way.
"""
from __future__ import annotations

import difflib
import json
import random
import re
import time
import unicodedata

import torch

from ..common import ROOT
from .core import Item, Iv, Model, iv_interchange, iv_set
from .das import Subspace, surfaces
from .evalkit import load_survivor
from .families import RAVEL_DIR, Template
from .sae import SAE, iv_sae
from .stage1 import flat_cands, value_scores

TASKS = {"Continent": 10, "Language": 12}
PASS, KL_BUDGET = 0.5, None   # KL_BUDGET fixed from reference runs before any agent episode (see calibrate_kl)
N_DEV, N_BASE, N_SRC = 150, 30, 2
NEUTRAL = ["I visited %s last summer and", "My cousin moved to %s because", "The weather in %s today is"]
BUDGET = dict(tool_calls=40, train_calls=4, train_seconds=2400, max_examples=3000, max_k=256, max_steps=600,
              max_pairs_per_call=64, max_cities_per_call=64)
INST_DIR = ROOT / "results" / "screen" / "env" / "instances"
_GEN = "not available: this city or template is not part of the agent-accessible data"


def _norm(s: str) -> str:
    """refusal key: accents stripped, case/punctuation/zero-width collapsed, saint/st and sainte/ste unified,
    spaces removed (audit M3: spelling aliases must not reach a refused city)."""
    s = unicodedata.normalize("NFKD", s.replace("\u200b", ""))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = re.sub(r"[\W_]+", " ", s).strip()
    s = re.sub(r"\bsainte\b", "ste", re.sub(r"\bsaint\b", "st", s))
    return s.replace(" ", "")


_ALIASES: dict = {}


def aliases() -> dict:
    """normalised name -> set of normalised GeoNames alternate names (cities15000; ~/ravel)."""
    if not _ALIASES:
        path = RAVEL_DIR.parent / "cities15000.txt"
        if path.exists():
            for line in path.read_text().splitlines():
                p = line.split("\t")
                names = {_norm(x) for x in [p[1], p[2]] + p[3].split(",") if len(x) >= 3}
                for n in {_norm(p[1]), _norm(p[2])}:
                    _ALIASES.setdefault(n, set()).update(names)
    return _ALIASES


# ------------------------------------------------------------------ instances
def make_instance(target: str, seed: int) -> dict:
    F = load_survivor("city_pop")
    rng = random.Random(seed * 101 + (0 if target == "Continent" else 7))
    dev = sorted(rng.sample(F.ents["train"], N_DEV))
    private = F.ents["test"]
    bases = rng.sample(private, N_BASE)
    items = []
    for b in bases:
        others = [e for e in private if e != b and F.labels[e][target] != F.labels[b][target]]
        for s in rng.sample(others, N_SRC):
            items.append([b, s])
    all_city = set(json.loads((RAVEL_DIR / "ravel_city_entity_attributes.json").read_text()))
    decoys = sorted(rng.sample(sorted(all_city - set(F.ents["train"]) - set(private)), 300))
    return dict(id=f"{target}_s{seed}", target=target, layer=TASKS[target], keep=[a for a in F.attrs if a != target],
                dev=dev, grade_pairs=items, refused=sorted(set(private) | set(decoys)), seed=seed)


def save_instances():
    INST_DIR.mkdir(parents=True, exist_ok=True)
    for t in TASKS:
        for s in range(4):
            inst = make_instance(t, s)
            (INST_DIR / f"{inst['id']}.json").write_text(json.dumps(inst, indent=1))


def load_instance(iid: str) -> dict:
    return json.loads((INST_DIR / f"{iid}.json").read_text())


# ------------------------------------------------------------------ environment
class CityEnv:
    def __init__(self, M: Model, inst: dict, sae_tools: bool = False):
        self.M, self.inst, self.sae_tools = M, inst, sae_tools
        self.F = load_survivor("city_pop")
        self.surf = surfaces(M, self.F)
        self.dev = set(inst["dev"])
        al = aliases()
        self.refused = {_norm(c) for c in inst["refused"]}
        self.refused |= {x for c in list(self.refused) for x in al.get(c, ())} - {_norm(c) for c in inst["dev"]}
        self.public_t = {a: self.F.tpls.get((a, "train"), []) for a in self.F.attrs}
        self.heldout_t = [t.text for a in self.F.attrs for t in self.F.tpls.get((a, "test"), [])]
        self.tid = {f"{a[:4].lower()}{i}": t for a, ts in self.public_t.items() for i, t in enumerate(ts)}
        self.subspaces: dict = {}
        self.calls = self.train_calls = 0
        self.train_secs = 0.0
        self.log, self.done, self.submission = [], False, None
        self._sae = None

    # ---------------- guards
    def _city(self, c):
        if not isinstance(c, str) or _norm(c) in self.refused:
            raise PermissionError(_GEN)
        return c

    def _template(self, t):
        """template id or raw text with exactly one %s -> (Template-like object, attribute or None)."""
        if not isinstance(t, str):
            raise ValueError("template must be a string (an id from describe_task or text containing %s)")
        if t in self.tid:
            tp = self.tid[t]
            return tp, tp.attr
        if t.count("%s") != 1:
            raise ValueError("a raw template must contain exactly one %s")
        nt = _norm(t)
        for h in self.heldout_t:
            if difflib.SequenceMatcher(None, nt, _norm(h)).ratio() > 0.8:
                raise PermissionError(_GEN)
        return Template(t, None, "custom", {}), None

    def _attr_of(self, tp, attr):
        a = tp.attr or attr
        if a is None:
            return None, None
        if a not in self.F.attrs:
            raise ValueError(f"attribute must be one of {self.F.attrs}")
        verb = tp.verb or {v: [v] for v in sorted({self.F.labels[e][a] for e in self.F.labels})}
        return a, Template(tp.text, a, "custom", verb)

    def _enc(self, text, city):
        it = self.M.encode(text, city)
        if it is None:
            raise ValueError(f"cannot locate the last token of {city!r} in this template")
        return it

    # ---------------- dispatch
    def call(self, name: str, args: dict) -> str:
        t0 = time.time()
        self.calls += 1
        try:
            if self.done:
                raise RuntimeError("episode already submitted")
            if self.calls > BUDGET["tool_calls"]:
                raise RuntimeError("tool-call budget exhausted; call submit")
            fn = {"describe_task": self.describe_task, "run_prompts": self.run_prompts,
                  "cache_activations": self.cache_activations, "interchange": self.interchange,
                  "train_subspace": self.train_subspace, "submit": self.submit}
            if self.sae_tools:
                fn.update(sae_top_latents=self.sae_top_latents, sae_latent_info=self.sae_latent_info)
            if name not in fn:
                raise ValueError(f"unknown tool {name}")
            out = fn[name](**{k: v for k, v in (args or {}).items() if v not in (None, "", [], {})})
        except PermissionError as e:
            out = {"error": str(e)}
        except Exception as e:  # any tool error goes back to the agent
            out = {"error": f"{type(e).__name__}: {e}"}
        s = json.dumps(out, default=float)
        self.log.append(dict(tool=name, args=args, out=s[:4000], secs=round(time.time() - t0, 1)))
        return s

    # ---------------- tools
    def describe_task(self):
        i, F = self.inst, self.F
        return {
            "goal": (f"In {self.M.name} (bf16, {self.M.n_layers} layers, d_model {self.M.d}), find a featurizer at layer "
                     f"{i['layer']}, at the last token of a city's name, such that interchanging it from a SOURCE city into a "
                     f"BASE city's forward pass makes the model answer the source city's {i['target']} on {i['target']} "
                     f"questions about the base city, while its answers about {' and '.join(i['keep'])} stay those of the "
                     f"base city. Submit the featurizer."),
            "intervention": ("interchange at layer L (residual stream after block L) at the base city's last token: "
                             "h_base <- h_base + U U^T (h_source - h_base), U = the featurizer's orthonormal columns "
                             "(featurizer 'full' replaces h_base by h_source). The source representation is taken from "
                             "the source city in a prompt of your choice (default: a random dev template)."),
            "grading": (f"On private held-out cities (not accessible) and held-out prompt wordings: score = Cause x Iso x "
                        f"KL_factor. Cause = share of {i['target']} prompts answering the source's {i['target']}; Iso = "
                        f"share of {'/'.join(i['keep'])} prompts still answering the base's value; KL_factor = 1 - "
                        f"min(1, max(0, KL - {KL_BUDGET}) / {KL_BUDGET}) with KL = mean next-token KL(clean||edited) on neutral sentences about "
                        f"the base city. Pass: score >= {PASS}. The submitted featurizer must be at layer {i['layer']}."),
            "attributes": {a: sorted({F.labels[e][a] for e in F.labels}) for a in F.attrs},
            "dev_cities": {c: F.labels[c] for c in i["dev"]},
            "templates": {k: {"attribute": t.attr, "text": t.text} for k, t in self.tid.items()},
            "tools_note": ("templates: an id above or your own text with %s for the city. run_prompts/interchange answer "
                           "by exact log-prob over the attribute's value list when the attribute is known."),
            "budgets": BUDGET,
        }

    def run_prompts(self, template, cities, attribute=None, top_k=5):
        tp, a = self._template(template)
        cities = [self._city(c) for c in cities][:BUDGET["max_cities_per_call"]]
        items = [self._enc(tp.text, c) for c in cities]
        logits, _, _, _ = self.M.forward(items)
        lp = torch.log_softmax(logits, -1)
        top = lp.topk(min(int(top_k), 20), -1)
        out = []
        for j, c in enumerate(cities):
            out.append({"city": c, "top_tokens": [[self.M.tok.decode([int(t)]), round(float(v), 2)]
                                                  for t, v in zip(top.indices[j], top.values[j])]})
        a, tpa = self._attr_of(tp, attribute)
        if a:
            vals, cands, owner = flat_cands(self.M, tpa)
            vs = value_scores(self.M.score(items, cands), owner, len(vals))
            for j in range(len(cities)):
                out[j]["answer"] = vals[int(vs[j].argmax())]
        return {"attribute": a, "results": out}

    def cache_activations(self, template, cities, layer):
        tp, _ = self._template(template)
        cities = [self._city(c) for c in cities][:BUDGET["max_cities_per_call"]]
        reps = self.M.capture([self._enc(tp.text, c) for c in cities], [int(layer)])[int(layer)]
        n = reps.norm(dim=1)
        cs = torch.nn.functional.normalize(reps, dim=1)
        sim = (cs @ cs.T)
        return {"layer": int(layer), "norms": {c: round(float(x), 1) for c, x in zip(cities, n)},
                "mean_cosine_similarity": round(float((sim.sum() - len(cities)) / max(1, len(cities) * (len(cities) - 1))), 3),
                "note": "raw vectors stay server-side; use interchange / train_subspace to act on them"}

    def _featurizer(self, featurizer, layer=None):
        if featurizer == "full":
            if layer is None:
                raise ValueError("featurizer 'full' needs `layer`")
            return int(layer), None, None
        if isinstance(featurizer, str) and featurizer.startswith("sae:"):
            if not self.sae_tools:
                raise ValueError("SAE featurizers are not available in this variant")
            ids = [int(x) for x in re.findall(r"\d+", featurizer[4:])][:256]
            return self.inst["layer"], None, ids
        if featurizer not in self.subspaces:
            raise ValueError(f"unknown featurizer {featurizer!r}; use 'full', a train_subspace id"
                             + (" or 'sae:[ids]'" if self.sae_tools else ""))
        L, U = self.subspaces[featurizer]
        return L, U, None

    def _iv(self, L, U, sae_ids, src):
        if U is not None:
            return iv_interchange(L, U, src)
        if sae_ids is not None:
            return iv_sae(L, self._sae_at(L), sae_ids, src)
        return iv_set(L, src)

    def interchange(self, featurizer, pairs, template, attribute=None, source_template=None, layer=None):
        """template: one id/text or a list of them (each evaluated on all pairs). Per template: per-pair clean / edited /
        source answers and a summary {kept_base, took_source, other}."""
        L, U, sae_ids = self._featurizer(featurizer, layer)
        templates = template if isinstance(template, list) else [template]
        if len(templates) > 6:
            raise ValueError("at most 6 templates per call")
        pairs = pairs[:BUDGET["max_pairs_per_call"]]
        bases = [self._city(p["base"]) for p in pairs]
        srcs = [self._city(p["source"]) for p in pairs]
        res = []
        for t in templates:
            tp, _ = self._template(t)
            a, tpa = self._attr_of(tp, attribute)
            stp, _ = self._template(source_template) if source_template else (tp, None)
            src = self.M.capture([self._enc(stp.text, s) for s in srcs], [L])[L]
            items = [self._enc(tp.text, b) for b in bases]
            iv = self._iv(L, U, sae_ids, src)
            out = []
            if a:
                vals, cands, owner = flat_cands(self.M, tpa)
                clean = value_scores(self.M.score(items, cands), owner, len(vals))
                edit = value_scores(self.M.score(items, cands, iv), owner, len(vals))
                sclean = value_scores(self.M.score([self._enc(tp.text, s) for s in srcs], cands), owner, len(vals))
                summ = {"kept_base": 0, "took_source": 0, "other": 0, "n_pairs_where_base_and_source_differ": 0}
                for j in range(len(pairs)):
                    c, e, sa = (vals[int(x[j].argmax())] for x in (clean, edit, sclean))
                    out.append({"base": bases[j], "source": srcs[j], "clean": c, "edited": e, "source_answer": sa})
                    if c != sa:
                        summ["n_pairs_where_base_and_source_differ"] += 1
                        summ["kept_base" if e == c else "took_source" if e == sa else "other"] += 1
                res.append({"template": t if isinstance(t, str) and t in self.tid else tp.text, "attribute": a,
                            "summary_over_differing_pairs": summ, "results": out})
            else:
                logits, _, _, _ = self.M.forward(items, iv.chunk(0, len(items)), L)
                top = torch.log_softmax(logits, -1).topk(5, -1)
                for j in range(len(pairs)):
                    out.append({"base": bases[j], "source": srcs[j],
                                "edited_top_tokens": [self.M.tok.decode([int(x)]) for x in top.indices[j]]})
                res.append({"template": tp.text, "attribute": None, "results": out})
        return {"layer": L, "featurizer": featurizer, "per_template": res}

    def _expand_recipes(self, recipes, rng):
        """recipe {n, templates: [ids], answer: 'source:<Attr>'|'base:<Attr>', differ?: [attrs], weight?} -> examples
        over random pairs of distinct dev cities (optionally required to differ in the listed attributes)."""
        out, dev = [], self.inst["dev"]
        for r in recipes:
            n = int(r.get("n", 0))
            m = re.fullmatch(r"(source|base):(.+)", str(r.get("answer", "")).strip())
            if not m or m.group(2) not in self.F.attrs:
                raise ValueError(f"recipe answer must be 'source:<attribute>' or 'base:<attribute>', attributes {self.F.attrs}")
            who, attr = m.groups()
            tids = r.get("templates") or []
            if not tids or any(t not in self.tid for t in tids):
                raise ValueError("recipe templates must be a non-empty list of template ids from describe_task")
            differ = [d for d in (r.get("differ") or []) if d in self.F.attrs]
            tries = 0
            while n > 0 and tries < 50 * n:
                tries += 1
                b, s = rng.sample(dev, 2)
                if any(self.F.labels[b][d] == self.F.labels[s][d] for d in differ):
                    continue
                out.append(dict(base=b, source=s, template=tids[len(out) % len(tids)],
                                answer=self.F.labels[s if who == "source" else b][attr], weight=r.get("weight", 1.0)))
                n -= 1
        return out

    def train_subspace(self, layer, k, examples=None, recipes=None, steps=300, lr=2e-3, seed=0):
        """examples: [{base, source, template, answer, weight?, source_template?}] and/or recipes (see
        _expand_recipes) -> featurizer id. Trains U (orthonormal, k columns) so that interchanging it at `layer` makes
        each base prompt answer `answer`. 10% of the examples are held out and reported (loss, accuracy)."""
        if self.train_calls >= BUDGET["train_calls"]:
            raise RuntimeError("train_subspace call budget exhausted")
        if self.train_secs >= BUDGET["train_seconds"]:
            raise RuntimeError("training compute budget exhausted")
        L, k, steps = int(layer), int(k), min(int(steps), BUDGET["max_steps"])
        if not 1 <= k <= BUDGET["max_k"] or not 0 <= L < self.M.n_layers:
            raise ValueError(f"need 1 <= k <= {BUDGET['max_k']} and 0 <= layer < {self.M.n_layers}")
        rng = random.Random(seed)
        examples = list(examples or []) + self._expand_recipes(recipes or [], rng)
        examples = examples[:BUDGET["max_examples"]]
        if len(examples) < 10:
            raise ValueError("need at least 10 examples (explicit and/or from recipes)")
        self.train_calls += 1
        t0 = time.time()
        dev_t = [t for ts in self.public_t.values() for t in ts]
        rows = []  # (base enc, label ids, source enc, weight, template, answer)
        for ex in examples:
            tp, a = self._template(ex["template"])
            stp = self._template(ex["source_template"])[0] if ex.get("source_template") else rng.choice(dev_t)
            b, s = self._city(ex["base"]), self._city(ex["source"])
            ans = str(ex["answer"])
            surf = self.surf.get((tp.text, ans), (tp.verb.get(ans) or [ans])[0]) if a else ans
            lab = self.M.cand_ids(tp.lead, (surf,))[0]
            rows.append((self._enc(tp.text, b), lab, self._enc(stp.text, s), float(ex.get("weight", 1.0)), tp, ans, a))
        rng.shuffle(rows)
        n_val = max(1, len(rows) // 10)
        val, tr = rows[:n_val], rows[n_val:]
        src_all = self.M.capture([r[2] for r in rows], [L])[L]
        src_val, src = src_all[:n_val], src_all[n_val:]
        base_items = [Item(r[0].ids + r[1][:-1], r[0].pos) for r in tr]
        S = Subspace(self.M.d, k, seed).to(self.M.device)
        opt = torch.optim.Adam(S.parameters(), lr=float(lr))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: 1 - st / steps)
        hist, cap = [], BUDGET["train_seconds"] - self.train_secs

        def batch_loss(items, labs, ws, sr, U, grad):
            keep = max(len(l) for l in labs)
            logits, _, _, _ = self.M.forward(items, lambda x: x.float() + (sr - x.float()) @ U @ U.T, L, grad=grad,
                                             keep=keep)
            if keep == 1:
                logits = logits.unsqueeze(1)
            lp = torch.log_softmax(logits, -1)
            tot = 0
            for r, lab in enumerate(labs):
                tot = tot - ws[r] * lp[r, keep - len(lab):].gather(-1, torch.tensor(lab, device=self.M.device)
                                                                  .unsqueeze(-1)).sum()
            return tot / len(labs)
        for step in range(steps):
            if time.time() - t0 > cap:
                break
            idx = rng.sample(range(len(tr)), min(32, len(tr)))
            loss = batch_loss([base_items[i] for i in idx], [tr[i][1] for i in idx], [tr[i][3] for i in idx],
                              src[idx].to(self.M.device), S.U(), True)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            hist.append(loss.item())
        U = S.U().detach().float()
        fid = f"sub{len(self.subspaces)}"
        self.subspaces[fid] = (L, U.cpu())
        # held-out report: loss and answer accuracy (exact log-prob over the attribute's values where known)
        with torch.no_grad():
            vitems = [Item(r[0].ids + r[1][:-1], r[0].pos) for r in val]
            vloss = float(batch_loss(vitems, [r[1] for r in val], [1.0] * len(val), src_val.to(self.M.device), U, False))
            hits, n_acc = 0, 0
            by_t = {}
            for j, r in enumerate(val):
                if r[6]:
                    by_t.setdefault(r[4].text, []).append(j)
            for txt, idx in by_t.items():
                vals, cands, owner = flat_cands(self.M, val[idx[0]][4])
                vs = value_scores(self.M.score([val[j][0] for j in idx], cands,
                                               iv_interchange(L, U.cpu(), src_val[idx])), owner, len(vals))
                for jj, j in enumerate(idx):
                    hits += vals[int(vs[jj].argmax())] == val[j][5]
                    n_acc += 1
        secs = time.time() - t0
        self.train_secs += secs
        return {"featurizer": fid, "layer": L, "k": k, "n_train": len(tr), "n_heldout": len(val), "steps_run": len(hist),
                "loss_first10": round(sum(hist[:10]) / max(1, len(hist[:10])), 3),
                "loss_last10": round(sum(hist[-10:]) / max(1, len(hist[-10:])), 3),
                "heldout_loss": round(vloss, 3), "heldout_answer_accuracy": round(hits / max(1, n_acc), 3),
                "seconds": round(secs), "train_seconds_left": round(BUDGET["train_seconds"] - self.train_secs)}

    def _sae_at(self, L):
        if self._sae is None or self._sae.layer != L:
            self._sae = SAE(L)
        return self._sae

    def sae_top_latents(self, template, city, top_k=20):
        tp, _ = self._template(template)
        L = self.inst["layer"]
        h = self.M.capture([self._enc(tp.text, self._city(city))], [L])[L]
        z = self._sae_at(L).enc(h.cuda())[0]
        top = z.topk(min(int(top_k), 50))
        return {"layer": L, "sae": self._sae_at(L).id, "latents": [[int(i), round(float(v), 2)]
                                                                  for i, v in zip(top.indices, top.values) if v > 0]}

    def sae_latent_info(self, latent, n=10):
        """dev cities (over the dev templates of all attributes) where the latent fires most."""
        L, lat = self.inst["layer"], int(latent)
        if not hasattr(self, "_dev_codes"):
            items, who = [], []
            for c in self.inst["dev"]:
                for ts in self.public_t.values():
                    t = ts[0]
                    it = self.M.encode(t.text, c)
                    if it is not None:
                        items.append(it); who.append(c)
            reps = self.M.capture(items, [L])[L]
            with torch.no_grad():
                z = torch.cat([self._sae_at(L).enc(reps[i:i + 256].cuda()).cpu() for i in range(0, len(reps), 256)])
            self._dev_codes = (who, z)
        who, z = self._dev_codes
        col = z[:, lat]
        order = col.argsort(descending=True)[:int(n)]
        return {"latent": lat, "frac_active": round(float((col > 0).float().mean()), 3),
                "top_dev_cities": [[who[i], round(float(col[i]), 2)] for i in order.tolist() if col[i] > 0]}

    def submit(self, featurizer):
        L, U, sae_ids = self._featurizer(featurizer, self.inst["layer"])
        if L != self.inst["layer"]:
            raise ValueError(f"the submitted featurizer must be at layer {self.inst['layer']} (it is at {L})")
        self.submission = dict(featurizer=featurizer, layer=L, U=U, sae_ids=sae_ids)
        self.done = True
        return {"submitted": featurizer}


# ------------------------------------------------------------------ grader
def grade(M: Model, inst: dict, submission: dict | None, kl_budget: float | None = None, bases: set | None = None,
          items_out: bool = False) -> dict:
    """bases: restrict to grade pairs whose base city is in this set (cross-fitting). items_out: also return per-item
    records. Soft metrics: mean probability (softmax over the attribute's candidate values) of the expected value."""
    if not submission:
        return dict(score=0.0, passed=False, reason="no submission")
    kl_budget = kl_budget or KL_BUDGET
    F = load_survivor("city_pop")
    L, U, sae_ids = submission["layer"], submission["U"], submission["sae_ids"]
    tgt = inst["target"]
    rng = random.Random(inst["seed"])
    rows = []  # (attr, template, base, source)
    gpairs = [p for p in inst["grade_pairs"] if bases is None or p[0] in bases]
    for sp in ("train", "test"):
        for b, s in gpairs:
            for a in F.attrs:
                ts = [t for t in F.tpls.get((a, sp), []) if b in F.correct[(a, t.text)]]
                if not ts:
                    continue
                if a != tgt and F.labels[s][a] == F.labels[b][a]:
                    continue
                for t in rng.sample(ts, min(2, len(ts))):
                    rows.append((a, t, b, s, sp))
    src_t = [t for a in F.attrs for t in F.tpls.get((a, "train"), [])]
    s_items = [M.encode(rng.choice(src_t).text, s) for _, _, _, s, _ in rows]
    src = M.capture(s_items, [L])[L]
    ok = {"cause": [], **{a: [] for a in inst["keep"]}}
    soft = {"cause": [], **{a: [] for a in inst["keep"]}}
    recs = []
    by_t = {}
    for j, r in enumerate(rows):
        by_t.setdefault(r[1].text, []).append(j)
    for txt, idx in by_t.items():
        tp = rows[idx[0]][1]
        vals, cands, owner = flat_cands(M, tp)
        items = [M.encode(tp.text, rows[j][2]) for j in idx]
        sub = src[idx]
        iv = (iv_interchange(L, U, sub) if U is not None else
              iv_sae(L, SAE(L), sae_ids, sub) if sae_ids is not None else iv_set(L, sub))
        vs = value_scores(M.score(items, cands, iv), owner, len(vals))
        pv = torch.softmax(vs, -1)
        for jj, j in enumerate(idx):
            a, _, b, s, sp = rows[j]
            pred = vals[int(vs[jj].argmax())]
            want = F.labels[s][a] if a == tgt else F.labels[b][a]
            key = "cause" if a == tgt else a
            ok[key].append(pred == want)
            soft[key].append(float(pv[jj, vals.index(want)]))
            recs.append((key, b, s, sp, int(pred == want), round(soft[key][-1], 4)))
    cause = sum(ok["cause"]) / max(1, len(ok["cause"]))
    iso = sum(sum(v) / max(1, len(v)) for a, v in ok.items() if a != "cause") / len(inst["keep"])
    # KL on neutral prompts about the base city
    nb = [M.encode(t, b) for t in NEUTRAL for b, _ in gpairs]
    ns = torch.cat([src[[j for j, r in enumerate(rows) if r[2] == b][:1]] for t in NEUTRAL for b, _ in gpairs])
    clean, _, _, _ = M.forward(nb)
    iv = (iv_interchange(L, U, ns) if U is not None else iv_sae(L, SAE(L), sae_ids, ns) if sae_ids is not None
          else iv_set(L, ns))
    edit, _, _, _ = M.forward(nb, iv.chunk(0, len(nb)), L)
    pc, pe = torch.log_softmax(clean, -1), torch.log_softmax(edit, -1)
    kl = float((pc.exp() * (pc - pe)).sum(-1).mean())
    # hinge guard (2026-10-02, before the agent batch): no penalty up to the budget, linear to 0 at 2x budget
    factor = 1 - min(1.0, max(0.0, kl - kl_budget) / kl_budget) if kl_budget else 1.0
    score = cause * iso * factor
    s_c = sum(soft["cause"]) / max(1, len(soft["cause"]))
    s_i = sum(sum(v) / max(1, len(v)) for a, v in soft.items() if a != "cause") / len(inst["keep"])
    out = dict(score=score, passed=score >= PASS, cause=cause, iso=iso, kl=kl, kl_factor=factor,
               soft_cause=s_c, soft_iso=s_i, soft_score=s_c * s_i * factor,
               n_cause=len(ok["cause"]), n_iso={a: len(v) for a, v in ok.items() if a != "cause"},
               iso_by={a: sum(v) / max(1, len(v)) for a, v in ok.items() if a != "cause"})
    if items_out:
        out["items"] = recs  # (cause|keep-attr, base, source, template split, correct, p_expected)
    return out


def metrics_from_items(items, keep, kl_factor=1.0, bases=None):
    """recompute hard/soft scores from per-item records (for bootstrap / subsets)."""
    sel = [r for r in items if bases is None or r[1] in bases]
    def m(key, j):
        v = [r[j] for r in sel if r[0] == key]
        return sum(v) / max(1, len(v))
    c, sc = m("cause", 4), m("cause", 5)
    i = sum(m(a, 4) for a in keep) / len(keep)
    si = sum(m(a, 5) for a in keep) / len(keep)
    return dict(score=c * i * kl_factor, cause=c, iso=i, soft_score=sc * si * kl_factor)
