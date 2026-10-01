"""Agent tool API for EditHunt.

Vectors live server-side in named registers; the agent refers to them by name and only `submit`
turns registers into a submission. Every tool returns compact JSON text.

Budget: each prompt sequence run through the subject model costs 1 forward pass (a batch of 8
prompts = 8; one optimize_vector step on 8 prompts + 4 generic sentences = 12). Clean (unedited)
results are cached per prompt and only charged once. Out of budget -> error.

Tasks (env/tasks.py) choose a tool subset, an optional tool-call budget, and (detective) a planted edit that
every forward pass on the "planted" model applies at the last token of any source-state city.

Held-out data: tools reject (a) any city in the instance's held-out test set, (b) any raw prompt
containing such a city as a word, (c) any template/prompt matching a held-out template string.
Every other city (dev cities, other cities the agent knows) and paraphrased templates are allowed. Private
readout prompts (held-out state question, hidden readouts) are also rejected.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata

import torch

from ..data import CAPITALS, CITY2STATE, COUNTRY_Q, READOUTS, STATE_Q, STATE_Q_HO, STATES
from ..hooks import Intervention
from ..metrics import kl
from ..model import Subject, pos_rule
from .grader import CAPS, GENERIC_TOOLS, POSITIONS, build_ivs, check, edit_positions
from .instance import Instance

MAX_PROMPTS = 16


class ToolError(Exception):
    pass


def _r(x: float, k: int = 4) -> float:
    return round(float(x), k)


# ------------------------------------------------------------ vec_op mini-language
_TOK = re.compile(r"\s*(?:(\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?)|([A-Za-z_][A-Za-z0-9_]*)|(.))")
_FUNCS = {"normalize": 1, "proj_out": 2, "proj": 2, "dot": 2, "norm": 1, "cos": 2, "randn": 1}


def _lex(s: str) -> list[tuple[str, object]]:
    out, i = [], 0
    s = s.strip()
    while i < len(s):
        m = _TOK.match(s, i)
        if not m or m.end() == i:
            raise ToolError(f"bad expression near {s[i:]!r}")
        num, ident, op = m.groups()
        if num is not None:
            out.append(("num", float(num)))
        elif ident is not None:
            out.append(("id", ident))
        elif op.strip():
            if op not in "+-*/(),":
                raise ToolError(f"unexpected character {op!r}")
            out.append(("op", op))
        i = m.end()
    return out


class _Parser:
    """expr := term (('+'|'-') term)* ; term := unary (('*'|'/') unary)* ;
    unary := '-' unary | atom ; atom := NUMBER | REGISTER | FUNC '(' expr (',' expr)* ')' | '(' expr ')'
    Functions: normalize(v) proj_out(v,u) proj(v,u) dot(u,v) norm(v) cos(u,v) randn(seed)."""

    def __init__(self, toks, regs: dict, d: int):
        self.t, self.i, self.regs, self.d = toks, 0, regs, d

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else (None, None)

    def eat(self, v=None):
        tok = self.peek()
        if tok[0] is None or (v is not None and tok[1] != v):
            raise ToolError(f"expected {v!r}, got {tok[1]!r}")
        self.i += 1
        return tok

    def parse(self):
        v = self.expr()
        if self.i != len(self.t):
            raise ToolError(f"trailing tokens at {self.peek()[1]!r}")
        return v

    def expr(self):
        v = self.term()
        while self.peek() in (("op", "+"), ("op", "-")):
            op = self.eat()[1]; w = self.term()
            if isinstance(v, torch.Tensor) != isinstance(w, torch.Tensor):
                raise ToolError("cannot add a scalar and a vector")
            v = v + w if op == "+" else v - w
        return v

    def term(self):
        v = self.unary()
        while self.peek() in (("op", "*"), ("op", "/")):
            op = self.eat()[1]; w = self.unary()
            if isinstance(v, torch.Tensor) and isinstance(w, torch.Tensor):
                raise ToolError("vector*vector not allowed (use dot)")
            if op == "/":
                if isinstance(w, torch.Tensor) or w == 0:
                    raise ToolError("can only divide by a non-zero scalar")
                v = v / w
            else:
                v = v * w
        return v

    def unary(self):
        if self.peek() == ("op", "-"):
            self.eat(); return -self.unary()
        return self.atom()

    def atom(self):
        kind, val = self.eat()
        if kind == "num":
            return val
        if kind == "op" and val == "(":
            v = self.expr(); self.eat(")"); return v
        if kind == "id" and val in _FUNCS and self.peek() == ("op", "("):
            self.eat("(")
            args = [self.expr()]
            while self.peek() == ("op", ","):
                self.eat(); args.append(self.expr())
            self.eat(")")
            if len(args) != _FUNCS[val]:
                raise ToolError(f"{val} takes {_FUNCS[val]} args")
            return self.call(val, args)
        if kind == "id":
            if val not in self.regs:
                raise ToolError(f"unknown register {val!r}")
            return self.regs[val].clone()
        raise ToolError(f"unexpected {val!r}")

    def call(self, f, a):
        if f == "randn":
            if isinstance(a[0], torch.Tensor):
                raise ToolError("randn(seed) takes a number")
            g = torch.Generator().manual_seed(int(a[0]))
            v = torch.randn(self.d, generator=g); return v / v.norm()
        if not all(isinstance(x, torch.Tensor) for x in a):
            raise ToolError(f"{f} expects vector arguments")
        if f == "normalize":
            return a[0] / a[0].norm().clamp_min(1e-12)
        if f == "norm":
            return a[0].norm().item()
        if f == "dot":
            return (a[0] @ a[1]).item()
        if f == "cos":
            return (a[0] @ a[1] / (a[0].norm() * a[1].norm()).clamp_min(1e-12)).item()
        u = a[1] / a[1].norm().clamp_min(1e-12)
        p = (a[0] @ u) * u
        return p if f == "proj" else a[0] - p


# ------------------------------------------------------------ environment
ALL_CITIES = sorted(CITY2STATE, key=len, reverse=True)


class ToolEnv:
    def __init__(self, S: Subject, inst: Instance, bs: int = 16, tools: list[str] | None = None,
                 call_budget: int | None = None, plant_visible: bool = True):
        self.S, self.inst, self.bs = S, inst, bs
        self.task = getattr(inst, "task", "edit")
        self.tools = set(tools or TOOL_NAMES)
        self.call_budget = call_budget
        self.calls = 0
        self.budget = inst.constraints["fwd_budget"]
        self.used = 0
        self.regs: dict[str, torch.Tensor] = {}
        self.submission: dict | None = None
        self.log: list[dict] = []
        self._clean: dict = {}  # (prompt text, cands) -> (top candidate, logp final)
        held = list(inst.test_cities) + sorted({c for c, _ in getattr(inst, "leak_items", [])})  # grader-only cities
        held += list(getattr(inst, "extra", {}).get("private_cities", []))
        # guards compare normalised text (unicode/zero-width/case/spacing/punctuation), so spelling variants
        # ("Los  Angeles", "Los-Angeles", "LosAngeles", "Ames" + zero-width) and near-copies of templates are refused
        self._held = {_letters(c) for c in held}
        self._held_re = [re.compile(r"(?<![a-z0-9])" + " ?".join(map(re.escape, _norm(c).split())) + r"(?![a-z0-9])")
                         for c in held]
        private_tpls = list(inst.test_templates.values()) + [STATE_Q_HO] + [t for t, _ in READOUTS.values()]
        self._private_tpls = set(private_tpls)
        self._held_tpl = [re.compile(".*".join(re.escape(_norm(p)) for p in t.split("{city}")), re.S) for t in private_tpls]
        self.templates = dict(inst.dev_templates) | {"state_q": STATE_Q, "country_q": COUNTRY_Q}
        # T2 v2 (env/ravel.py): one generic refusal for every private city/template (the private set includes decoy
        # cities, so refusals do not reveal the Iso pool); extended city list for span detection in raw prompts
        self._generic = bool(inst.constraints.get("generic_refusal"))
        self._all_cities = ALL_CITIES
        if getattr(inst, "extra", {}).get("extended_cities"):
            from ..data import CITY2STATE_EXT
            self._all_cities = sorted(CITY2STATE_EXT, key=len, reverse=True)
        # no target capital in tasks where it is unknown (erase) or secret (detective)
        self.tgt_cap = inst.target_capital if self.task not in ("detective", "erase") and inst.target_capital else None
        pl = getattr(inst, "extra", {}).get("plant")
        self.plant = None
        if pl:
            self.plant = (pl["layer"], torch.tensor(pl["vector"], dtype=torch.float32), set(CITIES_OF(pl["source"])))

    # ---------------- plumbing
    @property
    def done(self) -> bool:
        return self.submission is not None

    def call(self, name: str, args: dict) -> str:
        fn = getattr(self, "t_" + name, None)
        try:
            if fn is None or name not in self.tools:
                raise ToolError(f"unknown tool {name!r}")
            if self.done:
                raise ToolError("episode finished (already submitted)")
            if self.call_budget is not None and name not in ("describe_task", "submit", "submit_report"):
                if self.calls >= self.call_budget:
                    raise ToolError("tool-call budget exhausted; submit now")
                self.calls += 1
            out = fn(**_drop_empty(args or {}))
        except ToolError as e:
            out = {"error": str(e)}
        except (TypeError, KeyError, ValueError) as e:
            out = {"error": f"bad arguments: {e}"}
        except AssertionError as e:
            out = {"error": f"encoding failed: {e}"}
        except Exception as e:  # e.g. CUDA OOM / linalg errors: a tool error, not a dead episode
            if isinstance(e, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
            out = {"error": f"internal error in {name}: {type(e).__name__}: {str(e)[:200]}"}
        out["budget_left"] = self.budget - self.used
        if self.call_budget is not None:
            out["tool_calls_left"] = self.call_budget - self.calls
        self.log.append({"tool": name, "args": args, "used": self.used, "calls": self.calls})
        return json.dumps(out, separators=(",", ":"))

    def _spend(self, n: int):
        if self.used + n > self.budget:
            raise ToolError(f"forward-pass budget exceeded (need {n}, left {self.budget - self.used})")
        self.used += n

    def _refuse(self, msg: str):
        raise ToolError("not accessible" if self._generic else msg)

    def _template(self, t: str) -> str:
        if t in self.templates:
            return self.templates[t]
        if not isinstance(t, str) or "{city}" not in t:
            raise ToolError("template must be a known key or a string containing '{city}'")
        if t in self._private_tpls or any(p.fullmatch(_norm(t)) for p in self._held_tpl):
            self._refuse("that template is held out")
        return t

    def _cities(self, cs) -> list[str]:
        if isinstance(cs, str):
            cs = [cs]
        if not cs or len(cs) > MAX_PROMPTS:
            raise ToolError(f"give 1..{MAX_PROMPTS} cities")
        if not all(isinstance(c, str) for c in cs):
            raise ToolError("cities must be strings")
        bad = [c for c in cs if _letters(c) in self._held]
        if bad:
            self._refuse(f"held-out cities are not accessible: {bad}")
        return list(cs)

    def _check_text(self, s: str):
        if not isinstance(s, str):
            raise ToolError("prompts must be strings")
        s = _norm(s)
        if any(r.search(s) for r in self._held_re):
            self._refuse("prompt mentions a held-out city")
        if any(p.fullmatch(s) for p in self._held_tpl):
            self._refuse("prompt matches a held-out template")

    def _encs(self, template: str, cities) -> list:
        t = self._template(template)
        cs = self._cities(cities)
        encs = [self.S.encode(t, c) for c in cs]
        for e in encs:
            self._check_text(e.text)
        return encs

    def _raw_encs(self, prompts: list[str]) -> list:
        """Raw prompts; if the text mentions a dataset city, encode it with that (last-mentioned) city's
        span so city-position edits (and a planted edit) can find it."""
        out = []
        for p in prompts:
            self._check_text(p)
            ms = [(c, x) for c in self._all_cities
                  for x in re.finditer(r"(?<![A-Za-z])" + re.escape(c) + r"(?![A-Za-z])", p)]
            # drop matches nested in a longer one ("Bend" inside "South Bend"), then take the last-mentioned city
            ms = [(c, x) for c, x in ms if not any(y.start() <= x.start() and x.end() <= y.end() and
                                                   (y.end() - y.start()) > (x.end() - x.start()) for _, y in ms)]
            hit = max(ms, key=lambda cm: cm[1].start()) if ms else None
            e = None
            if hit:
                c, m = hit
                tpl = p[:m.start()].replace("{", "{{").replace("}", "}}") + "{city}" + \
                    p[m.end():].replace("{", "{{").replace("}", "}}")
                try:
                    e = self.S.encode(tpl, c)
                    e = e if e.city_last > e.prefix_len - 1 else None
                except AssertionError:
                    e = None
            out.append(e or self.S.encode_text(p))
        return out

    def _plant_ivs(self, encs, model: str = "planted") -> list:
        """The planted edit (detective task) for these prompts, if any."""
        if self.plant is None or model == "clean":
            return []
        L, v, src = self.plant
        pos = [[e.city_last] if e.city in src and e.city_last >= 0 else [] for e in encs]
        return [Intervention(L, pos, "add", v)] if any(pos) else []

    def _model(self, model: str) -> str:
        if model not in ("planted", "clean"):
            raise ToolError("model must be 'planted' or 'clean'")
        if model == "clean" and self.plant is not None and "clean_access" not in self.inst.constraints.get("features", []):
            raise ToolError("only the planted model is available in this task")
        return model

    def _reg(self, name: str) -> torch.Tensor:
        if name not in self.regs:
            raise ToolError(f"unknown register {name!r}; have {sorted(self.regs)}")
        return self.regs[name]

    def _layer(self, L) -> int:
        if not isinstance(L, int) or isinstance(L, bool) or not 0 <= L < self.S.n_layers:
            raise ToolError(f"layer must be an int in [0, {self.S.n_layers - 1}]")
        return L

    def _pos(self, p: str) -> str:
        if p not in POSITIONS:
            raise ToolError(f"position must be one of {POSITIONS}")
        return p

    def _site(self, L: int, pos: str):
        """T2 v2: edits only at the instance's intervention site (constraints['layer'], city_last)."""
        c = self.inst.constraints
        if c.get("layer") is not None and (L != c["layer"] or pos not in c["positions"]):
            raise ToolError(f"edits are only allowed at layer {c['layer']}, position {c['positions'][0]}")

    def _edits(self, edits: list[dict]) -> list[tuple]:
        """Tool-side edits: {"register", "layer", "position", "scale"} (add) or
        {"basis": [registers], "center": register?, "layer", "position"} (projection)."""
        if not edits:
            raise ToolError("need at least one edit")
        out = []
        for e in edits:
            L, pos = self._layer(e["layer"]), self._pos(e.get("position", "city_last"))
            self._site(L, pos)
            if self.inst.constraints.get("edit_kind", "add") == "proj" or ("basis" in e and "register" not in e):
                B = torch.stack([self._reg(b) for b in e["basis"]]) if e["basis"] else None
                if B is None or len(B) > 64:
                    raise ToolError("basis must list 1..64 registers")
                Q, R = torch.linalg.qr(B.T)
                Q = Q[:, R.diagonal().abs() > 1e-6 * R.diagonal().abs().max()]
                c = self._reg(e["center"]) if e.get("center") else None
                out.append((L, pos, (Q, c), "proj"))
            else:
                out.append((L, pos, self._reg(e["register"]) * float(e.get("scale", 1.0))))
        return out

    def _cands(self, answers: str) -> list[str]:
        if answers not in ("capitals", "states"):
            raise ToolError("answers must be 'capitals' or 'states'")
        return CAPS if answers == "capitals" else STATES

    def _clean_run(self, encs, cands=CAPS, model: str = "planted"):
        """Cached unedited (but planted, if any) top candidate + final log-probs per prompt."""
        key = lambda e: (e.text, tuple(cands), model)
        todo = [e for e in encs if key(e) not in self._clean]
        if todo:
            self._spend(len(todo))
            sc, lp, _ = self.S.score(todo, cands, self._plant_ivs(todo, model), bs=self.bs, return_logp=True)
            for e, i, l in zip(todo, sc.argmax(1).tolist(), lp):
                self._clean[key(e)] = (cands[i], l)
        return [self._clean[key(e)] for e in encs]

    # ---------------- tools
    def t_describe_task(self) -> dict:
        from .tasks import describe
        return describe(self)

    def t_run_prompts(self, prompts: list[str] | None = None, template: str | None = None,
                      cities: list[str] | None = None, top_k: int = 5, answers: str = "capitals",
                      model: str = "planted") -> dict:
        model = self._model(model)
        cands = self._cands(answers)
        if prompts:
            if isinstance(prompts, str):
                prompts = [prompts]
            if len(prompts) > MAX_PROMPTS:
                raise ToolError(f"at most {MAX_PROMPTS} prompts")
            encs = self._raw_encs(prompts)
        elif template and cities:
            encs = self._encs(template, cities)
        else:
            raise ToolError("give prompts, or template + cities")
        self._spend(len(encs))
        sc, lp, _ = self.S.score(encs, cands, self._plant_ivs(encs, model), bs=self.bs, return_logp=True)
        k = max(1, min(int(top_k), 20))
        kc = max(1, min(int(top_k), len(cands)))
        res = []
        for e, s, l in zip(encs, sc, lp):
            tp, ti = l.exp().topk(k)
            order = s.argsort(descending=True)[:kc].tolist()
            r = {"prompt": e.text[-120:], "city": e.city or None,
                 "next_tokens": [[self.S.tok.decode([t]), _r(p)] for t, p in zip(ti.tolist(), tp.tolist())],
                 f"top_{answers}": [[cands[j], _r(s[j].item(), 2)] for j in order]}
            if self.tgt_cap and answers == "capitals":
                r["target_capital_rank"] = int((s > s[CAPS.index(self.tgt_cap)]).sum()) + 1
            res.append(r)
        return {"results": res, "note": f"top_{answers}: full-sequence log-prob ranking over the 50 {answers} (the top entry is exact; lower ranks of pruned multi-token candidates use a first-token bound)"}

    def t_logit_lens(self, template: str, city: str, layer: int, position="final", top_k: int = 10,
                     model: str = "planted") -> dict:
        model = self._model(model)
        L = self._layer(layer)
        (e,) = self._encs(template, [city])
        p = {"city_last": e.city_last, "final": e.final}.get(position, position)
        if not isinstance(p, int) or not 0 <= p <= e.final:
            raise ToolError("position must be city_last, final, or a token index")
        self._spend(1)
        _, st, _, _ = self.S.forward([e], self._plant_ivs([e], model), {L: [[p]]})
        h = st.captures[L][0, 0].to(self.S.device)
        with torch.no_grad():
            lp = torch.log_softmax(self.S.model.lm_head(self.S.model.model.norm(h.to(self.S.dtype))).float(), -1).cpu()
        tp, ti = lp.topk(max(1, min(int(top_k), 30)))
        first = torch.tensor([lp[t[0]] for t in self.S.cand_tokens(CAPS)])
        order = first.argsort(descending=True)[:5].tolist()
        out = {"token": self.S.tok.decode([e.ids[p]]), "index": p,
               "top_tokens": [[self.S.tok.decode([t]), _r(v, 2)] for t, v in zip(ti.tolist(), tp.tolist())],
               "top_capitals_first_token": [[CAPS[j], _r(first[j], 2)] for j in order]}
        if self.tgt_cap:
            out["target_capital_first_token_rank"] = int((first > first[CAPS.index(self.tgt_cap)]).sum()) + 1
        return out

    def _acts(self, encs, layers: list[int], pos: str, model: str) -> dict:
        """Per-prompt residual (mean over the position rule's tokens) at each layer: {L: [B,d]}."""
        out = {L: [None] * len(encs) for L in layers}
        groups: dict[int, list] = {}
        for i, e in enumerate(encs):
            groups.setdefault(len(pos_rule([e], pos)[0]), []).append(i)
        for ix in groups.values():
            es = [encs[i] for i in ix]
            _, st, _, _ = self.S.forward(es, self._plant_ivs(es, model), {L: pos_rule(es, pos) for L in layers})
            for L in layers:
                for j, i in enumerate(ix):
                    out[L][i] = st.captures[L][j].mean(0).cpu()
        return {L: torch.stack(v) for L, v in out.items()}

    def _layers(self, layer, layers) -> list[int]:
        ls = layers if layers is not None else [layer]
        if isinstance(ls, int):
            ls = [ls]
        if not ls or len(ls) > self.S.n_layers:
            raise ToolError("give layer or a non-empty list of layers")
        return [self._layer(L) for L in ls]

    def t_cache_mean(self, name: str, cities: list[str], template: str, layer: int | None = None,
                     position: str = "city_last", layers: list[int] | None = None, model: str = "planted") -> dict:
        model = self._model(model)
        ls, pos = self._layers(layer, layers), self._pos(position)
        self._name(name)
        encs = self._encs(template, cities)
        self._spend(len(encs))
        A = self._acts(encs, ls, pos, model)
        if len(ls) == 1:
            self.regs[name] = A[ls[0]].mean(0)
            return {"register": name, "n": len(encs), "norm": _r(self.regs[name].norm())}
        for L in ls:
            self.regs[f"{name}_L{L}"] = A[L].mean(0)
        return {"registers": [f"{name}_L{L}" for L in ls], "n": len(encs),
                "norms": [_r(A[L].mean(0).norm(), 2) for L in ls]}

    def t_act_diff(self, template: str, cities: list[str], layer: int | None = None, position: str = "city_last",
                   layers: list[int] | None = None, name: str | None = None) -> dict:
        """planted - clean residual per city (norm, and relative to the clean residual norm)."""
        ls, pos = self._layers(layer, layers), self._pos(position)
        if name:
            self._name(name)
        encs = self._encs(template, cities)
        self._spend(2 * len(encs))
        P, C = self._acts(encs, ls, pos, "planted"), self._acts(encs, ls, pos, "clean")
        res = {}
        for L in ls:
            D = P[L] - C[L]
            res[str(L)] = [[e.city, _r(D[i].norm(), 3), _r(D[i].norm() / C[L][i].norm().clamp_min(1e-9), 4)]
                           for i, e in enumerate(encs)]
            if name:
                self.regs[name if len(ls) == 1 else f"{name}_L{L}"] = D.mean(0)
        return {"diff_by_layer": res, "format": "[city, |planted - clean|, relative to |clean|]",
                **({"stored": name} if name else {})}

    def _name(self, name: str):
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or name in _FUNCS:
            raise ToolError("bad register name")

    def t_vec_op(self, name: str, expr: str) -> dict:
        self._name(name)
        if len(expr) > 500:
            raise ToolError("expression too long")
        v = _Parser(_lex(expr), self.regs, self.S.d_model).parse()
        if not isinstance(v, torch.Tensor):
            return {"value": _r(v, 6), "note": "scalar result; nothing stored"}
        if not torch.isfinite(v).all():
            raise ToolError("result is not finite")
        self.regs[name] = v.float()
        return {"register": name, "norm": _r(v.norm())}

    def t_vec_info(self, name: str, top_k: int = 8) -> dict:
        v = self._reg(name)
        cos = {k: _r(v @ u / (v.norm() * u.norm()).clamp_min(1e-12), 3) for k, u in self.regs.items() if k != name}
        with torch.no_grad():
            lg = self.S.model.lm_head(self.S.model.model.norm(v.to(self.S.device, self.S.dtype))).float().cpu()
        ti = lg.topk(max(1, min(int(top_k), 20))).indices.tolist()
        return {"register": name, "norm": _r(v.norm()), "cos": dict(list(cos.items())[:12]),
                "unembed_top_tokens": [self.S.tok.decode([t]) for t in ti]}

    def t_eval_intervention(self, template: str, cities: list[str], edits: list[dict] | None = None,
                            vector: str | None = None, layer: int | None = None, position: str = "city_last",
                            scale: float = 1.0, n_generic: int = 4, answers: str = "capitals") -> dict:
        edits = edits or [{"register": vector, "layer": layer, "position": position, "scale": scale}]
        E = self._edits(edits)
        cands = self._cands(answers)
        encs = self._encs(template, cities)
        clean = self._clean_run(encs, cands)
        ng = max(0, min(int(n_generic), len(GENERIC_TOOLS)))
        ge = [self.S.encode_text(t) for t in GENERIC_TOOLS[:ng]]
        gclean = self._clean_run(ge) if ge else []
        self._spend(len(encs) + len(ge))
        sc, lp, _ = self.S.score(encs, cands, self._plant_ivs(encs) + build_ivs(E, encs), bs=self.bs, return_logp=True)
        pred = [cands[i] for i in sc.argmax(1).tolist()]
        klc = kl(torch.stack([c[1] for c in clean]), lp)
        out = {"predictions": [{"city": e.city, "clean": c[0], "edited": p} for e, c, p in zip(encs, clean, pred)],
               "changed_rate": _r(sum(c[0] != p for c, p in zip(clean, pred)) / len(pred), 3),
               "kl_on_these_prompts": _r(klc.mean(), 3)}
        if self.tgt_cap and answers == "capitals":
            out["flip_rate"] = _r(sum(p == self.tgt_cap for p in pred) / len(pred), 3)
        if ge:
            ref = [encs[i % len(encs)] for i in range(len(ge))]
            lg = self.S.logp_final(ge, build_ivs(E, ge, ref), bs=self.bs)
            out["kl_generic"] = _r(kl(torch.stack([c[1] for c in gclean]), lg).mean(), 4)
        return out

    def t_optimize_vector(self, name: str, layer: int, position: str = "city_last", dev_cities: list[str] | None = None,
                          templates: list[str] | None = None, steps: int = 50, kl_weight: float = 0.0,
                          init: str | None = None, lr: float = 0.05, max_norm: float | None = None,
                          extra_examples: list[dict] | None = None) -> dict:
        """Adam on an additive vector v (added at `layer`/`position` of every training prompt). Loss =
        mean NLL of the target capital's first token on dev_cities x templates (edit tasks only)
        + sum over extra_examples of weight * NLL(first token of ' ' + answer) on that (template, city) prompt
        + kl_weight * KL on generic sentences. Negative weights push an answer down.
        lr is relative: step ~ lr*|resid|/sqrt(d)."""
        self._name(name)
        L, pos = self._layer(layer), self._pos(position)
        self._site(L, pos)
        steps = int(steps)
        if not 1 <= steps <= 500:
            raise ToolError("steps must be in [1, 500]")
        if isinstance(templates, str):
            templates = [templates]
        encs = []
        if dev_cities:
            if not self.tgt_cap:
                raise ToolError("this task has no target capital: use extra_examples to define the objective")
            encs = [e for t in (templates or []) for e in self._encs(t, dev_cities)]
            if not encs:
                raise ToolError("dev_cities needs templates")
        if len(encs) > MAX_PROMPTS:
            raise ToolError(f"cities x templates must be <= {MAX_PROMPTS}")
        ex = extra_examples or []
        if len(ex) > MAX_PROMPTS:
            raise ToolError(f"at most {MAX_PROMPTS} extra_examples")
        xencs, xids, xw = [], [], []
        for x in ex:
            (e,) = self._encs(x["template"], [x["city"]])
            a = x["answer"]
            if not isinstance(a, str) or not a.strip():
                raise ToolError("each extra example needs a non-empty answer string")
            xencs.append(e); xids.append(self.S.cand_tokens([a.strip()])[0][0]); xw.append(float(x.get("weight", 1.0)))
        if not encs and not xencs:
            raise ToolError("nothing to optimize: give dev_cities + templates and/or extra_examples")
        xids = torch.tensor(xids, device=self.S.device)
        xw = torch.tensor(xw, device=self.S.device)
        ge = [self.S.encode_text(t) for t in GENERIC_TOOLS[:4]] if kl_weight > 0 else []
        gclean = torch.stack([c[1] for c in self._clean_run(ge)]).to(self.S.device) if ge else None
        self._spend(steps * (len(encs) + len(ge) + len(xencs)))
        tok = self.S.cand_tokens([self.tgt_cap])[0][0] if encs else None
        cap_first = torch.tensor([t[0] for t in self.S.cand_tokens(CAPS)])
        maxn = self.inst.constraints.get("max_norm")
        if max_norm is not None:  # agent-chosen cap, never looser than the task's
            maxn = float(max_norm) if maxn is None else min(maxn, float(max_norm))
        v = (self._reg(init).clone() if init else torch.zeros(self.S.d_model)).to(self.S.device).requires_grad_(True)
        opt = torch.optim.Adam([v], lr=1.0)  # lr set after measuring the residual norm
        anchor = encs or xencs
        ref = [anchor[i % len(anchor)] for i in range(len(ge))]
        hist = []
        for s in range(steps):
            loss = torch.zeros((), device=self.S.device)
            rec = {"step": s}
            if s == 0:
                _, st, _, _ = self.S.forward(anchor, self._plant_ivs(anchor), {L: pos_rule(anchor, "city_last")})
                rn = st.captures[L].norm(dim=-1).mean().item()
                for g in opt.param_groups:
                    g["lr"] = lr * rn / math.sqrt(self.S.d_model)
            if encs:
                lp, *_ = self.S.forward(encs, self._plant_ivs(encs) + [Intervention(L, pos_rule(encs, pos), "add", v)], grad=True)
                nll = -lp[:, tok].mean()
                loss = loss + nll
                rec["p_target"] = nll
                rec["hit"] = lp[:, cap_first].argmax(1) == CAPS.index(self.tgt_cap)
            if xencs:
                lx, *_ = self.S.forward(xencs, self._plant_ivs(xencs) + [Intervention(L, pos_rule(xencs, pos), "add", v)], grad=True)
                px = lx[torch.arange(len(xencs)), xids]
                loss = loss - (xw * px).sum() / max(1, len(xencs))
                rec["px"] = px
            if ge:
                lg, *_ = self.S.forward(ge, [Intervention(L, edit_positions(pos, ge, ref), "add", v)], grad=True)
                loss = loss + kl_weight * kl(gclean, lg).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if maxn is not None:
                with torch.no_grad():
                    v.mul_(min(1.0, maxn / max(v.norm().item(), 1e-12)))
            if s % max(1, steps // 5) == 0 or s == steps - 1:
                h = {"step": s, "loss": _r(loss.item(), 3)}
                if encs:
                    h["p_target"] = _r(math.exp(-rec["p_target"].item()), 3)
                    h["target_first_token_top_among_capitals"] = _r(rec["hit"].float().mean().item(), 2)
                if xencs:
                    h["extra_p_answer"] = [_r(math.exp(x), 3) for x in rec["px"].tolist()]
                hist.append(h)
        self.regs[name] = v.detach().float().cpu()
        return {"register": name, "norm": _r(v.norm().item()), "history": hist,
                "note": "history metrics are from the forward pass before each update"}

    def t_submit(self, edits: list[dict]) -> dict:
        if not edits:
            raise ToolError("need at least one edit")
        kind = self.inst.constraints.get("edit_kind", "add")
        es = []
        for e in edits:
            d = {"layer": e["layer"], "position": e.get("position", "city_last")}
            if kind == "proj":
                if "basis" not in e:
                    raise ToolError("this task takes projection edits: {basis: [registers], center?, layer, position}")
                d["basis"] = [self._reg(b).tolist() for b in e["basis"]]
                if e.get("center"):
                    d["center"] = self._reg(e["center"]).tolist()
            else:
                if "register" not in e:
                    raise ToolError("this task takes additive edits: {register, layer, position, scale}")
                d |= {"vector": self._reg(e["register"]).tolist(), "scale": float(e.get("scale", 1.0))}
            es.append(d)
        sub = {"kind": kind, "edits": es}
        ok, why = check(sub, self.inst.constraints, self.S.d_model, self.S.n_layers)
        if ok is None:
            raise ToolError(f"submission rejected: {why}")
        self.submission = sub
        return {"status": "submitted", "n_edits": len(edits)}

    def t_submit_report(self, report: dict) -> dict:
        from .tasks import check_report
        why = check_report(self.inst, report)
        if why:
            raise ToolError(f"report rejected: {why}")
        self.submission = {"report": report}
        return {"status": "submitted"}


def _norm(s: str) -> str:
    """Canonical text for the held-out guards: NFKC, no format chars (zero-width), lower case, every run of
    non-alphanumerics -> one space."""
    s = "".join(ch for ch in unicodedata.normalize("NFKC", s) if unicodedata.category(ch) != "Cf")
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _letters(s: str) -> str:
    return _norm(s).replace(" ", "")


def _drop_empty(args: dict) -> dict:
    """Treat empty optional values ("", None, [], {}) as absent, at the top level and inside `edits` items.
    Some agent APIs fill every schema property with a placeholder (seen with gpt-6.1-sol, 2026-09-30)."""
    empty = lambda v: v is None or v == "" or v == [] or v == {}
    out = {k: v for k, v in args.items() if not empty(v)}
    if isinstance(out.get("edits"), list):
        out["edits"] = [{k: v for k, v in e.items() if not empty(v)} if isinstance(e, dict) else e for e in out["edits"]]
    return out


def CITIES_OF(state: str) -> list[str]:
    from ..data import CITIES
    return list(CITIES[state])


TOOL_NAMES = ["describe_task", "run_prompts", "logit_lens", "cache_mean", "vec_op", "vec_info",
              "eval_intervention", "optimize_vector", "submit"]
