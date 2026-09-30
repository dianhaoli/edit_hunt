"""Agent tool API for EditHunt.

Vectors live server-side in named registers; the agent refers to them by name and only `submit`
turns registers into a submission. Every tool returns compact JSON text.

Budget: each prompt sequence run through the subject model costs 1 forward pass (a batch of 8
prompts = 8; one optimize_vector step on 8 prompts + 4 generic sentences = 12). Clean (unedited)
results are cached per prompt and only charged once. Out of budget -> error.

Held-out data: tools reject (a) any city in the instance's held-out test set, (b) any raw prompt
containing such a city as a word, (c) any template/prompt matching a held-out template string.
Every other city (dev cities, other cities the agent knows) and paraphrased templates are allowed.
"""
from __future__ import annotations

import json
import math
import re

import torch

from ..data import CAPITALS, COUNTRY_Q, STATE_Q, STATES
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
class ToolEnv:
    def __init__(self, S: Subject, inst: Instance, bs: int = 16):
        self.S, self.inst, self.bs = S, inst, bs
        self.budget = inst.constraints["fwd_budget"]
        self.used = 0
        self.regs: dict[str, torch.Tensor] = {}
        self.submission: dict | None = None
        self.log: list[dict] = []
        self._clean: dict[str, tuple[str, torch.Tensor]] = {}  # prompt text -> (top capital, logp final)
        self._held = {c.lower() for c in inst.test_cities}
        self._held_re = [re.compile(r"\b" + re.escape(c) + r"\b", re.I) for c in inst.test_cities]
        self._held_tpl = inst.heldout_patterns()
        self.templates = dict(inst.dev_templates) | {"state_q": STATE_Q, "country_q": COUNTRY_Q}
        self.tgt_cap = inst.target_capital

    # ---------------- plumbing
    @property
    def done(self) -> bool:
        return self.submission is not None

    def call(self, name: str, args: dict) -> str:
        fn = getattr(self, "t_" + name, None)
        try:
            if fn is None:
                raise ToolError(f"unknown tool {name!r}")
            if self.done:
                raise ToolError("episode finished (already submitted)")
            out = fn(**(args or {}))
        except ToolError as e:
            out = {"error": str(e)}
        except TypeError as e:
            out = {"error": f"bad arguments: {e}"}
        except AssertionError as e:
            out = {"error": f"encoding failed: {e}"}
        out["budget_left"] = self.budget - self.used
        self.log.append({"tool": name, "args": args, "used": self.used})
        return json.dumps(out, separators=(",", ":"))

    def _spend(self, n: int):
        if self.used + n > self.budget:
            raise ToolError(f"forward-pass budget exceeded (need {n}, left {self.budget - self.used})")
        self.used += n

    def _template(self, t: str) -> str:
        if t in self.templates:
            return self.templates[t]
        if "{city}" not in t:
            raise ToolError("template must be a known key or a string containing '{city}'")
        if t in self.inst.test_templates.values():
            raise ToolError("that template is held out")
        return t

    def _cities(self, cs) -> list[str]:
        if isinstance(cs, str):
            cs = [cs]
        if not cs or len(cs) > MAX_PROMPTS:
            raise ToolError(f"give 1..{MAX_PROMPTS} cities")
        bad = [c for c in cs if c.lower() in self._held]
        if bad:
            raise ToolError(f"held-out cities are not accessible: {bad}")
        return list(cs)

    def _check_text(self, s: str):
        if any(r.search(s) for r in self._held_re):
            raise ToolError("prompt mentions a held-out city")
        if any(p.fullmatch(s) for p in self._held_tpl):
            raise ToolError("prompt matches a held-out template")

    def _encs(self, template: str, cities) -> list:
        t = self._template(template)
        cs = self._cities(cities)
        encs = [self.S.encode(t, c) for c in cs]
        for e in encs:
            self._check_text(e.text)
        return encs

    def _reg(self, name: str) -> torch.Tensor:
        if name not in self.regs:
            raise ToolError(f"unknown register {name!r}; have {sorted(self.regs)}")
        return self.regs[name]

    def _layer(self, L) -> int:
        if not isinstance(L, int) or not 0 <= L < self.S.n_layers:
            raise ToolError(f"layer must be an int in [0, {self.S.n_layers - 1}]")
        return L

    def _pos(self, p: str) -> str:
        if p not in POSITIONS:
            raise ToolError(f"position must be one of {POSITIONS}")
        return p

    def _edits(self, edits: list[dict]) -> list[tuple]:
        if not edits:
            raise ToolError("need at least one edit")
        return [(self._layer(e["layer"]), self._pos(e.get("position", "city_last")),
                 self._reg(e["register"]) * float(e.get("scale", 1.0))) for e in edits]

    def _clean_run(self, encs, cands=CAPS):
        """Cached clean top candidate + final log-probs per prompt."""
        todo = [e for e in encs if (e.text, tuple(cands)) not in self._clean]
        if todo:
            self._spend(len(todo))
            sc, lp, _ = self.S.score(todo, cands, bs=self.bs, return_logp=True)
            for e, i, l in zip(todo, sc.argmax(1).tolist(), lp):
                self._clean[(e.text, tuple(cands))] = (cands[i], l)
        return [self._clean[(e.text, tuple(cands))] for e in encs]

    # ---------------- tools
    def t_describe_task(self) -> dict:
        i = self.inst
        return {"task": f"Make prompts about cities in {i.source} answer the capital of {i.target} "
                        f"({i.target_capital}) instead of {i.source_capital}, generalizing to unseen {i.source} "
                        f"cities and unseen prompt templates, with minimal side effects on other text.",
                "source": i.source, "target": i.target, "source_capital": i.source_capital,
                "target_capital": i.target_capital, "dev_source_cities": i.dev_source,
                "dev_target_cities": i.dev_target, "templates": self.templates, "constraints": i.constraints,
                "subject": {"model": i.model, "n_layers": self.S.n_layers, "d_model": self.S.d_model},
                "positions": {"city_last": "last token of the city name", "city_all": "all city-name tokens",
                              "final": "last prompt token"},
                "registers": sorted(self.regs), "budget_total": self.budget}

    def t_run_prompts(self, prompts: list[str] | None = None, template: str | None = None,
                      cities: list[str] | None = None, top_k: int = 5) -> dict:
        if prompts:
            if len(prompts) > MAX_PROMPTS:
                raise ToolError(f"at most {MAX_PROMPTS} prompts")
            for p in prompts:
                self._check_text(p)
            encs = [self.S.encode_text(p) for p in prompts]
        elif template and cities:
            encs = self._encs(template, cities)
        else:
            raise ToolError("give prompts, or template + cities")
        self._spend(len(encs))
        sc, lp, _ = self.S.score(encs, CAPS, bs=self.bs, return_logp=True)
        k = max(1, min(int(top_k), 20))
        res = []
        for e, s, l in zip(encs, sc, lp):
            tp, ti = l.exp().topk(k)
            order = s.argsort(descending=True)[:k].tolist()
            res.append({"prompt": e.text[-120:], "city": e.city or None,
                        "next_tokens": [[self.S.tok.decode([t]), _r(p)] for t, p in zip(ti.tolist(), tp.tolist())],
                        "top_capitals": [[CAPS[j], _r(s[j].item(), 2)] for j in order],
                        "target_capital_rank": int((s > s[CAPS.index(self.tgt_cap)]).sum()) + 1})
        return {"results": res, "note": "top_capitals: exact full-sequence log-prob ranking over the 50 capitals"}

    def t_logit_lens(self, template: str, city: str, layer: int, position="final", top_k: int = 10) -> dict:
        L = self._layer(layer)
        (e,) = self._encs(template, [city])
        p = {"city_last": e.city_last, "final": e.final}.get(position, position)
        if not isinstance(p, int) or not 0 <= p <= e.final:
            raise ToolError("position must be city_last, final, or a token index")
        self._spend(1)
        _, st, _, _ = self.S.forward([e], (), {L: [[p]]})
        h = st.captures[L][0, 0].to(self.S.device)
        with torch.no_grad():
            lp = torch.log_softmax(self.S.model.lm_head(self.S.model.model.norm(h)).float(), -1).cpu()
        tp, ti = lp.topk(min(int(top_k), 30))
        first = torch.tensor([lp[t[0]] for t in self.S.cand_tokens(CAPS)])
        order = first.argsort(descending=True)[:5].tolist()
        return {"token": self.S.tok.decode([e.ids[p]]), "index": p,
                "top_tokens": [[self.S.tok.decode([t]), _r(v, 2)] for t, v in zip(ti.tolist(), tp.tolist())],
                "top_capitals_first_token": [[CAPS[j], _r(first[j], 2)] for j in order],
                "target_capital_first_token_rank": int((first > first[CAPS.index(self.tgt_cap)]).sum()) + 1}

    def t_cache_mean(self, name: str, cities: list[str], template: str, layer: int,
                     position: str = "city_last") -> dict:
        L, pos = self._layer(layer), self._pos(position)
        encs = self._encs(template, cities)
        self._spend(len(encs))
        vecs = []
        groups: dict[int, list] = {}
        for e in encs:
            groups.setdefault(len(pos_rule([e], pos)[0]), []).append(e)
        for es in groups.values():
            _, st, _, _ = self.S.forward(es, (), {L: pos_rule(es, pos)})
            vecs.append(st.captures[L].mean(1).cpu())
        v = torch.cat(vecs).mean(0)
        self.regs[name] = v
        return {"register": name, "n": len(encs), "norm": _r(v.norm())}

    def t_vec_op(self, name: str, expr: str) -> dict:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or name in {"normalize", "proj_out", "proj", "dot", "norm", "cos", "randn"}:
            raise ToolError("bad register name")
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
            lg = self.S.model.lm_head(self.S.model.model.norm(v.to(self.S.device))).float().cpu()
        ti = lg.topk(min(int(top_k), 20)).indices.tolist()
        return {"register": name, "norm": _r(v.norm()), "cos": dict(list(cos.items())[:12]),
                "unembed_top_tokens": [self.S.tok.decode([t]) for t in ti]}

    def t_eval_intervention(self, template: str, cities: list[str], edits: list[dict] | None = None,
                            vector: str | None = None, layer: int | None = None, position: str = "city_last",
                            scale: float = 1.0, n_generic: int = 4) -> dict:
        edits = edits or [{"register": vector, "layer": layer, "position": position, "scale": scale}]
        E = self._edits(edits)
        encs = self._encs(template, cities)
        clean = self._clean_run(encs)
        ng = max(0, min(int(n_generic), len(GENERIC_TOOLS)))
        ge = [self.S.encode_text(t) for t in GENERIC_TOOLS[:ng]]
        gclean = self._clean_run(ge) if ge else []
        self._spend(len(encs) + len(ge))
        sc, lp, _ = self.S.score(encs, CAPS, build_ivs(E, encs), bs=self.bs, return_logp=True)
        pred = [CAPS[i] for i in sc.argmax(1).tolist()]
        klc = kl(torch.stack([c[1] for c in clean]), lp)
        out = {"predictions": [{"city": e.city, "clean": c[0], "edited": p} for e, c, p in zip(encs, clean, pred)],
               "flip_rate": _r(sum(p == self.tgt_cap for p in pred) / len(pred), 3),
               "kl_on_these_prompts": _r(klc.mean(), 3)}
        if ge:
            ref = [encs[i % len(encs)] for i in range(len(ge))]
            lg = self.S.logp_final(ge, build_ivs(E, ge, ref), bs=self.bs)
            out["kl_generic"] = _r(kl(torch.stack([c[1] for c in gclean]), lg).mean(), 4)
        return out

    def t_optimize_vector(self, name: str, layer: int, position: str, dev_cities: list[str],
                          templates: list[str], steps: int = 50, kl_weight: float = 0.0,
                          init: str | None = None, lr: float = 0.05) -> dict:
        """Adam on an additive vector: maximize log p(first token of target capital) at the final
        position (+ kl_weight * KL on generic sentences). lr is relative: step ~ lr*|resid|/sqrt(d)."""
        L, pos = self._layer(layer), self._pos(position)
        steps = int(steps)
        if not 1 <= steps <= 500:
            raise ToolError("steps must be in [1, 500]")
        if isinstance(templates, str):
            templates = [templates]
        encs = [e for t in templates for e in self._encs(t, dev_cities)]
        if len(encs) > MAX_PROMPTS:
            raise ToolError(f"cities x templates must be <= {MAX_PROMPTS}")
        ge = [self.S.encode_text(t) for t in GENERIC_TOOLS[:4]] if kl_weight > 0 else []
        gclean = torch.stack([c[1] for c in self._clean_run(ge)]).to(self.S.device) if ge else None
        self._spend(steps * (len(encs) + len(ge)))
        tok = self.S.cand_tokens([self.tgt_cap])[0][0]
        cap_first = torch.tensor([t[0] for t in self.S.cand_tokens(CAPS)])
        maxn = self.inst.constraints.get("max_norm")
        v = (self._reg(init).clone() if init else torch.zeros(self.S.d_model)).to(self.S.device).requires_grad_(True)
        opt = torch.optim.Adam([v], lr=1.0)  # lr set after measuring the residual norm
        ref = [encs[i % len(encs)] for i in range(len(ge))]
        hist = []
        for s in range(steps):
            cap = {L: pos_rule(encs, "city_last")} if s == 0 else None
            lp, st, _, _ = self.S.forward(encs, [Intervention(L, pos_rule(encs, pos), "add", v)], cap, grad=True)
            if s == 0:
                rn = st.captures[L].norm(dim=-1).mean().item()
                for g in opt.param_groups:
                    g["lr"] = lr * rn / math.sqrt(self.S.d_model)
            nll = -lp[:, tok].mean()
            loss = nll
            if ge:
                lg, *_ = self.S.forward(ge, [Intervention(L, edit_positions(pos, ge, ref), "add", v)], grad=True)
                loss = loss + kl_weight * kl(gclean, lg).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if maxn is not None:
                with torch.no_grad():
                    v.mul_(min(1.0, maxn / max(v.norm().item(), 1e-12)))
            if s % max(1, steps // 5) == 0 or s == steps - 1:
                with torch.no_grad():
                    hit = (lp[:, cap_first].argmax(1) == CAPS.index(self.tgt_cap)).float().mean().item()
                hist.append({"step": s, "loss": _r(loss.item(), 3), "p_target": _r(math.exp(-nll.item()), 3),
                             "target_first_token_top_among_capitals": _r(hit, 2)})
        self.regs[name] = v.detach().float().cpu()
        return {"register": name, "norm": _r(v.norm().item()), "history": hist,
                "note": "history metrics are from the forward pass before each update"}

    def t_submit(self, edits: list[dict]) -> dict:
        if not edits:
            raise ToolError("need at least one edit")
        sub = {"kind": "add", "edits": [{"layer": e["layer"], "position": e.get("position", "city_last"),
                                         "vector": self._reg(e["register"]).tolist(),
                                         "scale": float(e.get("scale", 1.0))} for e in edits]}
        ok, why = check(sub, self.inst.constraints, self.S.d_model, self.S.n_layers)
        if ok is None:
            raise ToolError(f"submission rejected: {why}")
        self.submission = sub
        return {"status": "submitted", "n_edits": len(edits)}


TOOL_NAMES = ["describe_task", "run_prompts", "logit_lens", "cache_mean", "vec_op", "vec_info",
              "eval_intervention", "optimize_vector", "submit"]
