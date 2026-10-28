"""Deterministic arithmetic layer: parse equations out of a CoT, build the dependency graph,
inject a wrong value, and compute exactly what the final answer becomes if the error is followed.

Nothing here calls a model. Everything is plain Python so it can be unit-tested.
"""
from __future__ import annotations

import ast
import operator as op
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

# ----------------------------------------------------------------------------- numbers

NUM_RE = r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_RHS_RE = re.compile(r"=\s*\$?\s*(" + NUM_RE + r")(?![\d/])")
_NUM_IN_EXPR = re.compile(r"\d+(?:\.\d+)?")


def to_float(s: str) -> Optional[float]:
    s = s.replace(",", "").replace("$", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def close(a: Optional[float], b: Optional[float], rel: float = 1e-6, abs_: float = 1e-6) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))


def n_decimals(text: str) -> int:
    text = text.replace(",", "")
    return len(text.split(".")[1]) if "." in text else 0


def format_like(value: float, template: str) -> str:
    """Write `value` with the same style (commas, decimals) as `template`."""
    dec = n_decimals(template)
    if dec == 0:
        s = f"{int(round(value)):,}" if "," in template else str(int(round(value)))
    else:
        s = f"{value:,.{dec}f}" if "," in template else f"{value:.{dec}f}"
    return s


# ----------------------------------------------------------------------------- safe eval

_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv}


def safe_eval(expr: str) -> Optional[float]:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        raise ValueError("disallowed")

    try:
        return float(ev(tree))
    except (ValueError, ZeroDivisionError, TypeError, OverflowError):
        return None


# ----------------------------------------------------------------------------- normalisation

_FRAC_RE = re.compile(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}")
_ALLOWED_TAIL = re.compile(r"[\d\s\.\+\-\*/\(\)]*$")


def normalise_lhs(text: str) -> str:
    """Turn the text left of '=' into something `safe_eval` can read, keeping only the
    arithmetic tail (e.g. 'So the total is 3 x 4' -> '3 * 4')."""
    t = text
    for a, b in [("\\times", "*"), ("\\cdot", "*"), ("\\div", "/"), ("×", "*"), ("÷", "/"),
                 ("\\(", " "), ("\\)", " "), ("\\[", " "), ("\\]", " "), ("$", ""),
                 ("\\,", ""), ("{,}", ""), ("−", "-"), ("**", "*")]:
        t = t.replace(a, b)
    t = _FRAC_RE.sub(r"((\1)/(\2))", t)
    t = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", " * ", t)
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)  # 1,200 -> 1200
    tail = _ALLOWED_TAIL.search(t).group(0)
    tail = tail.strip()
    # drop unmatched leading ')' and trailing '('; balance by trimming the front
    while tail and (tail[0] in ")+*/" or tail.count("(") < tail.count(")")):
        if tail[0] == ")" or tail[0] in "+*/":
            tail = tail[1:].lstrip()
        else:
            break
    while tail.count("(") > tail.count(")"):
        i = tail.find("(")
        tail = tail[i + 1:].lstrip()
    tail = tail.rstrip("+-*/( ")
    return tail


# ----------------------------------------------------------------------------- equations

@dataclass
class Equation:
    idx: int
    lhs: str                       # normalised expression
    operands: list                 # float values appearing on the LHS, in order
    rhs_value: float
    rhs_text: str
    rhs_start: int                 # char span of the RHS number in the trace
    rhs_end: int
    parents: list = field(default_factory=list)   # parent[k] = eq idx that produced operand k, or None

    def to_dict(self):
        return asdict(self)


def num_tokens(expr: str) -> list[tuple[int, int, float]]:
    """Numbers in an expression with their spans, keeping a unary minus ('3 - -4' -> 3, -4)."""
    out = []
    for m in _NUM_IN_EXPR.finditer(expr):
        start, sign = m.start(), 1.0
        j = m.start() - 1
        while j >= 0 and expr[j] == " ":
            j -= 1
        if j >= 0 and expr[j] == "-":
            k = j - 1
            while k >= 0 and expr[k] == " ":
                k -= 1
            if k < 0 or expr[k] in "(+-*/":
                start, sign = j, -1.0
        out.append((start, m.end(), sign * float(m.group(0))))
    return out


def parse_equations(text: str, offset: int = 0) -> list[Equation]:
    """Find verified `<arithmetic> = <number>` statements in `text`.

    Only equations whose LHS evaluates to the written RHS are kept, so every kept equation is a
    real computation, not a restatement ('x = 5') or a slip by the model.
    """
    eqs: list[Equation] = []
    for m in _RHS_RE.finditer(text):
        rhs_text = m.group(1)
        rhs_val = to_float(rhs_text)
        if rhs_val is None:
            continue
        line_start = text.rfind("\n", 0, m.start()) + 1
        seg = text[line_start:m.start()]
        prev_eq = seg.rfind("=")
        if prev_eq != -1:            # chained "a = b + c = d": LHS is "b + c"
            seg = seg[prev_eq + 1:]
        lhs = normalise_lhs(seg)
        if not lhs or not re.search(r"[\+\-\*/]", lhs.lstrip("-")):
            continue
        toks = num_tokens(lhs)
        if len(toks) < 2:
            continue
        val = safe_eval(lhs)
        if val is None:
            continue
        dec = n_decimals(rhs_text)
        exact = close(val, rhs_val, rel=1e-6)
        rounded = dec > 0 and abs(round(val, dec) - rhs_val) < 1e-9
        if not (exact or rounded):
            continue
        eqs.append(Equation(idx=len(eqs), lhs=lhs, operands=[t[2] for t in toks],
                            rhs_value=rhs_val, rhs_text=rhs_text,
                            rhs_start=offset + m.start(1), rhs_end=offset + m.end(1)))
    _link(eqs)
    return eqs


def _link(eqs: list[Equation]) -> None:
    for j, e in enumerate(eqs):
        parents = []
        for v in e.operands:
            src = None
            for i in range(j - 1, -1, -1):
                if close(eqs[i].rhs_value, v):
                    src = i
                    break
            parents.append(src)
        e.parents = parents


def ancestors(eqs: list[Equation], k: int) -> set[int]:
    out, stack = set(), [k]
    while stack:
        j = stack.pop()
        for p in eqs[j].parents:
            if p is not None and p not in out:
                out.add(p)
                stack.append(p)
    return out


def descendants(eqs: list[Equation], i: int) -> set[int]:
    out = set()
    for j in range(i + 1, len(eqs)):
        if any(p == i or p in out for p in eqs[j].parents if p is not None):
            out.add(j)
    return out


def answer_equation(eqs: list[Equation], answer: float) -> Optional[int]:
    """Last equation whose result is the final answer."""
    for e in reversed(eqs):
        if close(e.rhs_value, answer, rel=1e-4):
            return e.idx
    return None


# ----------------------------------------------------------------------------- injection

OPERATORS = ("plus_one", "times_ten", "digit_swap")
# Full grid. The first three are fixed-type errors (prior work: Jia & Mu 2026, Knowing-Saying 2026,
# Phan et al. 2026). The last three spread error SIZE and PLAUSIBILITY continuously, so that
# "which operator" stops being a shortcut for "will it be caught":
#   random_scale        value * f, f log-uniform in [0.2, 5]          (continuous error size)
#   random_same_digits  uniform random number with the same digit count (looks plausible)
#   copy_operand        the result is replaced by one of its own operands ("forgot to compute")
ALL_OPERATORS = ("plus_one", "digit_swap", "times_ten", "random_scale", "random_same_digits", "copy_operand")


def corrupt_value(value: float, text: str, kind: str, rng=None, operands=None) -> Optional[float]:
    dec = n_decimals(text)
    if kind == "random_scale":
        import math
        rng = rng or __import__("random").Random(0)
        for _ in range(20):
            f = math.exp(rng.uniform(math.log(0.2), math.log(5.0)))
            if abs(f - 1) < 0.1:
                continue
            nv = round(value * f, dec) if dec else float(round(value * f))
            if nv != value and nv != 0:
                return nv
        return None
    if kind == "random_same_digits":
        rng = rng or __import__("random").Random(0)
        if value == 0:
            return None
        sign = -1.0 if value < 0 else 1.0
        ip = int(abs(value))
        nd = len(str(ip))
        lo, hi = (1, 9) if nd == 1 else (10 ** (nd - 1), 10 ** nd - 1)
        for _ in range(20):
            new_ip = rng.randint(lo, hi)
            frac = round(rng.random(), dec) if dec else 0.0
            nv = sign * (round(new_ip + frac, dec) if dec else float(new_ip))
            if nv != value:
                return nv
        return None
    if kind == "copy_operand":
        for o in operands or []:
            if not close(o, value) and o != 0:
                return float(o)
        return None
    if kind == "plus_one":
        return value + (1.0 if dec == 0 else 10 ** -dec)
    if kind == "times_ten":
        return value * 10 if value != 0 else None
    if kind == "digit_swap":
        if dec != 0:
            return None
        s = str(int(round(abs(value))))
        if len(s) < 2 or s[-1] == s[-2]:
            return None
        s = s[:-2] + s[-1] + s[-2]
        return float(s) * (-1 if value < 0 else 1)
    raise ValueError(kind)


def propagate(eqs: list[Equation], i: int, new_value: float) -> dict[int, float]:
    """Re-execute every equation after `i`, substituting changed upstream values.

    Returns {eq_idx: new_rhs} for every equation whose result changes.
    """
    changed = {i: new_value}
    for j in range(i + 1, len(eqs)):
        e = eqs[j]
        if not any(p in changed for p in e.parents if p is not None):
            continue
        new_ops = [changed[p] if p in changed else v for v, p in zip(e.operands, e.parents)]
        toks = num_tokens(e.lhs)
        expr = e.lhs
        for (s, t, _), v in sorted(zip(toks, new_ops), key=lambda z: -z[0][0]):
            expr = expr[:s] + f"({v!r})" + expr[t:]
        val = safe_eval(expr)
        if val is None:
            continue
        if not close(val, e.rhs_value):
            changed[j] = val
    return changed


@dataclass
class Injection:
    eq_idx: int
    kind: str
    true_value: float
    true_text: str
    new_value: float
    new_text: str
    on_path: bool
    propagated_answer: float
    changes_answer: bool
    n_downstream: int              # equations downstream of the injected one (original trace)
    step_frac: float               # position of the injected eq among all eqs (0..1)
    changed_values: list           # new values of every equation the error reaches


def candidate_injections(eqs: list[Equation], answer: float, question_numbers: list[float],
                         kinds=OPERATORS, seed: int = 0) -> list[Injection]:
    """All clean, unambiguous injections for one trace. Random operators are seeded per
    (seed, equation, operator), so the dataset is reproducible."""
    import random
    k = answer_equation(eqs, answer)
    if k is None:
        return []
    on_path = ancestors(eqs, k)
    out = []
    for e in eqs:
        if e.idx == k:
            continue  # injecting the answer itself is trivially followed
        # ambiguity filter: the value must be unique, otherwise we cannot tell what the model uses
        others = [x.rhs_value for x in eqs if x.idx != e.idx] + list(question_numbers)
        if any(close(e.rhs_value, v) for v in others):
            continue
        for ki, kind in enumerate(kinds):
            rng = random.Random(seed * 1_000_003 + e.idx * 101 + ki)
            nv = corrupt_value(e.rhs_value, e.rhs_text, kind, rng=rng, operands=e.operands)
            if nv is None or close(nv, e.rhs_value):
                continue
            # the new value must not collide with another number, except for copy_operand where
            # the collision is the point (its uptake/horizon metrics are therefore less clean)
            if kind != "copy_operand" and any(close(nv, v) for v in others):
                continue
            changed = propagate(eqs, e.idx, nv)
            prop = changed.get(k, answer)
            out.append(Injection(
                eq_idx=e.idx, kind=kind, true_value=e.rhs_value, true_text=e.rhs_text,
                new_value=nv, new_text=format_like(nv, e.rhs_text),
                on_path=e.idx in on_path, propagated_answer=prop,
                changes_answer=not close(prop, answer, rel=1e-4),
                n_downstream=len(descendants(eqs, e.idx)),
                step_frac=e.idx / max(1, len(eqs) - 1),
                changed_values=[v for j, v in changed.items()],
            ))
    return out


def select_injections(cands: list[Injection], max_on: int, max_off: int, seed: int) -> list[Injection]:
    """Spread on-path injections over positions and operators; add a few off-path controls.

    On-path candidates are kept only if the error actually changes the answer, so every on-path
    injection has a FOLLOW target that differs from gold.
    """
    import random
    rng = random.Random(seed)
    on = [c for c in cands if c.on_path and c.changes_answer]
    off = [c for c in cands if not c.on_path and not c.changes_answer]
    rng.shuffle(on)
    # round-robin over equation positions so we get early/mid/late rather than one step x3
    by_pos: dict[int, list[Injection]] = {}
    for c in on:
        by_pos.setdefault(c.eq_idx, []).append(c)
    picked, positions = [], sorted(by_pos)
    used: dict[str, int] = {}
    while len(picked) < max_on and any(by_pos.values()):
        for p in positions:
            if by_pos[p] and len(picked) < max_on:
                # take the operator used least so far, so every operator appears at every depth
                by_pos[p].sort(key=lambda c: (used.get(c.kind, 0), rng.random()))
                c = by_pos[p].pop(0)
                used[c.kind] = used.get(c.kind, 0) + 1
                picked.append(c)
    rng.shuffle(off)
    return picked + off[:max_off]


def numbers_in(text: str) -> list[float]:
    vals = []
    for s in re.findall(NUM_RE, text):
        v = to_float(s)
        if v is not None:
            vals.append(v)
    return vals


def mention_injections(body: str, answer: float, question_numbers: list[float], kinds=OPERATORS,
                       seed: int = 0) -> list[tuple[Injection, int, int]]:
    """Injections for answers that are recalled, not computed (TriviaQA: years, counts, heights).

    The site is the FIRST place the reasoning states the answer value before the final line. It is
    replaced by a corrupted value, i.e. a wrong recalled fact, and the model continues. There is no
    arithmetic between site and answer, so the FOLLOW target is the injected value itself:
        FOLLOW  = final answer == injected value     RECOVER = final answer == gold.
    Skipped when the value also appears in the question (then the model can simply copy it).
    Returns (injection, char_start, char_end) triples; all operators share the one site."""
    import random
    if any(close(answer, v) for v in question_numbers):
        return []
    site = None
    for m in re.finditer(NUM_RE, body):
        v = to_float(m.group(0))
        if v is not None and close(v, answer, rel=1e-9):
            site = m
            break
    if site is None:
        return []
    out = []
    for ki, kind in enumerate(k for k in kinds if k != "copy_operand"):
        rng = random.Random(seed * 1_000_003 + ki)
        nv = corrupt_value(answer, site.group(0), kind, rng=rng)
        if nv is None:
            continue
        if abs(answer - round(answer)) < 1e-9:
            nv = float(round(nv))          # recalled integers stay integers (years, counts)
        if close(nv, answer, rel=1e-4) or any(close(nv, v) for v in question_numbers):
            continue
        out.append((Injection(eq_idx=0, kind=kind, true_value=answer, true_text=site.group(0),
                              new_value=nv, new_text=format_like(nv, site.group(0)), on_path=True,
                              propagated_answer=nv, changes_answer=True, n_downstream=0,
                              step_frac=site.start() / max(1, len(body)), changed_values=[nv]),
                    site.start(), site.end()))
    return out
