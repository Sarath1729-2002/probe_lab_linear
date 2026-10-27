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
        nums = _NUM_IN_EXPR.findall(lhs)
        if len(nums) < 2:
            continue
        val = safe_eval(lhs)
        if val is None:
            continue
        dec = n_decimals(rhs_text)
        exact = close(val, rhs_val, rel=1e-6)
        rounded = dec > 0 and abs(round(val, dec) - rhs_val) < 1e-9
        if not (exact or rounded):
            continue
        eqs.append(Equation(idx=len(eqs), lhs=lhs, operands=[float(n) for n in nums],
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


def corrupt_value(value: float, text: str, kind: str) -> Optional[float]:
    dec = n_decimals(text)
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
        it = iter(new_ops)

        def repl(m):
            return repr(next(it))

        expr = _NUM_IN_EXPR.sub(repl, e.lhs)
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
                         kinds=OPERATORS) -> list[Injection]:
    """All clean, unambiguous injections for one trace."""
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
        for kind in kinds:
            nv = corrupt_value(e.rhs_value, e.rhs_text, kind)
            if nv is None or close(nv, e.rhs_value):
                continue
            if any(close(nv, v) for v in others):
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
    while len(picked) < max_on and any(by_pos.values()):
        for p in positions:
            if by_pos[p] and len(picked) < max_on:
                picked.append(by_pos[p].pop())
    rng.shuffle(off)
    return picked + off[:max_off]


def numbers_in(text: str) -> list[float]:
    vals = []
    for s in re.findall(NUM_RE, text):
        v = to_float(s)
        if v is not None:
            vals.append(v)
    return vals
