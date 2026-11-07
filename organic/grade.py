"""Deterministic grading. Every reply is reduced to its final answer (`Answer: ...` line, else the last
\\boxed{}, else the last line) and compared with the gold answer and with the planted wrong answer.
Outcome: GOLD (correct), WRONG (the planted wrong answer), OTHER (anything else)."""
from __future__ import annotations

import ast
import re

from decide.common import NUM, norm, text_match

ANS = re.compile(r"answer\s*(?:is)?\s*[:：]\s*(.+)", re.I)
_SENT = object()


def boxed(s: str | None) -> str | None:
    if not s:
        return None
    i = max(s.rfind("\\boxed"), s.rfind("\\fbox"))
    if i < 0:
        return None
    j = s.find("{", i)
    if j < 0:
        return None
    d = 0
    for k in range(j, len(s)):
        if s[k] == "{":
            d += 1
        elif s[k] == "}":
            d -= 1
            if d == 0:
                return s[j + 1:k]
    return None


def final_answer(text: str | None) -> str | None:
    if text is None:
        return None
    m = ANS.findall(text)
    if m:
        a = m[-1]
    else:
        b = boxed(text)
        if b is not None:
            return b.strip()
        lines = [l for l in text.strip().splitlines() if l.strip()]
        a = lines[-1] if lines else ""
    a = a.strip().strip("*`").strip()
    b = boxed(a)
    if b is not None:
        a = b
    a = re.sub(r"^\$+|\$+$", "", a).strip().rstrip(".").strip()
    return a


# ---------------------------------------------------------------------------- per type

def mc_letter(a: str | None, n: int) -> str | None:
    if not a:
        return None
    letters = "ABCDEFGHIJ"[:n]
    m = re.match(r"^\(?([A-J])\)?(?=[\s.:)]|$)", a.strip())
    if m and m.group(1) in letters:
        return m.group(1)
    m = re.findall(r"\(([A-J])\)", a)
    return m[-1] if m and m[-1] in letters else None


def norm_math(s: str | None) -> str | None:
    if s is None:
        return None
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s.strip())
    for t in ("\\left", "\\right", "\\!", "\\,", "\\;", "\\:", "$", " ", "^\\circ", "^{\\circ}", "\\%", "%",
              "\\displaystyle"):
        s = s.replace(t, "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").rstrip(".")
    return re.sub(r"^[a-zA-Z]=", "", s)


def _num(s: str | None) -> float | None:
    if s is None:
        return None
    s2 = s.replace(",", "")
    if re.fullmatch(r"-?\d+(\.\d+)?", s2):
        return float(s2)
    m = re.fullmatch(r"(-?)\\frac\{?(-?\d+)\}?\{?(\d+)\}?", s2)
    if m and float(m.group(3)) != 0:
        v = float(m.group(2)) / float(m.group(3))
        return -v if m.group(1) else v
    return None


def math_equal(a: str | None, b: str | None) -> bool:
    na, nb = norm_math(a), norm_math(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    fb = _num(nb)
    if fb is None:
        return False
    fa = _num(na)
    if fa is None:                                   # "18 dollars", "x = 18": last number in the answer
        nums = NUM.findall((a or "").replace("$", ""))
        fa = float(nums[-1].replace(",", "")) if nums else None
    return fa is not None and abs(fa - fb) <= 1e-6 * max(1.0, abs(fb))


def _lit(s: str):
    try:
        return ast.literal_eval(s)
    except Exception:
        return _SENT


def code_equal(a: str | None, gold: str) -> bool:
    if a is None:
        return False
    a = a.strip().strip("`").strip()
    a = re.sub(r"^(?:python\s+)?", "", a)
    a = re.sub(r"^f\(.*\)\s*(?:==|returns|->|=)\s*", "", a)
    g, v = _lit(gold), _lit(a)
    if v is _SENT or g is _SENT:
        return a == gold.strip()
    return type(v) is type(g) and v == g


def exact_norm(s: str | None) -> str:
    s = (s or "").strip().strip("\"'`“”‘’").strip().lower().rstrip(".")
    return re.sub(r"\s+", " ", s)


def exact_equal(a: str | None, gold: str) -> bool:
    x, y = exact_norm(a), exact_norm(gold)
    return bool(x) and (x == y or re.sub(r"[\s\-,]", "", x) == re.sub(r"[\s\-,]", "", y))


def _match(typ: str, a: str | None, target, n_opts: int = 10) -> bool:
    if target is None or a is None:
        return False
    if typ == "text":
        return text_match(a, target)
    if typ == "mc":
        return mc_letter(a, n_opts) == target
    if typ == "math":
        return math_equal(a, target)
    if typ == "code":
        return code_equal(a, target)
    return exact_equal(a, target)


def grade(item: dict, text: str | None) -> tuple[str, str | None]:
    a = final_answer(text)
    n = item.get("n_options", 10)
    g = _match(item["type"], a, item["gold"], n)
    w = _match(item["type"], a, item.get("wrong"), n)
    if g and w and item["type"] == "text":            # both alias sets hit: decide on exact equality
        na = norm(a)
        g = any(norm(x) == na for x in item["gold"])
        w = any(norm(x) == na for x in item["wrong"]) and not g
    if g and not w:
        return "GOLD", a
    if w and not g:
        return "WRONG", a
    return "OTHER", a
