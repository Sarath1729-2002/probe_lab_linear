"""Deterministic labels for a continuation after an injected error. No judges."""
from __future__ import annotations

import re
from typing import Optional

from .arith import NUM_RE, close, parse_equations, to_float, safe_eval

_BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
_HASH = re.compile(r"####\s*\$?\s*(" + NUM_RE + r"(?:\s*/\s*\d+)?)")
_ANSWER_IS = re.compile(r"answer is[:\s]*\$?\s*(" + NUM_RE + r")", re.I)
_FRAC = re.compile(r"\\d?frac\{(-?[\d\.]+)\}\{(-?[\d\.]+)\}")

CORRECTION_MARKERS = re.compile(
    r"\b(wait|actually|mistake|oops|incorrect|not correct|isn't right|is wrong|error|"
    r"recalculat\w*|re-?check\w*|let me re\w*|should be|correction|miscalculat\w*|double[- ]check\w*)\b",
    re.I,
)


def parse_number(s: str) -> Optional[float]:
    if s is None:
        return None
    s = re.sub(r"\\text\{[^{}]*\}", "", s)
    for a in ("\\!", "\\,", "$", "\\%", "%", "\\left", "\\right"):
        s = s.replace(a, "")
    s = s.strip().rstrip(".").strip()
    m = _FRAC.search(s)
    if m:
        a, b = to_float(m.group(1)), to_float(m.group(2))
        if a is not None and b not in (None, 0):
            return a / b
    if "/" in s:
        v = safe_eval(s.replace(",", ""))
        if v is not None:
            return v
    nums = re.findall(NUM_RE, s)
    if len(nums) == 1:
        return to_float(nums[0])
    return None


_STRICT = re.compile(r"-?(\d{1,3}(,\d{3})+|\d+)(\.\d+)?|-?\d*\.\d+")
_STRICT_FRAC = re.compile(r"(-?)\\d?frac\{(-?\d+)\}\{(\d+)\}|(-?\d+)/(\d+)")


def strict_number(s: Optional[str]) -> Optional[float]:
    """Gold answers only: the WHOLE string must be one plain number or a simple fraction, after removing
    \\text{...}, $, thin spaces, %, ^\\circ. Rejects \\sqrt{2}, 3\\pi, x^2, intervals, tuples."""
    if s is None:
        return None
    s = re.sub(r"\\text\{[^{}]*\}|\\mbox\{[^{}]*\}", "", str(s))
    for a in ("\\$", "\\!", "\\,", "\\ ", "$", "\\%", "%", "^\\circ", "^{\\circ}", "\\left", "\\right", " "):
        s = s.replace(a, "")
    s = s.strip().rstrip(".")
    if _STRICT.fullmatch(s):
        return float(s.replace(",", ""))
    m = _STRICT_FRAC.fullmatch(s)
    if m:
        if m.group(2) is not None:
            v = int(m.group(2)) / int(m.group(3))
            return -v if m.group(1) else v
        if int(m.group(5)) != 0:
            return int(m.group(4)) / int(m.group(5))
    return None


def extract_answer(text: str) -> Optional[float]:
    """Final numeric answer: '#### x', else \\boxed{x}, else 'answer is x', else last number."""
    for rx in (_HASH, _BOXED, _ANSWER_IS):
        ms = list(rx.finditer(text))
        if ms:
            v = parse_number(ms[-1].group(1))
            if v is not None:
                return v
    if "</think>" in text:          # reasoning models: only look after the thinking block
        text = text.split("</think>")[-1]
    nums = re.findall(NUM_RE, text)
    return to_float(nums[-1]) if nums else None


def outcome(answer: Optional[float], gold: float, propagated: float, tol: float = 1e-4) -> str:
    """FOLLOW / RECOVER / OTHER. 'NA' when the error cannot change the answer (off-path).
    `tol` is the dataset's relative answer tolerance (1e-4 exact sets; 2% for MMLU-Pro, whose
    options are rounded)."""
    if close(propagated, gold, rel=tol):
        return "NA_RECOVER" if close(answer, gold, rel=tol) else "NA_OTHER"
    if close(answer, propagated, rel=tol):
        return "FOLLOW"
    if close(answer, gold, rel=tol):
        return "RECOVER"
    return "OTHER"


def contains_value(text: str, value: float) -> bool:
    return any(close(to_float(s), value) for s in re.findall(NUM_RE, text))


def recovery_type(continuation: str, true_value: float) -> str:
    """For RECOVER runs: 'explicit' (correction language), 'restated' (true value re-written
    without correction language) or 'silent' (the visible chain never repairs the error)."""
    if CORRECTION_MARKERS.search(continuation):
        return "explicit"
    if contains_value(continuation, true_value):
        return "restated"
    return "silent"


def contamination_horizon(continuation: str, wrong_values: list[float]) -> int:
    """How many verified equations in the continuation use a wrong value (the injected one or
    anything it propagated to)."""
    n = 0
    for e in parse_equations(continuation):
        if any(any(close(o, w) for w in wrong_values) for o in e.operands):
            n += 1
    return n


def uptake(continuation: str, new_value: float) -> bool:
    """Does the injected number re-appear (as an operand or restatement) in the continuation?"""
    return contains_value(continuation, new_value)


def first_mention_char(continuation: str, value: float) -> Optional[int]:
    for m in re.finditer(NUM_RE, continuation):
        if close(to_float(m.group(0)), value):
            return m.start()
    return None


def label_continuation(cont: str, gold: float, inj: dict) -> dict:
    ans = extract_answer(cont)
    out = outcome(ans, gold, inj["propagated_answer"], inj.get("tol", 1e-4))
    wrong = [inj["new_value"]] + [v for v in inj["changed_values"] if not close(v, inj["new_value"])]
    rec = {
        "answer": ans,
        "outcome": out,
        "uptake": uptake(cont, inj["new_value"]),
        "horizon": contamination_horizon(cont, wrong),
        "marker": bool(CORRECTION_MARKERS.search(cont)),
        "true_restated_char": first_mention_char(cont, inj["true_value"]),
        "n_chars": len(cont),
        # the prefix ends right after the injected number; if the model keeps writing digits it has
        # changed the number itself (e.g. 84 -> 840), so the item is not what we injected
        "glued": bool(re.match(r"(?:\d|,\d|\.\d)", cont)),
        # the model read the injected number as the start of a new expression ("= 8 ‖ * 3.5 = 28"),
        # i.e. it never used it as a result. Seen mostly with copy_operand in fast_v1.
        "operand_reuse": bool(re.match(r"\s*[\*\+/×÷x]\s*\(?\d|\s*-\s*\(?\d", cont)),
    }
    rec["recovery_type"] = recovery_type(cont, inj["true_value"]) if out in ("RECOVER", "NA_RECOVER") else None
    return rec
