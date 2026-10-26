"""Pure-python answer parsing.  Everything the ground truth depends on lives here and in bench.py, and is unit-tested."""
import re
from typing import List, Optional

SPECIAL_TAILS = ("<|im_end|>", "<|endoftext|>", "<|im_start|>")
_INT = re.compile(r"-?\d+")


def strip_special(t: str) -> str:
    for s in SPECIAL_TAILS:
        t = t.replace(s, "")
    return t


def all_boxed(text: str) -> List[str]:
    """contents of every \\boxed{...} (brace matched), in order."""
    out, i = [], 0
    while True:
        i = text.find("\\boxed", i)
        if i < 0:
            return out
        j = text.find("{", i)
        if j < 0:
            return out
        depth, k = 0, j
        while k < len(text):
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
                if depth == 0:
                    out.append(text[j + 1:k])
                    break
            k += 1
        else:
            return out
        i = k + 1


def last_boxed(text: str) -> Optional[str]:
    b = all_boxed(text)
    return b[-1] if b else None


def canon_int(s: Optional[str]) -> Optional[str]:
    """'\\$1{,}000' -> '1000', '45^\\circ' -> '45', '12\\text{ cm}' -> '12', '-7' -> '-7', '3.0' -> '3'.
    None unless the WHOLE content is one integer (fractions, expressions, decimals, lists -> None)."""
    if s is None:
        return None
    s = re.sub(r"\\(?:text|mathrm|textbf)\{[^{}]*\}", "", s)
    for a in ("\\$", "$", "\\!", "\\,", "\\ ", "{,}", "\\%", "%", "^\\circ", "^{\\circ}", "\\circ", "\\left", "\\right", "\\dots"):
        s = s.replace(a, "" if a != "{,}" else ",")
    s = s.strip().rstrip(".").strip()
    m = re.fullmatch(r"(?:[A-Za-z]\s*=\s*)?(-?\d[\d,]*(?:\.0+)?)(?:\s*[A-Za-z][A-Za-z\s\.]*)?", s)       # optional 'x =' prefix, optional trailing unit words
    if not m:
        return None
    num = m.group(1).split(".")[0]
    if "," in num:
        if not re.fullmatch(r"-?\d{1,3}(,\d{3})+", num):
            return None
        num = num.replace(",", "")
    if len(num.lstrip("-")) > 12:
        return None
    return "0" if int(num) == 0 else str(int(num))


_MCQ = re.compile(r"^\(?([A-D])\)?(?:[\.\):\s].*)?$", re.S)


def canon_letter(s: Optional[str]) -> Optional[str]:
    """'B', '(B)', 'B. Paris', '\\text{B}' -> 'B'; anything else -> None."""
    if s is None:
        return None
    s = re.sub(r"\\(?:text|mathrm|textbf)\{([^{}]*)\}", r"\1", s).strip()
    m = _MCQ.match(s)
    return m.group(1) if m else None


def parse_answer(completion: str, kind: str) -> Optional[str]:
    """The completion's OWN final answer: content of its LAST \\boxed{}, canonicalised.  None = no usable answer."""
    b = last_boxed(strip_special(completion))
    return canon_letter(b) if kind == "mcq" else canon_int(b)
