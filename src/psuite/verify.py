"""Deterministic output constraints (IFEval-style).  Every verifier is a pure function of the response text, so compliance is exactly checkable.

A response that did NOT finish (hit the token cap) never complies: a truncated answer has not satisfied any of these.
  maxw50 / maxw100 / maxw200   "Keep your entire response under N words."          words = len(text.split()) <= N
  nocomma                      "Do not use any commas anywhere in your response."   ',' not in text
  endphrase                    "End your response with exactly this sentence: ..."  text.rstrip() ends with the phrase
"""
import math
import re
import zlib
from typing import Optional

from .textutil import strip_special

PHRASE = "That is my full answer."
CONSTRAINTS = ("maxw50", "maxw100", "maxw200", "nocomma", "endphrase")
CONSTRAINT_TEXT = {
    "maxw50": "Keep your entire response under 50 words.",
    "maxw100": "Keep your entire response under 100 words.",
    "maxw200": "Keep your entire response under 200 words.",
    "nocomma": "Do not use any commas anywhere in your response.",
    "endphrase": f"End your response with exactly this sentence: {PHRASE}",
}
LIMIT = {"maxw50": 50, "maxw100": 100, "maxw200": 200}


def constraint_for(pid: int) -> str:
    return CONSTRAINTS[zlib.crc32(f"constraint-{pid}".encode()) % len(CONSTRAINTS)]


def n_words(text: str) -> int:
    return len(strip_special(text).split())


def complies(ctype: str, text: str, finished: bool) -> bool:
    if not finished:
        return False
    t = strip_special(text)
    if ctype in LIMIT:
        return n_words(t) <= LIMIT[ctype]
    if ctype == "nocomma":
        return "," not in t
    if ctype == "endphrase":
        return t.rstrip().endswith(PHRASE)
    raise ValueError(ctype)


def overshoot(ctype: str, text: str, finished: bool) -> Optional[float]:
    """log(words / limit) for the word-limit constraints of finished responses (continuous 'how far over / under'); None otherwise."""
    if ctype not in LIMIT or not finished:
        return None
    return math.log(max(n_words(text), 1) / LIMIT[ctype])
