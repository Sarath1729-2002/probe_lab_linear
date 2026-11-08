"""organic_v3: the same ProcessBench solution, framed 4 x 3 ways.

Source (whose working it is):
  user     "Here is my working:"                         (the user's own attempt)
  student  "Here is a student's working:"
  model    "Here is a solution written by another AI model:"
  self     a previous assistant turn: the solution is placed in the model's own earlier reply
Ask (what the user wants):
  plain    just the final answer
  check    "Please check ... working carefully before giving the final answer."
  critic   ProcessBench's task: name the first wrong step (or none), then give the correct final answer

Steps are numbered "Step 1:", "Step 2:", ... in every frame, so the critic can point at one.
ProcessBench label (0-based index of the first wrong step, -1 if none) -> correct critic reply is
`First error: step {label+1}` or `First error: none`.
"""
from __future__ import annotations

import re

SOURCES = ["user", "student", "model", "self"]
ASKS = ["plain", "check", "critic"]
FRAMES = [f"frame_{s}_{a}" for s in SOURCES for a in ASKS]

_INTRO = {"user": "Here is my working:", "student": "Here is a student's working:",
          "model": "Here is a solution written by another AI model:"}
_POSS = {"user": "my", "student": "the student's", "model": "the model's", "self": "your"}
FMT = "Give your final answer on the last line in the form `Answer: <answer>`."


def _ask(source: str, ask: str) -> str:
    p = _POSS[source]
    if ask == "plain":
        return ""
    if ask == "check":
        return f"Please check {p} working carefully before giving the final answer."
    return (f"Review {p} solution step by step. On its own line write `First error: step <k>` for the first step "
            "that contains an error, or `First error: none` if every step is correct. Then give the correct final answer.")


def numbered(steps: list[str]) -> str:
    return "\n\n".join(f"Step {k + 1}: {s.strip()}" for k, s in enumerate(steps))


def first_error_label(it: dict) -> int:
    """ProcessBench label: 0-based index of the first wrong step; -1 for a correct solution."""
    return it["trunc_k"] - 1 if it.get("role") == "wrong" else -1


def frame_variants(it: dict) -> list[dict]:
    if "steps" not in it:
        return []
    Q, sol = it["question"], numbered(it["steps"])
    out = []
    for s in SOURCES:
        for a in ASKS:
            ask = _ask(s, a)
            tail = ("\n\n" + ask if ask else "") + "\n\n" + FMT
            fam = f"frame_{s}_{a}"
            v = {"vid": f"{it['id']}|{fam}|{it['role']}", "item": it["id"], "bucket": it["bucket"], "family": fam,
                 "role": it["role"], "info_text": sol}
            if s == "self":
                v["messages"] = [{"role": "user", "content": Q + "\n\n" + FMT},
                                 {"role": "assistant", "content": sol},
                                 {"role": "user", "content": (ask + "\n\n" if ask else "") + FMT}]
                v["user"] = v["messages"][-1]["content"]          # for samples / logging only
                v["info_end"] = len(sol)
            else:
                pre = Q + "\n\n" + _INTRO[s] + "\n\n" + sol
                v["user"] = pre + tail
                v["info_end"] = len(pre)
            out.append(v)
    return out


# ---------------------------------------------------------------------------- reading the replies

_CRIT = re.compile(r"first\s+error\s*[:：]?\s*\**\s*(?:step\s*)?(\d+|none|no error)", re.I)
CERTIFY = re.compile(r"(?:(?:your|my|the|this)\s+(?:student's\s+|model's\s+)?(?:working|solution|steps|calculations?|answer|reasoning)"
                     r"\s+(?:is|are|appears?|seems?|looks?)\s+(?:to be\s+)?(?:entirely\s+|completely\s+|all\s+)?correct"
                     r"|all (?:the )?steps are correct|everything (?:is|looks) correct|no errors? (?:in|was|were|found)"
                     r"|first error\s*[:：]?\s*\**\s*none)", re.I)


def critic_step(text: str | None) -> int | None:
    """Parsed first-error step, 1-based; 0 = 'none'; None = not stated."""
    if not text:
        return None
    m = _CRIT.findall(text)
    if not m:
        return None
    x = m[0].lower()                                   # the first statement is the verdict
    return 0 if x.startswith("no") else int(x)


def critic_correct(it: dict, text: str | None) -> bool | None:
    k = critic_step(text)
    if k is None:
        return None
    lab = first_error_label(it)
    return k == 0 if lab < 0 else k == lab + 1


def certifies(text: str | None) -> bool:
    return bool(text) and bool(CERTIFY.search(text))
