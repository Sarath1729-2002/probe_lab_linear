"""Problem sources. Only numeric-answer problems, so every label is an exact number match."""
from __future__ import annotations

import random

from .labels import parse_number

SYSTEM = "You are a careful math assistant."
INSTRUCTION = (
    "Solve the problem step by step. Write every calculation as a plain-text equation with its "
    "result on the same line, for example `12 * 4 = 48` (no LaTeX). "
    "Finish with a final line of the form `#### <number>`."
)


def load_gsm8k(n: int, seed: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    out = []
    for i in idx[:n]:
        r = ds[i]
        gold = parse_number(r["answer"].split("####")[-1])
        out.append({"id": f"gsm8k-{i}", "dataset": "gsm8k", "level": 0,
                    "question": r["question"], "gold": gold})
    return out


def load_math500(n: int, min_level: int, seed: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = []
    for i, r in enumerate(ds):
        gold = parse_number(r["answer"])
        if gold is None or int(r["level"]) < min_level:
            continue
        rows.append({"id": f"math-{i}", "dataset": "math", "level": int(r["level"]),
                     "subject": r.get("subject"), "question": r["problem"], "gold": gold})
    random.Random(seed).shuffle(rows)
    return rows[:n]


def messages(question: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{question}\n\n{INSTRUCTION}"}]


def render_prompt(tokenizer, question: str) -> str:
    return tokenizer.apply_chat_template(messages(question), tokenize=False, add_generation_prompt=True)
