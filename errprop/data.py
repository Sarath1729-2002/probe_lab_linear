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


def load_gsm_symbolic(config: str, n: int, seed: int) -> list[dict]:
    """GSM-Symbolic (Mirzadeh et al., ICLR 2025): templated GSM8K variants. p1/p2 add one/two
    reasoning clauses, i.e. longer dependency chains. Instances of one template share a group so
    they never straddle the dev/test split."""
    from datasets import load_dataset
    ds = load_dataset("apple/GSM-Symbolic", config, split="test")
    rows = []
    for i, r in enumerate(ds):
        ans_field = next((r[k] for k in ("answer", "final_answer") if k in r and r[k] is not None), None)
        if ans_field is None:
            continue
        gold = parse_number(str(ans_field).split("####")[-1])
        if gold is None:
            continue
        tmpl = r.get("id", i)
        rows.append({"id": f"gsmsym_{config}-{tmpl}-{r.get('instance', i)}", "dataset": f"gsmsym_{config}",
                     "level": {"p1": 1, "p2": 2}.get(config, 0), "question": r["question"], "gold": gold,
                     "group": f"gsmsym_{config}-{tmpl}"})
    random.Random(seed).shuffle(rows)
    return rows[:n]


def load_all(cfg: dict) -> list[dict]:
    from .synth import serial_chains, bbh_arithmetic
    d, seed = cfg["data"], cfg["seed"]
    probs = []
    if d.get("gsm8k_n"):
        probs += load_gsm8k(d["gsm8k_n"], seed)
    for c, n in (d.get("gsm_symbolic") or {}).items():
        if n:
            probs += load_gsm_symbolic(c, n, seed)
    if d.get("math_n"):
        probs += load_math500(d["math_n"], d.get("math_min_level", 1), seed)
    if d.get("chains_n"):
        probs += serial_chains(d["chains_n"], seed)
    if d.get("bbh_n"):
        probs += bbh_arithmetic(d["bbh_n"], seed)
    for p in probs:
        p.setdefault("group", p["id"])
    return probs


def messages(question: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{question}\n\n{INSTRUCTION}"}]


def render_prompt(tokenizer, question: str) -> str:
    return tokenizer.apply_chat_template(messages(question), tokenize=False, add_generation_prompt=True)
