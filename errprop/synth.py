"""Deterministic synthetic problems with exact answers and known structure.

serial_chains      word problems whose answer is a single dependency chain of `depth` steps
                   (Knowing-Saying Gap style multi-hop, but solved by the model in free text).
                   Every intermediate result feeds the next, so every injection is on-path and
                   depth / remaining steps are controlled by design.
bbh_arithmetic     BIG-Bench Hard `multistep_arithmetic_two` format: ((a op b op c op d) op (e ...)).
                   Jia & Mu (2026) found the highest propagation on exactly this task (~64%).
"""
from __future__ import annotations

import random

from .arith import safe_eval

NAMES = ["Ava", "Ben", "Chen", "Dara", "Eli", "Farah", "Gus", "Hana", "Ivan", "Jada", "Kofi", "Lena",
         "Maya", "Nico", "Omar", "Priya", "Quinn", "Rosa", "Sami", "Tara"]
ITEMS = [("marbles", "marble"), ("stickers", "sticker"), ("apples", "apple"), ("cards", "card"),
         ("bolts", "bolt"), ("cupcakes", "cupcake"), ("tickets", "ticket"), ("seeds", "seed")]


def _chain(rng: random.Random, depth: int):
    name = rng.choice(NAMES)
    items, item = rng.choice(ITEMS)
    a, b = rng.randint(3, 19), rng.randint(4, 25)
    q = a * b
    used = {a, b}
    lines = [f"{name} has {a} boxes with {b} {items} in each box."]
    ops = ["mul0"]
    for _ in range(depth - 1):
        choices = ["add", "sub", "mul", "div"]
        rng.shuffle(choices)
        for op in choices:
            if op == "add":
                c = rng.randint(7, 99)
                nq, text = q + c, f"Then {name} gets {c} more {items}."
            elif op == "sub":
                if q < 20:
                    continue
                c = rng.randint(3, max(4, q // 2))
                nq, text = q - c, f"After that, {name} gives away {c} {items}."
            elif op == "mul":
                if q > 3000:
                    continue
                c = rng.randint(2, 4)
                nq, text = q * c, f"A delivery arrives and {name}'s {items} become {c} times as many."
            else:
                divs = [d for d in range(2, 10) if q % d == 0]
                if not divs:
                    continue
                c = rng.choice(divs)
                nq, text = q // c, (f"{name} splits the {items} equally into {c} bags "
                                    f"and keeps only one bag.")
            if nq in used or nq <= 0:
                continue
            used.add(c)
            q = nq
            used.add(q)
            lines.append(text)
            ops.append(op)
            break
    question = " ".join(lines) + f" How many {items} does {name} have now?"
    return question, float(q), ops


def serial_chains(n: int, seed: int, depths=(3, 4, 5, 6, 7, 8)) -> list[dict]:
    rng = random.Random(seed + 7)
    out = []
    for i in range(n):
        d = depths[i % len(depths)]
        q, ans, ops = _chain(rng, d)
        out.append({"id": f"chain-{i}", "dataset": "chains", "level": d, "question": q, "gold": ans,
                    "meta": {"depth": d, "ops": ops}})
    return out


def _arith_group(rng):
    nums = [rng.randint(-9, 9) for _ in range(4)]
    ops = [rng.choice(["+", "-", "*"]) for _ in range(3)]
    s = str(nums[0])
    for o, x in zip(ops, nums[1:]):
        s += f" {o} {x}"
    return f"({s})"


def bbh_arithmetic(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed + 11)
    out = []
    while len(out) < n:
        expr = f"({_arith_group(rng)} {rng.choice(['+', '-', '*'])} {_arith_group(rng)})"
        val = safe_eval(expr.replace("--", "- -"))
        if val is None or abs(val) > 10_000:
            continue
        out.append({"id": f"bbh-{len(out)}", "dataset": "bbh_arith", "level": 0,
                    "question": f"Compute the value of the expression {expr} =", "gold": float(val)})
    return out
