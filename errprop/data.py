"""Problem sources. Only numeric-answer problems, so every label is an exact number match."""
from __future__ import annotations

import random

from .labels import parse_number, strict_number

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
                    "question": r["question"], "gold": gold, "group": f"gsm8k-{i}"})
    return out


def load_math500(n: int, min_level: int, seed: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = []
    for i, r in enumerate(ds):
        gold = strict_number(r["answer"])
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


def load_gsm_hard(n: int, seed: int) -> list[dict]:
    """GSM-Hard (Gao et al. 2023, PAL): GSM8K test problems with the numbers replaced by much
    larger ones. Same reasoning, unfamiliar magnitudes. Field names are detected, not assumed."""
    from datasets import load_dataset
    ds = load_dataset("reasoning-machines/gsm-hard", split="train")
    cols = ds.column_names
    qk = next(k for k in ("input", "question", "problem") if k in cols)
    ak = next(k for k in ("target", "answer", "final_answer") if k in cols)
    rows = []
    for i, r in enumerate(ds):
        gold = parse_number(str(r[ak]).split("####")[-1])
        if gold is None:
            continue
        # GSM-Hard row i rewrites GSM8K test problem i: share its group so the pooled split never puts
        # a problem in dev and its large-number twin in test
        rows.append({"id": f"gsm_hard-{i}", "dataset": "gsm_hard", "level": 0,
                     "question": r[qk], "gold": float(gold), "group": f"gsm8k-{i}"})
    random.Random(seed).shuffle(rows)
    return rows[:n]


# Recalled-answer sets (TriviaQA): the model must state the fact in its reasoning before the final
# line, so there is a site to corrupt.
RECALL_INSTRUCTION = (
    "Answer the question. First write one or two sentences recalling the relevant facts, stating the "
    "number you rely on explicitly in digits. Finish with a final line of the form `#### <number>`."
)


def load_prepared(name: str, data_dir: str, n: int, seed: int) -> list[dict]:
    """Datasets filtered on the login node by scripts/prepare_data.py into <data_dir>/<name>.jsonl
    (compute nodes are offline). Each row already has id, dataset, question, gold, group, tol, level."""
    import json
    import os
    path = os.path.join(os.path.expandvars(data_dir), f"{name}.jsonl")
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if name == "triviaqa":
        for r in rows:
            r.setdefault("instruction", RECALL_INSTRUCTION)
            r.setdefault("site", "mention")
    random.Random(seed).shuffle(rows)
    return rows[:n]


def _safe(loader, name, *a):
    try:
        out = loader(*a)
        print(f"[data] {name}: {len(out)} problems", flush=True)
        return out
    except Exception as e:                       # a missing/renamed dataset must not kill a 3 h job
        print(f"[data] WARNING: could not load {name} ({type(e).__name__}: {e}); skipping", flush=True)
        return []


def load_all(cfg: dict) -> list[dict]:
    from .synth import serial_chains, bbh_arithmetic
    d, seed = cfg["data"], cfg["seed"]
    probs = []
    if d.get("gsm8k_n"):
        probs += _safe(load_gsm8k, "gsm8k", d["gsm8k_n"], seed)
    if d.get("gsm_hard_n"):
        probs += _safe(load_gsm_hard, "gsm_hard", d["gsm_hard_n"], seed)
    for c, n in (d.get("gsm_symbolic") or {}).items():
        if n:
            probs += _safe(load_gsm_symbolic, f"gsmsym_{c}", c, n, seed)
    if d.get("math_n"):
        probs += _safe(load_math500, "math", d["math_n"], d.get("math_min_level", 1), seed)
    for name in ("math_full", "mmlu_pro", "openthoughts", "triviaqa"):
        if d.get(f"{name}_n"):
            probs += _safe(load_prepared, name, name, d.get("data_dir", "${SCRATCH}/errprop_data"), d[f"{name}_n"], seed)
    if d.get("chains_n"):
        probs += serial_chains(d["chains_n"], seed)
    if d.get("bbh_n"):
        probs += bbh_arithmetic(d["bbh_n"], seed)
    for p in probs:
        p.setdefault("group", p["id"])
    return probs


def messages(question: str, instruction: str | None = None) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{question}\n\n{instruction or INSTRUCTION}"}]


def render_prompt(tokenizer, question: str, instruction: str | None = None, no_system: bool = False) -> str:
    """Chat-template prompt. Some templates (older Mistral) reject a system role, and DeepSeek-R1
    distills are meant to run without one (`no_system: true`); then the system text is folded into
    the user turn, so every model sees the same words."""
    ins = instruction or INSTRUCTION
    folded = [{"role": "user", "content": f"{SYSTEM}\n\n{question}\n\n{ins}"}]
    if no_system:
        return tokenizer.apply_chat_template(folded, tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(messages(question, ins), tokenize=False, add_generation_prompt=True)
    except Exception:
        return tokenizer.apply_chat_template(folded, tokenize=False, add_generation_prompt=True)
