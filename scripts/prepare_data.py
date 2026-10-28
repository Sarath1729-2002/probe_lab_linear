"""Run once on a LOGIN node (needs internet). Builds the paper's five standard datasets, filtered to
the constraint this project needs (a numeric answer that can be checked exactly, no judge), and
writes them to <data_dir>/<name>.jsonl for the offline compute nodes.

    HF_HUB_OFFLINE=0 python scripts/prepare_data.py [data_dir]      # default $SCRATCH/errprop_data

| file              | source (paper §3.7)                    | filter                                               |
|-------------------|----------------------------------------|------------------------------------------------------|
| math_full         | MATH test, 7 subject configs           | \\boxed answer parses to one number                   |
| mmlu_pro          | MMLU-Pro test                          | >=8 of 10 options numeric, gold option numeric,       |
|                   |                                        | no "which of the following"; options WITHHELD, the   |
|                   |                                        | model answers the number; tolerance 2% (options are  |
|                   |                                        | rounded)                                              |
| openthoughts      | OpenThoughts-114k (metadata), math     | 1k-sample after filtering \\boxed numeric answers     |
| triviaqa          | TriviaQA rc.nocontext validation       | gold (value or an alias) is a plain number           |

GSM8K, GSM-Hard, GSM-Symbolic and MATH-500 are loaded directly by errprop/data.py.
Field names are detected and printed, so a renamed column fails loudly here, not on the cluster.
"""
import json
import os
import random
import re
import sys

for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ[k] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from datasets import load_dataset  # noqa: E402

from errprop.labels import parse_number, strict_number  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("SCRATCH", "."), "errprop_data")
os.makedirs(OUT, exist_ok=True)
BOX = re.compile(r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}")
PLAIN_NUM = re.compile(r"^-?(\d{1,3}(,\d{3})+|\d+)(\.\d+)?$")
REPORT = {}


def boxed(text):
    ms = BOX.findall(text or "")
    return strict_number(ms[-1]) if ms else None


def save(name, rows, cols):
    with open(os.path.join(OUT, f"{name}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    REPORT[name] = f"{len(rows)} rows   source columns: {cols}"
    print(f"[{name}] wrote {len(rows)} rows", flush=True)


def first_ok(attempts):
    err = None
    for fn in attempts:
        try:
            return fn()
        except Exception as e:      # try the next source name
            err = e
            print(f"   ... {type(e).__name__}: {str(e)[:150]}", flush=True)
    raise err


# ----------------------------------------------------------------------------- MATH (full test)
def math_full():
    configs = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
               "number_theory", "prealgebra", "precalculus"]
    rows, cols = [], None
    for c in configs:
        ds = load_dataset("EleutherAI/hendrycks_math", c, split="test")
        cols = ds.column_names
        for i, r in enumerate(ds):
            g = boxed(r["solution"])
            if g is None:
                continue
            lv = re.findall(r"\d", str(r.get("level", "")))
            rows.append({"id": f"math_full-{c}-{i}", "dataset": "math", "level": int(lv[0]) if lv else 0,
                         "subject": c, "question": r["problem"], "gold": g, "group": f"math_full-{c}-{i}",
                         "tol": 1e-4})
    save("math_full", rows, cols)


# ----------------------------------------------------------------------------- MMLU-Pro (numeric)
def mmlu_pro():
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    cols = ds.column_names
    rows = []
    for r in ds:
        opts = r["options"]
        vals = [None if re.search(r"\\|sqrt|π|\^|×|e[-+]\d", o) else parse_number(o) for o in opts]
        ai = r.get("answer_index")
        if ai is None:
            ai = "ABCDEFGHIJ".index(r["answer"])
        q = r["question"]
        if sum(v is not None for v in vals) < 8 or vals[ai] is None:
            continue
        if re.search(r"which of the following|which one|following (is|are)|statement", q, re.I):
            continue
        rows.append({"id": f"mmlu_pro-{r['question_id']}", "dataset": "mmlu_pro", "level": 0,
                     "subject": r.get("category"), "question": q, "gold": float(vals[ai]),
                     "group": f"mmlu_pro-{r['question_id']}", "tol": 0.02, "gold_option": opts[ai]})
    save("mmlu_pro", rows, cols)


# ----------------------------------------------------------------------------- OpenThoughts (math, 1k)
def openthoughts(n=1000):
    def meta():
        ds = load_dataset("open-thoughts/OpenThoughts-114k", "metadata", split="train", streaming=True)
        return ds, lambda r: r.get("domain") == "math", \
            lambda r: boxed(r.get("ground_truth_solution")) if boxed(r.get("ground_truth_solution")) is not None \
            else boxed(r.get("deepseek_solution")), lambda r: r["problem"]

    def default():
        ds = load_dataset("open-thoughts/OpenThoughts-114k", split="train", streaming=True)
        conv = lambda r: r["conversations"]                                       # noqa: E731
        return ds, lambda r: "\\boxed" in conv(r)[-1]["value"], lambda r: boxed(conv(r)[-1]["value"]), \
            lambda r: conv(r)[0]["value"]

    ds, keep, gold, question = first_ok([meta, default])
    ds = ds.shuffle(seed=0, buffer_size=20_000)
    rows, cols = [], None
    for i, r in enumerate(ds):
        cols = cols or list(r.keys())
        if not keep(r):
            continue
        g = gold(r)
        if g is None:
            continue
        rows.append({"id": f"openthoughts-{i}", "dataset": "openthoughts", "level": 0, "subject": r.get("source"),
                     "question": question(r), "gold": g, "group": f"openthoughts-{i}", "tol": 1e-4})
        if len(rows) >= n:
            break
    save("openthoughts", rows, cols)


# ----------------------------------------------------------------------------- TriviaQA (numeric)
def triviaqa(n=4000):
    ds = first_ok([lambda: load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation", streaming=True),
                   lambda: load_dataset("trivia_qa", "rc.nocontext", split="validation", streaming=True)])
    rows, cols = [], None
    for i, r in enumerate(ds):
        cols = cols or list(r.keys())
        a = r["answer"]
        cands = [a.get("value", "")] + list(a.get("aliases", []))
        num = next((c.strip() for c in cands if PLAIN_NUM.match(c.strip())), None)
        if num is None:
            continue
        g = float(num.replace(",", ""))
        rows.append({"id": f"triviaqa-{r.get('question_id', i)}", "dataset": "triviaqa", "level": 0,
                     "question": r["question"], "gold": g, "group": f"triviaqa-{r.get('question_id', i)}",
                     "tol": 1e-4, "answer_value": a.get("value")})
        if len(rows) >= n:
            break
    save("triviaqa", rows, cols)


for name, fn in [("math_full", math_full), ("mmlu_pro", mmlu_pro), ("openthoughts", openthoughts),
                 ("triviaqa", triviaqa)]:
    print(f"== {name}", flush=True)
    try:
        fn()
    except Exception as e:
        REPORT[name] = f"FAILED: {type(e).__name__}: {str(e)[:200]}"
print("\n".join(f"{k:14s} {v}" for k, v in REPORT.items()))
print("written to", OUT)
