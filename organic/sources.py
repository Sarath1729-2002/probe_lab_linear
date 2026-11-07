"""Download step (login node, internet). Each source is written to <data_dir>/raw/<name>.jsonl in a
small normalised schema, so everything after this step is independent of Hub field names.
A source that fails is reported and skipped; the rest still build."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .common import write_jsonl

from .grade import boxed

CUTE_TASKS = ["spell", "spell_inverse", "contains_char", "ins_char", "del_char", "sub_char", "swap_char"]


def _ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def triviaqa(out: Path):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("meg-tong/sycophancy-eval", "answer.jsonl", repo_type="dataset")
    seen, rows = set(), []
    for line in open(p):
        r = json.loads(line)
        b = r.get("base", {})
        if "trivia" not in str(b.get("dataset", "")).lower():
            continue
        q = b.get("question")
        if not q or q in seen or not b.get("correct_answer") or not b.get("incorrect_answer"):
            continue
        seen.add(q)
        al = b.get("answer") or []
        al = al if isinstance(al, list) else [al]
        al = [a for a in al if isinstance(a, str)]
        if b["correct_answer"] not in al:
            al = [b["correct_answer"]] + al
        rows.append({"question": q, "aliases": al, "correct": b["correct_answer"], "incorrect": b["incorrect_answer"]})
    write_jsonl(out, rows)
    return len(rows)


def nqswap(out: Path):
    from datasets import load_dataset
    dd = load_dataset("pminervini/NQ-Swap")
    ds = dd[list(dd.keys())[0]]
    rows = []
    for r in ds:
        oa, sa = r.get("org_answer"), r.get("sub_answer")
        oa = oa if isinstance(oa, list) else [oa]
        sa = sa if isinstance(sa, list) else [sa]
        if not oa or not sa or not r.get("sub_context"):
            continue
        rows.append({"question": r["question"], "org_context": r["org_context"], "org_answer": oa,
                     "sub_context": r["sub_context"], "sub_answer": sa})
    write_jsonl(out, rows)
    return len(rows)


def truthfulqa(out: Path):
    from datasets import load_dataset
    ds = None
    for name in ("truthfulqa/truthful_qa", "truthful_qa"):
        try:
            ds = load_dataset(name, "multiple_choice", split="validation")
            break
        except Exception:
            continue
    rows = []
    for r in ds:
        t = r["mc1_targets"]
        rows.append({"question": r["question"], "choices": list(t["choices"]), "gold_index": int(list(t["labels"]).index(1))})
    write_jsonl(out, rows)
    return len(rows)


def mmlu_pro(out: Path):
    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    rows = [{"question": r["question"], "options": list(r["options"]), "gold_index": int(r["answer_index"]),
             "category": r["category"]} for r in ds]
    write_jsonl(out, rows)
    return len(rows)


def _gold_maps():
    """Normalised problem text -> gold answer, for ProcessBench's GSM8K and MATH splits."""
    from datasets import load_dataset
    gsm, math = {}, {}
    for r in load_dataset("openai/gsm8k", "main", split="test"):
        gsm[_ws(r["question"])] = r["answer"].split("####")[-1].strip().replace(",", "")
    for name, cfgs, pf, af in (("HuggingFaceH4/MATH-500", [None], "problem", "answer"),
                               ("EleutherAI/hendrycks_math", ["algebra", "counting_and_probability", "geometry",
                                                              "intermediate_algebra", "number_theory", "prealgebra",
                                                              "precalculus"], "problem", "solution")):
        for c in cfgs:
            try:
                ds = load_dataset(name, c, split="test") if c else load_dataset(name, split="test")
            except Exception:
                continue
            for r in ds:
                a = r[af] if af == "answer" else boxed(r[af])
                if a:
                    math.setdefault(_ws(r[pf]), a.strip())
    return gsm, math


def processbench(out: Path):
    from datasets import load_dataset
    gsm, math = _gold_maps()
    rows, miss = [], 0
    for split, gm in (("gsm8k", gsm), ("math", math)):
        for r in load_dataset("Qwen/ProcessBench", split=split):
            gold = gm.get(_ws(r["problem"]))
            if gold is None:
                miss += 1
                continue
            rows.append({"split": split, "id": r.get("id"), "problem": r["problem"], "steps": list(r["steps"]),
                         "label": int(r["label"]), "final_answer_correct": bool(r["final_answer_correct"]),
                         "generator": r.get("generator"), "gold": gold})
    write_jsonl(out, rows)
    print(f"  processbench: {miss} problems without a matched gold answer (dropped)")
    return len(rows)


def cute(out: Path):
    from datasets import load_dataset
    rows = []
    for t in CUTE_TASKS:
        try:
            ds = load_dataset("leukas/cute", split=t)
        except Exception as e:
            print(f"  cute/{t}: {type(e).__name__}")
            continue
        cols = ds.column_names
        pk = next((c for c in ("prompt", "question", "input") if c in cols), cols[0])
        ak = next((c for c in ("answer", "target", "output", "label") if c in cols), cols[-1])
        for r in ds:
            rows.append({"task": t, "prompt": str(r[pk]), "answer": str(r[ak])})
    write_jsonl(out, rows)
    return len(rows)


def cruxeval(out: Path):
    from datasets import load_dataset
    ds = load_dataset("cruxeval-org/cruxeval", split="test")
    rows = [{"id": r["id"], "code": r["code"], "input": r["input"], "output": r["output"]} for r in ds]
    write_jsonl(out, rows)
    return len(rows)


SOURCES = {"triviaqa": triviaqa, "nqswap": nqswap, "truthfulqa": truthfulqa, "mmlu_pro": mmlu_pro,
           "processbench": processbench, "cute": cute, "cruxeval": cruxeval}


def download_all(raw: Path, only: list[str] | None = None) -> dict:
    raw.mkdir(parents=True, exist_ok=True)
    status = {}
    for name, fn in SOURCES.items():
        if only and name not in only:
            continue
        out = raw / f"{name}.jsonl"
        if out.exists() and out.stat().st_size > 0:
            status[name] = "already present"
            continue
        try:
            n = fn(out)
            status[name] = f"{n} rows"
            with open(out) as f:
                print(f"  {name} sample: {f.readline()[:300]}")
        except Exception as e:
            status[name] = f"FAILED ({type(e).__name__}: {str(e)[:150]})"
            if out.exists():
                out.unlink()
        print(f"[sources] {name}: {status[name]}", flush=True)
    return status
