"""E1/E2 labels by sampling: N answers per (problem, condition) at T>0.  Resumable chunks, one process per GPU."""
import json
import math
from collections import Counter
from pathlib import Path
from typing import List

import torch

from .hf_util import generate
from .prompts import CONDS, chat_prompt, parse_boxed, same, user_text


def answer_entropy(answers: List) -> float:
    c = Counter(answers)
    n = sum(c.values())
    return -sum(v / n * math.log(v / n) for v in c.values())


def run(problems, model, tok, out_dir: Path, shard=0, nshards=1, conds=CONDS, n=16, max_new=512, temperature=0.7, top_p=0.95,
        chunk=64, max_seqs=128, tag="", log=print) -> int:
    d = Path(out_dir) / "samples"
    d.mkdir(parents=True, exist_ok=True)
    mine = [p for p in problems if p["id"] % nshards == shard]
    done = 0
    for k in range(0, len(mine), chunk):
        f = d / f"s{shard:02d}{tag}_{k // chunk:05d}.jsonl"
        if f.exists():
            continue
        batch = mine[k:k + chunk]
        rows = []
        for cond in conds:
            prompts = [chat_prompt(tok, user_text(p["question"], cond, p["hint"])) for p in batch]
            outs = generate(model, tok, prompts, n, max_new, temperature, top_p, max_seqs)
            for p, texts in zip(batch, outs):
                ans = [parse_boxed(t) for t in texts]
                key = [None if a is None else round(a, 6) for a in ans]
                rows.append({"id": p["id"], "cond": cond, "n": n,
                             "n_correct": sum(same(a, p["gold"]) for a in ans),
                             "n_follow": sum(same(a, p["hint"]) for a in ans),
                             "n_none": sum(a is None for a in ans),
                             "ent": answer_entropy(key),
                             "mean_chars": sum(len(t) for t in texts) / n,
                             "answers": key, "ex": texts[0] if p["id"] < 60 else None})
        tmp = f.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        tmp.rename(f)
        done += len(batch)
        log(f"[sample shard {shard}] {done}/{len(mine)} problems")
    return done
