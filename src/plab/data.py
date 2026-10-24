"""GSM8K problem lists (LOGIN node, needs internet).  Compute nodes read only problems.jsonl."""
import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def parse_gold(answer: str) -> Optional[int]:
    tail = answer.split("####")[-1].replace(",", "").replace("$", "").strip()
    m = _NUM.search(tail)
    if not m:
        return None
    v = float(m.group())
    return int(v) if v == int(v) else None


def n_steps(answer: str) -> int:
    k = answer.count("<<")
    return k if k else max(1, len([l for l in answer.split("\n") if l.strip() and not l.startswith("####")]))


def hint_wrong(pid: int, gold: int) -> int:
    """A plausible WRONG answer, deterministic per problem."""
    rng = np.random.RandomState(1000 + pid)
    d = int(rng.choice([-10, -3, -2, -1, 1, 2, 3, 10]))
    x = gold + d
    if x < 0 or x == gold:
        x = gold + abs(d)
    return int(x)


def load_gsm8k(loader: Optional[Callable] = None) -> Dict[str, List[dict]]:
    if loader is None:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main")
        loader = lambda split: [dict(r) for r in ds[split]]      # noqa: E731
    return {s: loader(s) for s in ("train", "test")}


def build_problems(out_dir: Path, raw: Dict[str, List[dict]], n_train_extra: int = 2700, seed: int = 0, log=print) -> int:
    """All GSM8K test problems + n_train_extra random train problems, assigned to fit/val/test BY PROBLEM (70/10/20)."""
    rng = np.random.RandomState(seed)
    rows = []
    for split in ("test", "train"):
        recs = raw[split]
        idx = np.arange(len(recs))
        if split == "train":
            idx = rng.permutation(idx)[:n_train_extra]
        for i in idx:
            r = recs[int(i)]
            g = parse_gold(r["answer"])
            if g is None or g < 0:
                continue
            rows.append({"src": f"{split}{int(i)}", "question": r["question"].strip(), "gold": g, "steps": n_steps(r["answer"])})
    perm = rng.permutation(len(rows))
    n = len(rows)
    role = np.empty(n, dtype=object)
    role[perm[: int(0.7 * n)]] = "fit"
    role[perm[int(0.7 * n): int(0.8 * n)]] = "val"
    role[perm[int(0.8 * n):]] = "test"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "problems.jsonl", "w") as f:
        for pid, (r, ro) in enumerate(zip(rows, role)):
            r.update(id=pid, split=str(ro), hint=hint_wrong(pid, r["gold"]))
            f.write(json.dumps(r) + "\n")
    log(f"{n} problems -> {out_dir / 'problems.jsonl'}  (fit/val/test = {(role == 'fit').sum()}/{(role == 'val').sum()}/{(role == 'test').sum()})")
    return n


def read_problems(run_dir: Path) -> List[dict]:
    with open(Path(run_dir) / "problems.jsonl") as f:
        return [json.loads(l) for l in f if l.strip()]
