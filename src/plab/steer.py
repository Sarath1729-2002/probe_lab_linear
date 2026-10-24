"""M5 (causal): add the probe direction (or a random direction of the same norm) to the residual stream at the probe's layer and
measure the change in the rate at which the model ends on the planted wrong answer.  Problems = held-out TEST problems the model
solves reliably without a hint (so adoption is a genuine deviation).  Hook acts on every position, prompt and generated."""
import json
from pathlib import Path

import numpy as np
import torch

from .analysis import load_samples
from .data import read_problems
from .hf_util import generate
from .introspect import Injector, typical_norm
from .prompts import chat_prompt, parse_boxed, same, user_text

ALPHAS = (-0.4, -0.2, 0.2, 0.4)        # fractions of the typical residual-stream norm at the probe layer
N_RANDOM = 2


def pick_problems(run_dir: Path, n: int, solved_only: bool = True):
    S = load_samples(run_dir)
    probs = read_problems(run_dir)
    ok = [p for p in probs if p["split"] == "test" and (p["id"], "clean") in S
          and (not solved_only or S[(p["id"], "clean")]["n_correct"] == S[(p["id"], "clean")]["n"])]
    return ok[:n]


def run(model, tok, run_dir: Path, shard=0, nshards=1, n_problems=400, n=8, max_new=512, temperature=0.7, top_p=0.95, max_seqs=192,
        alphas=ALPHAS, n_random=N_RANDOM, solved_only=True, log=print) -> int:
    run_dir = Path(run_dir)
    d = run_dir / "steer"
    d.mkdir(exist_ok=True)
    dirs = np.load(run_dir / "analysis_v3" / "directions.npz")
    meta = json.loads((run_dir / "analysis_v3" / "directions.json").read_text())
    probs = pick_problems(run_dir, n_problems, solved_only)
    mine = [p for i, p in enumerate(probs) if i % nshards == shard]
    rng = np.random.RandomState(0)
    done = 0
    for f in dirs.files:
        L = meta[f]["layer"]
        prompts = [chat_prompt(tok, user_text(p["question"], f, p["hint"])) for p in mine]
        norm = typical_norm(model, tok, prompts[:8], L)
        D = dirs[f].shape[0]
        rand = []
        for _ in range(n_random):
            v = rng.randn(D).astype(np.float32)
            rand.append(v / np.linalg.norm(v))
        arms = [("none", 0.0, None)] + [(name, a, vec) for name, vec in [("probe", dirs[f])] + [(f"rand{i}", r) for i, r in enumerate(rand)] for a in alphas]
        for name, a, vec in arms:
            out = d / f"{f}_{name}_{a:+.2f}_s{shard:02d}.jsonl"
            if out.exists():
                continue
            with Injector(model, L) as inj:
                inj.vec = None if vec is None else torch.tensor(vec) * (a * norm)
                texts = generate(model, tok, prompts, n, max_new, temperature, top_p, max_seqs)
                inj.vec = None
            rows = []
            for p, tx in zip(mine, texts):
                ans = [parse_boxed(t) for t in tx]
                rows.append({"id": p["id"], "framing": f, "arm": name, "alpha": a, "layer": L, "n": n,
                             "n_follow": sum(same(x, p["hint"]) for x in ans), "n_correct": sum(same(x, p["gold"]) for x in ans),
                             "n_none": sum(x is None for x in ans), "mean_chars": sum(len(t) for t in tx) / n})
            tmp = out.with_suffix(".tmp")
            with open(tmp, "w") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            tmp.rename(out)
            done += len(mine)
            log(f"[steer shard {shard}] {f} {name} alpha={a:+.2f} done")
    return done
