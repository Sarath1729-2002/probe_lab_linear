import argparse
import json
from pathlib import Path

DEFAULT_SIZES = {"gsm8k": 2400, "math": 1800, "arc": 1200, "arith": 1000}


def _write_meta(run: Path, a):
    import time
    try:
        import torch
        import transformers
        ver = {"torch": torch.__version__, "transformers": transformers.__version__}
    except Exception:                                   # noqa: BLE001
        ver = {}
    d = run / "meta"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{a.cmd}_{a.shard:02d}.json").write_text(json.dumps({"args": vars(a), "versions": ver, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["data", "gen", "states", "trace", "analyze", "report"])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--n-spread", type=int, default=4, help="sampled plain solutions per problem (length spread, soft runaway rate)")
    ap.add_argument("--max-new", type=int, default=0, help="0 = per-dataset default")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--max-seqs", type=int, default=96)
    ap.add_argument("--fwd-bs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N problems of every dataset")
    ap.add_argument("--datasets", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--budget-s", type=float, default=0)
    ap.add_argument("--sub", default="", help="w/W: take every W-th chunk of this shard (rescue mode)")
    ap.add_argument("--block", default="P", help="analysis block: P, C, K, tables")
    ap.add_argument("--sizes", default="")
    a = ap.parse_args(argv)
    run = Path(a.run_dir)
    dss = tuple(x for x in a.datasets.split(",") if x) or None
    if a.cmd == "data":
        from . import bench
        bench.build(run, {**DEFAULT_SIZES, **(json.loads(a.sizes) if a.sizes else {})})
        return
    if a.cmd == "analyze":
        from . import study
        study.compute(run, a.block, dss)
        return
    if a.cmd == "report":
        from . import report
        report.write(run)
        return
    from .bench import read_problems
    from .hf_util import load_model
    _write_meta(run, a)
    model, tok = load_model(a.model_id)
    problems = read_problems(run, dss)
    if a.limit:
        keep, cnt = [], {}
        for p in problems:
            cnt[p["ds"]] = cnt.get(p["ds"], 0) + 1
            if cnt[p["ds"]] <= a.limit:
                keep.append(p)
        problems = keep
    if a.cmd == "gen":
        from . import gen
        gen.run(problems, model, tok, run, a.shard, a.nshards, chunk=a.chunk, max_new=a.max_new, n_spread=a.n_spread, max_seqs=a.max_seqs, tag=a.tag, budget_s=a.budget_s, sub=a.sub)
    elif a.cmd == "states":
        from . import states
        states.run(problems, model, tok, run, a.shard, a.nshards, fwd_bs=a.fwd_bs * 2, tag=a.tag, budget_s=a.budget_s, sub=a.sub)
    else:
        from . import trace
        trace.run(problems, model, tok, run, a.shard, a.nshards, fwd_bs=a.fwd_bs, chunk=a.chunk, tag=a.tag, budget_s=a.budget_s, sub=a.sub)


if __name__ == "__main__":
    main()
