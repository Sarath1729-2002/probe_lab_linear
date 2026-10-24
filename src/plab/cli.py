import argparse
import sys
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["data", "sample", "states", "introspect", "analyze"])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--max-seqs", type=int, default=128)
    ap.add_argument("--fwd-bs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--extra-train", type=int, default=2700)
    a = ap.parse_args(argv)
    run = Path(a.run_dir)
    if a.cmd == "data":
        from .data import build_problems, load_gsm8k
        build_problems(run, load_gsm8k(), a.extra_train)
        return
    if a.cmd == "analyze":
        from . import analysis
        main_res = analysis.analyse_main(run)
        intro = analysis.analyse_introspect(run)
        analysis.write_report(run, main_res, intro)
        return
    from .data import read_problems
    from .hf_util import load_model
    model, tok = load_model(a.model_id)
    if a.cmd == "introspect":
        from . import introspect
        introspect.run(model, tok, run)
        return
    problems = read_problems(run)
    if a.limit:
        problems = problems[: a.limit]
    if a.cmd == "sample":
        from . import sample
        sample.run(problems, model, tok, run, a.shard, a.nshards, n=a.n, max_new=a.max_new, temperature=a.temperature,
                   top_p=a.top_p, chunk=a.chunk, max_seqs=a.max_seqs)
    else:
        from . import states
        states.run(problems, model, tok, run, a.shard, a.nshards, fwd_bs=a.fwd_bs)


if __name__ == "__main__":
    main()
