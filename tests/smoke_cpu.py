"""CPU smoke test of the GPU stages (continue -> extract -> fit -> trajectory -> paper) on a tiny
random model, so a typo can't surface at the end of a 3-hour job. The random model can't solve
anything, so after the real `continue` runs its texts are swapped for planted FOLLOW/RECOVER
continuations (relabelled with the real labeller); everything else is the real code.

    python tests/smoke_cpu.py /path/to/tiny_model
"""
from __future__ import annotations

import json
import os
import random
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("ERRPROP_DEVICE", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from errprop.cli import select_model  # noqa: E402
from errprop.io import load_config, run_dir, write_jsonl, read_jsonl  # noqa: E402
from errprop.labels import label_continuation  # noqa: E402
from errprop.paper import stage_fit, stage_trajectory  # noqa: E402
from errprop.paper_report import write_paper  # noqa: E402
from errprop.stages import stage_inject, stage_continue, stage_extract  # noqa: E402
from synthetic_paper import problem  # noqa: E402


def main(tiny: str):
    tmp = tempfile.mkdtemp()
    base = load_config(str(Path(__file__).resolve().parents[1] / "config_paper.yaml"),
                       {"out_dir": tmp, "report_dir": str(Path(tmp) / "reports"), "gen.backend": "hf",
                        "gen.cont_max_tokens": 6, "extract.dtype": "float32", "probes.n_boot": 50,
                        "probes.n_perm": 5, "probes.n_jobs": 2, "trajectory.n_items": 60,
                        "trajectory.max_tokens": 40, "gen.noise_n": 60, "gen.noise_k": 3})
    base["models"] = [dict(base["models"][0], name=tiny)]
    cfg = select_model(json.loads(json.dumps(base)), base["models"][0]["tag"])
    rd = run_dir(cfg)
    rng = random.Random(0)
    probs = [problem(i, rng) for i in range(160)]
    for i in range(80):                       # TriviaQA-style recalled answers (mention sites)
        y_ = 1500 + 7 * i
        probs.append({"id": f"tq{i}", "dataset": "triviaqa", "level": 0, "group": f"tq{i}", "site": "mention",
                      "question": f"In which year did event {i} happen?", "gold": float(y_),
                      "trace": f"The event happened in {y_}, as recorded.\n#### {y_}", "greedy_answer": float(y_),
                      "correct": True, "pass_rate": None, "n_eq": 0, "has_final": True})
    from transformers import AutoTokenizer
    from errprop.data import render_prompt
    tok = AutoTokenizer.from_pretrained(tiny)
    for p in probs:
        p["prompt"] = render_prompt(tok, p["question"])
    write_jsonl(rd / "clean.jsonl", probs)
    stage_inject(cfg)
    injs = read_jsonl(rd / "injections.jsonl")
    injs = [i for i in injs if i["dataset"] != "triviaqa"][:200] + [i for i in injs if i["dataset"] == "triviaqa"][:120]
    assert any(i["site"] == "mention" for i in injs), "no mention injections"
    write_jsonl(rd / "injections.jsonl", injs)
    stage_continue(cfg)                                   # real generation path (hf backend, CPU)
    conts = read_jsonl(rd / "continuations.jsonl")
    r2 = random.Random(1)
    for c, inj in zip(conts, injs):
        if inj["on_path"] and r2.random() < 0.5:
            c["greedy_text"] = f" then {inj['new_text']} + 1 = {inj['new_value'] + 1:g}.\n#### {inj['propagated_answer']:g}"
        else:
            c["greedy_text"] = f" so Wait, {inj['true_text']} is right.\n#### {inj['gold']:g}"
        c["greedy"] = label_continuation(c["greedy_text"], inj["gold"], inj)
        c["clean_text"] = f" ok.\n#### {inj['gold']:g}"
        c["clean_cont_correct"] = True
        if c.get("noise") is not None:
            c["noise"] = ["FOLLOW" if r2.random() < 0.5 else "RECOVER" for _ in c["noise"]]
    write_jsonl(rd / "continuations.jsonl", conts)
    stage_extract(cfg)
    stage_fit(cfg)
    stage_trajectory(cfg)
    T = json.load(open(rd / "trajectories.json"))
    R = json.load(open(rd / "results_paper.json"))
    print("noise floor:", {k: R["noise_floor"][k] for k in ("n_injections", "brier_constant", "brier_loo_floor")})
    print("asymmetry:", R["transfer"]["asymmetry"])
    e = T["events"]["trigger_Wait"]
    print("Wait events:", e["n_events"], "d", round(e.get("cohens_d", float("nan")), 2), "p", e.get("p_signflip"))
    print("discovery top:", T["discovery"]["most_negative"][:3])
    print("trajectory n:", {k: v["n"] for k, v in T["curves"].items()},
          {k: v["n"] for k, v in T["aligned"].items()})
    write_paper(base)
    print("CPU SMOKE OK ->", Path(base["report_dir"]) / base["run_prefix"] / "PAPER.md")


if __name__ == "__main__":
    main(sys.argv[1])
