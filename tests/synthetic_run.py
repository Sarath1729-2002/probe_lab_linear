"""End-to-end smoke test of everything after the GPU: real `inject`, fabricated continuations and
activations with a planted signal, then real `probes` and `report`.

    python tests/synthetic_run.py      # writes runs/synthetic/REPORT.md

The planted signal: one direction carries 'will FOLLOW' in layers >= 10, independent of difficulty;
another carries difficulty everywhere. P1 should find the first and beat the covariates.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from errprop.io import load_config, run_dir, write_jsonl, read_jsonl  # noqa: E402
from errprop.stages import stage_inject  # noqa: E402
from errprop.labels import label_continuation  # noqa: E402
from errprop.analysis import run_probes  # noqa: E402
from errprop.report import write_report  # noqa: E402


def make_problem(i, rng):
    a, b, c, d = rng.randint(11, 60), rng.randint(3, 9), rng.randint(70, 140), rng.randint(2, 7)
    e = a * b
    f = e + c
    g = f * d
    h = rng.randint(13, 29)
    q = f"Item {i}: values {a}, {b}, {c}, {d} and {h} spare."
    trace = (f"First {a} * {b} = {e}.\nThen {e} + {c} = {f}.\nSide note {h} + 1000 = {h + 1000}.\n"
             f"Finally {f} * {d} = {g}.\n#### {g}")
    return {"id": f"syn-{i}", "dataset": "math" if i % 2 else "gsm8k", "level": i % 5,
            "question": q, "gold": float(g), "prompt": f"<prompt {i}> ", "trace": trace,
            "greedy_answer": float(g), "correct": True, "pass_rate": rng.random(), "n_eq": 4}


def main():
    cfg = load_config(str(Path(__file__).resolve().parents[1] / "config.yaml"),
                      {"run_name": "synthetic", "probes.n_boot": 200, "probes.layer_stride": 4,
                       "probes.n_jobs": 2})
    rd = run_dir(cfg)
    rng = random.Random(0)
    write_jsonl(rd / "clean.jsonl", [make_problem(i, rng) for i in range(160)])
    stage_inject(cfg)
    injs = read_jsonl(rd / "injections.jsonl")

    nrng = np.random.default_rng(0)
    L, D = 29, 64
    N = len(injs)
    latent = nrng.normal(size=N)                       # instance-level propensity to follow
    diff = np.array([i["pass_rate"] for i in injs])
    follow = (latent + 0.5 * (diff - 0.5) + 0.3 * nrng.normal(size=N)) > 0.3
    conts, meta = [], []
    for inj, fo in zip(injs, follow):
        if not inj["on_path"]:
            text = f" ok.\n#### {inj['gold']:g}"
        elif fo:
            text = f" so carrying {inj['new_text']} forward.\n#### {inj['propagated_answer']:g}"
        else:
            silent = nrng.random() < 0.5
            text = (" fine.\n" if silent else f" wait, that should be {inj['true_text']}.\n") + f"#### {inj['gold']:g}"
        lab = label_continuation(text, inj["gold"], inj)
        conts.append({"inj_id": inj["inj_id"], "greedy_text": text, "sample_texts": [text], "clean_text": "",
                      "greedy": lab, "samples": [lab], "clean_cont_correct": True,
                      "follow_rate": float(fo) if inj["on_path"] else None})
        meta.append({"inj_id": inj["inj_id"], "surprise_corrupt": float(nrng.normal(5, 1)),
                     "surprise_clean": float(nrng.normal(2, 1)), "n_num_tokens": 2, "prefix_ok": True,
                     "n_tokens": 300})
    write_jsonl(rd / "continuations.jsonl", conts)
    write_jsonl(rd / "extract_meta.jsonl", meta)

    base = nrng.normal(size=(N, L, D)).astype(np.float32)
    layer_gain = (np.arange(L) >= 10).astype(np.float32)
    clean = base.copy()
    corrupt = base.copy()
    corrupt[:, :, 0] += 3.0                                        # 'corrupted' direction
    corrupt[:, :, 1] += latent[:, None] * layer_gain[None, :] * 2  # 'will follow' direction
    for arr in (clean, corrupt):
        arr[:, :, 2] += (diff[:, None] - 0.5) * 3                  # difficulty direction
    # last-digit code for value tracking: true digit in dims 10-19 (stronger for RECOVER in mid layers)
    tl = np.array([int(abs(round(i["true_value"]))) % 10 for i in injs])
    nl = np.array([int(abs(round(i["new_value"]))) % 10 for i in injs])
    for n in range(N):
        clean[n, :, 10 + tl[n]] += 3
        corrupt[n, :, 10 + nl[n]] += 3
        if not follow[n]:
            corrupt[n, 12:20, 10 + tl[n]] += 5
    np.save(rd / "acts_clean.npy", clean.astype(np.float16))
    np.save(rd / "acts_corrupt.npy", corrupt.astype(np.float16))
    pids = sorted({i["problem_id"] for i in injs})
    write_jsonl(rd / "prompt_ids.jsonl", [{"problem_id": p} for p in pids])
    np.save(rd / "acts_prompt.npy", nrng.normal(size=(len(pids), L, D)).astype(np.float16))

    R = run_probes(cfg)
    write_report(cfg)
    p1 = R["probes"]["P1_follow_vs_recover"]
    assert p1["probe"]["auroc"] > 0.8, p1["probe"]
    assert p1["best_layer"] >= 8, p1["best_layer"]
    assert abs(p1["shuffled_label_test_auroc"] - 0.5) < 0.2
    assert R["probes"]["P0_detect"]["test"]["auroc"] > 0.95
    print("SYNTHETIC RUN OK")


if __name__ == "__main__":
    main()
