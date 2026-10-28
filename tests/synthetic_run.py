"""End-to-end smoke test of everything after the GPU: real `inject`, fabricated continuations
(including the Wait arm) and activations with a planted signal, then real `probes` and `report`.

    python tests/synthetic_run.py          # writes <out_dir>/synthetic/REPORT.md

Planted structure (mirrors what look_v1 showed):
  - small errors (<50%) are followed almost always; large errors are followed ~half the time
  - a 'will FOLLOW' direction exists from layer 10 on, independent of difficulty
  - 'Wait' rescues most FOLLOW cases and rarely breaks clean steps
"""
from __future__ import annotations

import random
import sys
import tempfile
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
    ds = ["gsm8k", "chains", "gsmsym_p1", "bbh_arith"][i % 4]
    return {"id": f"syn-{i}", "dataset": ds, "level": i % 5, "group": f"g-{i // 2}",
            "question": q, "gold": float(g), "prompt": f"<prompt {i}> ", "trace": trace,
            "greedy_answer": float(g), "correct": True, "pass_rate": rng.random(), "n_eq": 4}


def main():
    tmp = tempfile.mkdtemp()
    cfg = load_config(str(Path(__file__).resolve().parents[1] / "config.yaml"),
                      {"run_name": "synthetic", "out_dir": tmp, "report_dir": str(Path(tmp) / "reports"),
                       "probes.n_boot": 100, "probes.layer_stride": 4, "probes.n_jobs": 2,
                       "probes.n_perm": 30, "probes.split_seeds": [0, 1]})
    rd = run_dir(cfg)
    rng = random.Random(0)
    write_jsonl(rd / "clean.jsonl", [make_problem(i, rng) for i in range(240)])
    stage_inject(cfg)
    injs = read_jsonl(rd / "injections.jsonl")

    nrng = np.random.default_rng(0)
    L, D = 29, 64
    N = len(injs)
    latent = nrng.normal(size=N)
    diff = np.array([i["pass_rate"] for i in injs])
    rel = np.array([abs(i["new_value"] - i["true_value"]) / abs(i["true_value"]) for i in injs])
    big = rel >= 0.5
    follow = np.where(big, latent + 0.5 * (diff - 0.5) + 0.3 * nrng.normal(size=N) > 0,
                      nrng.random(N) < 0.95)
    conts, meta = [], []
    for k, (inj, fo) in enumerate(zip(injs, follow)):
        if not inj["on_path"]:
            text = f" ok.\n#### {inj['gold']:g}"
        elif fo:
            text = f" so carrying {inj['new_text']} forward.\n#### {inj['propagated_answer']:g}"
        else:
            silent = nrng.random() < 0.3
            text = (" fine.\n" if silent else f" so {inj['true_text']} it is.\n") + f"#### {inj['gold']:g}"
        rescued = (not fo) or (nrng.random() < 0.8)
        wait_text = f" that should be {inj['true_text']}.\n#### {inj['gold']:g}" if rescued else \
            f" yes.\n#### {inj['propagated_answer']:g}"
        lab, labw = label_continuation(text, inj["gold"], inj), label_continuation(wait_text, inj["gold"], inj)
        conts.append({"inj_id": inj["inj_id"], "greedy_text": text, "sample_texts": [text], "clean_text": "",
                      "wait_text": wait_text, "wait_clean_text": "", "greedy": lab, "samples": [lab], "wait": labw,
                      "clean_cont_correct": True, "wait_clean_correct": bool(nrng.random() > 0.05),
                      "follow_rate": float(fo) if inj["on_path"] else None})
        meta.append({"inj_id": inj["inj_id"], "surprise_corrupt": float(nrng.normal(5, 1) + 2 * big[k]),
                     "surprise_clean": float(nrng.normal(2, 1)), "n_num_tokens": 2, "prefix_ok": True,
                     "n_tokens": 300})
    write_jsonl(rd / "continuations.jsonl", conts)
    write_jsonl(rd / "extract_meta.jsonl", meta)

    base = nrng.normal(size=(N, L, D)).astype(np.float32)
    gain = (np.arange(L) >= 10).astype(np.float32)
    clean, corrupt = base.copy(), base.copy()
    corrupt[:, :, 0] += 3.0
    corrupt[:, :, 1] += (latent * big)[:, None] * gain[None, :] * 2
    for arr in (clean, corrupt):
        arr[:, :, 2] += (diff[:, None] - 0.5) * 3
    tl = np.array([int(abs(round(i["true_value"]))) % 10 for i in injs])
    nl = np.array([int(abs(round(i["new_value"]))) % 10 for i in injs])
    for n in range(N):
        clean[n, :, 10 + tl[n]] += 3
        corrupt[n, :, 10 + nl[n]] += 3
    np.save(rd / "acts_clean.npy", clean.astype(np.float16))
    np.save(rd / "acts_corrupt.npy", corrupt.astype(np.float16))
    pids = sorted({i["problem_id"] for i in injs})
    write_jsonl(rd / "prompt_ids.jsonl", [{"problem_id": p} for p in pids])
    np.save(rd / "acts_prompt.npy", nrng.normal(size=(len(pids), L, D)).astype(np.float16))

    R = run_probes(cfg)
    write_report(cfg)
    pc = R["probes"]["P1_catchable"]
    assert pc["probe"]["auroc"] > 0.8, pc["probe"]
    assert pc["permutation_null"]["p_value"] < 0.05
    assert R["probes"]["P0_detect"]["test"]["auroc"] > 0.95
    W = R["probes"]["W_wait_policy"]["corrupted_pool"]
    i20 = [r["budget"] for r in W["probe"]].index(0.2)
    assert W["probe"][i20]["accuracy"] >= W["random"][i20]["accuracy"], "probe policy should beat random"
    assert (Path(cfg["report_dir"]) / "synthetic" / "REPORT.md").exists()
    print("SYNTHETIC RUN OK ->", rd / "REPORT.md")


if __name__ == "__main__":
    main()
