"""End-to-end test of the paper recreation on fabricated data: real `inject`, fabricated
continuations/activations for two "models", real `fit` and `paper` (trajectory needs a GPU, so a
small fake trajectories.json is written for one model).

Planted structure: each problem has a susceptibility (readable at the prompt AND at the error token);
each injection adds its own term (readable only at the error token from layer 8 on). So the expected
ordering is error-token > prompt-only > constant, and error-token > black-box.

    python tests/synthetic_paper.py
"""
from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from errprop.cli import select_model  # noqa: E402
from errprop.io import load_config, run_dir, write_jsonl, read_jsonl  # noqa: E402
from errprop.labels import label_continuation  # noqa: E402
from errprop.paper import stage_fit  # noqa: E402
from errprop.paper_report import write_paper  # noqa: E402
from errprop.stages import stage_inject  # noqa: E402

DS = ["gsm8k", "chains", "bbh_arith", "mmlu_pro"]


def problem(i, rng):
    a, b, c, d = rng.randint(11, 60), rng.randint(3, 9), rng.randint(70, 140), rng.randint(2, 7)
    e, f = a * b, a * b + c
    g = f * d
    trace = f"First {a} * {b} = {e}.\nThen {e} + {c} = {f}.\nFinally {f} * {d} = {g}.\n#### {g}"
    return {"id": f"p{i}", "dataset": DS[i % 4], "level": 0, "group": f"p{i}", "question": f"Q {i}: {a} {b} {c} {d}",
            "gold": float(g), "prompt": f"<p{i}> ", "trace": trace, "greedy_answer": float(g), "correct": True,
            "pass_rate": None, "n_eq": 3}


def fake_model(cfg, seed):
    rd = run_dir(cfg)
    rng = random.Random(seed)
    write_jsonl(rd / "clean.jsonl", [problem(i, rng) for i in range(360)])
    stage_inject(cfg)
    injs = read_jsonl(rd / "injections.jsonl")
    nr = np.random.default_rng(seed)
    pids = sorted({i["problem_id"] for i in injs})
    zp = {p: nr.normal() for p in pids}                          # problem susceptibility
    N, L, D = len(injs), 17, 48
    zi = nr.normal(size=N)                                       # injection-specific term
    follow = np.array([zp[i["problem_id"]] + zi[k] + 0.4 * nr.normal() > 0 for k, i in enumerate(injs)])
    conts, meta = [], []
    for k, (inj, fo) in enumerate(zip(injs, follow)):
        nh = (k % 4) if fo else 0                  # contamination horizon varies 0-3
        uses = "".join(f"\n{inj['new_value']:g} + {q + 1} = {inj['new_value'] + q + 1:g}." for q in range(nh))
        text = (f" carry on.{uses}\n#### {inj['propagated_answer']:g}" if fo and inj["on_path"]
                else f" so {inj['true_text']}.\n#### {inj['gold']:g}")
        pf = 1 / (1 + np.exp(-(zp[inj["problem_id"]] + zi[k])))
        noise = ["FOLLOW" if nr.random() < pf else "RECOVER" for _ in range(4)] if k % 3 == 0 else None
        lab = label_continuation(text, inj["gold"], inj)
        conts.append({"inj_id": inj["inj_id"], "greedy_text": text, "sample_texts": [], "clean_text": "ok",
                      "greedy": lab, "samples": [], "wait": {"outcome": None}, "clean_cont_correct": True, "noise": noise,
                      "wait_clean_correct": None, "follow_rate": None})
        meta.append({"inj_id": inj["inj_id"], "surprise_corrupt": float(nr.normal(5, 1)),
                     "surprise_clean": float(nr.normal(2, 1)), "n_num_tokens": 2, "prefix_ok": True, "n_tokens": 200})
    write_jsonl(rd / "continuations.jsonl", conts)
    write_jsonl(rd / "extract_meta.jsonl", meta)
    gain = (np.arange(L) >= 8).astype(np.float32)
    A = nr.normal(size=(N, L, D)).astype(np.float32)
    A[:, :, 0] += np.array([zp[i["problem_id"]] for i in injs])[:, None] * 1.5
    A[:, :, 1] += zi[:, None] * gain[None, :] * 2.0
    np.save(rd / "acts_corrupt.npy", A.astype(np.float16))
    np.save(rd / "acts_clean.npy", nr.normal(size=(N, L, D)).astype(np.float16))
    P = nr.normal(size=(len(pids), L, D)).astype(np.float32)
    P[:, :, 0] += np.array([zp[p] for p in pids])[:, None] * 1.5
    np.save(rd / "acts_prompt.npy", P.astype(np.float16))
    write_jsonl(rd / "prompt_ids.jsonl", [{"problem_id": p} for p in pids])
    return rd


def _fake_event(m):
    from errprop.paper import _event_stats
    r = np.random.default_rng(0)
    ev = [(i % 7, float(r.normal(m, 0.5))) for i in range(30)]
    return _event_stats(ev, [r.normal(0, 0.5, 50) for _ in ev], None, 500, 0)


def main():
    tmp = tempfile.mkdtemp()
    base = load_config(str(Path(__file__).resolve().parents[1] / "config_paper.yaml"),
                       {"out_dir": tmp, "report_dir": str(Path(tmp) / "reports"),
                        "probes.n_boot": 100, "probes.n_perm": 20, "probes.n_jobs": 2})
    base["models"] = base["models"][:2]
    for k, m in enumerate(base["models"]):
        cfg = select_model(json.loads(json.dumps(base)), m["tag"])
        rd = fake_model(cfg, k)
        stage_fit(cfg)
        R = json.load(open(rd / "results_paper.json"))
        p = R["cells"]["pooled"]
        et, po, bb = p["error_token"]["auroc"][0], p["prompt_only"]["auroc"][0], p["blackbox"]["auroc"][0]
        print(f"{m['tag']}: error-token {et:.3f}  prompt-only {po:.3f}  black-box {bb:.3f}")
        assert et > po > 0.55 and et > bb + 0.1, (et, po, bb)
        assert R["pooled_seed0"]["permutation"]["p"] < 0.1
        assert len(R["transfer"]["probe"]) == len(R["dataset_list"])
        assert R["noise_floor"]["brier_loo_floor"] < R["noise_floor"]["brier_constant"], R["noise_floor"]
        assert "mae_probe" in R["horizon"], R.get("horizon")
        assert "asymmetry" in R["transfer"] and "surface_prompt" in p and "shuffle_within" in p
        if k == 0:                                     # fake trajectory output for the report
            rel = list(range(-8, 49))
            json.dump({"layer": 10, "rel": rel,
                       "curves": {o: {"n": 5, "mean": list(np.linspace(0.3, v, len(rel))), "se": [0.0] * len(rel)}
                                  for o, v in [("FOLLOW", 0.8), ("RECOVER", 0.2), ("CLEAN", 0.5)]},
                       "aligned_rel": list(range(-12, 13)),
                       "aligned": {k2: {"n": 4, "mean": [0.5] * 25, "se": [0.0] * 25}
                                   for k2 in ("RECOVER_at_true_value_rewritten", "FOLLOW_at_injected_value_reused")},
                       "examples": [{"inj_id": "p0#0", "outcome": "RECOVER", "kind": "times_ten", "true": "42",
                                     "new": "420", "inj_index": 2, "tokens": ["=", " 4", "20", "\n"],
                                     "p_follow": [0.4, 0.5, 0.2, 0.1]}],
                       "trigger": "Wait", "n_items": 50, "n_items_with_trigger": 20,
                       "events": {"trigger_Wait": {**_fake_event(-1.2), "confirmation_half": _fake_event(-1.1),
                                                   "by_outcome_FOLLOW": _fake_event(-0.4),
                                                   "by_outcome_RECOVER": _fake_event(-1.6),
                                                   "examples": [{"inj_id": "p1#0", "outcome": "RECOVER", "event_index": 2,
                                                                 "tokens": [".", "\n", "Wait", ",", " no"],
                                                                 "p_follow": [0.8, 0.8, 0.3, 0.25, 0.2]}]},
                                  "correction_words": {"n_events": 2}},
                       "discovery": {"n_token_types": 300, "global_drift": 0.001, "trigger_rank_most_negative": 1,
                                     "most_negative": [["Wait", -1.2, 30]], "most_positive": [["=", 0.3, 400]]}},
                      open(rd / "trajectories.json", "w"))
    write_paper(base)
    out = Path(base["report_dir"]) / base["run_prefix"] / "PAPER.md"
    txt = out.read_text()
    for must in ("Tables 4–5", "Tables 1 and 6", "Implied claims", "Transfer asymmetry", "Fisher", "retraction",
                 "Sampling-noise floor", "constant-median floor", "Surface features", "Limitations", "Appendix"):
        assert must in txt, must
    print("SYNTHETIC PAPER OK ->", out)


if __name__ == "__main__":
    main()
