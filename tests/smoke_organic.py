"""CPU smoke test of the organic pipeline on a tiny random model: build -> gen -> extract -> fit -> traj -> report.

Raw files for all seven sources are mocked in the normalised schema the download step writes.
The random model can't answer, so after `gen` the outcomes are replaced by planted ones, and after
`extract` the states get a planted signal: the wrong suggestion is readable where it ends, and at the
last prompt token only for items that were not adopted (flag-then-fade). The checks assert that the
analysis recovers exactly that.

    python tests/smoke_organic.py /path/to/tiny_model
"""
from __future__ import annotations

import json
import os
import random
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("ERRPROP_DEVICE", "cpu")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from organic.common import load_config, read_jsonl, write_jsonl  # noqa: E402
from organic.build import build  # noqa: E402
from organic.fit import stage_fit  # noqa: E402
from organic.grade import grade  # noqa: E402
from organic.report import write_report  # noqa: E402
from organic.run import bdir, stage_extract, stage_gen  # noqa: E402
from organic.traj import stage_traj  # noqa: E402


def mock_raw(raw: Path, rng):
    raw.mkdir(parents=True)
    write_jsonl(raw / "triviaqa.jsonl", [{"question": f"Which city hosted event {i}?", "aliases": [f"Town{i}", f"T{i}"],
                                          "correct": f"Town{i}", "incorrect": f"Ville{i}"} for i in range(150)])
    write_jsonl(raw / "nqswap.jsonl", [{"question": f"who wrote book {i}", "org_context": f"Book {i} was written by Ann{i}.",
                                        "org_answer": [f"Ann{i}"], "sub_context": f"Book {i} was written by Bob{i}.",
                                        "sub_answer": [f"Bob{i}"]} for i in range(150)])
    write_jsonl(raw / "truthfulqa.jsonl", [{"question": f"What happens if you do thing {i}?",
                                            "choices": [f"Nothing {i}", f"Myth {i}", f"Other myth {i}"], "gold_index": 0}
                                           for i in range(120)])
    write_jsonl(raw / "mmlu_pro.jsonl", [{"question": f"Q{i}: pick", "options": [f"o{i}{j}" for j in range(10)] + ["N/A"],
                                          "gold_index": i % 10, "category": ["math", "law"][i % 2]} for i in range(200)])
    pb = []
    for i in range(160):
        g = 10 + i
        bad = i % 2 == 0
        steps = [f"Step 1: {i} + 10 = {g}.", f"Step 2: so it is {g + 1 if bad else g}.",
                 f"The answer is $\\boxed{{{g + 1 if bad else g}}}$."]
        pb.append({"split": "gsm8k", "problem": f"What is {i} + 10?", "steps": steps, "label": 1 if bad else -1,
                   "final_answer_correct": not bad, "gold": str(g)})
    write_jsonl(raw / "processbench.jsonl", pb)
    write_jsonl(raw / "cute.jsonl", [{"task": t, "prompt": f"Spell out the word \"w{i}\".", "answer": " ".join(f"w{i}")}
                                     for t in ("spell", "swap_char") for i in range(70)] +
                [{"task": "contains_char", "prompt": f"Does \"w{i}\" contain \"x\"?", "answer": ["Yes", "No"][i % 2]}
                 for i in range(70)])
    write_jsonl(raw / "cruxeval.jsonl", [{"id": f"s{i}", "code": f"def f(x):\n    return x + {i}", "input": "1",
                                          "output": str(1 + i)} for i in range(120)])


def main(tiny: str):
    tmp = Path(tempfile.mkdtemp())
    cfg = load_config(str(ROOT / "config_organic.yaml"),
                      {"out_dir": str(tmp / "runs"), "data_dir": str(tmp / "data"), "report_dir": str(tmp / "reports"),
                       "model": tiny, "gen.backend": "hf", "gen.max_tokens": {"default": 4},
                       "extract.dtype": "float32", "probes.n_boot": 50, "traj.n_items": 20, "traj.max_tokens": 4,
                       "buckets": {"triviaqa": 150, "nqswap": 150, "truthfulqa": 120, "mmlu_pro": 100,
                                   "processbench": 80, "cute": 70, "cruxeval": 120}})
    rng = random.Random(0)
    dd = tmp / "data"
    mock_raw(dd / "raw", rng)
    items, V = build(cfg, dd, tmp / "reports" / cfg["run_prefix"] / "SAMPLES.md")
    assert {it["bucket"] for it in items} == set(cfg["buckets"]), {it["bucket"] for it in items}
    I = {it["id"]: it for it in items}
    # grader sanity on real-looking replies
    tq = next(it for it in items if it["bucket"] == "triviaqa")
    assert grade(tq, f"It is {tq['gold_text']}.\nAnswer: {tq['gold_text']}")[0] == "GOLD"
    assert grade(tq, f"Answer: {tq['wrong_text']}")[0] == "WRONG"
    mc = next(it for it in items if it["type"] == "mc")
    assert grade(mc, f"**Answer:** ({mc['gold']})")[0] == "GOLD" and grade(mc, f"Answer: {mc['wrong']}")[0] == "WRONG"
    pbi = next(it for it in items if it["bucket"] == "processbench" and it["wrong"])
    assert grade(pbi, f"so \\boxed{{{pbi['gold']}}}")[0] == "GOLD" and grade(pbi, f"Answer: {pbi['wrong']} apples")[0] == "WRONG"
    cx = next(it for it in items if it["bucket"] == "cruxeval")
    assert grade(cx, f"Answer: `{cx['gold']}`")[0] == "GOLD"

    buckets = sorted({it["bucket"] for it in items})
    stage_gen(cfg, buckets)                                  # real generation path (hf backend)
    # plant outcomes: 70% known; wrong suggestion adopted by a third of known items
    adopt = {}
    for b in buckets:
        rows = read_jsonl(bdir(cfg, b) / "gen.jsonl")
        for r in rows:
            iid, fam, role = r["vid"].split("|")
            h = random.Random(iid)
            known, ad = h.random() < 0.7, h.random() < 0.35
            adopt[iid] = ad
            if fam == "base":
                r["outcome"] = "GOLD" if known else "OTHER"
            elif role == "wrong":
                r["outcome"] = "WRONG" if ad else "GOLD"
            else:
                r["outcome"] = "GOLD"
            r["text"] = r["text"] or "x"
        write_jsonl(bdir(cfg, b) / "gen.jsonl", rows)
    stage_extract(cfg, buckets)
    nr = np.random.default_rng(0)
    for b in buckets:
        d = bdir(cfg, b)
        idx = json.load(open(d / "acts_index.json"))
        A = nr.normal(size=np.load(d / "acts.npy").shape).astype(np.float32)
        for k, v in enumerate(idx["vids"]):
            iid, fam, role = v.split("|")
            if role == "wrong":
                A[k, 0, :, 0] += 3.0                                 # registered where it ends ...
                if not adopt[iid]:
                    A[k, 1, :, 0] += 3.0                             # ... and kept only when not adopted
        np.save(d / "acts.npy", A.astype(np.float16))
    R = stage_fit(cfg)
    stage_traj(cfg, buckets)
    out = write_report(cfg)
    for b in buckets:
        p = R["buckets"][b]["probes"]["sugg"]
        a0, a1 = p["info_end"], p["last_prompt_token"]
        print(f"{b:13s} det@info {a0.get('detection_auroc', float('nan')):.3f}  flagged adopted/held @info "
              f"{a0.get('flagged_adopted')}/{a0.get('flagged_held')}  @last {a1.get('flagged_adopted')}/{a1.get('flagged_held')}")
        assert a0["detection_auroc"] > 0.75, (b, a0)   # tiny model: few dims, one planted
        assert a1["flagged_adopted"] < a1["flagged_held"] and a1["detection_score_predicts_adopt_auroc"] < 0.4, (b, a1)
    assert "matrix_train_row_test_col" in R["transfer"], R["transfer"]
    txt = out.read_text()
    for must in ("## Data", "## Behaviour on known items", "## Does the model register", "## One signal",
                 "## From the suggestion to the answer", "work_trunc", "doc"):
        assert must in txt, must
    assert (out.parent / "fig_transfer.png").exists() and (out.parent / "fig_trajectory.png").exists()
    assert (out.parent / "SAMPLES.md").exists()
    print("\nSMOKE ORGANIC OK ->", out)


if __name__ == "__main__":
    main(sys.argv[1])
