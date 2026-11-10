"""CPU smoke test of organic_v3 on a tiny random model, two model tags:
variants -> gen -> confidence -> extract -> fit -> traj -> steer -> report, then compare.

Planted: knowledge, follow/recover outcomes per frame (user/plain follows most, critic least), critic
verdict lines, confidence, and an activation signal for following at the last prompt token. The checks
assert that the frame metrics, critic scoring, outcome probe, frame transfer, steering and comparison
all come out and read the planted structure back.

    python tests/smoke_organic_v3.py /path/to/tiny_model
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
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402

from organic.build import build, rebuild_variants  # noqa: E402
from organic.common import load_config, read_jsonl, write_jsonl  # noqa: E402
from organic.fit import stage_fit  # noqa: E402
from organic.frames import FRAMES, critic_correct, critic_step, first_error_label  # noqa: E402
from organic.report_v3 import write_compare, write_report_v3  # noqa: E402
from organic.run import bdir, stage_confidence, stage_extract, stage_gen  # noqa: E402
from organic.steer import stage_steer  # noqa: E402
from organic.traj import stage_traj  # noqa: E402
from smoke_organic import mock_raw  # noqa: E402

FOLLOW = {"plain": 0.75, "check": 0.5, "critic": 0.3}


def main(tiny: str):
    tmp = Path(tempfile.mkdtemp())
    # ---- v1-style frozen items (only ProcessBench is used by v3)
    c1 = load_config(str(ROOT / "config_organic.yaml"),
                     {"out_dir": str(tmp / "runs"), "data_dir": str(tmp / "data1"), "report_dir": str(tmp / "reports"),
                      "buckets": {"triviaqa": 0, "nqswap": 0, "truthfulqa": 0, "mmlu_pro": 0, "processbench": 160,
                                  "cute": 0, "cruxeval": 0}})
    mock_raw(tmp / "data1" / "raw", random.Random(0))
    build(c1, tmp / "data1")
    # ---- v3
    cfg = load_config(str(ROOT / "config_organic_v3.yaml"),
                      {"out_dir": str(tmp / "runs"), "data_dir": str(tmp / "data3"), "report_dir": str(tmp / "reports"),
                       "items_from": str(tmp / "data1" / "items.jsonl"), "gen.backend": "hf",
                       "gen.max_tokens": {"default": 4}, "extract.dtype": "float32", "probes.n_boot": 30,
                       "traj.n_items": 40, "steer.n_wrong": 12, "steer.n_right": 6, "steer.batch": 6,
                       "steer.max_tokens": 4, "steer.alphas": [0, 1, -1], "confidence.n_samples": 2})
    prefix = cfg["run_prefix"]
    V = rebuild_variants(Path(cfg["data_dir"]), tmp / "reports" / prefix / "SAMPLES.md", 0, cfg["families"],
                         Path(cfg["items_from"]), ["processbench"])
    fams = {v["family"] for v in V}
    assert set(FRAMES) <= fams and "sugg" in fams and "base" in fams, fams
    selfv = next(v for v in V if v["family"] == "frame_self_check")
    assert selfv["messages"][1]["role"] == "assistant" and "Step 1:" in selfv["messages"][1]["content"]
    rebuild_variants(Path(cfg["data_dir"]), None, 0, cfg["families"], None, None)      # idempotent
    items = {it["id"]: it for it in read_jsonl(Path(cfg["data_dir"]) / "items.jsonl")}
    # critic parsing
    itw = next(it for it in items.values() if it["role"] == "wrong")
    itr = next(it for it in items.values() if it["role"] == "right")
    k = first_error_label(itw) + 1
    assert critic_correct(itw, f"Let me look.\nFirst error: step {k}\nAnswer: 3") is True
    assert critic_correct(itw, "**First error:** none\nAnswer: 3") is False
    assert critic_correct(itr, "First error: none") is True and critic_step("no verdict") is None

    tags = []
    for tag in ("qwen7b", "llama8b"):
        c = dict(cfg, model=tiny, run_prefix=f"{prefix}/{tag}")
        tags.append(tag)
        stage_gen(c, ["processbench"])
        stage_confidence(c, ["processbench"])
        b = bdir(c, "processbench")
        rows = read_jsonl(b / "gen.jsonl")
        fol = {}
        for r in rows:                                   # plant outcomes and critic lines
            iid, fam, role = r["vid"].split("|")
            it = items[iid]
            h = random.Random(iid + tag)
            if fam == "base":
                r["outcome"] = "GOLD" if h.random() < 0.85 else "OTHER"
                continue
            if role == "right":
                r["outcome"] = "GOLD"
                r["text"] = "First error: none\nAnswer: x" if fam.endswith("critic") else "Answer: x"
                continue
            ask = fam.split("_")[-1] if fam.startswith("frame_") else "plain"
            p = FOLLOW[ask] if fam.startswith("frame_") else 0.05
            f = random.Random(iid + fam + tag).random() < p
            fol[r["vid"]] = f
            r["outcome"] = "WRONG" if f else "GOLD"
            if tag == "llama8b" and fam == "frame_user_check" and not f:
                r["outcome"] = "OTHER"                   # Mistral-like: never recovers in the steering frame
            if fam.endswith("critic"):
                r["text"] = ("First error: none\nYour working is correct.\nAnswer: x" if f
                             else f"First error: step {first_error_label(it) + 1}\nAnswer: y")
            elif fam.endswith("check") and f:
                r["text"] = "Your working is correct.\nAnswer: x"
        write_jsonl(b / "gen.jsonl", rows)
        conf = read_jsonl(b / "confidence.jsonl")
        for r in conf:
            r["conf"] = 1.0
        write_jsonl(b / "confidence.jsonl", conf)
        stage_extract(c, ["processbench"])
        idx = json.load(open(b / "acts_index.json"))
        A = np.random.default_rng(1).normal(size=np.load(b / "acts.npy").shape).astype(np.float32)
        for j, v in enumerate(idx["vids"]):              # planted: wrong solution readable; following readable
            if v.split("|")[2] == "wrong":
                A[j, :, :, 0] += 2.0
                if fol.get(v):
                    A[j, 1, :, 1] += 3.0
        np.save(b / "acts.npy", A.astype(np.float16))
        R = stage_fit(c)
        stage_traj(c, ["processbench"])
        stage_steer(c)
        write_report_v3(c)
        P = R["buckets"]["processbench"]
        fr = P["frames"]
        print(tag, {f: round(fr[f]["follow"], 2) for f in ("frame_user_plain", "frame_user_check", "frame_user_critic")},
              "critic acc", round(fr["frame_user_critic"]["critic_acc_wrong"], 2),
              "outcome@last", (P["probes"]["frame_user_check"]["last_prompt_token"].get("outcome_probe") or {}).get("auroc_oof"))
        assert fr["frame_user_plain"]["follow"] > fr["frame_user_critic"]["follow"]
        assert fr["frame_user_critic"]["critic_acc_wrong"] > 0.5 and fr["frame_user_critic"]["critic_acc_right"] == 1.0
        assert fr["frame_user_check"]["certify_when_following"] == 1.0
        if tag == "qwen7b":
            assert P["probes"]["frame_user_check"]["last_prompt_token"]["outcome_probe"]["auroc_oof"] > 0.8
        else:                                            # no recoveries -> no follow-vs-recover probe, handled
            assert P["probes"]["frame_user_plain"]["last_prompt_token"]["outcome_probe"]["auroc_oof"] > 0.8
        assert "matrix_train_row_test_col" in P["frame_transfer"]
        S = json.load(open(b / "steer.json"))
        assert len(S["conditions"]) == 4 and S["n_test_wrong"] > 0, S
        assert S["contrast"] == ("not follow (recover or other) - follow" if tag == "llama8b" else "recover - follow"), S["contrast"]
        txt = (tmp / "reports" / prefix / tag / "REPORT.md").read_text()
        for must in ("## How often it inherits the error", "## Can it find the error", "## What the residual stream",
                     "## Is 'about to inherit", "## From the end of the solution", "## Causal test"):
            assert must in txt, must
    cmp = write_compare(dict(cfg, run_prefix=prefix), tags)
    t = cmp.read_text()
    assert "qwen7b" in t and "llama8b" in t and "## Steering" in t
    print(t[:1500])
    print("\nSMOKE ORGANIC v3 OK ->", tmp / "reports" / prefix)


if __name__ == "__main__":
    main(sys.argv[1])
