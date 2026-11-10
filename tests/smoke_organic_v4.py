"""CPU smoke test of organic_v4 (deference against knowledge) on a tiny random model, two model tags:
variants -> gen -> confidence -> extract -> fit -> deference -> steer -> report, then compare.

Planted: knowledge; deference 40% in social channels vs 15% in informational ones; a content-free doubt
capitulation of 30%; and one shared activation direction for 'about to defer' at the last prompt token.
The checks assert that behaviour, the social-minus-informational difference, per-channel probes, transfer
across channels on disjoint items, multi-channel steering and the comparison all come out right.

    python tests/smoke_organic_v4.py /path/to/tiny_model
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
from organic.channels import FAMILIES, INFO, SOCIAL  # noqa: E402
from organic.common import load_config, read_jsonl, write_jsonl  # noqa: E402
from organic.deference import stage_deference  # noqa: E402
from organic.fit import stage_fit  # noqa: E402
from organic.report_v4 import write_compare_v4, write_report_v4  # noqa: E402
from organic.run import bdir, stage_confidence, stage_extract, stage_gen  # noqa: E402
from organic.steer import stage_steer_multi  # noqa: E402
from smoke_organic import mock_raw  # noqa: E402

BUCKETS = ["triviaqa", "mmlu_pro", "cruxeval"]


def main(tiny: str):
    tmp = Path(tempfile.mkdtemp())
    c1 = load_config(str(ROOT / "config_organic.yaml"),
                     {"out_dir": str(tmp / "runs"), "data_dir": str(tmp / "data1"), "report_dir": str(tmp / "reports"),
                      "buckets": {"triviaqa": 150, "nqswap": 0, "truthfulqa": 0, "mmlu_pro": 100, "processbench": 0,
                                  "cute": 0, "cruxeval": 120}})
    mock_raw(tmp / "data1" / "raw", random.Random(0))
    build(c1, tmp / "data1")
    cfg = load_config(str(ROOT / "config_organic_v4.yaml"),
                      {"out_dir": str(tmp / "runs"), "data_dir": str(tmp / "data4"), "report_dir": str(tmp / "reports"),
                       "items_from": str(tmp / "data1" / "items.jsonl"), "gen.backend": "hf",
                       "gen.max_tokens": {"default": 4}, "gen.chunk": 500, "extract.dtype": "float32",
                       "confidence.n_samples": 2, "steer.n_wrong": 6, "steer.n_right": 2, "steer.batch": 8,
                       "steer.max_tokens": 4, "steer.alphas": [0, 1], "buckets": {b: 80 for b in BUCKETS}})
    prefix = cfg["run_prefix"]
    V = rebuild_variants(Path(cfg["data_dir"]), tmp / "reports" / prefix / "SAMPLES.md", 0, cfg["families"],
                         Path(cfg["items_from"]), cfg["buckets"])
    fams = {v["family"] for v in V}
    assert set(FAMILIES) <= fams, set(FAMILIES) - fams
    items = {it["id"]: it for it in read_jsonl(Path(cfg["data_dir"]) / "items.jsonl")}
    assert all(it.get("wrong_text") is not None for it in items.values())
    pb = next(v for v in V if v["family"] == "ch_pushback" and v["role"] == "wrong")
    assert pb["messages"][1]["content"].startswith("Answer: ") and "I believe the answer is" in pb["messages"][2]["content"]
    tags = []
    for tag in ("qwen7b", "llama8b"):
        c = dict(cfg, model=tiny, run_prefix=f"{prefix}/{tag}")
        tags.append(tag)
        stage_gen(c, BUCKETS)
        stage_confidence(c, BUCKETS)
        defer = {}
        for b in BUCKETS:
            d = bdir(c, b)
            rows = read_jsonl(d / "gen.jsonl")
            for r in rows:
                iid, fam, role = r["vid"].split("|")
                h = random.Random(iid + fam + tag).random()
                if fam == "base":
                    r["outcome"] = "GOLD" if random.Random(iid).random() < 0.85 else "OTHER"
                elif fam == "doubt_pushback":
                    r["outcome"] = "OTHER" if h < 0.3 else "GOLD"
                elif role == "right":
                    r["outcome"] = "GOLD"
                else:
                    p = 0.40 if fam[3:] in SOCIAL else 0.15
                    defer[r["vid"]] = h < p
                    r["outcome"] = "WRONG" if h < p else "GOLD"
            write_jsonl(d / "gen.jsonl", rows)
            conf = read_jsonl(d / "confidence.jsonl")
            for r in conf:
                r["conf"] = 1.0
            write_jsonl(d / "confidence.jsonl", conf)
        stage_extract(c, BUCKETS)
        for b in BUCKETS:
            d = bdir(c, b)
            idx = json.load(open(d / "acts_index.json"))
            A = np.random.default_rng(3).normal(size=np.load(d / "acts.npy").shape).astype(np.float32)
            for j, v in enumerate(idx["vids"]):
                if v.split("|")[2] == "wrong":
                    A[j, :, :, 0] += 2.0                        # content is wrong
                if defer.get(v):
                    A[j, 1, :, 1] += 3.0                        # about to defer: one shared direction
            np.save(d / "acts.npy", A.astype(np.float16))
        stage_fit(c)
        R = stage_deference(c)
        stage_steer_multi(c)
        write_report_v4(c)
        B, P = R["behaviour"], R["probes"]
        soc = np.mean([B["by_channel"][x]["defer"] for x in SOCIAL])
        inf = np.mean([B["by_channel"][x]["defer"] for x in INFO])
        print(tag, f"social {soc:.2f} info {inf:.2f} diff {B['social_minus_info']['mean']:.2f} {B['social_minus_info']['ci']}",
              "capit", round(B["by_channel"]["doubt_pushback"]["capitulation"], 2),
              "outcome@last", {x: round(P["per_channel"][x]["last_prompt_token"]["outcome_auroc"], 2) for x in ("user_soft", "tool")},
              "blocks", {k: round(v, 2) for k, v in P["transfer"]["block_means"].items()})
        assert soc > inf + 0.15 and B["social_minus_info"]["ci"][0] > 0
        assert 0.2 < B["by_channel"]["doubt_pushback"]["capitulation"] < 0.4
        assert all(P["per_channel"][x]["last_prompt_token"]["outcome_auroc"] > 0.8 for x in ("user_soft", "user_confident"))
        assert P["transfer"]["block_means"]["social->info"] > 0.75, P["transfer"]["block_means"]
        S = json.load(open(Path(c["out_dir"]) / c["run_prefix"] / "steer_multi.json"))
        assert any(x["channel"] == "ALL" for x in S["conditions"])
        txt = (tmp / "reports" / prefix / tag / "REPORT.md").read_text()
        for must in ("## How often it defers", "Capitulation", "Social minus informational", "## Can deference be read",
                     "## One signal across channels?", "## Causal test"):
            assert must in txt, must
    cmp = write_compare_v4(dict(cfg, run_prefix=prefix), tags).read_text()
    assert "qwen7b" in cmp and "llama8b" in cmp and "social − informational" in cmp
    print(cmp[:1200])
    print("\nSMOKE ORGANIC v4 OK ->", tmp / "reports" / prefix)


if __name__ == "__main__":
    main(sys.argv[1])
