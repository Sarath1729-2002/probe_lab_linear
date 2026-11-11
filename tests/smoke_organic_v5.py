"""CPU smoke test of organic_v5 (option B) on a tiny random model, two model tags:
variants -> gen -> confidence -> extract -> score -> deference (behaviour, probes, regression, anchors) -> report,
then compare.

Planted: knowledge; deference 35% for the user channels vs 15% for the source channels; a 30% capitulation; a
continuous shift = 0.5 * prior + a component written into the activations (so the probe must explain more than prior
strength alone); and a defer direction. Checks: the published anchors are computed, Level 1 > Level 2 with a CI
above 0, the regression probe beats the prior and fails on shuffled targets, the binary control task is near 0.5,
transfer exists, no steering stage, and the reports render.

    python tests/smoke_organic_v5.py /path/to/tiny_model
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

from organic import protocols as PR  # noqa: E402
from organic.build import build, rebuild_variants  # noqa: E402
from organic.common import load_config, read_jsonl, write_jsonl  # noqa: E402
from organic.deference import stage_deference  # noqa: E402
from organic.report_v5 import write_compare_v5, write_report_v5  # noqa: E402
from organic.run import bdir, stage_confidence, stage_extract, stage_gen  # noqa: E402
from organic.score import stage_score  # noqa: E402
from smoke_organic import mock_raw  # noqa: E402

BUCKETS = ["triviaqa", "nqswap", "mmlu_pro", "processbench"]


def main(tiny: str):
    tmp = Path(tempfile.mkdtemp())
    c1 = load_config(str(ROOT / "config_organic.yaml"),
                     {"out_dir": str(tmp / "runs"), "data_dir": str(tmp / "data1"), "report_dir": str(tmp / "reports"),
                      "buckets": {"triviaqa": 150, "nqswap": 150, "truthfulqa": 0, "mmlu_pro": 100, "processbench": 80,
                                  "cute": 0, "cruxeval": 0}})
    mock_raw(tmp / "data1" / "raw", random.Random(0))
    build(c1, tmp / "data1")
    cfg = load_config(str(ROOT / "config_organic_v5.yaml"),
                      {"out_dir": str(tmp / "runs"), "data_dir": str(tmp / "data5"), "report_dir": str(tmp / "reports"),
                       "items_from": str(tmp / "data1" / "items.jsonl"), "gen.backend": "hf",
                       "gen.max_tokens": {"default": 4}, "gen.chunk": 400, "extract.dtype": "float32",
                       "confidence.n_samples": 2, "buckets": {b: 70 for b in BUCKETS}})
    prefix = cfg["run_prefix"]
    V = rebuild_variants(Path(cfg["data_dir"]), tmp / "reports" / prefix / "SAMPLES.md", 0, cfg["families"],
                         Path(cfg["items_from"]), cfg["buckets"])
    assert {v["family"] for v in V} == set(PR.FAMILIES + ["base"]), {v["family"] for v in V}
    items = {it["id"]: it for it in read_jsonl(Path(cfg["data_dir"]) / "items.jsonl")}
    nq = next(v for v in V if v["family"] == "p_document" and v["bucket"] == "nqswap" and v["role"] == "wrong")
    assert items[nq["item"]]["doc_wrong"].strip() in nq["user"], "NQ-Swap must use the real substituted passage"
    tags = []
    for tag in ("qwen7b", "llama8b"):
        c = dict(cfg, model=tiny, run_prefix=f"{prefix}/{tag}")
        tags.append(tag)
        stage_gen(c, BUCKETS)
        stage_confidence(c, BUCKETS)
        stage_score(c, BUCKETS)                           # real teacher-forced scoring path
        sc = read_jsonl(bdir(c, "triviaqa") / "score.jsonl")
        assert sc and all(np.isfinite(r["margin"]) for r in sc), "scoring produced NaN"
        defer, latent, prior = {}, {}, {}
        for b in BUCKETS:
            d = bdir(c, b)
            rows = read_jsonl(d / "gen.jsonl")
            for r in rows:
                iid, fam, role = r["vid"].split("|")
                h = random.Random(iid + fam + tag).random()
                if fam == "base":
                    r["outcome"] = "GOLD" if random.Random(iid).random() < 0.85 else "OTHER"
                elif fam == PR.DOUBT:
                    r["outcome"] = "OTHER" if h < 0.3 else "GOLD"
                elif role == "right":
                    r["outcome"] = "GOLD"
                else:
                    p = 0.35 if fam[2:] in PR.SOCIAL else 0.15
                    defer[r["vid"]] = h < p
                    r["outcome"] = "WRONG" if h < p else "GOLD"
            write_jsonl(d / "gen.jsonl", rows)
            conf = read_jsonl(d / "confidence.jsonl")
            for r in conf:
                r["conf"] = 1.0
            write_jsonl(d / "confidence.jsonl", conf)
            S = read_jsonl(d / "score.jsonl")                # plant: shift = 0.5 * prior + latent (latent is in the states)
            for r in S:
                iid, fam, role = r["vid"].split("|")
                prior.setdefault(iid, 2 + 2 * random.Random(iid + "p").random())
                if fam == "base":
                    r["margin"] = prior[iid]
                elif role == "wrong":
                    latent[r["vid"]] = random.Random(r["vid"] + "z").gauss(0, 1)
                    r["margin"] = prior[iid] - (0.5 * prior[iid] + latent[r["vid"]])
            write_jsonl(d / "score.jsonl", S)
        stage_extract(c, BUCKETS)
        for b in BUCKETS:
            d = bdir(c, b)
            idx = json.load(open(d / "acts_index.json"))
            A = np.random.default_rng(5).normal(size=np.load(d / "acts.npy").shape).astype(np.float32)
            for j, v in enumerate(idx["vids"]):
                if v.split("|")[2] == "wrong":
                    A[j, :, :, 0] += 2.0
                    A[j, 1, :, 2] += 2.0 * latent.get(v, 0.0)
                if defer.get(v):
                    A[j, 1, :, 1] += 3.0
            np.save(d / "acts.npy", A.astype(np.float16))
        R = stage_deference(c)
        write_report_v5(c)
        B, P, G, An = R["behaviour"], R["probes"], R["regression"], R["anchors"]
        rs = G["per_channel"]["suggest"]
        print(tag, "L1-L2", round(B["social_minus_info"]["mean"], 3), B["social_minus_info"]["ci"],
              "| suggest posterior R2", round(rs["r2"], 3), "prior-only", round(rs["prior_only_r2"], 3), "beyond", round(rs["partial_r2_beyond_prior"], 3),
              "shuffled", round(rs["control_shuffled_r2"], 3), "| binary shuffled",
              round(P["per_channel"]["suggest"]["last_prompt_token"]["control_shuffled_auroc"], 3),
              "| anchors", {k: v for k, v in An.items() if k != "published_reference"})
        assert B["social_minus_info"]["mean"] > 0.1 and B["social_minus_info"]["ci"][0] > 0
        assert rs["r2"] > rs["prior_only_r2"] and rs["partial_r2_beyond_prior"] > 0.3 and rs["control_shuffled_r2"] < 0.1
        assert abs(P["per_channel"]["suggest"]["last_prompt_token"]["control_shuffled_auroc"] - 0.5) < 0.12
        assert {"sharma_triviaqa", "longpre_nqswap", "benchform_conformity", "flipflop"} <= set(An)
        assert "block_means" in G["transfer"] and "block_means" in P["transfer"]
        txt = (tmp / "reports" / prefix / tag / "REPORT.md").read_text()
        for must in ("## 1. Does our setup reproduce", "## 2. How often it defers", "## 3. How far the content moves",
                     "## 4. Can deference be read", "## 5. One signal"):
            assert must in txt, must
    cmp = write_compare_v5(dict(cfg, run_prefix=prefix), tags).read_text()
    assert "qwen7b" in cmp and "llama8b" in cmp and "Replication anchors" in cmp
    import subprocess
    r = subprocess.run([sys.executable, "-m", "organic.cli", "steer", "--config", str(ROOT / "config_organic_v5.yaml")],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "observation-only" in (r.stderr + r.stdout), "v5 must refuse steering"
    print(cmp[:1500])
    print("\nSMOKE ORGANIC v5 OK ->", tmp / "reports" / prefix)


if __name__ == "__main__":
    main(sys.argv[1])
