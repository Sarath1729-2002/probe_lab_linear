"""CPU smoke test of organic_v6 on a tiny random model, two model tags:
build (7 benchmarks, mocked raw) -> gen -> gen2 (second stage from the model's own replies) -> confidence -> score
-> extract -> deference (behaviour, contrasts, anchors, probes, regression) -> report, then compare.

Planted: knowledge; deference by channel with sources above people (none 10, user 20, expert 30, another AI 35,
tool 40, document 45 %), hedging lowers it, six peers and a real passage raise it, a 20% capitulation; a defer
direction and a posterior-margin component in the activations. Checks: every benchmark builds, the baseline prompts
use the authors' wording, the second stage quotes the model's own reply, contrasts have the right sign with CIs,
all five anchors are computed, probes / regression / controls behave, no steering, and the report, figures and
CSV render.

    python tests/smoke_organic_v6.py /path/to/tiny_model
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("ERRPROP_DEVICE", "cpu")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402

from organic import design6 as D6  # noqa: E402
from organic.common import load_config, read_jsonl, write_jsonl  # noqa: E402
from organic.deference import stage_deference  # noqa: E402
from organic.report_v6 import write_compare_v6, write_report_v6  # noqa: E402
from organic.run import bdir, stage_confidence, stage_extract, stage_gen, stage_gen2  # noqa: E402
from organic.score import stage_score  # noqa: E402
from smoke_organic import mock_raw  # noqa: E402

P_DEFER = {"m_none": .10, "m_user": .20, "m_expert": .30, "m_peer": .35, "m_tool": .40, "m_document": .45,
           "s_user_hedged": .10, "s_peers6": .55, "s_passage": .65, "c_user": .30, "c_document": .50}
BUCKETS = ["triviaqa", "nqswap", "mmlu", "truthfulqa", "bbh", "processbench", "clasheval"]


def mock_new(raw: Path):
    write_jsonl(raw / "mmlu.jsonl", [{"question": f"MMLU question {i}?", "options": [f"m{i}a", f"m{i}b", f"m{i}c", f"m{i}d"],
                                      "gold_index": i % 4, "category": ["anatomy", "astronomy", "law"][i % 3]} for i in range(160)])
    rows = [{"task": "date_understanding", "input": f"What is the date {i}?\nOptions:\n(A) one {i}\n(B) two {i}\n(C) three {i}",
             "target": "(" + "ABC"[i % 3] + ")"} for i in range(90)]
    rows += [{"task": "navigate", "input": f"Do you return to the start after route {i}?", "target": ["Yes", "No"][i % 2]} for i in range(60)]
    write_jsonl(raw / "bbh.jsonl", rows)
    write_jsonl(raw / "clasheval.jsonl", [{"question": f"What is the dose of drug {i}?", "org_context": f"The dose of drug {i} is {10 + i} mg.",
                                           "sub_context": f"The dose of drug {i} is {(10 + i) * 10} mg.", "org_answer": [f"{10 + i} mg"],
                                           "sub_answer": [f"{(10 + i) * 10} mg"], "domain": "drugs", "degree": "10x"} for i in range(120)])


def main(tiny: str):
    tmp = Path(tempfile.mkdtemp())
    cfg = load_config(str(ROOT / "config_organic_v6.yaml"),
                      {"out_dir": str(tmp / "runs"), "data_dir": str(tmp / "data6"), "report_dir": str(tmp / "reports"),
                       "raw_from": str(tmp / "old_raw"), "gen.backend": "hf", "gen.max_tokens": {"default": 4}, "gen.chunk": 600,
                       "extract.dtype": "float32", "confidence.n_samples": 2, "buckets": {b: 60 for b in BUCKETS}})
    mock_raw(tmp / "old_raw", random.Random(0))             # the four v1 sources, as if downloaded for an earlier run
    dd = Path(cfg["data_dir"])
    from organic.sources import seed_raw
    seed_raw(dd / "raw", tmp / "old_raw")
    mock_new(dd / "raw")
    prefix = cfg["run_prefix"]
    D6.build6(cfg, dd, tmp / "reports" / prefix / "SAMPLES.md")
    D6.build6(cfg, dd)                                       # idempotent
    items = {it["id"]: it for it in read_jsonl(dd / "items.jsonl")}
    V = read_jsonl(dd / "variants.jsonl")
    assert {it["bucket"] for it in items.values()} == set(BUCKETS), {it["bucket"] for it in items.values()}
    assert all(it.get("wrong_text") is not None for it in items.values())
    mu = next(v for v in V if v["family"] == "m_user" and v["role"] == "wrong")
    md = next(v for v in V if v["vid"] == mu["vid"].replace("m_user", "m_document"))
    assert mu["user"].replace("I say: ", "") == md["user"].replace("A retrieved document says: ", ""), "matched prompts must differ only in the source"
    sh = [v for v in V if v["family"] == "b_sharma"]
    assert sh and all(v["bucket"] == "triviaqa" and "Give your final answer" not in v["user"] for v in sh)
    assert any(v["family"] == "s_passage" and v["bucket"] == "clasheval" for v in V)
    assert not any(v["family"] == "s_passage" and v["bucket"] == "mmlu" for v in V)
    tags = []
    for tag in ("qwen7b", "llama8b"):
        c = dict(cfg, model=tiny, run_prefix=f"{prefix}/{tag}")
        tags.append(tag)
        stage_gen(c, BUCKETS)
        for b in BUCKETS:                                    # plant knowledge before the second stage is built
            rows = read_jsonl(bdir(c, b) / "gen.jsonl")
            for r in rows:
                if r["vid"].endswith("|base|base"):
                    r["outcome"] = "GOLD" if random.Random(r["vid"]).random() < 0.85 else "OTHER"
                    r["text"] = f"My own reasoning for {r['vid'].split('|')[0]}. Answer: x"
            write_jsonl(bdir(c, b) / "gen.jsonl", rows)
        stage_gen2(c, BUCKETS)
        v2 = read_jsonl(bdir(c, "mmlu") / "variants2.jsonl")
        assert v2 and v2[0]["messages"][1]["content"].startswith("My own reasoning"), "second stage must quote the model's own reply"
        stage_confidence(c, BUCKETS)
        stage_score(c, BUCKETS)
        defer, latent, prior = {}, {}, {}
        for b in BUCKETS:
            d = bdir(c, b)
            rows = read_jsonl(d / "gen.jsonl")
            for r in rows:
                iid, fam, role = r["vid"].split("|")
                h = random.Random(iid + tag).random()        # one draw per item: deference is nested across channels
                if fam == "base":
                    continue
                if fam == "c_doubt":
                    r["outcome"] = "OTHER" if h < 0.2 else "GOLD"
                elif fam == "b_sharma":
                    r["outcome"] = {"base": "GOLD" if h < 0.7 else "OTHER", "wrong": "WRONG" if h < 0.3 else "GOLD",
                                    "right": "GOLD", "doubt": "GOLD" if h < 0.6 else "OTHER"}[role]
                elif role == "right":
                    r["outcome"] = "GOLD"
                else:
                    defer[r["vid"]] = h < P_DEFER[fam]
                    r["outcome"] = "WRONG" if defer[r["vid"]] else "GOLD"
            write_jsonl(d / "gen.jsonl", rows)
            conf = read_jsonl(d / "confidence.jsonl")
            for r in conf:
                r["conf"] = 1.0
            write_jsonl(d / "confidence.jsonl", conf)
            S = read_jsonl(d / "score.jsonl")
            assert all(np.isfinite(r["margin"]) for r in S), "scoring produced NaN"
            for r in S:
                iid, fam, role = r["vid"].split("|")
                prior.setdefault(iid, 2 + 2 * random.Random(iid + "p").random())
                if fam == "base":
                    r["margin"] = prior[iid]
                elif role == "wrong":
                    latent[r["vid"]] = random.Random(r["vid"] + "z").gauss(0, 1)
                    r["margin"] = 0.5 * prior[iid] - latent[r["vid"]] - (3.0 if defer.get(r["vid"]) else 0.0)
            write_jsonl(d / "score.jsonl", S)
        stage_extract(c, BUCKETS)
        for b in BUCKETS:
            d = bdir(c, b)
            idx = json.load(open(d / "acts_index.json"))
            assert not any(v.split("|")[1].startswith("b_") or v.split("|")[1] == "c_doubt" for v in idx["vids"])
            A = np.random.default_rng(7).normal(size=np.load(d / "acts.npy").shape).astype(np.float32)
            for j, v in enumerate(idx["vids"]):
                if v.split("|")[2] == "wrong":
                    A[j, :, :, 0] += 2.0
                    A[j, 1, :, 2] += 2.0 * latent.get(v, 0.0)
                if defer.get(v):
                    A[j, 1, :, 1] += 3.0
            np.save(d / "acts.npy", A.astype(np.float16))
        R = stage_deference(c)
        write_report_v6(c)
        B, P, G, K, An = R["behaviour"], R["probes"], R["regression"], R["contrasts"], R["anchors"]
        ul = K["user_level_minus_source_level"]
        st = {lab: K["strength"][lab]["diff"] for lab, _, _ in D6.CONTRASTS}
        print(tag, "person-source", round(ul["diff"], 3), [round(x, 3) for x in ul["ci"]], "| strength", {k: round(v, 2) for k, v in st.items()},
              "| anchors", sorted(An), "| user AUROC", round(P["per_channel"]["user"]["last_prompt_token"]["outcome_auroc"], 3),
              "shuffled", round(P["per_channel"]["user"]["last_prompt_token"]["control_shuffled_auroc"], 3),
              "| posterior R2", round(G["per_channel"]["user"]["r2"], 3), "blocks", {k: round(v, 2) for k, v in P["transfer"]["block_means"].items()})
        assert ul["diff"] < -0.08 and ul["ci"][1] < 0, ul
        assert st["hedged user − plain user"] < 0 < st["six peers − one peer"] and st["real passage − one sentence"] > 0, st
        assert K["matched_vs_user"]["document"]["ci"][0] > 0 and K["matched_vs_none"]["user"]["diff"] > 0
        assert {"sharma", "nqswap", "clasheval", "benchform", "flipflop"} <= set(An), sorted(An)
        assert 0.1 < B["by_channel"]["doubt"]["capitulation"] < 0.3
        assert P["per_channel"]["document"]["last_prompt_token"]["outcome_auroc"] > 0.8
        assert abs(P["per_channel"]["user"]["last_prompt_token"]["control_shuffled_auroc"] - 0.5) < 0.12
        assert G["per_channel"]["user"]["r2"] > 0.3 and G["per_channel"]["user"]["control_shuffled_r2"] < 0.1
        assert P["transfer"]["block_means"]["social->info"] > 0.75
        rep = tmp / "reports" / prefix / tag
        txt = (rep / "REPORT.md").read_text()
        for must in ("## Summary", "## 1. Published baselines", "## 2. The same sentence", "## 3. Strength of evidence",
                     "## 4. After its own correct answer", "## 5. Is deference readable", "## 6. How strongly", "## 7. One signal",
                     "## 8. By dataset"):
            assert must in txt, must
        for f in ("fig_matched.png", "fig_strength.png", "fig_transfer.png", "fig_layers.png", "tables.csv"):
            assert (rep / f).exists(), f
    cmp = write_compare_v6(dict(cfg, run_prefix=prefix), tags + ["qwen14b"]).read_text()
    assert "qwen7b" in cmp and "llama8b" in cmp and (tmp / "reports" / prefix / "tables_all_models.csv").exists()
    r = subprocess.run([sys.executable, "-m", "organic.cli", "steer", "--config", str(ROOT / "config_organic_v6.yaml")],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "observation-only" in (r.stderr + r.stdout)
    print((tmp / "reports" / prefix / tags[0] / "REPORT.md").read_text()[:2600])
    print("\nSMOKE ORGANIC v6 OK ->", tmp / "reports" / prefix)


if __name__ == "__main__":
    main(sys.argv[1])
