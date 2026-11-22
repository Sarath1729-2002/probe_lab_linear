"""CPU smoke test of organic/knowsay.py on top of the sel smoke run (tiny random model; replies re-labelled at random).

    ERRPROP_DEVICE=cpu python tests/smoke_knowsay.py /path/to/tiny_model
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
os.environ.setdefault("ERRPROP_DEVICE", "cpu")

import numpy as np  # noqa: E402

import smoke_sel  # noqa: E402
from organic import knowsay as K  # noqa: E402
from organic import selective as S  # noqa: E402
from organic.common import read_jsonl, write_jsonl  # noqa: E402


def main(tiny):
    cfg = smoke_sel.main(tiny)
    cfg = dict(cfg, knowsay={"n_train_items": 40, "n_steer_known": 6, "n_steer_unknown": 4, "min_n": 1})
    for tag in ("m1", "m2"):
        c = dict(cfg, model=tiny, run_prefix=f"{cfg['run_prefix']}/{tag}")
        items, B = S.load_items(c), S._base(c)
        it = next(v for v in items.values() if v["set"] == "test")
        pool = [v for v in items.values() if v["set"] == "train"]
        b_ok = {"correct": True, "ans": S.L5[it["gold"]], "text": "(A)"}
        fr = K.family_rows(it, b_ok, pool, 2)
        assert sorted({r["fam"] for r in fr}) == sorted(K.FAMS), {r["fam"] for r in fr}
        assert all(r["claim"] != S.L5[it["gold"]] for r in fr if r["claim"])
        fw = K.family_rows(it, {"correct": False, "ans": "Z", "text": "x"}, pool, 2)
        assert {r["fam"] for r in fw} == {"sugg", "auth", "push", "doc", "incentive"} and all(r["role"] == "right" for r in fw)
        assert "Answer: (" in next(r for r in fr if r["fam"] == "fewshot")["msgs"][0]["content"]
        K.stage_gen(c)
        rng = np.random.default_rng(5)
        G = read_jsonl(K._p(c, "ks_gen.jsonl"))
        for r in G:
            r["correct"] = bool(rng.random() < 0.5)
            r["ans"] = S.L5[items[r["item"]]["gold"]] if r["correct"] else "A"
        write_jsonl(K._p(c, "ks_gen.jsonl"), G)
        for r in G:
            assert K.label(r, B) in ("honest", "dishonest", None)
        K.stage_follow(c)
        Fo = read_jsonl(K._p(c, "ks_follow.jsonl"))
        assert {"own", "self", "gold", "dist"} <= {r["key"] for r in Fo} and any(r["key"].startswith("opt") for r in Fo)
        Fj = K.stage_fit(c)
        assert Fj["det"] and Fj["groups"] and Fj["recovery"] and "pooled" in Fj["mus"]
        St = K.stage_steer(c)
        assert "none" in St["cross_ablation"] and "pooled" in St["cross_ablation"] and set(St["gate"]) == {"stated", "gate_decode", "gate_steer", "always_decode"}
        K.stage_steer(c)                                                   # resumable
        txt = K.write_report(c).read_text()
        for must in ("Pre-registered", "Detecting a dishonest reply", "graded know-say gap", "Recovery", "Routes", "cross" if False else "Causal", "gated"):
            assert must in txt, must
        assert {"K1 belief readout transfers", "K5 recovery", "K6 routes differ"} <= set(json.load(open(K._p(c, "ks_verdict.json"))))
    txt = K.write_compare(cfg, ["m1", "m2", "nope"]).read_text()
    assert "Know vs say across 2 models" in txt and "Recovery on departed" in txt
    print(txt[:1500])
    print("\nSMOKE KNOWSAY OK")


if __name__ == "__main__":
    main(sys.argv[1])
