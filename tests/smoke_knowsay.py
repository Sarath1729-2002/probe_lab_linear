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
    # peer-pressure variant (separate prefix, 17 families)
    cfgp = dict(cfg, knowsay=dict(S.load_config(str(ROOT / "config_sel_v2.yaml"))["knowsay_peer"], n_train_items=40,
                                  n_steer_known=6, n_steer_unknown=4, min_n=1))
    K.configure(cfgp)
    assert K.PFX == "kp" and "floor" in K.FAMS and K.GROUP_F["dose"] == ["peers1", "peers3", "peers6"]
    c = dict(cfgp, model=tiny, run_prefix=f"{cfg['run_prefix']}/m1")
    items, B = S.load_items(c), S._base(c)
    it = next(v for v in items.values() if v["set"] == "test")
    pr = K.peer_rows(it, {"correct": True, "ans": S.L5[it["gold"]], "text": "(A)"})
    fams = {r["fam"] for r in pr}
    assert fams == {f for f in K.PEER_F}, fams ^ set(K.PEER_F)
    ally = next(r for r in pr if r["fam"] == "peers6_ally")["msgs"][0]["content"]
    assert ally.count(f"({S.L5[it['gold']]})") >= 1 and "Participant 6" in ally
    assert next(r for r in pr if r["fam"] == "floor")["msgs"][0]["content"].count("The answer is (") == 6
    assert "peers6_ally" not in {r["fam"] for r in K.peer_rows(it, {"correct": False, "ans": "Z", "text": "x"})}
    K.stage_gen(c)
    G = read_jsonl(K._p(c, "ks_gen.jsonl"))
    assert K._p(c, "ks_gen.jsonl").name == "kp_gen.jsonl" and {r["fam"] for r in G} >= {"floor", "peers6", "lie"}
    rng = np.random.default_rng(7)
    for r in G:
        r["correct"] = bool(rng.random() < 0.5)
        r["ans"] = S.L5[items[r["item"]]["gold"]] if r["correct"] else "A"
    write_jsonl(K._p(c, "ks_gen.jsonl"), G)
    K.stage_follow(c)
    Fj = K.stage_fit(c)
    assert Fj["fams"] == K.FAMS and "peers6" in Fj["departure"]
    K.stage_steer(c)
    txt = K.write_report(c).read_text()
    assert "Behaviour per family" in txt and (S.repdir(c) / "REPORT_knowsay_kp.md").exists()
    V = json.load(open(K._p(c, "ks_verdict.json")))
    assert {"P1 dose", "P2 speaker-free floor", "P4 Asch ally", "P5 normative keeps the belief", "P6 publicness"} <= set(V), set(V)
    txt = K.write_compare(cfgp, ["m1"]).read_text()
    assert "Departure on known items" in txt and "still knows" in txt
    # natural-form variant (8 forms per process, readout phrasings, test-awareness)
    cfgn = dict(cfg, knowsay=dict(S.load_config(str(ROOT / "config_sel_v2.yaml"))["knowsay_nat"], n_train_items=40,
                                  n_steer_known=6, n_steer_unknown=4, min_n=1))
    K.configure(cfgn)
    assert K.PFX == "kn" and K.VARIANTS and K.AWARE and "natural" in K.GROUP_F
    c = dict(cfgn, model=tiny, run_prefix=f"{cfg['run_prefix']}/m1")
    items, B = S.load_items(c), S._base(c)
    it = next(v for v in items.values() if v["set"] == "test")
    allf = K.nat_rows(it, {"correct": True, "ans": S.L5[it["gold"]], "text": "(A)"}, forms=list(range(8)))
    assert len(allf) == 6 * 8 and len({r["pid"] for r in allf}) == 48
    for r in allf:
        txt = r["msgs"][-1]["content"]
        assert "{" not in txt and S.L5[it["gold"]] not in [x for x in txt.split("(")[1:2]] or r["fam"] == "nat_doubt"
    assert len(K.nat_rows(it, {"correct": True, "ans": "A", "text": "x"})) == 6
    K.stage_gen(c)
    G = read_jsonl(K._p(c, "ks_gen.jsonl"))
    assert K._p(c, "ks_gen.jsonl").name == "kn_gen.jsonl" and {r["fam"] for r in G} >= {"nat_user", "nat_doubt", "sugg"}
    rng = np.random.default_rng(9)
    for r in G:
        r["correct"] = bool(rng.random() < 0.5)
        r["ans"] = S.L5[items[r["item"]]["gold"]] if r["correct"] else "A"
    write_jsonl(K._p(c, "ks_gen.jsonl"), G)
    K.stage_follow(c)
    Fo = read_jsonl(K._p(c, "ks_follow.jsonl"))
    assert {"own_b", "own_c", "test", "gold_b", "dist_c"} <= {r["key"] for r in Fo}
    Fj = K.stage_fit(c)
    assert Fj["variants"] and Fj["awareness"] and any(k.startswith("nat_user|") for k in Fj["forms"])
    K.stage_steer(c)
    txt = K.write_report(c).read_text()
    assert "Natural forms" in txt and "Perceived test-ness" in txt and "Readout phrasing" in txt
    V = json.load(open(K._p(c, "ks_verdict.json")))
    assert {"R2 not a test artefact", "R3 readout phrasing"} <= set(V), set(V)
    txt = K.write_compare(cfgn, ["m1"]).read_text()
    assert "spread across the 8 forms" in txt
    print("\nSMOKE KNOWSAY OK")


if __name__ == "__main__":
    main(sys.argv[1])
