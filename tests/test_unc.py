"""Planted-signal tests of the uncertainty POC ANALYSIS (no model): fabricated rollouts whose answer distribution depends on a latent v carried linearly by the LATE layers of the
prompt-end and checkpoint states.  Checks: the labels are computed exactly; the probe recovers the planted effect and the decision says GO; the same pipeline given
labels from an INDEPENDENT latent does not say GO; the reliability machinery measures label noise."""
import json

import numpy as np
import pytest

from psuite import bench, table, unc, unc_study
from test_signal import LATE, fabricate

QUIET = lambda *_: None            # noqa: E731


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sim(gold, q, n, rng):
    return [gold if rng.rand() < q else str(int(gold) + int(rng.randint(1, 4))) for _ in range(n)]


def fabricate_unc(tmp, planted=True, seed=3, n_roll=16, n_cont=8, n=250):
    fabricate(tmp)
    rng = np.random.RandomState(seed)
    probs = bench.read_problems(tmp)
    N = len(probs)
    v = rng.randn(N)                                       # latent carried by the states
    vl = v if planted else rng.randn(N)                    # latent behind the labels (null: independent of the states)
    logmag = np.array([np.log10(int(p["gold"]) + 1) if p["kind"] == "int" else 0.0 for p in probs])
    isint = np.array([p["kind"] == "int" for p in probs])
    g = np.where(isint, (logmag - logmag[isint].mean()) / logmag[isint].std(), 0.0)
    z = np.load(tmp / "states" / "x00_all_0000.npz")
    d = {k: z[k] for k in z.files}
    Hp = d["plain"].astype(np.float32)
    Hp[:, LATE, 5] += 2.0 * v[:, None]
    Hp[:, LATE, 7] += 2.0 * g[:, None]
    d["plain"] = Hp.astype(np.float16)
    np.savez(tmp / "states" / "x00_all_0000.npz", **d)
    z = np.load(tmp / "trace" / "t00_all_0000.npz")
    d = {k: z[k] for k in z.files}
    H = d["H"].astype(np.float32)
    lay = d["layers"]
    for k in (2, 3):
        H[:, k, lay >= 9, 5] += 2.0 * v[:, None]
    d["H"] = H.astype(np.float16)
    np.savez(tmp / "trace" / "t00_all_0000.npz", **d)
    q0 = sigmoid(2.0 * vl + 1.0)
    gf = tmp / "gen" / "g00_all_0000.jsonl"
    rows = [json.loads(l) for l in open(gf)]
    for r, p, q in zip(rows, probs, q0):                   # greedy correctness follows the same latent
        if p["kind"] == "int" and r["plain_ans"] is not None:
            a = p["gold"] if rng.rand() < q else str(int(p["gold"]) + 1)
            r["plain_ans"], r["plain_text"] = a, "work\n" * 3 + f"\\boxed{{{a}}}"
    with open(gf, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    G = table.load_gen(tmp)
    sub = unc.pick_subset(probs, G, n)
    out = tmp / f"unc_n{n}"
    out.mkdir()
    with open(out / "u00_all_0000.jsonl", "w") as f:
        for p in sub:
            i, Tc = p["id"], len(G[p["id"]]["plain_ids"])
            row = {"id": i, "ds": p["ds"], "n_greedy": Tc, "roll_ans": sim(p["gold"], q0[i], n_roll, rng), "roll_fin": [True] * n_roll, "roll_ntok": [100] * n_roll,
                   "ent_all": float(rng.rand()), "nll_all": float(rng.rand()), "ent48": float(rng.rand()), "ent96": float(rng.rand()), "nll48": float(rng.rand()), "nll96": float(rng.rand()), "cont": {}}
            for cp, sh in ((48, 0.2), (96, 0.4)):
                if cp <= Tc - 8:
                    row["cont"][str(cp)] = {"ans": sim(p["gold"], sigmoid(2.0 * vl[i] + 1.0 + sh), n_cont, rng), "fin": [True] * n_cont}
            f.write(json.dumps(row) + "\n")
    return out


@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("unc_planted")
    out = fabricate_unc(tmp, planted=True)
    unc_study.compute(tmp, out, log=QUIET)
    unc_study.write(out, log=QUIET)
    return out


@pytest.fixture(scope="module")
def null(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("unc_null")
    out = fabricate_unc(tmp, planted=False, seed=5)
    unc_study.compute(tmp, out, log=QUIET)
    unc_study.write(out, log=QUIET)
    return out


def test_answer_stats_exact():
    s = unc_study.answer_stats(["4", "4", "5", None], "4")
    assert s["pass"] == 0.5 and s["maj"] == 0.5 and s["none"] == 0.25
    assert abs(s["ent"] - (-(0.5 * np.log(0.5) + 2 * 0.25 * np.log(0.25)))) < 1e-12
    u = unc_study.answer_stats(["7"] * 8, "7")
    assert u["ent"] == 0 and u["maj"] == 1 and u["pass"] == 1
    assert np.isnan(unc_study.answer_stats([], "1")["ent"])


def test_split_half_reliability_measures_noise():
    rng = np.random.RandomState(0)
    true = rng.rand(3000)
    a, b = true + 0.3 * rng.randn(3000), true + 0.3 * rng.randn(3000)      # each half: var(true)=1/12 vs noise 0.09 -> r ~ 0.48
    r, rel = unc_study.spearman_brown(a, b)
    assert 0.40 < r < 0.56 and abs(rel - 2 * r / (1 + r)) < 1e-12 and rel > r
    r0, rel0 = unc_study.spearman_brown(rng.randn(3000), rng.randn(3000))
    assert abs(r0) < 0.06 and rel0 < 0.12
    assert all(np.isnan(x) for x in unc_study.spearman_brown(np.ones(50), rng.randn(50)))


def test_planted_signal_recovered_and_go(planted):
    sp = __import__("pandas").read_csv(planted / "analysis" / "unc_prompt_results.csv")
    rel = __import__("pandas").read_csv(planted / "analysis" / "unc_reliability.csv")
    for lab in ("pass_rate", "ans_ent"):
        r = sp[(sp.prefix == "U[pooled]") & (sp.target == lab) & (sp.regime == "iid")].iloc[0]
        assert r.verdict_obs == "PASS" and r.probe_test_r2 > 0.15 and r.probe_layer >= 9, (lab, r.to_dict())
        assert (sp[(sp.prefix == "U[pooled]") & (sp.target == lab) & (sp.regime != "iid")].verdict_obs == "PASS").sum() >= 2
    assert rel[(rel.label == "pass_rate") & (rel.scope == "pooled")].reliability.iloc[0] > 0.6
    txt = (planted / "analysis" / "UNC_REPORT.md").read_text()
    assert "**GO" in txt and "Positive controls" in txt and "Label reliability" in txt and "Rule fixed before the run" in txt and "SUPPORTED" in txt
    st = __import__("pandas").read_csv(planted / "analysis" / "unc_status.csv")
    ck = st[(st.kind == "checkpoint") & (st.label == "c_pass")]
    assert len(ck) == 2 and (ck.status == "SUPPORTED").all(), ck.to_dict("records")


def test_controls_reproduce_in_planted(planted):
    import pickle
    z = pickle.load(open(planted / "analysis" / "unc.pkl", "rb"))
    c = unc_study.controls(z["sp"], z["sc"], z["lab"])
    assert c["ok_prompt"] and c["ok_ck"] and c["ans_logmag_r2"] > 0.3 and c["label_corr"] > 0.4 and c["log_rem_o48_r2"] > 0.3, c


def test_null_labels_do_not_say_go(null):
    st = __import__("pandas").read_csv(null / "analysis" / "unc_status.csv")
    prim = st[(st.scope == "pooled") & (st.kind == "prompt-end") & st.label.isin(["ans_ent", "pass_rate"])]
    assert len(prim) == 2 and not prim.status.isin(["SUPPORTED", "PARTIAL"]).any(), prim.to_dict("records")
    assert not st[st.kind == "checkpoint"].status.isin(["SUPPORTED"]).any()
    txt = (null / "analysis" / "UNC_REPORT.md").read_text()
    assert "**GO" not in txt and ("STOP" in txt or "INCONCLUSIVE" in txt or "LEAD" in txt)
    assert set(prim.status) <= {"NEGATIVE", "INCONCLUSIVE"}


def test_decision_requires_controls():
    import pandas as pd
    stat = pd.DataFrame([{"kind": "prompt-end", "scope": "pooled", "label": "ans_ent", "status": "SUPPORTED"}])
    ok = {"ok_prompt": True, "ok_ck": True}
    assert unc_study.decide(stat, ok).startswith("**GO")
    assert "CONTROLS FAILED" in unc_study.decide(stat, {"ok_prompt": False, "ok_ck": False})
    inc = stat.assign(status="INCONCLUSIVE")
    assert unc_study.decide(inc, ok).startswith("**INCONCLUSIVE")
    assert unc_study.decide(stat.assign(status="NEGATIVE"), ok).startswith("**STOP")


def test_cli_unc_analyze(planted):
    from psuite import cli
    rep = planted / "analysis" / "UNC_REPORT.md"
    rep.unlink()
    cli.main(["unc-analyze", "--run-dir", str(planted.parent), "--unc-n", "250"])
    assert rep.exists() and "DECISION" in rep.read_text()
