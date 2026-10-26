"""Planted-signal tests of the ANALYSIS half (no model): fabricate generation records, prompt-end states and checkpoint states in the exact on-disk formats,
with a known linear signal in the LATE layers only (so early-layer baselines cannot see it), and check that the pipeline recovers it, rejects the null, and
that the strict scorecard says what it should."""
import json

import numpy as np
import pytest

from psuite import bench, report, study, table, trace, verify

NL, D = 12, 24                      # hidden states: layers 0..12
LATE = np.arange(NL + 1) >= 9
DSN = {"gsm8k": 400, "math": 300, "arith": 250, "arc": 250}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def fabricate(tmp, seed=0):
    rng = np.random.RandomState(seed)
    probs, nid = [], 0
    for ds, n in DSN.items():
        for i in range(n):
            gold = "ABCD"[int(rng.randint(4))] if ds == "arc" else str(int(rng.randint(1, 5000)))
            probs.append({"id": nid, "ds": ds, "kind": "mcq" if ds == "arc" else "int", "tier": "B" if ds == "arc" else "A", "question": f"Compute thing {i} with {rng.randint(100)} and {rng.randint(100)} in {ds}.", "gold": gold,
                          "d1": float(rng.randint(1, 8)), "split": rng.choice(["fit", "val", "test"], p=[0.7, 0.1, 0.2])})
            nid += 1
    bench.write(tmp, probs, [{"dataset": ds, "tier": "A", "n_in": n, "n_kept": n, "pass_rate": 1.0, "admitted": True, "fails": {}} for ds, n in DSN.items()])
    N = len(probs)
    z, w, u = rng.randn(N), rng.randn(N), rng.randn(N)       # runaway propensity, compliance propensity, log-length of finished solutions
    gen_dir = tmp / "gen"
    gen_dir.mkdir()
    rows = []
    for i, p in enumerate(probs):
        runaway = (z[i] + 0.4 * rng.randn() > 0.6)
        T = 512 if runaway else int(min(500, np.exp(4.5 + 0.5 * u[i] + 0.1 * rng.randn())))
        ans = None if (runaway and rng.rand() < 0.7) else (p["gold"] if rng.rand() < 0.7 else ("A" if p["gold"] != "A" else "B") if p["kind"] == "mcq" else str(int(p["gold"]) + 1))
        ctype = verify.constraint_for(p["id"])
        comply = bool(w[i] + 0.4 * rng.randn() > 0)
        sn = [512 if (z[i] + 0.8 * rng.randn() > 0.6) else int(np.exp(4.5 + 0.2 * rng.randn())) for _ in range(4)]
        rows.append({"id": p["id"], "ds": p["ds"], "ctype": ctype, "plain_ids": [5] * T, "plain_fin": not runaway, "plain_text": "work\n" * 3 + (f"\\boxed{{{ans}}}" if ans else ""),
                     "plain_ans": ans, "plain_words": T // 2, "spread_n": sn, "spread_fin": [x < 512 for x in sn], "con_ntok": 100, "con_fin": True, "con_text": "x", "con_ans": ans,
                     "con_words": 60, "con_comply": comply, "con_over": float(np.log(60 / verify.LIMIT[ctype])) if ctype in verify.LIMIT else None})
    with open(gen_dir / "g00_all_0000.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (tmp / "states").mkdir()
    Hp, Hc = rng.randn(N, NL + 1, D).astype(np.float32), rng.randn(N, NL + 1, D).astype(np.float32)
    Hp[:, LATE, 0] += 2.0 * z[:, None]
    Hp[:, LATE, 3] += 2.0 * u[:, None]
    Hc[:, LATE, 1] += 2.0 * w[:, None]
    np.savez(tmp / "states" / "x00_all_0000.npz", ids=np.arange(N), plain=Hp.astype(np.float16), con=Hc.astype(np.float16), plain_ntok=np.zeros(N, np.int32), con_ntok=np.zeros(N, np.int32))
    # checkpoints: settledness s_k rises with k; carried by late layers (coordinate 2); the text can only see a weak, noisy trace of it
    K, layers = len(trace.CK_NAMES), np.array(trace.ck_layers(NL + 1))
    H = rng.randn(N, K, len(layers), D).astype(np.float32)
    s = rng.randn(N)
    lp_final, lp_gold, men, rem, Tc = (np.full((N, K), np.nan) for _ in range(5))
    pf = rng.randn(N, K, 8).astype(np.float32)
    for k in range(K):
        sk = s + 0.8 * k + 0.3 * rng.randn(N)
        H[:, k, layers >= 9, 2] += 2.0 * sk[:, None]
        lp_final[:, k] = np.log(np.clip(sigmoid(1.5 * (sk - 2.0)), 1e-6, 1))
        lp_gold[:, k] = lp_final[:, k] - 0.5
        men[:, k] = (sk + 1.0 * rng.randn(N) > 3.0)                       # the visible mention is a weak, noisy proxy
        pf[:, k, 0] += 0.3 * sk
    Tcs = np.array([len(r["plain_ids"]) for r in rows])
    offs = np.array([0, 24, 48, 96, 192])
    t = np.concatenate([np.tile(offs, (N, 1)), np.maximum(Tcs - 3, 0)[:, None]], 1)
    fin = np.array([r["plain_fin"] for r in rows])
    valid = np.zeros((N, K), dtype=bool)
    valid[:, 0] = True
    for k in range(1, 5):
        valid[:, k] = offs[k] <= Tcs - 8
    valid[:, 5] = [r["plain_ans"] is not None for r in rows]
    rem = np.where(fin[:, None] & valid, Tcs[:, None] - t, np.nan)
    for k in range(K):
        H[:, k, layers >= 9, 3] += 2.0 * u[:, None]                      # the model's (noisy) knowledge of how long the solution is
    for i, r in enumerate(rows):                                          # no parsable answer => settledness undefined
        if r["plain_ans"] is None:
            lp_final[i] = np.nan
    (tmp / "trace").mkdir()
    np.savez(tmp / "trace" / "t00_all_0000.npz", ids=np.arange(N), H=H.astype(np.float16), layers=layers, t=t.astype(np.int32), valid=valid, lp_final=lp_final, lp_gold=lp_gold, rem=rem, Tc=Tcs.astype(np.int32),
             fin=fin, men_final=men.astype(bool), men_gold=men.astype(bool), pf=pf)
    return N


@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("planted")
    fabricate(tmp)
    for b in ("tables", "P", "C", "K"):
        study.compute(tmp, b, log=lambda *_: None)
    report.write(tmp, log=lambda *_: None)
    return tmp


def _res(tmp, blk):
    return report._load(tmp / "analysis", blk)


def test_prompt_end_signals_recovered_and_null_rejected(planted):
    P, C = _res(planted, "P"), _res(planted, "C")
    r = report._rows(P["res"], "P[pooled]", "runaway", "iid")[0]
    assert report._v(r) == "PASS" and r["probe_test_r2"] > 0.2 and r["probe_layer"] >= 9 and r["auroc_probe"] > 0.85 and r["auroc_probe"] > r["auroc_obs"] + 0.1
    assert report._v(report._rows(P["res"], "P[pooled]", "logT", "iid")[0]) == "PASS"
    c = report._rows(C["res"], "C[pooled]", "comply", "iid")[0]
    assert report._v(c) == "PASS" and c["probe_layer"] >= 9 and c["auroc_probe"] > 0.8
    # nothing was planted for these: must not PASS
    assert report._v(report._rows(P["res"], "P[pooled]", "ans_ndig", "iid")[0]) != "PASS"
    assert report._v(report._rows(P["res"], "P[pooled]", "steps", "iid")[0]) != "PASS" or report._rows(P["res"], "P[pooled]", "steps", "iid")[0]["probe_test_r2"] < 0.1
    # leave-one-dataset-out really holds out the dataset
    lo = report._rows(P["res"], "P[pooled]", "runaway", "lodo:math")[0]
    assert set(lo["_ds"]) == {"math"} and report._v(lo) == "PASS"
    assert len(C["ctype"]) == len(verify.CONSTRAINTS) and C["ctype"].auroc_probe.min() > 0.7


def test_checkpoint_signals_and_silent(planted):
    K = _res(planted, "K")
    for ck in ("o24", "o48", "pre"):
        r = next(x for x in K["res"] if x["ck"] == ck and x["target"] == "settled_p" and x["regime"] == "iid")
        assert report._v(r) == "PASS" and r["probe_layer"] >= 9 and r["probe_test_r2"] > 0.2
    sil = K["silent"]
    assert len(sil) == 6 and (sil.n_silent > 0).all()
    good = sil.dropna(subset=["auroc_probe"])
    good = good[(good.n_pos >= 20) & (good.n_silent >= 50)]            # tiny silent subsets (e.g. 6 positives) are noise by construction
    assert len(good) >= 3 and (good.auroc_probe > good.auroc_obs).all()


def test_strict_scorecard(planted):
    P, C, K = _res(planted, "P"), _res(planted, "C"), _res(planted, "K")
    sc = report.scorecard(P, C, K).set_index("signal")
    for s in ("finish", "length", "comply", "settled", "ready", "remaining"):
        assert sc.loc[s, "status"] == "SUPPORTED", (s, sc.loc[s].to_dict())
    assert sc.loc["answer_size", "status"] != "SUPPORTED"
    txt = (planted / "analysis" / "REPORT.md").read_text()
    for needle in ("Scorecard", "Plain prompt-end state", "Constrained prompt-end state", "Silent settledness", "Limitations", "SUPPORTED"):
        assert needle in txt
    assert (planted / "analysis" / "fig1_layers_and_checkpoints.png").exists() and (planted / "analysis" / "scorecard.csv").exists()


def test_signal_that_does_not_generalise_is_not_supported(tmp_path):
    """plant the runaway signal in ONE dataset only: iid (pooled) may pass, but leave-dataset-out must stop it from being SUPPORTED."""
    fabricate(tmp_path, seed=1)
    z = np.load(tmp_path / "states" / "x00_all_0000.npz")
    d = {k: z[k] for k in z.files}
    ids = np.arange(len(d["ids"]))
    probs = bench.read_problems(tmp_path)
    ds = np.array([p["ds"] for p in probs])
    # erase the planted coordinate everywhere except gsm8k
    rng = np.random.RandomState(5)
    plain = d["plain"].astype(np.float32)
    plain[ds != "gsm8k"][:, :, 0] = 0
    m = ds != "gsm8k"
    plain[m, :, 0] = rng.randn(int(m.sum()), plain.shape[1])
    d["plain"] = plain.astype(np.float16)
    np.savez(tmp_path / "states" / "x00_all_0000.npz", **d)
    study.compute(tmp_path, "tables", log=lambda *_: None)
    study.compute(tmp_path, "P", log=lambda *_: None)
    P = _res(tmp_path, "P")
    st = report._status(P["res"], "P[pooled]", "runaway")
    assert st["status"] != "SUPPORTED", st
