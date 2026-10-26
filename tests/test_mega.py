"""Tests of the mega-run machinery: step extraction / editing (pure), the step-edit stage and the steering hook on the tiny model, the cross-validated evaluator on planted signals,
the steering analysis on fabricated dose-response files, and the whole report on fabricated on-disk data of all four threads."""
import json

import numpy as np
import pytest
import torch

from psuite import cv_eval, mega, steer, steps, table
from psuite.hf_util import enc, generate_ids
from psuite.prompts import chat_prompt, plain_text
from test_pipeline import tok, model, run   # noqa: F401  (fixtures)
from test_signal import LATE, NL, D as FD, fabricate

QUIET = lambda *_: None            # noqa: E731

TEXT = "Step 1: Calculate the cost: \\( 3 \\times 4 = 12 \\).\nThen add 5: \\( 12 + 5 = 17 \\)\nShe has 17 apples at $18.50 each.\nThe answer is \\boxed{17}\n"


# ----------------------------------------------------------------------------- pure
def test_candidate_steps_and_edits():
    cs = steps.candidate_steps(TEXT)
    assert [c["value"] for c in cs] == [12, 17, 17] or [c["value"] for c in cs][:2] == [12, 17]
    assert all("\\boxed" not in c["line"] for c in cs)
    c0 = cs[0]
    assert TEXT[c0["start"]:c0["end"]] == c0["line"] and c0["line"][c0["span"][0]:c0["span"][1]] == "12"
    assert steps.edited_prefix(TEXT, c0, "n1") == TEXT[:c0["start"]] + c0["line"].replace("= 12", "= 13") + "\n"
    assert steps.edited_prefix(TEXT, c0, "n2").count("= 22") == 1
    assert steps.edited_prefix(TEXT, c0, "del") == "" and steps.edited_prefix(TEXT, cs[1], "del") == TEXT[:cs[1]["start"]]
    assert steps.reuse_count(TEXT, c0) == 1                              # the 12 is used again in the next line
    assert steps.candidate_steps("price is $18.50 and 2.5 more\n") == []   # decimals are never edited
    big = steps.candidate_steps("total = 1,250\nmore\n")[0]
    assert steps.edited_prefix("total = 1,250\nmore\n", big, "n1").startswith("total = 1,251")
    assert steps.edited_prefix("a = 5", steps.candidate_steps("a = 5")[0], "n1") is None      # last line has nothing after it
    many = "\n".join(f"s{i} = {i + 1}" for i in range(30)) + "\n"
    cm = steps.candidate_steps(many, max_steps=8)
    assert len(cm) == 8 and [c["k"] for c in cm] == list(range(8)) and cm[0]["value"] == 1 and cm[-1]["value"] == 30


def test_end_token_index(tok):
    ids = tok.encode("ab\n\ncd = 12\nxyz")
    for end in (2, 8, 11):
        t = steps.end_token_index(tok, ids, end)
        assert len(tok.decode(ids[:t])) >= end and (t == 1 or len(tok.decode(ids[:t - 1])) < end)


# ----------------------------------------------------------------------------- tiny model: step stage and steering hook
def test_steps_stage_tiny_model(tmp_path, tok, model):
    text = "Step 1: add: 3 + 4 = 7\nThen: 7 * 2 = 14\nSo 14 apples.\n\\boxed{14}"
    comp = tok.encode(text)
    probs = [{"id": i, "ds": "gsm8k", "kind": "int", "question": f"What is {i}+{i}?", "gold": "14", "d1": 1.0, "split": "fit"} for i in range(12)]
    G = {p["id"]: {"plain_ids": comp, "plain_fin": True, "plain_ans": "14", "plain_text": text} for p in probs}
    out = tmp_path / "steps"
    for sh in (0, 1):
        steps.run(probs, G, model, tok, out, sh, 2, chunk=3, max_steps=3, max_new=6, max_seqs=16, log=QUIET)
    files = sorted(p.name for p in out.glob("s*.jsonl"))
    steps.run(probs, G, model, tok, out, 0, 2, chunk=3, max_steps=3, max_new=6, max_seqs=16, log=QUIET)
    assert files and files == sorted(p.name for p in out.glob("s*.jsonl"))            # resumable
    rows, H, layers = mega.load_steps(out)
    assert H.shape == (len(rows), len(layers), 32) and len(rows) >= 12
    r0 = next(r for r in rows if r["k"] == 0)
    assert {"flip_n1", "flip_n2", "flip_del", "flip_null", "load", "reuse", "t_end"} <= set(r0) and 0 <= r0["load"] <= 1
    assert all(("flip_null" in r) == (r["k"] == 0) for r in rows)
    # the stored state is the natural state at the LAST TOKEN of the step: compare with a direct forward
    r = rows[3]
    p = probs[r["id"]]
    pr = enc(tok, chat_prompt(tok, plain_text(p)))
    with torch.no_grad():
        hs = model(input_ids=torch.tensor([pr + comp]), output_hidden_states=True).hidden_states
    li = len(layers) - 1
    np.testing.assert_allclose(H[3, li].astype(np.float32), hs[int(layers[li])][0, len(pr) + r["t_end"] - 1].numpy(), atol=3e-2, rtol=3e-2)
    assert text[:r["end"]].endswith(text[r["start"]:r["end"]]) and len(tok.decode(comp[:r["t_end"]])) >= r["end"]


def test_steer_hook_adds_exactly(tok, model):
    ids = torch.tensor([enc(tok, chat_prompt(tok, "What is 2+2?"))])
    with torch.no_grad():
        base = model(input_ids=ids, output_hidden_states=True).hidden_states
    vec = torch.randn(32)
    with steer.Steer(model, 2, vec), torch.no_grad():
        hs = model(input_ids=ids, output_hidden_states=True).hidden_states
    assert torch.allclose(hs[1], base[1])                                 # earlier layers untouched
    assert not torch.allclose(hs[3], base[3])                             # later layers see it (the recorded hs[2] may be taken before the hook, library detail)
    with steer.Steer(model, 2, 2 * vec), torch.no_grad():
        hs2 = model(input_ids=ids, output_hidden_states=True).hidden_states
    assert (hs2[3] - base[3]).norm() > (hs[3] - base[3]).norm()           # dose-response in the right direction
    with steer.Steer(model, 2, None), torch.no_grad():                    # alpha = 0 -> no hook at all
        z = model(input_ids=ids, output_hidden_states=True).hidden_states
    assert torch.allclose(z[3], base[3])
    with torch.no_grad():                                                 # the hook is removed on exit
        again = model(input_ids=ids, output_hidden_states=True).hidden_states
    assert torch.allclose(again[3], base[3])
    p = chat_prompt(tok, "What is 2+2?")
    a = generate_ids(model, tok, [p], 6, 1, 0.0)[0][0]
    with steer.Steer(model, 2, 50 * vec):
        b = generate_ids(model, tok, [p], 6, 1, 0.0)[0][0]
    assert a != b


# ----------------------------------------------------------------------------- cross-validated evaluator
def planted_rows(seed=0, n=700, signal=True):
    rng = np.random.RandomState(seed)
    g = np.repeat(np.arange(n), 3)                                # 3 rows per problem
    u = rng.randn(n)[g] + 0.5 * rng.randn(len(g))
    NLc, Dc = 8, 16
    H = rng.randn(len(g), NLc, Dc).astype(np.float32)
    if signal:
        H[:, 4:, 2] += 2.0 * u[:, None]
    y = (u if signal else rng.randn(len(g))) + 0.5 * rng.randn(len(g))
    split = np.where(rng.rand(n) < 0.8, "fit", "test")[g]
    return g, H, y, split, rng


def test_cv_eval_recovers_planted_and_rejects_null():
    for signal in (True, False):
        g, H, y, split, rng = planted_rows(signal=signal)
        bl = {"Bnoise": rng.randn(len(g), 5).astype(np.float32), "H1(early-layer)": H[:, 1, :]}
        rows = cv_eval.run_block(split, np.zeros(len(g)), g, lambda l: H[:, l, :], list(range(8)), bl, y[:, None], ["y"], regimes=("iid",), prefix="T", log=QUIET, B=200)
        r = rows[0]
        if signal:
            assert r["verdict_obs"] == "PASS" and r["probe_layer"] >= 4 and r["probe_test_r2"] > 0.3 and r["n_test_groups"] < r["n_test"]
            assert max(r["curve"], key=r["curve"].get) >= 4                # cross-validation prefers a layer that carries the signal
        else:
            assert r["verdict_obs"] != "PASS"


def test_cluster_bootstrap_is_wider_than_naive():
    rng = np.random.RandomState(1)
    n, m = 60, 8
    g = np.repeat(np.arange(n), m)
    shared = rng.randn(n)[g]
    y = shared + 0.3 * rng.randn(len(g))
    pa, pb = y + 0.5 * shared * 0 + 0.0 * rng.randn(len(g)), np.zeros(len(g))
    d, lo, hi = cv_eval.boot_delta_cluster(y, pa, pb, g, B=400)
    from psuite import core
    d2, lo2, hi2 = core.boot_delta(y, pa, pb, B=400)
    assert (hi - lo) >= (hi2 - lo2) * 0.9 and d > 0.5


# ----------------------------------------------------------------------------- steering analysis on fabricated dose-response data
def fab_steer(out, slope, seed=0, n=120, damage=0.0):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    base = 5.0 + 0.5 * rng.randn(n)
    ok = rng.rand(n)
    for name, fam, a in steer.conditions():
        eff = slope * a if fam == "probe" else (0.01 * a * rng.randn() if fam else 0.0)
        with open(out / f"r00_{name}.jsonl", "w") as f:
            for i in range(n):
                nt = int(np.exp(base[i] + eff + 0.05 * rng.randn()))
                f.write(json.dumps({"id": i, "cond": name, "ntok": nt, "fin": True, "ans": "1", "correct": float(ok[i] > 0.3 + damage * abs(a) / 4 * (fam is not None))}) + "\n")


def test_steering_analysis_decisions(tmp_path):
    fab_steer(tmp_path / "pos", slope=0.08)
    r = steer.analyze(tmp_path / "pos")
    assert r["status"] == "SUPPORTED" and r["slopes"]["probe"][0] > 0.05 and r["slopes"]["probe"][1] > 0 and r["n"] == 120
    fab_steer(tmp_path / "null", slope=0.0, seed=1)
    assert steer.analyze(tmp_path / "null")["status"] == "NEGATIVE"
    fab_steer(tmp_path / "dmg", slope=0.08, seed=2, damage=0.35)
    assert steer.analyze(tmp_path / "dmg")["status"] == "PARTIAL"            # moves length but also breaks accuracy
    assert steer.analyze(tmp_path / "empty")["status"] == "n/a"


def test_steer_stage_tiny_model(run, tok, model):
    tmp, problems = run
    out = tmp / "steer_t"
    with pytest.raises(RuntimeError):                                         # the random tiny model finishes nothing: prep refuses instead of fitting noise
        steer.prep(tmp, out, log=QUIET)
    out.mkdir(exist_ok=True)
    rng = np.random.RandomState(0)
    rand = rng.randn(steer.N_RAND, 32)
    np.savez(out / "dirs.npz", probe=np.eye(32)[0], diff=np.eye(32)[1], rand=rand / np.linalg.norm(rand, axis=1, keepdims=True), layer=2, s=1.5)
    test = [p for p in problems if p["ds"] == "gsm8k" and p["split"] == "test"][:6]
    steer.run(test, model, tok, out, 0, 1, max_new=6, max_seqs=16, log=QUIET)
    assert len(list(out.glob("r00_*.jsonl"))) == len(steer.conditions())
    R = steer.load(out)
    assert set(R["base"]) == {p["id"] for p in test}
    assert steer.analyze(out)["status"] == "n/a"                              # 6 problems are too few: refuses instead of guessing


# ----------------------------------------------------------------------------- the whole report on fabricated on-disk data
def fabricate_mega(tmp, seed=7):
    from test_unc import fabricate_unc
    unc_out = fabricate_unc(tmp, planted=True, seed=seed, n=400)
    mega_dir = tmp / "mega"
    (mega_dir / "unc").mkdir(parents=True)
    for f in unc_out.glob("u*.jsonl"):
        (mega_dir / "unc" / f.name).write_text(f.read_text())
    rng = np.random.RandomState(seed)
    probs = [p for p in __import__("psuite").bench.read_problems(tmp) if p["ds"] == "gsm8k"]
    sd = mega_dir / "steps"
    sd.mkdir()
    layers = np.arange(2, NL + 1, 2)
    rows, Hs = [], []
    for p in probs:
        for k in range(4):
            u = rng.randn()
            H = rng.randn(len(layers), FD).astype(np.float32)
            H[layers >= 10, 2] += 2.0 * u
            pf = 1 / (1 + np.exp(-2 * u))
            fl = [float(rng.rand() < pf) for _ in range(3)]
            r = {"id": p["id"], "k": k, "n_steps": 4, "li": k, "start": 5 * k, "end": 5 * k + 4, "value": int(rng.randint(2, 90)), "t_end": 20 * (k + 1), "n_tok": 120, "reuse": int(rng.randint(0, 3)),
                 "ans_orig": "1", "ans_n1": "2", "ans_n2": "2", "ans_del": "2", "flip_n1": fl[0], "flip_n2": fl[1], "flip_del": fl[2], "load": float(np.mean(fl))}
            if k == 0:
                r["flip_null"] = float(rng.rand() < 0.03)
            rows.append(r)
            Hs.append(H)
    with open(sd / "s00_0000.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    np.savez(sd / "s00_0000.npz", H=np.stack(Hs).astype(np.float16), layers=layers)
    fab_steer(mega_dir / "steer", slope=0.08, seed=seed)
    (mega_dir / "steer" / "prep.json").write_text(json.dumps({"layer": 10, "cv_r2": 0.7, "test_r2": 0.68, "scale_s": 3.0, "cos_probe_diff": 0.4, "cos_probe_rand_max": 0.05}))
    return mega_dir


@pytest.fixture(scope="module")
def mega_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mega")
    mega_dir = fabricate_mega(tmp)
    rep = mega.run_all(tmp, mega_dir, log=QUIET)
    return tmp, mega_dir, rep


def test_mega_report_end_to_end(mega_run):
    import pandas as pd
    tmp, mega_dir, rep = mega_run
    txt = rep.read_text()
    for needle in ("Where to pull", "Status table", "Thread 1", "Thread 2", "Thread 3", "Thread 4", "Corrected protocol"):
        assert needle in txt
    st = pd.read_csv(mega_dir / "analysis" / "mega_status.csv").set_index(["thread", "label"])
    for lab in ("load", "flip_num"):
        assert st.loc[("steps", lab), "status"] == "SUPPORTED", st.loc[("steps", lab)].to_dict()
    assert st.loc[("steer", "length direction"), "status"] == "SUPPORTED"
    assert st.loc[("unc", "pass_rate"), "status"] in ("SUPPORTED", "PARTIAL"), st.loc[("unc", "pass_rate")].to_dict()
    assert st.loc[("suite", "P:logT"), "status"] in ("SUPPORTED", "PARTIAL")
    assert "SUPPORTED" in txt.split("## Status table")[0]
    assert (mega_dir / "analysis" / "steps_results.csv").exists()


def test_mega_survives_a_broken_thread(tmp_path):
    tmp = tmp_path
    fabricate(tmp)
    rep = mega.run_all(tmp, tmp / "mega", parts=("steps", "steer"), log=QUIET)         # no step / steer data at all: reports n/a, does not crash
    txt = rep.read_text()
    assert "Thread 1" in txt and "not enough data" in txt and "no baseline condition" in txt


def test_mega_cli_commands(run, tok, model, monkeypatch):
    """the exact command lines of cluster/run.sbatch for the mega stages (arguments, output dirs) on the tiny model; the random tiny model finishes no solutions, so the point is wiring, not content."""
    from psuite import cli
    tmp, _ = run
    monkeypatch.setattr("psuite.hf_util.load_model", lambda mid: (model, tok))
    md = tmp / "mega_cli"
    common = ["--run-dir", str(tmp), "--mega-dir", str(md)]
    for sh in (0, 1):
        cli.main(["steps", *common, "--shard", str(sh), "--nshards", "2", "--chunk", "4", "--max-steps", "3", "--max-new", "6", "--max-seqs", "16", "--budget-s", "600"])
        cli.main(["unc-roll", *common, "--shard", str(sh), "--nshards", "2", "--unc-datasets", "gsm8k", "--unc-n", "10", "--unc-skip", "5", "--n-roll", "3", "--n-cont", "0", "--max-new", "6", "--max-seqs", "16"])
    assert any((md / "unc").glob("u*.jsonl"))
    r = [json.loads(l) for f in (md / "unc").glob("u*.jsonl") for l in open(f)]
    assert r and all(not x["cont"] for x in r) and all(len(x["roll_ans"]) == 3 for x in r)
    (md / "steer").mkdir(parents=True, exist_ok=True)
    np.savez(md / "steer" / "dirs.npz", probe=np.eye(32)[0], diff=np.eye(32)[1], rand=np.eye(32)[2:2 + steer.N_RAND], layer=2, s=1.0)
    cli.main(["steer-run", *common, "--shard", "0", "--nshards", "1", "--steer-n", "4", "--max-new", "6", "--max-seqs", "16", "--budget-s", "600"])
    assert len(list((md / "steer").glob("r00_*.jsonl"))) == len(steer.conditions())
    cli.main(["mega-analyze", *common, "--parts", "steer"])
    assert (md / "analysis" / "MEGA_REPORT.md").exists()
