import json

import numpy as np
import pytest
import torch

from psuite import bench, cli, core, gen, report, states, study, table, textutil, trace, verify
from psuite.hf_util import enc, eos_set, generate_ids
from psuite.prompts import CUE, chat_prompt, con_text, plain_text

CHARS = list(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,=+-*/\\{}()$:#?!<>|_'\n;%\"&`[]")


class FakeTok:
    def __init__(self):
        self.vocab = ["<pad>", "<eos>"] + CHARS + ["\n\n"]
        self.idx = {t: i for i, t in enumerate(self.vocab)}
        self.eos_token_id, self.pad_token_id = 1, 0

    def __len__(self):
        return len(self.vocab)

    def encode(self, s, add_special_tokens=False):
        out, i = [], 0
        while i < len(s):
            if s[i:i + 2] == "\n\n":
                out.append(self.idx["\n\n"]); i += 2
            else:
                out.append(self.idx.get(s[i], 2)); i += 1
        return out

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.vocab[int(i)] for i in ids if not (skip_special_tokens and int(i) in (0, 1)))


@pytest.fixture(scope="module")
def tok():
    return FakeTok()


@pytest.fixture(scope="module")
def model(tok):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=4096, eos_token_id=1, pad_token_id=0, tie_word_embeddings=False)
    return Qwen2ForCausalLM(cfg).eval()


def fake_loaders():
    import pandas as pd
    rng = np.random.RandomState(0)
    gs = {"train": [], "test": []}
    for sp in gs:
        for i in range(60):
            a, b = int(rng.randint(2, 30)), int(rng.randint(2, 30))
            gs[sp].append({"question": f"Tom has {a} apples and gets {b} more ({sp}{i}). How many?", "answer": f"step <<{a}+{b}={a + b}>> \n#### {a + b}"})
    mth = {s: pd.DataFrame([{"problem": f"What is {i}+{i} ({s})?", "solution": f"It is $\\boxed{{{2 * i}}}$.", "level": f"Level {1 + i % 5}", "type": "algebra"} for i in range(80)])
           for s in ("train", "test")}
    arc = {s: [{"id": f"{s}{i}", "question": f"Which is largest ({s}{i})?", "choices": {"label": list("ABCD"), "text": ["a", "b", "c", "d"]}, "answerKey": "ABCD"[i % 4]}
               for i in range(60)] for s in ("train", "validation", "test")}
    return {"gsm8k": lambda: gs, "math": lambda: mth, "arc": lambda split: arc[split]}


# ----------------------------------------------------------------------------- pure functions
def test_textutil_and_audit():
    assert textutil.parse_answer("so \\boxed{\\$1,250} then \\boxed{12 apples}", "int") == "12"
    rows = bench.rows_arith(60, seed=1)
    for i, r in enumerate(rows):
        r["id"], r["kind"] = i, "int"
        r["hint"] = bench.make_hint("int", i, r["gold"])
        r["hint_off"] = bench.hint_distance("int", r["gold"], r["hint"])
    assert bench.audit(rows, log=lambda *_: None)["report"]["admitted"]
    bad = [dict(r) for r in rows]
    for r in bad[:10]:
        r["gold"] = str(int(r["gold"]) + 1)
    rep = bench.audit(bad, log=lambda *_: None)["report"]
    assert not rep["admitted"] and "G2_arith_recompute" in rep["fails"]


def test_verifiers():
    P = verify.PHRASE
    assert verify.complies("maxw50", "word " * 50, True) and not verify.complies("maxw50", "word " * 51, True)
    assert verify.complies("maxw200", "word " * 150, True) and not verify.complies("maxw100", "word " * 101, True)
    assert verify.complies("nocomma", "a b. c", True) and not verify.complies("nocomma", "a, b", True)
    assert verify.complies("endphrase", f"x\n{P}\n", True) and not verify.complies("endphrase", f"{P} x", True)
    for c in verify.CONSTRAINTS:                                   # a truncated response never complies, whatever its text
        assert not verify.complies(c, f"ok {P}", False)
    assert abs(verify.overshoot("maxw100", "w " * 200, True) - np.log(2)) < 1e-9 and verify.overshoot("nocomma", "x", True) is None and verify.overshoot("maxw50", "w", False) is None
    assert {verify.constraint_for(i) for i in range(500)} == set(verify.CONSTRAINTS) and verify.constraint_for(17) == verify.constraint_for(17)
    p = {"id": 3, "question": "Q?", "kind": "int"}
    assert plain_text(p) in con_text(p) and verify.CONSTRAINT_TEXT[verify.constraint_for(3)] in con_text(p) and verify.CONSTRAINT_TEXT[verify.constraint_for(3)] not in plain_text(p)


def test_generate_ids_contract(tok, model):
    prompts = [chat_prompt(tok, f"Q{i}? " * (1 + i % 3)) for i in range(7)]
    a = generate_ids(model, tok, prompts, 6, 1, 0.0, max_seqs=3)
    b = generate_ids(model, tok, prompts, 6, 1, 0.0, max_seqs=7)
    assert len(a) == 7 and all(len(x) == 1 for x in a)
    for (ia, fa), (ib, fb) in zip((x[0] for x in a), (x[0] for x in b)):
        assert fa == fb and len(ia) <= 6 and (fa or len(ia) == 6) and 1 not in ia
    s = generate_ids(model, tok, prompts[:3], 6, 4, 0.7)
    assert all(len(x) == 4 for x in s) and eos_set(model, tok) == {1}


# ----------------------------------------------------------------------------- the real pipeline on a tiny random model
@pytest.fixture(scope="module")
def run(tmp_path_factory, tok, model):
    tmp = tmp_path_factory.mktemp("run")
    sizes = {"gsm8k": 70, "math": 60, "arc": 60, "arith": 70}
    problems, reports = bench.build(tmp, sizes, loaders=fake_loaders(), log=lambda *_: None)
    assert {r["dataset"] for r in reports} == set(sizes) and all(r["admitted"] for r in reports)
    for sh in (0, 1):
        gen.run(problems, model, tok, tmp, sh, 2, chunk=30, max_new=8, n_spread=3, max_seqs=16, log=lambda *_: None)
        states.run(problems, model, tok, tmp, sh, 2, fwd_bs=5, chunk=40, log=lambda *_: None)
        trace.run(problems, model, tok, tmp, sh, 2, fwd_bs=3, chunk=30, offsets=(0, 2, 3, 4, 5), min_left=1, log=lambda *_: None)
    return tmp, problems


def test_gen_resumable_and_labels(run, tok, model):
    tmp, problems = run
    before = sorted(p.name for p in (tmp / "gen").iterdir())
    gen.run(problems, model, tok, tmp, 0, 2, chunk=30, max_new=8, n_spread=3, max_seqs=16, log=lambda *_: None)
    assert before == sorted(p.name for p in (tmp / "gen").iterdir())
    G = table.load_gen(tmp)
    assert len(G) == len(problems)
    r = G[problems[5]["id"]]
    assert r["ctype"] == verify.constraint_for(r["id"]) and len(r["spread_n"]) == 3
    assert r["con_comply"] == verify.complies(r["ctype"], r["con_text"], r["con_fin"])
    assert (len(r["plain_ids"]) == 8) or r["plain_fin"]
    # greedy => regenerating one problem alone reproduces its stored completion
    p = next(q for q in problems if q["id"] == r["id"])
    ids, fin = generate_ids(model, tok, [chat_prompt(tok, plain_text(p))], 8, 1, 0.0)[0][0]
    assert ids == r["plain_ids"] and fin == r["plain_fin"]


def test_states_match_direct_forward(run, tok, model):
    tmp, problems = run
    st = table.load_states(tmp)
    n = len(problems)
    assert st["states"]["plain"].shape == (n, 5, 32) and st["states"]["con"].shape == (n, 5, 32) and (np.diff(st["ids"]) > 0).all()
    p = problems[9]
    k = int(np.where(st["ids"] == p["id"])[0][0])
    for name, fn in (("plain", plain_text), ("con", con_text)):
        ids = torch.tensor([enc(tok, chat_prompt(tok, fn(p)))])
        with torch.no_grad():
            hs = model(input_ids=ids, output_hidden_states=True).hidden_states
        ref = torch.stack([h[0, -1] for h in hs]).numpy()
        np.testing.assert_allclose(st["states"][name][k].astype(np.float32), ref, atol=2e-2, rtol=2e-2)
    assert not np.allclose(st["states"]["plain"][k], st["states"]["con"][k])           # the constraint is really in the prompt


def test_trace_labels_and_states(run, tok, model):
    tmp, problems = run
    tr = table.load_trace(tmp)
    G = table.load_gen(tmp)
    n, K = len(problems), len(trace.CK_NAMES)
    assert tr["H"].shape == (n, K, len(tr["layers"]), 32) and tr["pf"].shape == (n, K, 8) and (np.diff(tr["ids"]) > 0).all() and tr["valid"].shape == (n, K)
    assert (tr["t"][:, 0] == 0).all() and tr["valid"][:, 0].all()
    assert (tr["t"][:, :5] == np.array([0, 2, 3, 4, 5])).all()
    assert (tr["t"][tr["valid"]] <= np.broadcast_to(tr["Tc"][:, None], tr["t"].shape)[tr["valid"]]).all()
    assert tr["valid"][:, 1].sum() > 0 and not tr["valid"].all()
    assert (tr["valid"][:, 1] == (tr["Tc"] >= 3)).all()               # offset 2 + min_left 1
    # checkpoint 0 state == the prompt-end state of the states stage (same position, same layers)
    st = table.load_states(tmp)
    assert (st["ids"] == tr["ids"]).all()
    for li, l in enumerate(tr["layers"]):
        np.testing.assert_allclose(tr["H"][:, 0, li].astype(np.float32), st["states"]["plain"][:, l].astype(np.float32), atol=3e-2, rtol=3e-2)
    # the forced-answer log-prob equals a direct, unpadded computation; the state is the state WITHOUT the cue (causality); invalid checkpoints carry no labels
    checked = 0
    for j in range(n):
        p = next(q for q in problems if q["id"] == tr["ids"][j])
        g = G[p["id"]]
        for kk in range(6):
            if not tr["valid"][j, kk]:
                assert np.isnan(tr["lp_gold"][j, kk]) and np.isnan(tr["lp_final"][j, kk]) and np.isnan(tr["rem"][j, kk])
                continue
            t = int(tr["t"][j, kk])
            pr = enc(tok, chat_prompt(tok, plain_text(p)))
            base = pr + g["plain_ids"][:t] + enc(tok, CUE)
            a = enc(tok, f"{p['gold']}}}")
            ids = torch.tensor([base + a])
            with torch.no_grad():
                o = model(input_ids=ids, output_hidden_states=True)
            lp = torch.log_softmax(o.logits[0].float(), -1)
            ref = sum(float(lp[len(base) - 1 + i, tok_id]) for i, tok_id in enumerate(a))
            assert abs(ref - tr["lp_gold"][j, kk]) < 5e-3
            li = len(tr["layers"]) - 1
            np.testing.assert_allclose(tr["H"][j, kk, li].astype(np.float32), o.hidden_states[tr["layers"][li]][0, len(pr) + t - 1].numpy(), atol=3e-2, rtol=3e-2)
            if g["plain_ans"] is not None:
                assert np.isfinite(tr["lp_final"][j, kk])
                if g["plain_ans"] == p["gold"]:
                    assert tr["lp_final"][j, kk] == tr["lp_gold"][j, kk]
            else:
                assert np.isnan(tr["lp_final"][j, kk])
            assert np.isnan(tr["rem"][j, kk]) if not g["plain_fin"] else tr["rem"][j, kk] == tr["Tc"][j] - t
            checked += 1
        if checked > 60:
            break
    assert checked > 20


def test_no_length_leak_through_position(run):
    """checkpoints are at ABSOLUTE offsets, so within a checkpoint the position is constant and carries no information about the total length."""
    tmp, _ = run
    df, tr, frames, tks = table.build_trace(tmp, None, log=lambda *_: None)
    for k in range(5):
        assert frames[k].t_k.nunique() == 1


def test_tables_and_cli_smoke(run):
    tmp, problems = run
    df, sub = table.build(tmp, None, need_states=True, log=lambda *_: None)
    assert len(df) == len(problems) and {"runaway", "logT", "comply", "over", "ans_logmag", "len_sd", "runaway_rate"} <= set(df.columns)
    assert sub["plain"].shape[0] == len(df) and np.isnan(df.loc[df.runaway == 1, "logT"]).all()
    df2, tr, frames, tks = table.build_trace(tmp, None, log=lambda *_: None)
    assert len(frames) == 6 and (frames[0].t_k == 0).all() and frames[0].settled_p.dropna().between(0, 1).all() and len(tks) == 6 and len(tks[1]["H"]) == len(frames[1])
    assert frames[5].log_rem.isna().all() and df2.id.is_unique
    for blk in ("tables",):
        cli.main(["analyze", "--run-dir", str(tmp), "--block", blk])
    assert (tmp / "analysis" / "tables.pkl").exists()


def test_cli_stages_end_to_end(tmp_path, tok, model, monkeypatch):
    """the exact command lines the sbatch script uses (arguments, sharding, file naming) on the tiny model."""
    bench.build(tmp_path, {"gsm8k": 70, "arith": 70}, loaders=fake_loaders(), log=lambda *_: None)   # the audit rejects tiny sets (< 50 problems)
    monkeypatch.setattr("psuite.hf_util.load_model", lambda mid: (model, tok))
    for sh in (0, 1):
        common = ["--run-dir", str(tmp_path), "--shard", str(sh), "--nshards", "2"]
        cli.main(["states", *common, "--fwd-bs", "4"])
        cli.main(["gen", *common, "--chunk", "20", "--max-new", "6", "--n-spread", "2", "--max-seqs", "16", "--budget-s", "600"])
        cli.main(["trace", *common, "--chunk", "20", "--fwd-bs", "4"])
    assert len(list((tmp_path / "gen").glob("g*.jsonl"))) == len(list((tmp_path / "trace").glob("t*.npz"))) > 0 and len(list((tmp_path / "meta").glob("*.json"))) >= 6
    cli.main(["analyze", "--run-dir", str(tmp_path), "--block", "tables"])
    assert len(table.load_gen(tmp_path)) == 140
    for sh in (0, 1):                                                       # uncertainty POC stage through the CLI (gsm8k only exists here)
        cli.main(["unc-roll", "--run-dir", str(tmp_path), "--shard", str(sh), "--nshards", "2", "--unc-n", "30", "--n-roll", "3", "--n-cont", "2", "--max-new", "8", "--max-seqs", "16"])
    assert len(list((tmp_path / "unc_n30").glob("u*.jsonl"))) >= 2


# ----------------------------------------------------------------------------- uncertainty POC: sampling stage on the tiny model
def test_unc_stage(run, tok, model):
    from psuite import unc, unc_study
    tmp, problems = run
    G = table.load_gen(tmp)
    sub = unc.pick_subset(problems, G, 12)
    assert len(sub) == 24 and {p["ds"] for p in sub} == {"gsm8k", "math"}
    assert {p["id"] for p in unc.pick_subset(problems, G, 6)} <= {p["id"] for p in sub}              # nested subsets: n=40 is part of n=400
    out = tmp / "unc_t"
    kw = dict(n_roll=4, n_cont=3, max_seqs=16, chunk=5, cps=(2, 3), min_left=1, max_new=10, log=lambda *_: None)
    for sh in (0, 1):
        unc.run(sub, G, model, tok, out, sh, 2, **kw)
    files = sorted(p.name for p in out.glob("u*.jsonl"))
    unc.run(sub, G, model, tok, out, 0, 2, **kw)
    assert files == sorted(p.name for p in out.glob("u*.jsonl")) and files                              # resumable: nothing redone
    rows = unc_study.load_unc(out)
    assert len(rows) == 24
    for r in rows.values():
        assert len(r["roll_ans"]) == len(r["roll_fin"]) == len(r["roll_ntok"]) == 4 and max(r["roll_ntok"]) <= 10
        for t in (2, 3):                                           # a continuation exists exactly where the checkpoint is valid
            assert (str(t) in r["cont"]) == (t <= r["n_greedy"] - 1)
            if str(t) in r["cont"]:
                assert len(r["cont"][str(t)]["ans"]) == 3
    # per-token entropy of the greedy solution == a direct forward
    checked = 0
    for p in sub:
        comp = G[p["id"]]["plain_ids"]
        if len(comp) < 3:
            continue
        pr = enc(tok, chat_prompt(tok, plain_text(p)))
        with torch.no_grad():
            lg = model(input_ids=torch.tensor([pr + comp])).logits[0, len(pr) - 1:-1].float()
        lp = torch.log_softmax(lg, -1)
        ent = -(lp.exp() * lp).sum(-1)
        nll = -lp.gather(1, torch.tensor(comp)[:, None])[:, 0]
        r = rows[p["id"]]
        assert abs(r["ent_all"] - float(ent.mean())) < 1e-3 and abs(r["nll_all"] - float(nll.mean())) < 1e-3
        if len(comp) >= 3:
            assert abs(r["ent2"] - float(ent[:2].mean())) < 1e-3 and abs(r["ent3"] - float(ent[:3].mean())) < 1e-3
        checked += 1
    assert checked > 5
    # a continuation prompt given as token ids is the same prompt as the string (greedy => identical)
    p = sub[0]
    s = chat_prompt(tok, plain_text(p))
    a = generate_ids(model, tok, [s], 6, 1, 0.0)[0][0]
    b = generate_ids(model, tok, [enc(tok, s)], 6, 1, 0.0)[0][0]
    assert a == b

