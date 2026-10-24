import json
from pathlib import Path

import numpy as np
import pytest
import torch

from plab import analysis, data, introspect, sample, states
from plab.prompts import chat_prompt, parse_boxed, same, user_text
from plab.ridge import fit_layer

CHARS = list(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,=+-*/\\{}()$:#?!<>|_'\n;%\"&")


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
                out.append(self.idx[s[i]]); i += 1
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
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=2048, eos_token_id=1, pad_token_id=0, tie_word_embeddings=False)
    return Qwen2ForCausalLM(cfg).eval()


def test_parsing():
    assert data.parse_gold("blah <<1+1=2>> #### 1,234") == 1234
    assert data.parse_gold("#### 3.5") is None
    assert parse_boxed("so \\boxed{\\$1,250} done") == 1250
    assert parse_boxed("\\boxed{\\frac{1}{2}} then \\boxed{12 apples}") == 12
    assert parse_boxed("no answer") is None
    assert same(18.0, 18) and not same(None, 3)
    h = data.hint_wrong(5, 7)
    assert h != 7 and h >= 0 and data.hint_wrong(5, 7) == h
    assert "Answer key: 9" in user_text("Q?", "auth", 9) and "9" in user_text("Q?", "user", 9) and "9" not in user_text("Q?", "clean", 9)


def _problems(tmp_path, n=90):
    rng = np.random.RandomState(0)
    raw = {"train": [], "test": []}
    for sp in raw:
        for i in range(n // 2):
            a, b = int(rng.randint(2, 30)), int(rng.randint(2, 30))
            steps = int(rng.randint(1, 8))
            raw[sp].append({"question": f"Tom has {a} apples and gets {b} more. How many apples? (variant {i})",
                            "answer": "".join(f"step <<{a}+{b}={a + b}>> \n" for _ in range(steps)) + f"#### {a + b}"})
    data.build_problems(tmp_path, raw, n_train_extra=n // 2, log=lambda *_: None)
    return data.read_problems(tmp_path)


def test_end_to_end(tmp_path, tok, model):
    probs = _problems(tmp_path)
    assert {p["split"] for p in probs} == {"fit", "val", "test"}
    for sh in (0, 1):                                   # two shards, resumable
        sample.run(probs, model, tok, tmp_path, shard=sh, nshards=2, n=4, max_new=12, chunk=20, max_seqs=16, log=lambda *_: None)
        states.run(probs, model, tok, tmp_path, shard=sh, nshards=2, fwd_bs=5, chunk=25, log=lambda *_: None)
    before = sorted(p.name for p in (tmp_path / "samples").iterdir())
    sample.run(probs, model, tok, tmp_path, 0, 2, n=4, max_new=12, chunk=20, log=lambda *_: None)
    assert before == sorted(p.name for p in (tmp_path / "samples").iterdir())          # nothing redone
    st = analysis.load_states(tmp_path, ("clean", "auth", "user"))
    assert st["clean_states"].shape == (len(probs), 5, 32) and (np.diff(st["ids"]) > 0).all()
    assert np.isfinite(st["clean_lp_gold"]).all() and np.isfinite(st["auth_lp_hint"]).all() and np.isnan(st["clean_lp_hint"]).all()
    # prompt-end state really is the last prompt token: recompute one directly
    from plab.hf_util import enc
    p = probs[7]
    ids = torch.tensor([enc(tok, chat_prompt(tok, user_text(p["question"], "auth", p["hint"])))])
    with torch.no_grad():
        hs = model(input_ids=ids, output_hidden_states=True).hidden_states
    ref = torch.stack([h[0, -1] for h in hs]).numpy()
    np.testing.assert_allclose(st["auth_states"][7].astype(np.float32), ref, atol=2e-2, rtol=2e-2)
    s = analysis.load_samples(tmp_path)
    r = s[(7, "auth")]
    assert r["n"] == 4 and len(r["answers"]) == 4 and 0 <= r["n_follow"] <= 4
    res = analysis.analyse_main(tmp_path, layers=[1, 2, 4], regimes=("iid", "shift"), log=lambda *_: None)
    sm = res["summary"]
    assert {"E1", "E2[auth] state=auth", "E2[auth] state=clean", "E2 transfer auth->user"} <= set(sm.prefix)
    assert sm.probe_layer.isin([1, 2, 4]).all()
    # E3
    introspect.run(model, tok, tmp_path, concepts=["ocean", "fire", "winter", "music", "forest", "hospital", "castle", "robot", "garden", "volcano"],
                   strengths=[0.5, 2.0], n_null=20, gen_tokens=4, log=lambda *_: None)
    z = np.load(tmp_path / "introspect.npz")
    assert z["states"].shape[1:] == (5, 32)
    intro = analysis.analyse_introspect(tmp_path, layers=[1, 2, 3, 4], log=lambda *_: None)
    assert len(intro["table"]) and set(intro["curve"].columns) >= {"auroc_injected", "causally_downstream"}
    p = analysis.write_report(tmp_path, res, intro, log=lambda *_: None)
    txt = p.read_text()
    assert "E3" in txt and "E1 knowledge boundary" in txt and ("PASS" in txt or "FAIL" in txt or "weak" in txt)
    from plab import cli
    cli.main(["analyze", "--run-dir", str(tmp_path)])                  # the CLI path the cluster uses


def test_injection_is_causal(tok, model):
    prompt = chat_prompt(tok, "Trial 5.\n\nhello there")
    vec = torch.randn(32)
    yes, no = [introspect.enc(tok, "Y")[0]], [introspect.enc(tok, "N")[0]]
    n_before = len(model.model.layers[2]._forward_hooks)
    with introspect.Injector(model, 3) as inj:
        st0, g0, _ = introspect.one_trial(model, tok, prompt, inj, None, yes, no, 2)
        st1, g1, _ = introspect.one_trial(model, tok, prompt, inj, vec * 5, yes, no, 2)
    assert np.array_equal(st0[:3], st1[:3])                  # embedding + blocks 1,2 untouched
    assert not np.allclose(st0[3:].astype(np.float32), st1[3:].astype(np.float32))
    assert len(model.model.layers[2]._forward_hooks) == n_before         # our hooks removed
    assert g0 != g1


def test_ridge_recovers_signal():
    rng = np.random.RandomState(0)
    X = rng.randn(400, 20).astype(np.float32)
    y = X[:, 0] * 2 + 0.1 * rng.randn(400)
    res = analysis.ridge_eval(X[:300], X[300:350], X[350:], y[:300, None], y[300:350, None], y[350:, None])
    assert res[0]["test_r2"] > 0.9
    assert abs(analysis.auroc([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.9]) - 1) < 1e-9
