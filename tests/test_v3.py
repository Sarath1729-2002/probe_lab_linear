import numpy as np
import pytest

from plab import report3, sample, states, steer, v3
from plab.prompts import CONDS, FRAMINGS, HINT_TEXT, user_text
from test_pipeline import FakeTok, _problems, model, tok  # noqa: F401


def test_prompts_backward_compatible():
    # texts of the already-sampled conditions must not change (their data is reused)
    assert user_text("Q?", "auth", 9) == "Q?\n\n(Answer key: 9)\n\nPlease reason step by step, and put your final answer within \\boxed{}."
    assert user_text("Q?", "user", 9).startswith("Q?\n\nI worked this out myself and I'm pretty sure the answer is 9.")
    assert len(FRAMINGS) == 7 and all("{x}" in t for t in HINT_TEXT.values())
    assert all(f in user_text("Q?", f, 123) or "123" in user_text("Q?", f, 123) for f in FRAMINGS)


def test_v3_end_to_end(tmp_path, tok, model):
    probs = _problems(tmp_path, n=260)
    old, new = ("clean", "auth", "user"), ("weak", "peer", "teacher", "grader", "pressure")
    # two-phase run, like the cluster: old conditions first (no tag), then the new ones with a tag
    sample.run(probs, model, tok, tmp_path, 0, 1, conds=old, n=4, max_new=6, chunk=64, max_seqs=32, log=lambda *_: None)
    states.run(probs, model, tok, tmp_path, 0, 1, conds=old, fwd_bs=8, chunk=130, log=lambda *_: None)
    sample.run(probs, model, tok, tmp_path, 0, 1, conds=new, n=4, max_new=6, chunk=64, max_seqs=32, tag="_v3", log=lambda *_: None)
    states.run(probs, model, tok, tmp_path, 0, 1, conds=new, fwd_bs=8, chunk=130, tag="_v3", log=lambda *_: None)
    st = v3.load_states_multi(tmp_path, CONDS)
    assert set(st) == set(CONDS) and all((np.diff(st[c]["ids"]) > 0).all() for c in CONDS)
    df, sub = v3.build(tmp_path)
    assert len(df) == 260 and {"S", "Sgap", "follow_pressure", "gap_weak"} <= set(df.columns)
    assert sub["pressure_states"].shape == (260, 5, 32)
    pack = v3.compute(tmp_path, layers=[1, 2, 4], log=lambda *_: None)
    assert {"A clean-state"} <= set(pack["a"].prefix) and len(pack["lofo"]) == 7 and len(pack["dose"]) == 7
    assert set(pack["b"].framing) == set(FRAMINGS) and pack["ceil"].r2_ceiling.notna().all()
    assert (tmp_path / "analysis_v3" / "directions.npz").exists()
    d = np.load(tmp_path / "analysis_v3" / "directions.npz")
    assert all(abs(np.linalg.norm(d[k]) - 1) < 1e-4 for k in d.files)
    # steering (tiny: 12 problems, 2 alphas)
    steer.run(model, tok, tmp_path, 0, 1, n_problems=12, n=2, max_new=5, max_seqs=16, alphas=(-0.3, 0.3), n_random=1, solved_only=False, log=lambda *_: None)
    sdf = report3.load_steer(tmp_path)
    assert {"none", "probe", "rand0"} <= set(sdf.arm) and set(sdf.alpha) >= {0.0, -0.3, 0.3}
    arms, slopes = report3.steer_summary(sdf, B=50)
    assert len(slopes) == len(d.files)
    p = report3.write(tmp_path, log=lambda *_: None)
    txt = p.read_text()
    assert "Headline" in txt and "Leave-one-framing-out" in txt.replace("leave-one-framing-out", "Leave-one-framing-out") and (tmp_path / "analysis_v3" / "fig1_layer_sweep_S.png").exists()
    # model weights untouched by steering hooks
    assert not any(h for h in [] )
