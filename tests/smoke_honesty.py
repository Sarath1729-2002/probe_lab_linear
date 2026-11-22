"""CPU smoke test of organic/honesty.py on top of the sel smoke run (tiny random model; labels re-drawn).

    ERRPROP_DEVICE=cpu python tests/smoke_honesty.py /path/to/tiny_model
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
from organic import honesty as H  # noqa: E402
from organic import selective as S  # noqa: E402
from organic.common import read_jsonl, write_jsonl  # noqa: E402


def main(tiny):
    cfg = smoke_sel.main(tiny)
    cfg = dict(cfg, honesty={"n_steer_items": 6, "n_boot": 20})
    for tag in ("m1", "m2"):
        c = dict(cfg, model=tiny, run_prefix=f"{cfg['run_prefix']}/{tag}")
        H.stage_lie(c)
        rng = np.random.default_rng(3)
        L = read_jsonl(H._p(c, "lie.jsonl"))
        for r in L:
            r["correct"] = bool(rng.random() < 0.5)
            r["ans"] = "A" if r["correct"] is False else r.get("ans")
            r["lied"] = r["role"] == "lie" and not r["correct"]
        write_jsonl(H._p(c, "lie.jsonl"), L)
        items = S.load_items(c)
        it = next(v for v in items.values() if v["set"] == "test")
        assert H.LIE in H.lie_rows(it)[1]["msgs"][0]["content"] and H.HONEST in H.lie_rows(it)[0]["msgs"][0]["content"]
        m = H.post_msgs([{"role": "user", "content": "q"}], "B", "C")
        assert [x["role"] for x in m] == ["user", "assistant", "user"] and "(C)" in m[-1]["content"]
        H.stage_post(c)
        P = read_jsonl(H._p(c, "post.jsonl"))
        assert {r["key"] for r in P} == {"gold", "other"} and all(0 <= r["p_yes"] <= 1 for r in P)
        assert {"base", "lie", "honest", "wrong", "right", "doubt"} & {r["role"] for r in P} >= {"base", "lie", "honest"}
        Hf = H.stage_hfit(c)
        assert Hf["geometry"] and Hf["h_layers"] and "test|bare (right)" in Hf["table"] or "test|bare (wrong)" in Hf["table"]
        M = H.stage_hsteer(c)
        assert {"none", "ablate_d", "ablate_h", "ablate_rand", "ablate_t"} <= set(M)
        H.stage_hsteer(c)                                                 # resumable
        txt = H.write_report(c).read_text()
        for must in ("Pre-registered", "still know", "Layer profile", "Geometry", "Causal"):
            assert must in txt, must
        assert set(json.load(open(H._p(c, "verdict_hon.json")))) >= {"H3 shared dishonesty basis", "H4 deference disjoint from truth"}
    txt = H.write_compare(cfg, ["m1", "m2", "missing"]).read_text()
    assert "Hypotheses" in txt and "Headline" in txt
    print(txt)
    print("\nSMOKE HONESTY OK")


if __name__ == "__main__":
    main(sys.argv[1])
