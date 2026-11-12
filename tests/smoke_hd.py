"""CPU smoke test of organic/honest_defer.py on top of a finished v6 smoke run (tiny random model).
The v6 smoke plants a deference direction in the activations, so Q1 must find it on unseen datasets; the honesty
reads, steering hooks (every condition), grading and reports must all run.

    python tests/smoke_hd.py /path/to/tiny_model
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import smoke_organic_v6 as S6  # noqa: E402
from organic import honest_defer as HD  # noqa: E402
from organic.common import read_jsonl  # noqa: E402


def main(tiny):
    cfg, tmp, tags = S6.main(tiny)
    cfg = dict(cfg, hd={"n_honesty": 40, "n_per_bucket": 3, "batch": 4, "conditions": HD.CONDS})
    for t in tags:
        c = dict(cfg, model=tiny, run_prefix=f"{cfg['run_prefix']}/{t}")
        G = HD.stage_hd_dirs(c)
        p = G["per_layer"][G["steer"]["layer_index"]]
        assert p["lodo_mean"] > 0.8 and p["def_auroc"] > 0.8, p
        assert G["steer"]["layer"] >= 1
        HD.stage_hd_steer(c, ["none", "def_gate", "hon_gate"])
        HD.stage_hd_steer(c, ["def_uniform", "hon_uniform", "rand_uniform"])
        HD.stage_hd_steer(c, ["none"])                                    # resumable: nothing redone
        n = read_jsonl(HD.hdir(c) / "steer_none.jsonl")
        u = read_jsonl(HD.hdir(c) / "steer_def_uniform.jsonl")
        assert len(n) == len(u) > 0 and len({r["vid"] for r in n}) == len(n)
        assert any(a["text"] != b["text"] for a, b in zip(sorted(n, key=lambda r: r["vid"]), sorted(u, key=lambda r: r["vid"]))), \
            "steering must change at least some replies"
        f = HD.write_report_hd(c)
        txt = f.read_text()
        for must in ("## Summary", "## 1. Is the deference direction", "## 2. Is it the honesty", "## 3. Steering"):
            assert must in txt, must
        print(txt[:1500])
    cmp = HD.write_compare_hd(dict(cfg), tags + ["qwen14b"]).read_text()
    assert all(t in cmp for t in tags)
    print(cmp)
    print("\nSMOKE HD OK")


if __name__ == "__main__":
    main(sys.argv[1])
