"""CPU smoke test of organic/steer2.py on top of a finished v6 smoke run (tiny random model): directions on the training
half, gate calibration on generated tokens, every steering condition on the test half, report and comparison.

    python tests/smoke_s2.py /path/to/tiny_model
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import smoke_organic_v6 as S6  # noqa: E402
from organic import steer2 as S2  # noqa: E402
from organic.common import read_jsonl  # noqa: E402


def main(tiny):
    cfg, tmp, tags = S6.main(tiny)
    cfg = dict(cfg, s2={"n_per_bucket": 3, "n_per_bucket_train": 3, "batch": 4, "gate_quantile": 0.95, "conditions": S2.CONDS})
    for t in tags:
        c = dict(cfg, model=tiny, run_prefix=f"{cfg['run_prefix']}/{t}")
        D = S2.stage_s2_dirs(c)
        assert len(D["layers"]) >= 2 and all(L >= 1 for L in D["layers"]), D["layers"]
        assert all(p["test_auroc"] > 0.8 for p in D["per_layer"]), D["per_layer"]          # planted direction, held-out items
        _, tr = S2._sel(c, True)
        _, te = S2._sel(c, False)
        assert not ({v["item"] for v in tr} & {v["item"] for v in te}), "train and test items must not overlap"
        assert {v["family"] for v in te} >= {"base", "m_user", "m_document", "c_user", "c_doubt"}, {v["family"] for v in te}
        C = S2.stage_s2_calib(c)
        assert all(p["tau"] == p["tau"] for p in C["per_layer"])
        S2.stage_s2_steer(c, ["none", "ablate", "gate", "rand_2"])
        S2.stage_s2_steer(c, ["add_0.5", "add_1", "add_2"])
        S2.stage_s2_steer(c, ["none"])                                     # resumable
        n = {r["vid"]: r["text"] for r in read_jsonl(S2.sdir(c) / "steer_none.jsonl")}
        a = {r["vid"]: r["text"] for r in read_jsonl(S2.sdir(c) / "steer_add_2.jsonl")}
        assert set(n) == set(a) and len(n) == len(te) and any(n[k] != a[k] for k in n), "steering must change some replies"
        txt = S2.write_report_s2(c).read_text()
        for must in ("## Trade-off", "## Every metric", "flips after"):
            assert must in txt, must
        A = json.load(open(S2.sdir(c) / "summary.json"))
        assert set(A) == set(S2.CONDS) and A["none"]["pressure cut (mean, points)"] == 0
        print(txt[:2500])
    cmp = S2.write_compare_s2(dict(cfg), tags + ["qwen14b"]).read_text()
    assert all(t in cmp for t in tags)
    assert (tmp / "reports" / cfg["run_prefix"] / "fig_s2_tradeoff.png").exists()
    print(cmp)
    print("\nSMOKE S2 OK")


if __name__ == "__main__":
    main(sys.argv[1])
