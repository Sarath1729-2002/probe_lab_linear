"""CPU smoke test of organic/kill.py on top of the v6 + steering v2 smoke pipeline (tiny random model).

    python tests/smoke_kill.py /path/to/tiny_model
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import smoke_organic_v6 as S6  # noqa: E402
from organic import kill as K  # noqa: E402
from organic import steer2 as S2  # noqa: E402


def main(tiny):
    cfg, tmp, tags = S6.main(tiny)
    cfg = dict(cfg, s2={"n_per_bucket": 3, "n_per_bucket_train": 3, "batch": 4, "gate_quantile": 0.95}, kill={"n_per_bucket": 3})
    for t in tags:
        c = dict(cfg, model=tiny, run_prefix=f"{cfg['run_prefix']}/{t}")
        S2.stage_s2_dirs(c)
        S2.stage_s2_calib(c)
        R2 = K.stage_k2(c)
        b = R2["at_best_layer"]
        assert b["pooled"] > 0.8, b                      # planted direction
        assert R2["source_prior"]["within_item"] == 0.5 or R2["source_prior"]["within_item"] is not None
        _, sel = K._unknown_sel(c)
        tr = set(json.load(open(S2.sdir(c) / "dirs.json"))["train_items"])
        assert sel and not ({v["item"] for v in sel} & tr) and {v["role"] for v in sel} == {"right"}
        K1 = K.stage_k1(c)
        assert set(K1) == {"m_user", "m_document", "both"} and "gate" in K1["both"]
        txt = K.write_report_kill(c).read_text()
        assert "## K1" in txt and "## K2" in txt
        print(txt)
    print("\nSMOKE KILL OK")


if __name__ == "__main__":
    main(sys.argv[1])
