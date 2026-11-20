"""CPU smoke test of organic/audit.py on top of the v6 smoke pipeline (tiny random model).

    python tests/smoke_audit.py /path/to/tiny_model
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import smoke_organic_v6 as S6  # noqa: E402
from organic import audit as AU  # noqa: E402


def main(tiny):
    cfg, tmp, tags = S6.main(tiny)
    for t in tags:
        c = dict(cfg, model=tiny, run_prefix=f"{cfg['run_prefix']}/{t}")
        A = AU.stage_a1(c)
        P = A["positions"]["last_prompt_token"]
        assert P["families"] and all(k.split("@")[0] in ("logistic", "mlp", "ridge_margin", "paired") for k in P["families"])
        mm = next(p for p in P["mass_mean_per_layer"] if p["layer"] == P["best_layer"])
        assert mm["pooled"] > 0.8 and 0 <= mm["var_item"] <= 1, mm
        B = AU.stage_a2(c)
        assert B["n_items"] > 5 and "probe on BARE question" in B["spearman_fragility"]
        AU.stage_a2(c)                                         # reuses stored bare-question activations
        txt = AU.write_report_audit(c).read_text()
        assert "## KT1" in txt and "## KT2" in txt
        print(txt)
    print("\nSMOKE AUDIT OK")


if __name__ == "__main__":
    main(sys.argv[1])
