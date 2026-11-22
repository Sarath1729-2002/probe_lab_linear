"""CPU smoke test of organic/sel_more.py on top of the sel smoke run (tiny random model).

    ERRPROP_DEVICE=cpu python tests/smoke_more.py /path/to/tiny_model
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

import smoke_sel  # noqa: E402
from organic import sel_more as M  # noqa: E402
from organic import selective as S  # noqa: E402
from organic.common import read_jsonl  # noqa: E402


def main(tiny):
    cfg = smoke_sel.main(tiny)
    cfg = dict(cfg, more={"n_causal_items": 4, "causal_alpha": 3.0, "lora_n": 24, "lora_batch": 4, "lora_max_len": 256,
                          "lora_variants": ["anti", "sel", "mix25", "bal", "bal_s1", "bal_s2", "mix75"]})
    for tag in ("m1", "m2"):
        c = dict(cfg, model=tiny, run_prefix=f"{cfg['run_prefix']}/{tag}")
        M.stage_verify(c)
        V = read_jsonl(M._p(c, "verify.jsonl"))
        items = S.load_items(c)
        assert {r["key"] for r in V} >= {"gold", "dist"} and all(0 <= r["p_yes"] <= 1 for r in V)
        assert len({r["item"] for r in V}) == len(items)
        it = next(v for v in items.values() if v["set"] == "ays")
        assert "Proposed answer: (" in M.verify_text(it, "B") and "Please answer just" not in M.verify_text(it, "B")
        F = M.stage_vfit(c)
        assert "test|model wrong" in F["gap"] and set(F["taus"]) == {"policy_verify", "policy_truth", "policy_combo", "policy_lp"}
        M.stage_veval(c)
        C = M.stage_causal(c)
        assert len(C["layers"]) >= 1
        M.stage_ceval(c)
        M.stage_lora(c)
        M.stage_veval(c)                                                  # resumable
        for cond in M.NEW:
            assert M._p(c, f"eval_{cond}.jsonl").exists(), cond
        txt = M.write_report(c).read_text()
        for must in ("Verdicts", "## V.", "## C.", "## ays", "policy", "LoRA"):
            assert must in txt, must
        assert set(json.load(open(M._p(c, "verdict_more.json")))) >= {"V checking gap", "G claim gate", "L fine-tune", "C causal layers"}
    txt = M.write_compare(cfg, ["m1", "m2", "m3"]).read_text()          # m3 has no results: must be skipped
    for must in ("frontier", "3 training seeds", "shared geometry", "decode vs layers that steer", "## test"):
        assert must in txt, must
    print("\nSMOKE MORE OK")


if __name__ == "__main__":
    main(sys.argv[1])
