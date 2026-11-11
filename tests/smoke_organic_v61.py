"""CPU smoke test of the v6.1 checks, on top of a finished v6 smoke run (tiny random model):
variants61 appends prompts without touching v6 ones -> gen only the new prompts -> extend cut-off replies ->
deference + v6 report still run -> report61 / compare61 recover planted effects.

Planted: Qwen-like style, where "bare" replies defer more; under reason-first everything is reasoned and the
person-minus-source gap is zero; `Note:` behaves like the answer key; half of "Are you sure?" replies never finish.

    python tests/smoke_organic_v61.py /path/to/tiny_model
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import smoke_organic_v6 as S6  # noqa: E402
from organic import design61 as D61  # noqa: E402
from organic.common import read_jsonl, write_jsonl  # noqa: E402
from organic.deference import stage_deference  # noqa: E402
from organic.report_v6 import write_report_v6  # noqa: E402
from organic.run import bdir, is_probe_family, stage_extend, stage_gen  # noqa: E402

P = {"m_none": .5, "m_key": .5, "m_bare": .2, "m_user": .2, "m_expert": .3, "m_document": .5, "m_tool": .45, "m_peer": .15}


def main(tiny):
    cfg, tmp, tags = S6.main(tiny)
    dd = Path(cfg["data_dir"])
    n6 = len(read_jsonl(dd / "variants.jsonl"))
    before = {v["vid"]: v["user"] for v in read_jsonl(dd / "variants.jsonl")}
    cfg = dict(cfg, gen=dict(cfg["gen"], max_tokens_reason={"default": 6}, extend_to={"default": 8}))
    D61.stage_variants61(cfg, dd)
    D61.stage_variants61(cfg, dd)                                   # idempotent
    V = read_jsonl(dd / "variants.jsonl")
    assert all(before[v["vid"]] == v["user"] for v in V if v["vid"] in before), "v6 prompts must not change"
    new = [v for v in V if v["vid"] not in before]
    assert len(V) == n6 + len(new) and {v["family"] for v in new} == set(D61.NEW_FAMS), {v["family"] for v in new}
    assert not any(is_probe_family(v["family"]) for v in new)
    nb = next(v for v in new if v["family"] == "m_bare" and v["role"] == "wrong")
    nn = before[nb["vid"].replace("m_bare", "m_none")]
    assert nn.replace('Note: "The answer is', '"The answer is') == nb["user"], (nn, nb["user"])
    ru = next(v for v in new if v["family"] == "r_user")
    assert "step by step" in ru["user"] and 'I say: "The answer is' in ru["user"]
    for tag in tags:
        c = dict(cfg, model=tiny, run_prefix=f"{cfg['run_prefix']}/{tag}")
        n_old = len(read_jsonl(bdir(c, "mmlu") / "gen.jsonl"))
        stage_gen(c, S6.BUCKETS)
        n_mid = len(read_jsonl(bdir(c, "mmlu") / "gen.jsonl"))
        assert n_mid == n_old + sum(v["bucket"] == "mmlu" for v in new), (n_old, n_mid)
        stage_extend(c, S6.BUCKETS)
        rows = read_jsonl(bdir(c, "mmlu") / "gen.jsonl")
        assert any(r.get("extended") for r in rows) and all("text_v0" in r for r in rows if r.get("extended"))
        stage_extend(c, S6.BUCKETS)                                 # nothing re-extended
        for b in S6.BUCKETS:                                        # plant behaviour on all replies
            rows = read_jsonl(bdir(c, b) / "gen.jsonl")
            for r in rows:
                iid, fam, role = r["vid"].split("|")
                h = random.Random(iid + tag).random()
                k = random.Random(r["vid"] + "s").random()
                if fam in ("base", "r_base"):
                    r["outcome"], r["text"] = ("GOLD" if h < .85 else "OTHER"), "Answer: x"
                elif fam == "c_doubt":
                    fin = k < .5
                    r["outcome"] = "GOLD" if (fin and random.Random(r["vid"] + "f").random() < .8) else "OTHER"
                    r["text"] = "Hmm, let me reconsider this again. Answer: x" if fin else "loop " * 40
                elif fam.startswith("r_") and role == "wrong":
                    r["outcome"] = "WRONG" if h < .3 else "GOLD"
                    r["text"] = "Step one, I consider the question carefully. Answer: x"
                elif fam in P and role == "wrong":
                    bare = k < .6
                    r["outcome"] = "WRONG" if h < P[fam] else "GOLD"
                    r["text"] = "Answer: x" if bare else "I consider it carefully and conclude. Answer: x"
                elif role == "right":
                    r["outcome"] = "GOLD"
                else:
                    r["outcome"] = "WRONG" if h < .3 else "GOLD"
            write_jsonl(bdir(c, b) / "gen.jsonl", rows)
        stage_deference(c)
        write_report_v6(c)
        D61.write_report61(c)
        A = D61.analyse61(c)
        rf = A["reason_first"]
        print(tag, "person-source default", round(rf["person_minus_source"]["default"]["diff"], 3),
              "reason-first", round(rf["person_minus_source"]["reason_first"]["diff"], 3),
              "| key-none", round(A["controls"]["key − none"]["diff"], 3), "bare-none", round(A["controls"]["bare − none"]["diff"], 3),
              "| flip all/final", round(A["flipflop"]["flip_rate_all"], 2), round(A["flipflop"]["flip_rate_final_only"], 2))
        assert rf["person_minus_source"]["default"]["diff"] < -0.05
        assert abs(rf["person_minus_source"]["reason_first"]["diff"]) < 1e-9
        assert abs(A["controls"]["key − none"]["diff"]) < 1e-9 and A["controls"]["bare − none"]["diff"] < -0.2
        assert 0.4 < A["flipflop"]["no_final_answer"] < 0.6 and 0.1 < A["flipflop"]["flip_rate_final_only"] < 0.3
        assert 0.5 < A["style"]["m_user"]["bare"] < 0.7 and A["style"]["r_user"]["bare"] == 0
        txt = (tmp / "reports" / c["run_prefix"] / "REPORT_v61.md").read_text()
        for must in ("## 1. Reply style", "## 2. Is the no-source line", "## 3. Reason first", "## 4. Flip rate", "## 5. Cut-off"):
            assert must in txt, must
    cmp = D61.write_compare61(dict(cfg, run_prefix=cfg["run_prefix"]), tags + ["qwen14b"]).read_text()
    assert "reason first" in cmp and (tmp / "reports" / cfg["run_prefix"] / "fig_reason_first.png").exists()
    print(txt[:1800])
    print(cmp)
    print("\nSMOKE ORGANIC v6.1 OK")


if __name__ == "__main__":
    main(sys.argv[1])
