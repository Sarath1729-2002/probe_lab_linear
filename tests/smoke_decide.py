"""CPU smoke test of the whole decision run on a tiny random model: data -> a1 -> a2 -> b -> c -> fit -> report.

A random model can't know or follow anything, so after each generation stage its labels are replaced
by planted ones; after b_extract the states are replaced by ones with a planted knowledge confound
(truth readable on known facts only), so the B criterion must come out LIVE.

    python tests/smoke_decide.py /path/to/tiny_model [/path/to/ext_clones]
"""
from __future__ import annotations

import json
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("ERRPROP_DEVICE", "cpu")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402

from errprop.cli import select_model  # noqa: E402
from errprop.io import load_config, read_jsonl, write_jsonl  # noqa: E402
from decide import thread_a, thread_b, thread_c  # noqa: E402
from decide.common import run_dir  # noqa: E402
from decide.report import write_report  # noqa: E402
from synthetic_paper import fake_model  # noqa: E402


def mock_data(dd: Path, ext: Path | None):
    dd.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    props = ["occupation", "place of birth", "country", "capital", "genre", "author"]
    rows = []
    for i in range(900):
        p = props[i % len(props)]
        o = f"Obj{p[:3]}{rng.randint(0, 40)}"
        rows.append({"id": i, "prop": p, "subj": f"Subject {i}", "obj": o, "answers": [o],
                     "s_pop": rng.randint(1, 100000), "question": f"What is the {p} of Subject {i}?"})
    write_jsonl(dd / "popqa.jsonl", rows)
    for sub, src, files in (("truth_is_universal", "Truth_is_Universal/datasets", ["cities.csv", "neg_cities.csv"]),
                            ("poulis", "llm-truth-directions-limits/datasets/arithmetic",
                             ["arith_1op.csv", "arith_2ops.csv", "arith_3ops.csv"])):
        (dd / sub).mkdir(exist_ok=True)
        for f in files:
            s = ext / src / f if ext else None
            if s and s.exists():
                lines = s.read_text().splitlines()
                (dd / sub / f).write_text("\n".join(lines[:301]) + "\n")
    if not (dd / "truth_is_universal" / "cities.csv").exists():      # no clones: tiny stand-ins
        for fam, neg in (("cities", ""), ("neg_cities", "not ")):
            with open(dd / "truth_is_universal" / f"{fam}.csv", "w") as f:
                f.write("statement,label,city,country,correct_country\n")
                for i in range(150):
                    c, w = f"Country{i % 30}", f"Country{(i + 7) % 30}"
                    for lab, cc in ((1, c), (0, w)):
                        f.write(f"The city of Town{i} is {neg}in {cc}.,{lab if not neg else 1 - lab},Town{i},{cc},{c}\n")


def main(tiny: str, ext: str | None):
    tmp = Path(tempfile.mkdtemp())
    # ---- fake paper_v1 run for qwen7b (A1 / A2 read it)
    pcfg = load_config(str(ROOT / "config_paper.yaml"), {"out_dir": str(tmp / "errprop_runs")})
    pcfg = select_model(pcfg, "qwen7b")
    fake_model(pcfg, 0)
    # the fake prompts are placeholders; give A2 a real question per problem
    from transformers import AutoTokenizer  # noqa: F401
    cfg = load_config(str(ROOT / "config_decide.yaml"),
                      {"out_dir": str(tmp / "decide_runs"), "data_dir": str(tmp / "decide_data"),
                       "report_dir": str(tmp / "reports"), "paper_out_dir": str(tmp / "errprop_runs"),
                       "gen.backend": "hf", "a2.n_items": 160, "a2.max_tokens": 4, "a2.batch": 16,
                       "b.popqa_n": 300, "b.know_max_tokens": 4, "c.max_tokens": 4})
    cfg["models"] = [dict(cfg["models"][0], name=tiny)]
    tag = "qwen7b"
    dd = tmp / "decide_data"
    mock_data(dd, Path(ext) if ext else None)
    thread_b.build_items(cfg)
    thread_c.build_items(cfg)
    rd = run_dir(cfg, tag)

    # ---- A
    thread_a.stage_a1(cfg, tag)
    a1 = json.load(open(rd / "a1.json"))
    assert "pooled" in a1, a1
    print("a1 pooled:", {k: round(v, 3) for k, v in a1["pooled"]["at_detection_layer"].items()
                         if isinstance(v, float)})
    thread_a.stage_a2_gen(cfg, tag, tiny)
    items = read_jsonl(rd / "a2_items.jsonl")
    assert items and "prompt_corrupt" in items[0]
    r = random.Random(2)
    for it in items:                                   # plant: user attribution lowers FOLLOW
        it["clean_ok"] = True
        it["user_outcome"] = it["self_outcome"] if r.random() < 0.5 else "RECOVER"
    write_jsonl(rd / "a2_items.jsonl", items)
    thread_a.stage_a2_extract(cfg, tag, tiny)

    # ---- B
    thread_b.stage_b_know(cfg, tag, tiny)
    bitems = read_jsonl(dd / "b_items.jsonl")
    groups = sorted({it["group"] for it in bitems})
    know = {g: (1 if random.Random(g).random() < 0.6 else 0) for g in groups}
    json.dump(know, open(rd / "b_know.json", "w"))
    thread_b.stage_b_extract(cfg, tag, tiny)
    H = np.load(rd / "b_acts.npy").astype(np.float32)
    nr = np.random.default_rng(0)
    H = nr.normal(size=H.shape).astype(np.float32)
    for j, it in enumerate(bitems):                    # plant: truth readable only where the fact is known
        if know[it["group"]] == 1:
            H[j, :, 0] += 2.0 * (2 * it["label"] - 1)
    np.save(rd / "b_acts.npy", H.astype(np.float16))

    # ---- C
    thread_c.stage_c_gen(cfg, tag, tiny)
    citems = read_jsonl(rd / "c_items.jsonl")
    for it in citems:
        it["known"] = True
        for cond in ("opt", "plain"):
            it[f"label_{cond}"] = (r.choice(["ADOPT", "REJECT", "ADOPT", "OVERRIDE"]) if it["false"] else "CORRECT")
    write_jsonl(rd / "c_items.jsonl", citems)
    thread_c.stage_c_extract(cfg, tag, tiny)

    # ---- fit + report through the CLI
    from decide import cli
    for argv in (["fit", "--model", tag], ["report"]):
        sys.argv = ["decide.cli", *argv, "--config", str(ROOT / "config_decide.yaml"), "--set",
                    f"out_dir={tmp / 'decide_runs'}", f"data_dir={dd}", f"report_dir={tmp / 'reports'}",
                    f"paper_out_dir={tmp / 'errprop_runs'}"]
        cli.main()
    a2 = json.load(open(rd / "a2.json"))
    b = json.load(open(rd / "b.json"))
    c = json.load(open(rd / "c.json"))
    print("a2:", {k: a2.get(k) for k in ("n_paired", "follow_rate_self", "follow_rate_user", "follow_diff_ci")})
    assert a2["follow_rate_user"] < a2["follow_rate_self"]
    r1 = b["B1_popqa"]["logistic"]["result"]
    print("b1:", {k: r1[k] for k in ("n_known", "n_unknown", "auroc_known_matched", "auroc_unknown", "gap_matched", "gap_ci")})
    assert r1["gap_matched"] > 0.2 and r1["gap_ci"][0] > 0, r1
    print("b families:", b["families"])
    print("c:", c["opt"]["adopt_rate"], c["opt"]["reject_or_override_rate"], c["passes_gate_any"])
    assert c["passes_gate_any"]
    txt = (tmp / "reports" / "decide_v1" / "DECISION.md").read_text()
    for must in ("## A.", "## B.", "## C.", "## Verdict", "PopQA → PopQA"):
        assert must in txt, must
    print(txt[:3000])
    print("\nSMOKE DECIDE OK ->", tmp / "reports" / "decide_v1" / "DECISION.md")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
