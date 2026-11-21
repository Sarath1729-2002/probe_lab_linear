"""Sanity check for the selective-deference run: does steering beat a policy that never touches activations?

Policy(tau): under pressure, keep the model's own bare-question answer if its confidence >= tau, else return the
model's ordinary (unsteered) pressured reply. Confidence = output probability of its own answer (c_lp), or the
bare-question probe (half fitted without the tuning items). tau is chosen on the VALIDATION half of the training
renditions (press.jsonl), same objective as the alphas. Then scored on test / ays / answer with the same metrics.

Also: "confidence-behaviour gap" = selectivity the model could reach by using its own confidence (policy) minus what
it actually does (none). CPU only; reads the finished sel_v1 run.

    python -m organic.sel_policy --config config_sel.yaml --models qwen7b llama8b mistral7b qwen3_8b
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from . import selective as S
from .common import load_config, read_jsonl

TAUS = np.round(np.arange(0.30, 1.0001, 0.025), 3)


def _apply(rows, B, conf, tau):
    out = []
    for r in rows:
        if r["role"] != "base" and conf.get(r["item"], 0.0) >= tau:
            b = B[r["item"]]
            out.append(dict(r, correct=b["correct"], ans=b.get("ans")))
        else:
            out.append(r)
    return out


def run_model(cfg) -> dict:
    items, B = S.load_items(cfg), S._base(cfg)
    F = json.load(open(S.rdir(cfg) / "fit.json"))
    fit_h = set(F["fit_items"])
    val = [i for i, it in items.items() if it["set"] == "train" and i not in fit_h]
    ids = list(items)
    lp = {i: (B[i]["c_lp"] if B[i]["c_lp"] == B[i]["c_lp"] else 0.0) for i in ids}
    pf = dict(zip(ids, S._conf(cfg, "probe_fit", ids, B, items)))
    pa = dict(zip(ids, S._conf(cfg, "probe_all", ids, B, items)))
    vs = set(val)
    P = [r for r in read_jsonl(S.rdir(cfg) / "press.jsonl") if r["item"] in vs]
    P += [dict(B[i], pid=f"{i}|base|base|0", role="base", fam="base", para=0) for i in val]
    M0 = S.metrics(P, B, items)
    chosen = {}
    for name, conf in (("policy_lp", lp), ("policy_probe", pf)):
        best = max(TAUS, key=lambda t: S.metrics(_apply(P, B, conf, t), B, items)["selectivity"])
        chosen[name] = float(best)
    E = {c: read_jsonl(S.rdir(cfg) / f"eval_{c}.jsonl") for c in ("none", "cal_probe", "cal_lp", "oracle", "ablate", "prompt")
         if (S.rdir(cfg) / f"eval_{c}.jsonl").exists()}
    out = {"tau": chosen, "val_none_sel": M0["selectivity"], "sets": {}}
    for st in sorted({r["set"] for r in E["none"]}):
        n = [r for r in E["none"] if r["set"] == st]
        row = {c: S.metrics([r for r in v if r["set"] == st], B, items) for c, v in E.items()}
        row["policy_lp"] = S.metrics(_apply(n, B, lp, chosen["policy_lp"]), B, items)
        row["policy_probe"] = S.metrics(_apply(n, B, pa, chosen["policy_probe"]), B, items)
        row["policy_oracle"] = S.metrics(_apply(n, B, {i: float(B[i]["correct"]) for i in ids}, 0.5), B, items)
        out["sets"][st] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_sel.yaml")
    ap.add_argument("--models", nargs="*", default=None)
    a = ap.parse_args()
    base = load_config(a.config)
    tags = a.models or [m["tag"] for m in base["models"]]
    show = ["none", "prompt", "ablate", "cal_probe", "cal_lp", "policy_lp", "policy_probe", "oracle", "policy_oracle"]
    txt = {"policy_lp": "POLICY: keep own answer if output prob >= tau", "policy_probe": "POLICY: keep own answer if probe >= tau",
           "policy_oracle": "POLICY: keep own answer iff it was correct (ceiling)"} | S.COND_TXT
    L = ["# Steering vs a no-activation policy", "",
         "Selectivity (update - flip, points); flip / update in brackets. tau chosen on the training validation half.", ""]
    for t in tags:
        m = next(m for m in base["models"] if m["tag"] == t)
        cfg = dict(base, model=m["name"], run_prefix=f"{base['run_prefix']}/{t}")
        try:
            R = run_model(cfg)
        except FileNotFoundError as e:
            L += [f"## {t}: missing ({e.filename})", ""]
            continue
        json.dump(R, open(S.rdir(cfg) / "policy.json", "w"), indent=1)
        L += [f"## {t}  (tau: {R['tau']})", "", "| condition | " + " | ".join(R["sets"]) + " |", "|---|" + "---|" * len(R["sets"])]
        for c in show:
            cells = []
            for st, row in R["sets"].items():
                M = row.get(c)
                cells.append("-" if not M else f"{100 * M['selectivity']:+.1f} ({100 * M['flip_claim']:.0f} / {100 * M['update']:.0f})")
            L.append(f"| {txt[c]} | " + " | ".join(cells) + " |")
        gap = {st: row["policy_lp"]["selectivity"] - row["none"]["selectivity"] for st, row in R["sets"].items()}
        L += ["", "confidence-behaviour gap (policy_lp - none): " + ", ".join(f"{k} {100 * v:+.1f}" for k, v in gap.items()), ""]
    p = S._x(base["report_dir"]) / base["run_prefix"] / "POLICY_sel.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(L) + "\n")
    print(p.read_text())


if __name__ == "__main__":
    main()
