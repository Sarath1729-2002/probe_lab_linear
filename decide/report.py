"""DECISION.md: one page, three verdicts, each from a criterion fixed before the run."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .common import run_dir


def _load(rd, name):
    p = rd / name
    return json.load(open(p)) if p.exists() else None


def _f(x, d=3):
    return "–" if x is None or x != x else f"{x:.{d}f}"


def write_report(cfg: dict) -> Path:
    tags = [m["tag"] for m in cfg["models"]]
    R = {t: {k: _load(run_dir(cfg, t), f"{k}.json") for k in ("a1", "a2", "b", "c")} for t in tags}
    names = {m["tag"]: m["short"] for m in cfg["models"]}
    L = ["# Decision run: which thread to pursue", "",
         "Three experiments, each with a kill criterion fixed before the run. A thread is **LIVE** when its "
         "criterion holds in at least 2 of the models.", ""]

    # ---------------------------------------------------------------- A
    L += ["## A. Detection vs propagation", "",
          "**Question.** Goel et al. report that corruption is detectable (AUROC > 0.98), but failure prediction is "
          "near chance (~0.5). paper_v1 found propagation readable at 0.91–0.94. Is the difference the readout, "
          "or where the error sits?", "",
          "**A1: readout, on the paper_v1 injections.** Same token, same items.",
          "- **Detection:** a probe for corrupted vs clean prefix.",
          "- **R1 (Goel et al.'s readout):** the detection score, reused to predict FOLLOW vs RECOVER.",
          "- **R2 (paper_v1's readout):** a probe trained on FOLLOW vs RECOVER.", "",
          "| Model | n | FOLLOW rate | Detection AUROC | R1 (detection score → outcome) | R2 (outcome probe) | R2 − R1 [95% CI] | cos(detection, outcome) |",
          "|---|---|---|---|---|---|---|---|"]
    a1_live = 0
    for t in tags:
        a = R[t]["a1"]
        if not a or "pooled" not in a:
            L.append(f"| {names[t]} | – | – | – | – | – | {'skipped' if a else 'not run'} | – |")
            continue
        p = a["pooled"]["at_detection_layer"]
        ci = p["R2_minus_R1_ci"]
        L.append(f"| {names[t]} | {a['n_kept']} | {_f(a['follow_rate'], 2)} | {_f(p['detection_auroc'])} | "
                 f"{_f(p['R1_detection_score_for_outcome'])} | {_f(p['R2_outcome_probe'])} | "
                 f"{_f(p['R2_minus_R1'])} [{_f(ci[0])}, {_f(ci[1])}] | {_f(p['cos_detection_outcome'], 2)} |")
        a1_live += p["R2_minus_R1"] >= 0.2 and ci[0] > 0
    L += ["", "*At the layer that is best for detection, as in Goel et al.*", "",
          "**A2: where the error sits.** The same corrupted working is placed in the user's message (\"here is my "
          "working so far\") instead of the model's own reply. The same injections are compared both ways.", "",
          "| Model | Paired items | FOLLOW, own reply | FOLLOW, user's working | Difference [95% CI] | R2 at value token | R2 at last prompt token | R1 at last prompt token |",
          "|---|---|---|---|---|---|---|---|"]
    a2_live = 0
    for t in tags:
        a = R[t]["a2"]
        if not a:
            L.append(f"| {names[t]} | not run | | | | | | |")
            continue
        ci = a.get("follow_diff_ci", [None, None])
        vt = (a.get("value_token") or {}).get("at_outcome_layer", {})
        lp = (a.get("last_prompt_token") or {}).get("at_outcome_layer", {})
        lpd = (a.get("last_prompt_token") or {}).get("at_detection_layer", {})
        d = a.get("follow_diff_user_minus_self")
        L.append(f"| {names[t]} | {a['n_paired']} | {_f(a['follow_rate_self'], 2)} | {_f(a['follow_rate_user'], 2)} | "
                 f"{_f(d, 2)} [{_f(ci[0], 2)}, {_f(ci[1], 2)}] | {_f(vt.get('R2_outcome_probe'))} | "
                 f"{_f(lp.get('R2_outcome_probe'))} | {_f(lpd.get('R1_detection_score_for_outcome'))} |")
        if d is not None and abs(d) >= 0.10 and (ci[0] > 0 or ci[1] < 0):
            a2_live += 1
    a_verdict = ("LIVE: the readout explains the disagreement" if a1_live >= 2 else
                 "NARROW: the readouts agree, so the disagreement is about the data")
    L += ["", f"**A1 criterion:** R2 − R1 ≥ 0.20 with CI above 0 → met in {a1_live}/{len(tags)} models. **{a_verdict}**.",
          f"**A2 criterion:** FOLLOW changes by ≥ 10 points when the working is attributed to the user → met in "
          f"{a2_live}/{len(tags)} models. **{'LIVE: attribution matters' if a2_live >= 2 else 'not met'}**.", ""]

    # ---------------------------------------------------------------- B
    L += ["## B. Is the truth direction a knowledge direction?", "",
          "**Question.** Do truth-probe failures concentrate on facts the model doesn't know? A probe is trained on "
          "statements whose facts the model knows, then tested on held-out known and unknown facts. Known items "
          "are matched to unknown ones on statement surprisal.", "",
          "| Model | Test | Known / unknown (n) | AUROC known (all) | AUROC known (surprisal-matched) | AUROC unknown | Gap [95% CI] |",
          "|---|---|---|---|---|---|---|"]
    b_live = 0
    for t in tags:
        b = R[t]["b"]
        if not b:
            L.append(f"| {names[t]} | not run | | | | | |")
            continue
        hit = False
        for test, lab in (("B1_popqa", "PopQA → PopQA"), ("B2_cities", "cities → cities"),
                          ("B4_cities_to_popqa", "cities → PopQA"), ("B3_cities_to_arith1", "cities → arith 1 op"),
                          ("B3_cities_to_arith2", "cities → arith 2 ops"), ("B3_cities_to_arith3", "cities → arith 3 ops"),
                          ("B3b_arith1_to_arith23", "arith 1 → arith 2–3")):
            r = ((b.get(test) or {}).get("logistic") or {}).get("result")
            if not r:
                continue
            ci = r["gap_ci"]
            L.append(f"| {names[t]} | {lab} | {r['n_known']} / {r['n_unknown']} | {_f(r['auroc_known_all'])} | "
                     f"{_f(r['auroc_known_matched'])} | {_f(r['auroc_unknown'])} | {_f(r['gap_matched'])} "
                     f"[{_f(ci[0])}, {_f(ci[1])}] |")
            if test == "B1_popqa" and r["gap_matched"] >= 0.10 and ci[0] > 0:
                hit = True
        b_live += hit
        fams = b.get("families", {})
        L.append(f"| {names[t]} | *coverage* | " + "; ".join(
            f"{k}: {v['known']}/{v['unknown']}" for k, v in fams.items()) + " | | | | |")
    L += ["", f"**Criterion:** PopQA matched gap ≥ 0.10 with CI above 0 → met in {b_live}/{len(tags)} models. "
          f"**{'LIVE' if b_live >= 2 else 'KILL'}**. An unknown-item AUROC well above 0.5 would mean the probe reads "
          "plausibility rather than knowledge (E6 in the ledger).", ""]

    # ---------------------------------------------------------------- C
    L += ["## C. Silent-acceptance gate", "",
          "**Question.** Given a false premise about a number it demonstrably knows, does the model sometimes "
          "adopt it and sometimes reject it?", "",
          "| Model | Facts known | Condition | Adopt | Reject or override | False alarms on true items | Detection AUROC (premise end) | Silent acceptance | Gate |",
          "|---|---|---|---|---|---|---|---|---|"]
    c_live = 0
    for t in tags:
        c = R[t]["c"]
        if not c:
            L.append(f"| {names[t]} | not run | | | | | | | |")
            continue
        for cond, lab in (("opt", "with FALSE PREMISE option"), ("plain", "answer only")):
            r = c[cond]
            pe = r.get("premise_end", {})
            L.append(f"| {names[t]} | {c['facts_known']}/{c['facts_total']} | {lab} | {_f(r['adopt_rate'], 2)} | "
                     f"{_f(r['reject_or_override_rate'], 2)} | {_f(r['false_alarm_rate_true_items'], 2)} | "
                     f"{_f(pe.get('detection_auroc_oof'))} | {_f(pe.get('silent_acceptance_rate'), 2)} | "
                     f"{'pass' if r['passes_gate'] else 'fail'} |")
        c_live += c["passes_gate_any"]
    L += ["", f"**Criterion:** ≥ 15% adopt and ≥ 15% reject-or-override in some condition → met in "
          f"{c_live}/{len(tags)} models. **{'LIVE' if c_live >= 2 else 'KILL'}**.",
          "*Silent acceptance = share of adopted false premises the detection probe flags at a 5% false-alarm rate.*", ""]

    # ---------------------------------------------------------------- verdict
    live = [x for x, ok in (("A", a1_live >= 2 or a2_live >= 2), ("B", b_live >= 2), ("C", c_live >= 2)) if ok]
    L += ["## Verdict", "",
          f"Live threads: **{', '.join(live) if live else 'none'}**.", "",
          "- **A and B both live:** they combine into one paper on what these probes actually read: the model's "
          "knowledge, surface cues, or an artefact of how the probe is read out.",
          "- **Only A:** a focused paper on detection vs propagation, with the readout result as the headline.",
          "- **Only B:** the knowledge-confound paper, with the full factorial as the next run.",
          "- **Only C:** the silent-acceptance project, with probes as the next step.",
          "- **None:** stop and talk to Amine and Adam before choosing anything else.", ""]
    out = Path(os.path.expandvars(cfg["report_dir"])) / cfg["run_prefix"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "DECISION.md").write_text("\n".join(L))
    for t in tags:
        for k in ("a1", "a2", "b", "c"):
            p = run_dir(cfg, t) / f"{k}.json"
            if p.exists():
                (out / f"{t}_{k}.json").write_text(p.read_text())
    print(f"[decide] wrote {out / 'DECISION.md'}; live: {live}")
    return out / "DECISION.md"
