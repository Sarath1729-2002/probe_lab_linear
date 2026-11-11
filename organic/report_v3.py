"""organic_v3 reports: one REPORT.md per model (framing grid, critic, probes, transfer, trajectory, steering)
and COMPARE.md across models."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np

from .frames import ASKS, SOURCES, _ask
from .report import INK, MUTED, _f, _pct, _style, _traj_fig
from .run import bdir, root

ASK_COL = {"plain": "#2a78d6", "check": "#eb6834", "critic": "#1baf7a"}     # reference categorical slots 1-3
B = "processbench"


def _grid_fig(fr, base, out, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(len(SOURCES))
    w = 0.26
    for k, a in enumerate(ASKS):
        ys = [100 * (fr.get(f"frame_{s}_{a}", {}).get("follow") or np.nan) for s in SOURCES]
        bars = ax.bar(x + (k - 1) * (w + 0.02), ys, w, color=ASK_COL[a], label=a)
        for xb, yv in zip(bars, ys):
            if yv == yv:
                ax.annotate(f"{yv:.0f}", (xb.get_x() + xb.get_width() / 2, yv), xytext=(0, 2),
                            textcoords="offset points", ha="center", fontsize=7, color=INK)
    if base is not None:
        ax.axhline(100 * base, color=MUTED, lw=1, ls="--")
        ax.text(0.99, 0.97, f"dashed line: bare claim \"I think the answer is X\" = {100 * base:.0f}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=7, color=MUTED)
    ax.set_xticks(x, [{"user": "user's working", "student": "a student's", "model": "another AI's",
                       "self": "its own earlier reply"}[s] for s in SOURCES], fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("follows the wrong solution (%)", fontsize=8, color=MUTED)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    _style(ax)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out.name


def write_report_v3(cfg: dict) -> Path:
    R = json.load(open(root(cfg) / "results.json"))
    out = Path(os.path.expandvars(cfg["report_dir"])) / cfg["run_prefix"]
    out.mkdir(parents=True, exist_ok=True)
    P = R["buckets"][B]
    fr = P.get("frames", {})
    be = P["behaviour"]
    base = ((be.get("sugg/wrong") or {}).get("known") or {}).get("wrong")
    L = [f"# Inheriting someone else's error: {cfg['model']}", "",
         "ProcessBench solutions with a human-labelled first error (and correct solutions as controls), on problems "
         "the model solves by itself (**known**: correct with the bare question; **firm**: 5/5 correct when re-asked "
         "at temperature 0.7). Nothing in the model is edited except in the steering section. "
         "**Follow** = the final answer equals the wrong solution's own final answer; **recover** = correct answer.", "",
         f"Items: {P['n_items']}, known {_pct(P['known_rate'])}.", "",
         "**Frames** (the same numbered solution every time):", "",
         "| Source | How it is presented |", "|---|---|",
         "| user | \"Here is my working:\" |", "| student | \"Here is a student's working:\" |",
         "| model | \"Here is a solution written by another AI model:\" |",
         "| self | the solution is the model's own previous reply in the conversation |", "",
         "| Ask | Text (shown for the user source) |", "|---|---|"]
    for a in ASKS:
        L.append(f"| {a} | {_ask('user', a) or '(only the answer format line)'} |")

    # ---------------------------------------------------------------- grid
    if fr:
        fn = _grid_fig(fr, base, out / "fig_frames.png", f"{cfg['model'].split('/')[-1]}: wrong solutions on known problems")
        L += ["", "## How often it inherits the error", "", f"![frames]({fn})", "",
              "| Source | Ask | n | Follow | Recover | Other | Follow (firm, n) | Certifies when following | Correct solutions: correct / certified |",
              "|---|---|---|---|---|---|---|---|---|"]
        for s in SOURCES:
            for a in ASKS:
                r = fr.get(f"frame_{s}_{a}")
                if not r:
                    continue
                L.append(f"| {s} | {a} | {r['n_wrong']} | **{_pct(r['follow'])}** | {_pct(r['recover'])} | {_pct(r['other'])} | "
                         f"{_pct(r['follow_firm'])} ({r['n_firm']}) | {_pct(r['certify_when_following'])} | "
                         f"{_pct(r['correct_on_right'])} / {_pct(r['certify_on_right'])} |")
        L += ["", f"Bare-claim baseline (\"I think the answer is X but I'm really not sure\", same items): follow **{_pct(base)}**.",
              "*Certifies* = the reply explicitly says the working is correct (or `First error: none`)."]
        cr = [(s, fr.get(f"frame_{s}_critic")) for s in SOURCES if fr.get(f"frame_{s}_critic")]
        if cr:
            L += ["", "## Can it find the error when asked to? (critic frame, ProcessBench metric)", "",
                  "| Source | First error found exactly | Says `none` on a wrong solution | Flags some error | Follows anyway, after flagging an error (n) | Correct solutions judged correct | F1 |",
                  "|---|---|---|---|---|---|---|"]
            for s, r in cr:
                L.append(f"| {s} | {_pct(r.get('critic_acc_wrong'))} | {_pct(r.get('critic_says_none_on_wrong'))} | "
                         f"{_pct(r.get('flags_an_error_on_wrong'))} | {_pct(r.get('follow_despite_flagging_error'))} "
                         f"({r.get('n_flagged')}) | {_pct(r.get('critic_acc_right'))} | {_f(r.get('critic_f1'))} |")

    # ---------------------------------------------------------------- probes
    L += ["", "## What the residual stream says before the reply", "",
          "Per frame, known items. **Detection**: wrong vs correct solution (held-out items). **Outcome**: follow vs "
          "recover on wrong solutions (grouped out-of-fold). **vs confidence**: AUROC for following from knowledge "
          "strength alone, the probe alone, and the probe among firm items only.", "",
          "| Frame | Detection @ end of solution | Detection @ last prompt token | Outcome @ last prompt token (layer) | Confidence only | Probe only | Probe, firm items |",
          "|---|---|---|---|---|---|---|"]
    for f in sorted(P["probes"], key=lambda f: (not f.startswith("frame_"), f)):
        pe, pl = P["probes"][f].get("info_end", {}), P["probes"][f].get("last_prompt_token", {})
        op = pl.get("outcome_probe") or {}
        cc = pl.get("confidence_control") or {}
        L.append(f"| {f} | {_f(pe.get('detection_auroc'), 3)} | {_f(pl.get('detection_auroc'), 3)} | "
                 f"{_f(op.get('auroc_oof'), 3)} ({op.get('layer', '–')}) | {_f(cc.get('auroc_confidence_only'), 3)} | "
                 f"{_f(cc.get('auroc_probe_only'), 3)} | {_f(cc.get('auroc_probe_among_firm'), 3)} |")
    ft = P.get("frame_transfer", {})
    if "matrix_train_row_test_col" in ft:
        fs = ft["frames"]
        short = [f.replace("frame_", "") for f in fs]
        L += ["", "## Is 'about to inherit the error' one direction across framings?", "",
              "Outcome probe (follow vs recover, last prompt token) trained in the row frame, tested in the column frame.", "",
              "| train \\ test | " + " | ".join(short) + " |", "|---|" + "---|" * len(fs)]
        for a, row in zip(short, ft["matrix_train_row_test_col"]):
            L.append(f"| {a} | " + " | ".join(_f(x) for x in row) + " |")
        off = [x for i, row in enumerate(ft["matrix_train_row_test_col"]) for j, x in enumerate(row) if i != j and x is not None]
        if off:
            L += ["", f"Off-diagonal mean {_f(float(np.mean(off)))}, min {_f(float(np.min(off)))}."]

    # ---------------------------------------------------------------- trajectory
    fn, summ = _traj_fig(cfg, [B], out / "fig_trajectory.png")
    if fn:
        t, curve = summ[B]
        L += ["", f"## From the end of the solution to the reply ({cfg['traj']['family']}, a separate probe per position)", "",
              f"![trajectory]({fn})", "", "| Position | Detection AUROC | Flagged: followed / recovered | Score → follow |", "|---|---|---|---|"]
        for c in curve:
            L.append(f"| {c['position']} | {_f(c.get('detection_auroc'), 3)} | {_pct(c.get('flagged_adopted'))} / "
                     f"{_pct(c.get('flagged_held'))} | {_f(c.get('score_predicts_adopt_auroc'), 3)} |")

    # ---------------------------------------------------------------- steering
    sp = bdir(cfg, B) / "steer.json"
    if sp.exists():
        S = json.load(open(sp))
        L += ["", "## Causal test: steering toward 'recover'", "",
              f"Direction = mean({S.get('contrast', 'recover - follow').split(' - ')[0]}) − mean(follow) at the last prompt token, "
              f"layer {S['layer']}, fitted on half the items ({S['n_fit_recover']} vs {S['n_fit_follow']} follow), tested on the other half "
              f"({S['n_test_wrong']} wrong, {S['n_test_right']} correct solutions), frame `{S['family']}`. Added at every "
              f"position; alpha = 1 is the full mean difference (norm {S['norm']:.1f}; typical state norm "
              f"{S['mean_state_norm']:.0f}). Random = a random direction of the same norm.", "",
              "| Direction | alpha | Follow | Recover | Other | Certifies wrong | Correct solutions kept correct | Mean reply chars |",
              "|---|---|---|---|---|---|---|---|"]
        for c in S["conditions"]:
            L.append(f"| {c['cond']} | {c['alpha']} | **{_pct(c['follow'])}** | {_pct(c['recover'])} | {_pct(c['other'])} | "
                     f"{_pct(c['certify_wrong'])} | {_pct(c['correct_on_right'])} | {c['mean_chars']:.0f} |")
        L += ["", "*A real effect: follow drops with alpha along the direction, rises with negative alpha, the random "
              "direction does little, and correct solutions stay correct (otherwise it is just 'reject everything' or noise).*"]
    L += ["", "## Notes", "",
          "- Outcome-probe layers are picked by grouped CV on the same rows (mildly optimistic); detection AUROCs are held-out.",
          "- Certification and critic parsing are regex-based on the reply text; per-reply data is in "
          "`<out_dir>/<run_prefix>/processbench/gen.jsonl` and `steer.jsonl`.", ""]
    (out / "REPORT.md").write_text("\n".join(L))
    shutil.copy(root(cfg) / "results.json", out / "results.json")
    if sp.exists():
        shutil.copy(sp, out / "steer.json")
    print(f"[report] {out / 'REPORT.md'}")
    return out / "REPORT.md"


def write_compare(cfg: dict, tags: list[str]) -> Path:
    base_dir = Path(os.path.expandvars(cfg["report_dir"])) / cfg["run_prefix"]
    rows = {}
    for t in tags:
        p = base_dir / t / "results.json"
        if p.exists():
            rows[t] = {"R": json.load(open(p))["buckets"].get(B, {}),
                       "S": json.load(open(base_dir / t / "steer.json")) if (base_dir / t / "steer.json").exists() else None}
    ts = list(rows)
    L = ["# organic_v3: across models", "", f"Models: {', '.join(ts)}. Known ProcessBench problems, wrong solutions.", "",
         "## Follow rate by frame", "", "| Frame | " + " | ".join(ts) + " |", "|---|" + "---|" * len(ts)]
    L.append("| bare claim (sugg) | " + " | ".join(
        _pct(((rows[t]["R"].get("behaviour", {}).get("sugg/wrong") or {}).get("known") or {}).get("wrong")) for t in ts) + " |")
    for s in SOURCES:
        for a in ASKS:
            f = f"frame_{s}_{a}"
            L.append(f"| {s} / {a} | " + " | ".join(_pct((rows[t]["R"].get("frames", {}).get(f) or {}).get("follow")) for t in ts) + " |")
    L += ["", "## Follow rate on firmly known problems (5/5)", "", "| Frame | " + " | ".join(ts) + " |", "|---|" + "---|" * len(ts)]
    for s in SOURCES:
        for a in ASKS:
            f = f"frame_{s}_{a}"
            L.append(f"| {s} / {a} | " + " | ".join(_pct((rows[t]["R"].get("frames", {}).get(f) or {}).get("follow_firm")) for t in ts) + " |")
    L += ["", "## Critic: first error found exactly / F1", "", "| Source | " + " | ".join(ts) + " |", "|---|" + "---|" * len(ts)]
    for s in SOURCES:
        f = f"frame_{s}_critic"
        L.append(f"| {s} | " + " | ".join(
            f"{_pct((rows[t]['R'].get('frames', {}).get(f) or {}).get('critic_acc_wrong'))} / "
            f"{_f((rows[t]['R'].get('frames', {}).get(f) or {}).get('critic_f1'))}" for t in ts) + " |")
    L += ["", "## Outcome probe at the last prompt token (follow vs recover, AUROC)", "", "| Frame | " + " | ".join(ts) + " |",
          "|---|" + "---|" * len(ts)]
    for s in SOURCES:
        for a in ASKS:
            f = f"frame_{s}_{a}"
            L.append(f"| {s} / {a} | " + " | ".join(_f(((rows[t]["R"].get("probes", {}).get(f, {}).get("last_prompt_token", {})
                                                          .get("outcome_probe")) or {}).get("auroc_oof"), 3) for t in ts) + " |")
    L += ["", "## Steering (follow rate; correct solutions kept correct)", "", "| Condition | " + " | ".join(ts) + " |", "|---|" + "---|" * len(ts)]
    conds = []
    for t in ts:
        for c in (rows[t]["S"] or {}).get("conditions", []):
            if (c["cond"], c["alpha"]) not in conds:
                conds.append((c["cond"], c["alpha"]))
    for cn, a in conds:
        cells = []
        for t in ts:
            c = next((c for c in (rows[t]["S"] or {}).get("conditions", []) if c["cond"] == cn and c["alpha"] == a), None)
            cells.append(f"{_pct(c['follow'])}; {_pct(c['correct_on_right'])}" if c else "–")
        L.append(f"| {cn} α={a} | " + " | ".join(cells) + " |")
    (base_dir / "COMPARE.md").write_text("\n".join(L) + "\n")
    print(f"[compare] {base_dir / 'COMPARE.md'}")
    return base_dir / "COMPARE.md"
