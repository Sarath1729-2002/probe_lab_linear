"""Stage `report`: turn results.json into REPORT.md + a few figures you can look through."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .io import run_dir, read_jsonl  # noqa: E402

# reference categorical slots 1-3 (blue, orange, aqua); chance lines in muted grey
C1, C2, C3, GRID, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#e6e5e0", "#2b2b29", "#8a8984"


def _axes(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel("layer", color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)


def _line(ax, xs, ys, color, label):
    # legend carries identity; end-of-line labels collided when curves converge
    ys = [float("nan") if y is None else y for y in ys]
    ax.plot(xs, ys, color=color, linewidth=2, label=label, marker="o", markersize=3)


def figures(R, rd):
    figs = []
    p = R["probes"]
    # ---- classification curves
    fig, ax = plt.subplots(figsize=(7.5, 4))
    _axes(ax, "Dev cross-validated AUROC by layer", "AUROC")
    if p.get("P0_detect", {}).get("curve"):
        c = p["P0_detect"]["curve"]
        _line(ax, [r["layer"] for r in c], [r["dev_cv"] for r in c], C3, "P0 detect")
    if p.get("P1_follow_vs_recover", {}).get("curve"):
        c = p["P1_follow_vs_recover"]["curve"]
        _line(ax, [r["layer"] for r in c], [r["dev_cv"] for r in c], C1, "P1 follow (all pairs)")
        _line(ax, [r["layer"] for r in c], [r["dev_within"] for r in c], C2, "P1 follow (within problem)")
    ax.axhline(0.5, color=MUTED, linestyle="--", linewidth=1)
    ax.set_ylim(0.4, 1.02)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(rd / "fig_classification_layers.png", dpi=150)
    plt.close(fig)
    figs.append("fig_classification_layers.png")

    # ---- value tracking
    if p.get("P2_value_tracking"):
        c = p["P2_value_tracking"]["curve"]
        xs = [r["layer"] for r in c]
        fig, ax = plt.subplots(figsize=(7.5, 4))
        _axes(ax, "P2: which digit does a clean-trained probe read on corrupted runs?", "fraction of test runs")
        _line(ax, xs, [r["corrupt_reads_injected"] for r in c], C2, "reads injected digit")
        _line(ax, xs, [r["reads_true_RECOVER"] for r in c], C1, "reads true digit (RECOVER)")
        _line(ax, xs, [r["reads_true_FOLLOW"] for r in c], C3, "reads true digit (FOLLOW)")
        ax.axhline(0.1, color=MUTED, linestyle="--", linewidth=1)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(frameon=False, fontsize=8, loc="center right")
        fig.tight_layout()
        fig.savefig(rd / "fig_value_tracking.png", dpi=150)
        plt.close(fig)
        figs.append("fig_value_tracking.png")

    # ---- regression curves
    regs = [("P1r_follow_rate", C1, "P1r follow rate"), ("P4_horizon", C2, "P4 horizon"),
            ("P5_susceptibility", C3, "P5 susceptibility")]
    if any(p.get(k) for k, *_ in regs):
        fig, ax = plt.subplots(figsize=(7.5, 4))
        _axes(ax, "Dev cross-validated Spearman by layer (regression probes)", "Spearman")
        for k, col, lab in regs:
            if p.get(k):
                c = p[k]["curve"]
                _line(ax, [r["layer"] for r in c], [r["dev_cv"] for r in c], col, lab)
        ax.axhline(0, color=MUTED, linestyle="--", linewidth=1)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(rd / "fig_regression_layers.png", dpi=150)
        plt.close(fig)
        figs.append("fig_regression_layers.png")
    return figs


def _f3(x):
    return "n/a" if x is None or x != x else f"{x:.3f}"


def _ci(x):
    return f"[{x[0]:.3f}, {x[1]:.3f}]" if x and x[0] == x[0] else "n/a"


def _row(name, r):
    return (f"| {name} | {r['auroc']:.3f} {_ci(r['auroc_ci'])} | {r['within_auroc']:.3f} "
            f"{_ci(r['within_auroc_ci'])} | {r['n']} |")


def _reg_row(name, r):
    return f"| {name} | {_f3(r['spearman'])} {_ci(r['spearman_ci'])} | {_f3(r['within_spearman'])} | {r['n']} |"


def examples(rd, k=2):
    injs = {r["inj_id"]: r for r in read_jsonl(rd / "injections.jsonl")}
    conts = read_jsonl(rd / "continuations.jsonl")
    blocks = []
    for want, rtype in [("FOLLOW", None), ("RECOVER", "silent"), ("RECOVER", "explicit"), ("OTHER", None)]:
        picked = [c for c in conts if c["greedy"]["outcome"] == want
                  and (rtype is None or c["greedy"]["recovery_type"] == rtype)][:k]
        for c in picked:
            inj = injs[c["inj_id"]]
            head = inj["prefix_corrupt"][-220:].replace("\n", " ⏎ ")
            tail = c["greedy_text"][:350].replace("\n", " ⏎ ")
            blocks.append(
                f"**{want}{' / ' + rtype if rtype else ''}** · `{c['inj_id']}` · {inj['kind']} "
                f"{inj['true_text']} → **{inj['new_text']}** · gold {inj['gold']:g}, "
                f"propagated {inj['propagated_answer']:g}, got {c['greedy']['answer']}\n\n"
                f"> …{head} ‖ {tail}…\n")
    return "\n".join(blocks)


def write_report(cfg):
    rd = run_dir(cfg)
    R = json.load(open(rd / "results.json"))
    figs = figures(R, rd)
    p, c, v = R["probes"], R["counts"], R["verdict"]
    L = [f"# Error-propagation probes: {cfg['run_name']}", "",
         f"Model `{cfg['model']}` · {c['problems']} problems · {c['injections']} injections", "",
         "## Verdict (pre-registered rule)", "", f"**{v['summary']}**", "",
         "```", json.dumps(v, indent=2), "```", "",
         "## Labels", "",
         f"- On-path greedy outcomes: `{c['greedy_outcomes_on_path']}`",
         f"- Off-path greedy outcomes (control, cannot change the answer): `{c['greedy_outcomes_off_path']}`",
         f"- On-path by dataset: `{c['by_dataset']}`",
         f"- On-path by operator: `{c['by_kind']}`",
         f"- Recovery types (on-path RECOVER): `{c['recovery_types']}`",
         f"- Injected value re-used in continuation (uptake), on-path: {c['uptake_rate_on_path']:.3f}",
         f"- Sanity: clean truncated prefix still reaches gold: {c['clean_cont_correct_rate']:.3f}; "
         f"model extended the injected number with more digits: {c['glued_rate']:.3f}; "
         f"propagated answer rounds to gold (FOLLOW and RECOVER indistinguishable): "
         f"{c.get('ambiguous_label_n', 0)} injections. "
         f"Rows failing any of these are excluded from every probe (valid: {c['valid_rate']:.3f}).", ""]

    p0 = p.get("P0_detect", {})
    if p0.get("test"):
        L += ["## P0 · Corruption detection (sanity)", "",
              f"Test AUROC {p0['test']['auroc']:.3f} at layer {p0['best_layer']}. Expected near 1.0: "
              "the injected digits are literally in the text.", ""]

    p1 = p.get("P1_follow_vs_recover", {})
    L += ["## P1 · FOLLOW vs RECOVER at the injection point (main)", ""]
    if "probe" in p1:
        L += [f"Dev {p1['n_dev']} / test {p1['n_test']} injections · FOLLOW rate {p1['follow_rate']:.3f} · "
              f"best layer {p1['best_layer']} · mixed-outcome test problems "
              f"{p1['probe']['n_mixed_problems']}", "",
              "| Scorer | Test AUROC [95% CI] | Within-problem AUROC [CI] | n |", "|---|---|---|---|",
              _row("**Linear probe**", p1["probe"])]
        for k, r in p1["baselines"].items():
            L.append(_row(f"Covariates: {k}", r))
        L += [_row("All covariates + probe", p1["stacked"]), "",
              f"Δ AUROC (probe added to all covariates): {p1['delta_auroc_probe_over_covariates']['value']:.3f} "
              f"{_ci(p1['delta_auroc_probe_over_covariates']['ci'])}", "",
              f"- Shuffled-label control, test AUROC: {p1['shuffled_label_test_auroc']:.3f} (should be ≈ 0.5)"]
        if "off_path_control" in p1:
            o = p1["off_path_control"]
            L.append(f"- Off-path control, mean P(FOLLOW): off-path {o['mean_p_follow_off_path']:.3f} (n={o['n_off']}) "
                     f"vs test FOLLOW {o['mean_p_follow_test_FOLLOW']:.3f} / RECOVER {o['mean_p_follow_test_RECOVER']:.3f}. "
                     "If off-path looks like FOLLOW, the probe reads 'corrupted', not 'will propagate'.")
        L.append("")
    else:
        L += [f"Skipped: {p1.get('skipped')}. Counts above show why.", ""]

    if p.get("P1_by_operator"):
        L += ["### P1 within each operator", "",
              "The operator alone predicts a lot (+1 is mostly followed, ×10 mostly caught). Inside one "
              "operator that shortcut is gone, so this is the stricter test.", "",
              "| Operator | n dev / test | FOLLOW rate | Probe AUROC [CI] | Within-problem | Covariates AUROC | Δ (probe over covariates) [CI] | layer |",
              "|---|---|---|---|---|---|---|---|"]
        for k, q in p["P1_by_operator"].items():
            if "skipped" in q:
                L.append(f"| {k} | {q['n_dev']} / {q['n_test']} | {_f3(q['follow_rate'])} | skipped: {q['skipped']} | | | | |")
            else:
                L.append(f"| {k} | {q['n_dev']} / {q['n_test']} | {_f3(q['follow_rate'])} | "
                         f"{_f3(q['probe_auroc'])} {_ci(q.get('probe_auroc_ci'))} | {_f3(q['probe_within'])} "
                         f"(n={q['n_mixed']}) | {_f3(q['cov_auroc'])} | {_f3(q['delta'])} {_ci(q.get('delta_ci'))} | "
                         f"{q['best_layer']} |")
        L.append("")
    if p.get("P1_split_stability"):
        q = p["P1_split_stability"]
        L += ["### P1 across random dev/test splits", "",
              "| Split seed | Layer | Probe AUROC | Within-problem (n mixed) | Covariates AUROC | Δ |",
              "|---|---|---|---|---|---|"]
        for r in q["runs"]:
            L.append(f"| {r['seed']} | {r['best_layer']} | {_f3(r['probe_auroc'])} | {_f3(r['probe_within'])} "
                     f"({r['n_mixed']}) | {_f3(r['cov_auroc'])} | {_f3(r['delta'])} |")
        m = q["mean_sd"]
        L += [f"| **mean ± sd** | | {m['probe_auroc'][0]:.3f} ± {m['probe_auroc'][1]:.3f} | "
              f"{m['probe_within'][0]:.3f} ± {m['probe_within'][1]:.3f} | "
              f"{m['cov_auroc'][0]:.3f} ± {m['cov_auroc'][1]:.3f} | {m['delta'][0]:.3f} ± {m['delta'][1]:.3f} |", ""]

    if p.get("P1r_follow_rate"):
        q = p["P1r_follow_rate"]
        L += ["## P1r · Follow rate over greedy + samples (regression)", "",
              "| Predictor | Test Spearman [CI] | Within-problem Spearman | n |", "|---|---|---|---|",
              _reg_row(f"Linear probe (layer {q['best_layer']})", q["probe"]),
              _reg_row("All covariates", q["all_covariates"]), ""]

    if p.get("P2_value_tracking"):
        q = p["P2_value_tracking"]
        best = max(q["curve"], key=lambda r: (r.get("reads_true_RECOVER") or 0) - (r.get("reads_true_FOLLOW") or 0))
        L += ["## P2 · Value tracking", "",
              f"{q['n']} injections where the last digit changes ({q['n_test']} test). A 10-way last-digit "
              "probe is trained on clean runs and read on corrupted runs.", "",
              f"Largest RECOVER-vs-FOLLOW gap at layer {best['layer']}: reads the TRUE digit on "
              f"{_f3(best['reads_true_RECOVER'])} of RECOVER vs {_f3(best['reads_true_FOLLOW'])} of FOLLOW runs; "
              f"margin AUROC for recovery {_f3(best.get('margin_auroc_recover'))}.", ""]

    q = p.get("P3_silent_recovery", {})
    L += ["## P3 · Silent vs visible recovery", "",
          f"{q.get('n')} on-path RECOVER runs, silent rate {_f3(q.get('silent_rate'))}."]
    if "probe" in q:
        L += ["", "| Scorer | Test AUROC [CI] | Within-problem AUROC [CI] | n |", "|---|---|---|---|",
              _row(f"Linear probe (layer {q['best_layer']})", q["probe"]), _row("All covariates", q["all_covariates"])]
    L.append("")

    for key, title in [("P4_horizon", "P4 · Contamination horizon"), ("P5_susceptibility", "P5 · Prompt-time susceptibility")]:
        if p.get(key):
            q = p[key]
            other = "all_covariates" if "all_covariates" in q else "difficulty_covariates"
            L += [f"## {title}", "", "| Predictor | Test Spearman [CI] | Within-problem Spearman | n |",
                  "|---|---|---|---|", _reg_row(f"Linear probe (layer {q['best_layer']})", q["probe"]),
                  _reg_row(other.replace("_", " "), q[other]), ""]

    L += ["## Figures", ""] + [f"![{f}]({f})" for f in figs] + ["",
          "## Examples to read (greedy; ‖ marks where the injected prefix ends)", "", examples(rd)]
    (rd / "REPORT.md").write_text("\n".join(L))
    print(f"[report] wrote {rd / 'REPORT.md'}")
