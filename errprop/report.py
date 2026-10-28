"""Stage `report`: results.json (+ steer_summary.json) -> REPORT.md + figures, copied to report_dir."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .io import run_dir, read_jsonl  # noqa: E402

# reference categorical slots (blue, orange, aqua, yellow, magenta); muted grey for chance/reference
C1, C2, C3, C4, C5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GRID, INK, MUTED = "#e6e5e0", "#2b2b29", "#8a8984"


# ============================================================================ helpers

def _f3(x):
    return "n/a" if x is None or x != x else f"{x:.3f}"


def _ci(x):
    return f"[{x[0]:.3f}, {x[1]:.3f}]" if x and x[0] == x[0] else ""


def _axes(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)


def _line(ax, xs, ys, color, label, ls="-"):
    ys = [float("nan") if y is None else y for y in ys]
    ax.plot(xs, ys, color=color, linewidth=2, label=label, marker="o", markersize=3, linestyle=ls)


def _save(fig, rd, name, figs):
    fig.tight_layout()
    fig.savefig(rd / name, dpi=150)
    plt.close(fig)
    figs.append(name)


def _row(name, r):
    return (f"| {name} | {_f3(r['auroc'])} {_ci(r.get('auroc_ci'))} | {_f3(r['within_auroc'])} "
            f"{_ci(r.get('within_auroc_ci'))} | {r['n']} |")


def _reg_row(name, r):
    return f"| {name} | {_f3(r['spearman'])} {_ci(r.get('spearman_ci'))} | {_f3(r['within_spearman'])} | {r['n']} |"


# ============================================================================ figures

def figures(R, S, rd):
    figs, p = [], R["probes"]
    # P1 layer curves
    fig, ax = plt.subplots(figsize=(7.5, 4))
    _axes(ax, "P1 follow vs recover: dev cross-validated AUROC by layer", "layer", "AUROC")
    for key, col, lab in [("P1_catchable", C1, "catchable (headline)"), ("P1_all", C2, "all injections")]:
        c = p.get(key, {}).get("curve")
        if c:
            _line(ax, [r["layer"] for r in c], [r["dev_cv"] for r in c], col, f"{lab}, all pairs")
            _line(ax, [r["layer"] for r in c], [r["dev_within"] for r in c], col, f"{lab}, within problem", "--")
    c0 = p.get("P0_detect", {}).get("curve")
    if c0:
        _line(ax, [r["layer"] for r in c0], [r["dev_cv"] for r in c0], C3, "P0 corruption detection")
    ax.axhline(0.5, color=MUTED, linestyle=":", linewidth=1)
    ax.set_ylim(0.4, 1.02)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _save(fig, rd, "fig_p1_layers.png", figs)

    # wait policy
    W = p.get("W_wait_policy", {})
    for pool, title in [("corrupted_pool", "Wait intervention on corrupted steps"),
                        ("mixed_pool", "Wait intervention, corrupted + clean steps")]:
        A = W.get(pool)
        if not A:
            continue
        fig, ax = plt.subplots(figsize=(7.5, 4))
        _axes(ax, f"{title} (test, n={A['n']})", "fraction of steps intervened on", "final-answer accuracy")
        for key, col, lab in [("probe", C1, "probe-ranked"), ("surprise", C4, "surprisal-ranked"),
                              ("covariates", C3, "covariate-ranked"), ("random", MUTED, "random"),
                              ("oracle", C5, "oracle")]:
            if key in A:
                _line(ax, [r["budget"] for r in A[key]], [r["accuracy"] for r in A[key]], col, lab,
                      "--" if key in ("random", "oracle") else "-")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        _save(fig, rd, f"fig_wait_{pool}.png", figs)

    # steering
    if S:
        conds = S["conditions"]
        pr = [c for c in conds if c["cond"] == "probe"]
        fig, ax = plt.subplots(figsize=(7.5, 4))
        _axes(ax, f"Steering along the probe direction (layer {S['layer']})", "alpha (projection s.d.)", "rate")
        _line(ax, [c["alpha"] for c in pr], [c["follow_rate"] for c in pr], C1, "FOLLOW rate")
        _line(ax, [c["alpha"] for c in pr], [c["recover_rate"] for c in pr], C2, "RECOVER rate")
        rnd = [c for c in conds if c["cond"].startswith("random")]
        if rnd:
            ax.scatter([c["alpha"] for c in rnd], [c["follow_rate"] for c in rnd], color=C1, marker="x",
                       s=40, label="FOLLOW rate, random direction", zorder=3)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(frameon=False, fontsize=8, loc="center right")
        _save(fig, rd, "fig_steering.png", figs)

    # value tracking
    if p.get("P2_value_tracking"):
        c = p["P2_value_tracking"]["curve"]
        xs = [r["layer"] for r in c]
        fig, ax = plt.subplots(figsize=(7.5, 4))
        _axes(ax, "P2: which digit does a clean-trained probe read on corrupted runs?", "layer", "fraction of test runs")
        _line(ax, xs, [r["corrupt_reads_injected"] for r in c], C2, "reads injected digit")
        _line(ax, xs, [r["reads_true_RECOVER"] for r in c], C1, "reads true digit (RECOVER)")
        _line(ax, xs, [r["reads_true_FOLLOW"] for r in c], C3, "reads true digit (FOLLOW)")
        ax.axhline(0.1, color=MUTED, linestyle=":", linewidth=1)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(frameon=False, fontsize=8, loc="center right")
        _save(fig, rd, "fig_value_tracking.png", figs)

    # regression probes
    regs = [("P1r_follow_rate", C1, "P1r follow rate"), ("P4_horizon", C2, "P4 horizon"),
            ("P5_susceptibility", C3, "P5 prompt-time susceptibility")]
    if any(p.get(k) for k, *_ in regs):
        fig, ax = plt.subplots(figsize=(7.5, 4))
        _axes(ax, "Regression probes: dev cross-validated Spearman by layer", "layer", "Spearman")
        for k, col, lab in regs:
            if p.get(k):
                c = p[k]["curve"]
                _line(ax, [r["layer"] for r in c], [r["dev_cv"] for r in c], col, lab)
        ax.axhline(0, color=MUTED, linestyle=":", linewidth=1)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        _save(fig, rd, "fig_regression_layers.png", figs)
    return figs


# ============================================================================ sections

def _p1_section(title, q, note=""):
    L = [f"## {title}", ""]
    if note:
        L += [note, ""]
    if "probe" not in q:
        return L + [f"Skipped: {q.get('skipped')}", ""]
    pn = q["permutation_null"]
    L += [f"Dev {q['n_dev']} / test {q['n_test']} injections · FOLLOW rate {_f3(q['follow_rate'])} · "
          f"layer {q['best_layer']} · mixed-outcome test problems {q['probe']['n_mixed_problems']}", "",
          "| Scorer | Test AUROC [95% CI] | Within-problem AUROC [CI] | n |", "|---|---|---|---|",
          _row("**Linear probe**", q["probe"])]
    for k, r in q["baselines"].items():
        L.append(_row(f"Covariates: {k.replace('_', ' ')}", r))
    L += [_row("All covariates + probe", q["stacked"]), "",
          f"- **Gain of the probe over all covariates:** {_f3(q['delta'])} {_ci(q.get('delta_ci'))}",
          f"- **Permutation null** ({pn['n']} label shuffles): mean {_f3(pn['mean'])} ± {_f3(pn['sd'])}, "
          f"max {_f3(pn['max'])}; p = {pn['p_value']:.3f}"]
    if "off_path_control" in q:
        o = q["off_path_control"]
        L.append(f"- **Off-path control** (corrupted but cannot change the answer, n={o['n_off']}): mean "
                 f"P(FOLLOW) {_f3(o['mean_p_follow_off_path'])} vs test FOLLOW "
                 f"{_f3(o['mean_p_follow_test_FOLLOW'])} / RECOVER {_f3(o['mean_p_follow_test_RECOVER'])}")
    return L + [""]


def _rob_table(title, d, note=""):
    L = [f"### {title}", ""] + ([note, ""] if note else [])
    L += ["| Subset | n dev / test | FOLLOW rate | Probe AUROC [CI] | Within-problem (n) | Covariates AUROC | "
          "Gain [CI] | layer |", "|---|---|---|---|---|---|---|---|"]
    for k, q in d.items():
        if "skipped" in q:
            L.append(f"| {k} | {q['n_dev']} / {q['n_test']} | {_f3(q['follow_rate'])} | skipped: {q['skipped']} | | | | |")
        else:
            L.append(f"| {k} | {q['n_dev']} / {q['n_test']} | {_f3(q['follow_rate'])} | {_f3(q['probe_auroc'])} "
                     f"{_ci(q.get('probe_auroc_ci'))} | {_f3(q['probe_within'])} ({q['n_mixed']}) | "
                     f"{_f3(q['cov_auroc'])} | {_f3(q['delta'])} {_ci(q.get('delta_ci'))} | {q['best_layer']} |")
    return L + [""]


def _outcome_table(title, d):
    keys = ["FOLLOW", "RECOVER", "OTHER"]
    L = [f"**{title}**", "", "| | n | FOLLOW | RECOVER | OTHER |", "|---|---|---|---|---|"]
    for k, v in d.items():
        n = sum(v.get(x, 0) for x in keys)
        if n:
            L.append(f"| {k} | {n} | " + " | ".join(f"{v.get(x, 0) / n:.2f}" for x in keys) + " |")
    return L + [""]


def _wait_section(W):
    L = ["## W · Probe-gated \"Wait\" intervention", "",
         "Protocol borrowed from two lines of work. Reasoning Theater (Boppana et al. 2026) uses a probe "
         "to decide *when to act* on a reasoning trace (stop early); Self-Correction Bench (Tsui 2025) shows "
         "that appending \"Wait\" activates correction. Here the probe decides *where* to append \"Wait\", "
         "with a fixed intervention budget. Both continuation arms were generated for every test step, so "
         "each policy is simulated exactly, no extra sampling.", ""]
    if "skipped" in W:
        return L + [f"Skipped: {W['skipped']}", ""]
    for pool, desc in [("corrupted_pool", "Corrupted steps only"),
                       ("mixed_pool", "Corrupted steps + their clean twins (false alarms cost accuracy)")]:
        A = W[pool]
        L += [f"**{desc}** · n = {A['n']} · no intervention {A['no_intervention_acc']:.3f} · Wait everywhere "
              f"{A['all_wait_acc']:.3f}"
              + (f" · clean steps broken by Wait: {A['clean_broken_by_wait_rate']:.3f}"
                 if "clean_broken_by_wait_rate" in A else ""), ""]
        pols = [k for k in ("probe", "surprise", "covariates", "random", "oracle") if k in A]
        L += ["| Budget | " + " | ".join(pols) + " |", "|---|" + "---|" * len(pols)]
        for i, b in enumerate(A["probe"]):
            cells = []
            for k in pols:
                r = A[k][i]
                extra = f" (+{r['rescued']}/−{r['broken']})" if "rescued" in r and k == "probe" else ""
                cells.append(f"{r['accuracy']:.3f}{extra}")
            L.append(f"| {int(b['budget'] * 100)}% | " + " | ".join(cells) + " |")
        L.append("")
    L.append("Probe cells show (+rescued / −broken) at that budget.")
    return L + [""]


def _steer_section(S):
    L = ["## Steering (causal test)", ""]
    if not S:
        return L + ["Not run yet (`python -m errprop.cli steer`).", ""]
    L += [f"Direction: difference of FOLLOW and RECOVER means on dev, layer {S['layer']}, added at every "
          f"position; alpha in units of the projection s.d. ({S['proj_sd']:.3f}). Random directions of the "
          "same size are the control.", "",
          "| Condition | alpha | FOLLOW | RECOVER | OTHER | FOLLOW→RECOVER | RECOVER→FOLLOW |",
          "|---|---|---|---|---|---|---|"]
    for c in S["conditions"]:
        L.append(f"| {c['cond']} | {c['alpha']:+g} | {c['follow_rate']:.3f} | {c['recover_rate']:.3f} | "
                 f"{c['other_rate']:.3f} | {c['FOLLOW_to_RECOVER']:.3f} | {c['RECOVER_to_FOLLOW']:.3f} |")
    L += ["", "If the direction is causal: FOLLOW rises with +alpha, RECOVER rises with −alpha, and the random "
          "directions move little.", ""]
    return L


def examples(rd, k=2):
    injs = {r["inj_id"]: r for r in read_jsonl(rd / "injections.jsonl")}
    conts = read_jsonl(rd / "continuations.jsonl")
    blocks = []
    for want, rtype in [("FOLLOW", None), ("RECOVER", "silent"), ("RECOVER", "restated"), ("OTHER", None)]:
        picked = [c for c in conts if c["greedy"]["outcome"] == want and injs[c["inj_id"]]["on_path"]
                  and (rtype is None or c["greedy"]["recovery_type"] == rtype)]
        seen, chosen = set(), []
        for c in picked:                          # spread examples over datasets
            ds = injs[c["inj_id"]]["dataset"]
            if ds not in seen:
                chosen.append(c)
                seen.add(ds)
            if len(chosen) >= k:
                break
        for c in chosen:
            inj = injs[c["inj_id"]]
            head = inj["prefix_corrupt"][-200:].replace("\n", " ⏎ ")
            tail = c["greedy_text"][:300].replace("\n", " ⏎ ")
            w = c.get("wait", {}).get("outcome")
            blocks.append(
                f"**{want}{' / ' + rtype if rtype else ''}** · `{c['inj_id']}` ({inj['dataset']}) · {inj['kind']} "
                f"{inj['true_text']} → **{inj['new_text']}** · gold {inj['gold']:g}, propagated "
                f"{inj['propagated_answer']:g}, got {c['greedy']['answer']} · with Wait: {w}\n\n"
                f"> …{head} ‖ {tail}…\n")
    return "\n".join(blocks)


# ============================================================================ main

def write_report(cfg):
    rd = run_dir(cfg)
    R = json.load(open(rd / "results.json"))
    S = json.load(open(rd / "steer_summary.json")) if (rd / "steer_summary.json").exists() else None
    figs = figures(R, S, rd)
    p, c, v = R["probes"], R["counts"], R["verdict"]
    L = [f"# Error propagation in chain of thought: {cfg['run_name']}", "",
         f"Model `{cfg['model']}` · {c['problems']} problems · {c['injections']} injections · "
         f"valid {c['valid_rate']:.3f}", "",
         "## Verdict (pre-registered rule, DIRECTION.md)", ""]
    for key in ("P1_catchable", "P1_all"):
        q = v.get(key, {})
        L.append(f"- **{key.replace('_', ' ')}**: {q.get('summary')}"
                 + (f" (AUROC {_f3(q.get('auroc'))}, within-problem {_f3(q.get('within_auroc'))} on "
                    f"{q.get('n_mixed_problems')} problems, gain {_f3(q.get('delta'))} {_ci(q.get('delta_ci'))}, "
                    f"permutation p {q.get('perm_p', float('nan')):.3f})" if 'auroc' in q else ""))
    L += ["", "## Labels", "",
          f"- On-path outcomes (valid): `{c['outcomes_on_path']}`; off-path controls: `{c['outcomes_off_path']}`",
          f"- Recovery types (on-path RECOVER): `{c['recovery_types']}`",
          f"- Excluded: clean prefix not solved {1 - c['clean_cont_correct_rate']:.3f}; model extended the "
          f"injected number {c['glued_rate']:.3f}; propagated answer rounds to gold {c['ambiguous_label_n']} rows",
          f"- Catchable (relative error ≥ 50%) share of on-path: {c['catchable_rate_on_path']:.3f}; "
          f"uptake of the injected number: {c['uptake_rate_on_path']:.3f}", ""]
    L += _outcome_table("Outcome rates by dataset", c["by_dataset"])
    L += _outcome_table("Outcome rates by operator", c["by_kind"])
    L += _outcome_table("Outcome rates by error size (relative error)", c["by_err_bin"])
    if c.get("chains_follow_by_depth"):
        L += [f"**Synthetic chains, FOLLOW rate by chain depth:** `{c['chains_follow_by_depth']}`  ",
              f"**by number of downstream steps:** `{c['chains_follow_by_downstream']}`", ""]

    p0 = p.get("P0_detect", {})
    if p0.get("test"):
        L += ["## P0 · Corruption detection (sanity)", "",
              f"Test AUROC {p0['test']['auroc']:.3f} at layer {p0['best_layer']} (expected ≈ 1: the wrong digits "
              "are in the text).", ""]
    L += _p1_section("P1 · FOLLOW vs RECOVER, catchable errors (headline)", p.get("P1_catchable", {}),
                     "Pre-registered headline subset: relative error ≥ 50%. Smaller errors are followed almost "
                     "always, so they carry little information about *catching*.")
    L += _p1_section("P1 · FOLLOW vs RECOVER, all on-path injections", p.get("P1_all", {}))
    rob = p.get("P1_robustness", {})
    if rob:
        L += ["## P1 robustness", "",
              "Layer chosen from a ±3 window around the headline layer, on each subset's own dev split."]
        L += _rob_table("By operator", rob.get("by_operator", {}),
                        "Inside one operator, 'which kind of error' can no longer be the shortcut.")
        L += _rob_table("By dataset", rob.get("by_dataset", {}))
        L += _rob_table("By error size", rob.get("by_err_bin", {}))
        st = rob.get("split_stability")
        if st:
            L += [f"### Across 5 random group-level splits ({st['subset']})", "",
                  "| Seed | Layer | Probe AUROC | Within-problem (n) | Covariates AUROC | Gain |", "|---|---|---|---|---|---|"]
            for r in st["runs"]:
                L.append(f"| {r['seed']} | {r['best_layer']} | {_f3(r['probe_auroc'])} | {_f3(r['probe_within'])} "
                         f"({r['n_mixed']}) | {_f3(r['cov_auroc'])} | {_f3(r['delta'])} |")
            m = st["mean_sd"]
            L += [f"| **mean ± sd** | | {m['probe_auroc'][0]:.3f} ± {m['probe_auroc'][1]:.3f} | "
                  f"{m['probe_within'][0]:.3f} ± {m['probe_within'][1]:.3f} | {m['cov_auroc'][0]:.3f} ± "
                  f"{m['cov_auroc'][1]:.3f} | {m['delta'][0]:.3f} ± {m['delta'][1]:.3f} |", ""]

    if p.get("W_wait_policy"):
        L += _wait_section(p["W_wait_policy"])
    L += _steer_section(S)

    for key, title in [("P1r_follow_rate", "P1r · Follow rate over greedy + 4 samples"),
                       ("P4_horizon", "P4 · Contamination horizon")]:
        if p.get(key):
            q = p[key]
            L += [f"## {title}", "", "| Predictor | Test Spearman [CI] | Within-problem Spearman | n |",
                  "|---|---|---|---|", _reg_row(f"Linear probe (layer {q['best_layer']})", q["probe"]),
                  _reg_row("All covariates", q["all_covariates"]), ""]
    if p.get("P2_value_tracking"):
        q = p["P2_value_tracking"]
        best = max(q["curve"], key=lambda r: (r.get("reads_true_RECOVER") or 0) - (r.get("reads_true_FOLLOW") or 0))
        L += ["## P2 · Value tracking", "",
              f"{q['n']} injections where the last digit changes ({q['n_test']} test). Largest RECOVER-vs-FOLLOW gap "
              f"at layer {best['layer']}: true digit read on {_f3(best['reads_true_RECOVER'])} of RECOVER vs "
              f"{_f3(best['reads_true_FOLLOW'])} of FOLLOW runs; margin AUROC {_f3(best.get('margin_auroc_recover'))}.", ""]
    q = p.get("P3_silent_recovery", {})
    L += ["## P3 · Silent vs visible recovery", "", f"{q.get('n')} on-path RECOVER runs, silent rate {_f3(q.get('silent_rate'))}."]
    if "probe" in q:
        L += ["", "| Scorer | Test AUROC [CI] | Within-problem AUROC [CI] | n |", "|---|---|---|---|",
              _row(f"Linear probe (layer {q['best_layer']})", q["probe"]), _row("All covariates", q["all_covariates"])]
    L.append("")
    if p.get("P5_susceptibility"):
        q = p["P5_susceptibility"]
        L += ["## P5 · Prompt-time susceptibility (before any reasoning)", "",
              "| Predictor | Test Spearman [CI] | Within-problem | n problems |", "|---|---|---|---|",
              _reg_row(f"Linear probe (layer {q['best_layer']})", q["probe"]),
              _reg_row("Difficulty covariates", q["difficulty_covariates"]), ""]

    L += ["## Figures", ""] + [f"![{f}]({f})" for f in figs] + ["",
          "## Examples (greedy; ‖ marks where the injected prefix ends)", "", examples(rd)]
    (rd / "REPORT.md").write_text("\n".join(L))
    print(f"[report] wrote {rd / 'REPORT.md'}")

    # copy the small, shareable outputs next to the code
    out = Path(cfg.get("report_dir", "reports")) / cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    for name in ["REPORT.md", "results.json", "steer_summary.json", "p1_scores_all.csv",
                 "p1_scores_catchable.csv"] + figs:
        if (rd / name).exists():
            shutil.copy(rd / name, out / name)
    print(f"[report] copied to {out}")
