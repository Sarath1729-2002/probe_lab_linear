"""Stage `paper`: gather every model's results_paper.json (+ trajectories.json) into PAPER.md,
laid out like "How Much is Left?" (Merzouk et al. 2026): setup, Table 1 (prompt-only vs constant),
Table 2 (probe family), robustness, cross-dataset transfer, dynamic readout, limitations, appendix.
Safe to run on a login node; models whose results are missing are skipped."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .data import SYSTEM, INSTRUCTION  # noqa: E402
from .io import read_jsonl  # noqa: E402

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
GRID, INK, MUTED = "#e6e5e0", "#2b2b29", "#8a8984"
ORDER = ["chains", "bbh_arith", "gsm8k", "math", "mmlu_pro", "openthoughts", "triviaqa",
         "gsm_hard", "gsmsym_p1", "gsmsym_p2"]
NAME = {"chains": "Chains", "bbh_arith": "BBH-Arith", "gsm8k": "GSM8K", "gsm_hard": "GSM-Hard",
        "gsmsym_p1": "GSM-Sym P1", "gsmsym_p2": "GSM-Sym P2", "math": "MATH", "mmlu_pro": "MMLU-Pro",
        "openthoughts": "OpenThoughts-1k", "triviaqa": "TriviaQA", "pooled": "Pooled"}
PRED_NAME = {"constant": "Constant baseline", "blackbox": "Black-box predictor",
             "prompt_only": "Prompt-only probe", "error_token": "Error-token probe",
             "prompt_struct": "Prompt-only + structure", "stacked": "Black-box + error-token",
             "surface_prompt": "Surface features (prompt)", "surface_context": "Surface features (prompt + context)",
             "shuffle_global": "Shuffled labels (global)", "shuffle_within": "Shuffled labels (within problem)"}


def _f(x, d=3):
    return "–" if x is None or x != x else f"{x:.{d}f}"


def _m(cell, pred, metric="auroc"):
    v = (cell or {}).get(pred, {}).get(metric)
    return v[0] if v else None


def _sd(cell, pred, metric="auroc"):
    v = (cell or {}).get(pred, {}).get(metric)
    return v[1] if v else None


def _axes(ax, title, xl, yl):
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel(xl, color=MUTED)
    ax.set_ylabel(yl, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


# ============================================================================ load

def load(cfg):
    base = Path(os.path.expandvars(cfg["out_dir"]))
    runs = []
    for m in cfg["models"]:
        rd = base / f"{cfg['run_prefix']}-{m['tag']}"
        if not (rd / "results_paper.json").exists():
            print(f"[paper] no results for {m['tag']} ({rd}); skipping")
            continue
        R = json.load(open(rd / "results_paper.json"))
        T = json.load(open(rd / "trajectories.json")) if (rd / "trajectories.json").exists() else None
        runs.append({"tag": m["tag"], "name": m["name"], "short": m.get("short", m["tag"]), "rd": rd, "R": R, "T": T})
    return runs


def _cols(runs):
    present = set()
    for r in runs:
        present |= set(r["R"]["cells"])
    return [d for d in ORDER if d in present] + ["pooled"]


# ============================================================================ tables

def _bold(vals, higher=True):
    ok = [v for v in vals if v is not None and v == v]
    if not ok:
        return [_f(v) for v in vals]
    best = max(ok) if higher else min(ok)
    return [f"**{_f(v)}**" if v is not None and v == v and abs(v - best) < 1e-12 else _f(v) for v in vals]


def table_setup(runs):
    L = ["| Model | Dataset | Problems | Greedy acc. | Problems injected | Kept injections | FOLLOW | RECOVER | OTHER |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        for d in ORDER:
            x = r["R"]["datasets"].get(d)
            if x:
                L.append(f"| {r['short']} | {NAME[d]} | {x['problems']} | {_f(x['accuracy'], 2)} | "
                         f"{x['problems_injected']} | {x['kept']} | {_f(x['follow'], 2)} | {_f(x['recover'], 2)} | "
                         f"{_f(x['other'], 2)} |")
    return L


def table_metric(runs, cols, preds, metric, higher, title_note=""):
    L = ["| Predictor | " + " | ".join(NAME[c] for c in cols) + " |", "|---|" + "---|" * len(cols)]
    for r in runs:
        L.append(f"| *{r['short']}* |" + " |" * len(cols))
        cells = r["R"]["cells"]
        grid = {p: [_m(cells.get(c), p, metric) for c in cols] for p in preds}
        bolded = {p: [] for p in preds}
        for j in range(len(cols)):
            col = [grid[p][j] for p in preds]
            b = _bold(col, higher)
            for i, p in enumerate(preds):
                bolded[p].append(b[i])
        for p in preds:
            L.append(f"| {PRED_NAME[p]} | " + " | ".join(bolded[p]) + " |")
    return L


def table_sd(runs, cols):
    L = ["| Model | Predictor | " + " | ".join(NAME[c] for c in cols) + " |", "|---|---|" + "---|" * len(cols)]
    for r in runs:
        for p in ("blackbox", "prompt_only", "error_token"):
            L.append(f"| {r['short']} | {PRED_NAME[p]} | " + " | ".join(
                f"{_f(_m(r['R']['cells'].get(c), p))} ± {_f(_sd(r['R']['cells'].get(c), p))}" for c in cols) + " |")
    return L


def implied_claims(runs, cols):
    n = gt_c = gt_bb = gt_po = po_c = gt_ps = 0
    for r in runs:
        for c in cols:
            cell = r["R"]["cells"].get(c) or {}
            et = _m(cell, "error_token")
            if et is None:
                continue
            n += 1
            gt_c += et > 0.5
            gt_bb += et > (_m(cell, "blackbox") or 0)
            gt_po += et > (_m(cell, "prompt_only") or 0)
            gt_ps += et > (_m(cell, "prompt_struct") or 0)
            po_c += (_m(cell, "prompt_only", "rel_brier") or 9) < 1.0
    return {"cells": n, "et_gt_constant": gt_c, "et_gt_blackbox": gt_bb, "et_gt_prompt": gt_po, "po_beats_const_brier": po_c,
            "et_gt_prompt_struct": gt_ps}


def table_robust(runs):
    L = ["| Model | Error-token AUROC | Within-problem AUROC (n problems) | Gain over black-box (mean ± sd) | "
         "Gain, 95% CI (split 0) | Permutation null: max / p | Catchable: probe / black-box | Horizon ρ: probe / black-box |",
         "|---|---|---|---|---|---|---|---|"]
    for r in runs:
        R = r["R"]
        p = R["cells"].get("pooled", {})
        s0 = R.get("pooled_seed0", {})
        ci = (s0.get("ci") or {}).get("delta_over_blackbox")
        perm = s0.get("permutation", {})
        cat = R.get("robustness", {}).get("catchable", {})
        h = R.get("horizon", {})
        L.append(f"| {r['short']} | {_f(_m(p, 'error_token'))} | {_f(_m(p, 'error_token', 'within'))} "
                 f"({_f(_m(p, 'error_token', 'n_mixed'), 0)}) | {_f((p.get('delta_over_blackbox') or [None])[0])} ± "
                 f"{_f((p.get('delta_over_blackbox') or [None, None])[1])} | "
                 f"{'[' + _f(ci[0]) + ', ' + _f(ci[1]) + ']' if ci else '–'} | "
                 f"{_f(perm.get('max'))} / {_f(perm.get('p'))} | {_f(_m(cat, 'error_token'))} / {_f(_m(cat, 'blackbox'))} | "
                 f"{_f(h.get('probe_spearman'))} / {_f(h.get('blackbox_spearman'))} |")
    return L


def table_operator(runs):
    kinds = ["plus_one", "digit_swap", "times_ten", "random_scale", "random_same_digits"]
    L = ["| Model | " + " | ".join(kinds) + " |", "|---|" + "---|" * len(kinds)]
    for r in runs:
        ops = r["R"].get("robustness", {}).get("by_operator", {})
        cells = []
        for k in kinds:
            o = ops.get(k, {})
            if "skipped" in o or not o:
                cells.append(f"– (FOLLOW {_f(o.get('follow_rate'), 2)})")
            else:
                cells.append(f"{_f(_m(o, 'error_token'))} / {_f(_m(o, 'blackbox'))} (F {_f(o.get('follow_rate'), 2)})")
        L.append(f"| {r['short']} | " + " | ".join(cells) + " |")
    return L


def table_transfer(r, key="probe_rel_brier", bb=False):
    """Paper Table 3/13: rows = train dataset, columns = eval dataset. Default entries are Brier score
    divided by the eval dataset's own constant (its test-split FOLLOW rate), so < 1 beats the floor."""
    T = r["R"].get("transfer")
    if not T or key not in T:
        return ["Not available."]
    ds = T["datasets"]
    order = [d for d in ORDER if d in ds]
    idx = [ds.index(d) for d in order]
    syn = set(T.get("synthetic", []))
    L = ["| Train \\ Eval | " + " | ".join(NAME[d] for d in order) + " |", "|---|" + "---|" * len(order)]
    if key == "probe_rel_brier":
        L.append("| *eval floor (Brier)* | " + " | ".join(_f(T["floor_brier"][j]) for j in idx) + " |")
    better = (lambda v: v < 1) if key == "probe_rel_brier" else (lambda v: v > 0.5)
    for i in idx:
        cells = []
        for j in idx:
            v = T[key][i][j]
            t = _f(v)
            cells.append(f"*{t}*" if i == j else (f"**{t}**" if v == v and v is not None and better(v) else t))
        tag = " (syn.)" if ds[i] in syn else ""
        L.append(f"| {NAME[ds[i]]}{tag} | " + " | ".join(cells) + " |")
        if bb:
            L.append("| *black-box* | " + " | ".join(_f(T["blackbox"][i][j]) for j in idx) + " |")
    return L


def asymmetry(runs, crit="rel_brier_below_1"):
    """Paper §5 / App. A.10: off-diagonal cells beating the eval floor, natural- vs synthetic-trained,
    per model and pooled, with a one-sided Fisher exact test (cells sharing a train row are not
    independent, so the per-model replication is reported alongside, as in the paper)."""
    from scipy.stats import fisher_exact
    L = ["| Model | Natural-trained: beat floor / cells | Synthetic-trained: beat floor / cells |", "|---|---|---|"]
    tot = {"natural": [0, 0], "synthetic": [0, 0]}
    for r in runs:
        a = ((r["R"].get("transfer") or {}).get("asymmetry") or {}).get(crit)
        if not a:
            continue
        for k in tot:
            tot[k][0] += a[k][0]
            tot[k][1] += a[k][1]
        L.append(f"| {r['short']} | {a['natural'][0]} / {a['natural'][1]} | {a['synthetic'][0]} / {a['synthetic'][1]} |")
    n, sy = tot["natural"], tot["synthetic"]
    p = None
    if n[1] and sy[1]:
        p = fisher_exact([[n[0], n[1] - n[0]], [sy[0], sy[1] - sy[0]]], alternative="greater")[1]
    L.append(f"| **Pooled** | {n[0]} / {n[1]} ({100 * n[0] / max(1, n[1]):.1f}%) | "
             f"{sy[0]} / {sy[1]} ({100 * sy[0] / max(1, sy[1]):.1f}%) |")
    return L, p, tot


def table_controls(runs):
    """Paper App. A.12 (surface features, shuffled labels) on the pooled cell."""
    preds = ["constant", "surface_prompt", "prompt_only", "surface_context", "shuffle_global", "shuffle_within",
             "error_token"]
    L = ["| Model | " + " | ".join(PRED_NAME[p] for p in preds) + " |", "|---|" + "---|" * len(preds)]
    for r in runs:
        c = r["R"]["cells"].get("pooled")
        L.append(f"| {r['short']} | " + " | ".join(f"{_f(_m(c, p))} / {_f(_m(c, p, 'rel_brier'))}" for p in preds) + " |")
    return L


def table_noise(runs):
    L = ["| Model | Injections × samples | Greedy-label agreement | Mixed outcomes | Brier: constant | Brier: error-token | "
         "Brier: prompt-only | Brier: LOO floor | Closable range captured (error-token / prompt-only) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        z = r["R"].get("noise_floor")
        if not z:
            continue
        cap = lambda v: "–" if v is None else f"{100 * v:.0f}%"   # noqa: E731
        L.append(f"| {r['short']} | {z['n_injections']} × ~{z['n_samples'] // max(1, z['n_injections'])} | "
                 f"{_f(z['greedy_agreement'], 2)} | {_f(z['frac_mixed'], 2)} | {_f(z['brier_constant'])} | "
                 f"{_f(z['brier_error_token'])} | {_f(z['brier_prompt_only'])} | {_f(z['brier_loo_floor'])} | "
                 f"{cap(z['captured_error_token'])} / {cap(z['captured_prompt_only'])} |")
    return L


def table_horizon(runs):
    L = ["| Model | n | Mean horizon | Constant-median MAE | Black-box MAE | Probe MAE | Probe / floor | Spearman ρ (probe / black-box) |",
         "|---|---|---|---|---|---|---|---|"]
    for r in runs:
        h = r["R"].get("horizon")
        if not h or "mae_probe" not in h:
            continue
        L.append(f"| {r['short']} | {h['n']} | {_f(h['mean_horizon'], 2)} | {_f(h['mae_constant_median'])} | "
                 f"{_f(h['mae_blackbox'])} | **{_f(h['mae_probe'])}** | {_f(h['mae_probe'] / max(h['mae_constant_median'], 1e-9), 2)} | "
                 f"{_f(h['probe_spearman'])} / {_f(h['blackbox_spearman'])} |")
    return L


def table_events(runs):
    """Paper §4.3 / App. A.16 statistics."""
    L = ["| Model | Event | Events (continuations) | Δ logit at event: mean / median | Matched controls: mean / median | "
         "Down: event / control | Cohen's d | p sign-flip | p label shuffle |", "|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        T = r["T"]
        if not T or "events" not in T:
            continue
        for name, e in T["events"].items():
            if "event_mean" not in e:
                L.append(f"| {r['short']} | {name} | {e.get('n_events', 0)} | – | – | – | – | – | – |")
                continue
            L.append(f"| {r['short']} | {name} | {e['n_events']} ({e['n_continuations']}) | {_f(e['event_mean'], 2)} / "
                     f"{_f(e['event_median'], 2)} | {_f(e['control_mean'], 2)} / {_f(e['control_median'], 2)} | "
                     f"{100 * e['event_frac_down']:.0f}% / {100 * e['control_frac_down']:.0f}% | {_f(e['cohens_d'], 2)} | "
                     f"{e['p_signflip']:.1e} | {e['p_label_shuffle']:.1e} |")
    return L


def table_event_breakdown(runs):
    L = ["| Model | Trigger in continuations | Rank of trigger among most negative token types (discovery half) | "
         "Confirmation half: d / p | In FOLLOW: n / d | In RECOVER: n / d |", "|---|---|---|---|---|---|"]
    for r in runs:
        T = r["T"]
        if not T or "events" not in T:
            continue
        e = T["events"].get(f"trigger_{T['trigger']}", {})
        ch, fo, re_ = e.get("confirmation_half", {}), e.get("by_outcome_FOLLOW", {}), e.get("by_outcome_RECOVER", {})
        D = T["discovery"]
        L.append(f"| {r['short']} | {T['n_items_with_trigger']} / {T['n_items']} | "
                 f"{D.get('trigger_rank_most_negative') or '–'} of {D['n_token_types']} | "
                 f"{_f(ch.get('cohens_d'), 2)} / {('%.1e' % ch['p_signflip']) if 'p_signflip' in ch else '–'} | "
                 f"{fo.get('n_events', 0)} / {_f(fo.get('cohens_d'), 2)} | {re_.get('n_events', 0)} / {_f(re_.get('cohens_d'), 2)} |")
    return L


def discovery_lists(r):
    T = r["T"]
    if not T or "discovery" not in T:
        return []
    D = T["discovery"]
    fmt = lambda xs: ", ".join(f"`{t.replace('|', '/')}` {m:+.2f} (n={n})" for t, m, n in xs)   # noqa: E731
    return [f"*{r['short']}* — global drift {_f(D['global_drift'], 3)} per token. Most negative: {fmt(D['most_negative'][:8])}. "
            f"Most positive: {fmt(D['most_positive'][:5])}.", ""]


def table_kappa(runs, cols):
    preds = ["blackbox", "prompt_only", "prompt_struct", "error_token"]
    L = ["| Model | Predictor | " + " | ".join(NAME[c] for c in cols) + " |", "|---|---|" + "---|" * len(cols)]
    for r in runs:
        for p in preds:
            L.append(f"| {r['short']} | {PRED_NAME[p]} | " + " | ".join(
                _f(_m(r["R"]["cells"].get(c), p, "kappa")) for c in cols) + " |")
    return L


# ============================================================================ figures

def fig_layers(runs, out):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    _axes(ax, "Dev cross-validated AUROC by relative depth (pooled, split 0)", "layer / number of layers", "AUROC")
    for i, r in enumerate(runs):
        cv = (r["R"].get("pooled_seed0") or {}).get("curves")
        if not cv:
            continue
        L = r["R"]["n_layers"] - 1
        for key, ls, lab in [("error_token", "-", "error-token"), ("prompt_only", "--", "prompt-only")]:
            c = cv[key]
            ax.plot([x["layer"] / L for x in c], [x["dev_cv"] for x in c], color=C[i % len(C)], linestyle=ls,
                    linewidth=2, marker="o", markersize=2.5, label=f"{r['short']}, {lab}")
    ax.axhline(0.5, color=MUTED, linestyle=":", linewidth=1)
    ax.legend(frameon=False, fontsize=7, loc="center right", ncol=1)
    fig.tight_layout()
    fig.savefig(out / "fig_layers.png", dpi=150)
    plt.close(fig)
    return "fig_layers.png"


def fig_traj(runs, out):
    rs = [r for r in runs if r["T"]]
    if not rs:
        return []
    names = []
    for key, fname, xl, title in [
        ("curves", "fig_trajectory.png", "tokens relative to the injected number", "P(FOLLOW) read at every token"),
        ("aligned", "fig_aligned.png", "tokens relative to the event", "Aligned at the first re-use of a value")]:
        fig, axes = plt.subplots(1, len(rs), figsize=(3.6 * len(rs), 3.2), squeeze=False)
        for ax, r in zip(axes[0], rs):
            T = r["T"]
            _axes(ax, f"{r['short']} (layer {T['layer']})", xl, "probe P(FOLLOW)")
            if key == "curves":
                series = [("FOLLOW", C[1], "FOLLOW"), ("RECOVER", C[0], "RECOVER"), ("CLEAN", MUTED, "clean prefix")]
                xs = T["rel"]
                for k, col, lab in series:
                    s = T["curves"][k]
                    if s["n"]:
                        ax.plot(xs, s["mean"], color=col, linewidth=2, label=f"{lab} (n={s['n']})")
                ax.axvline(0, color=INK, linewidth=0.8, linestyle=":")
            else:
                xs = T["aligned_rel"]
                for k, col, lab in [("FOLLOW_at_injected_value_reused", C[1], "FOLLOW: wrong value re-used"),
                                    ("RECOVER_at_true_value_rewritten", C[0], "RECOVER: true value re-written")]:
                    s = T["aligned"][k]
                    if s["n"]:
                        ax.plot(xs, s["mean"], color=col, linewidth=2, label=f"{lab} (n={s['n']})")
                ax.axvline(0, color=INK, linewidth=0.8, linestyle=":")
            ax.set_ylim(-0.02, 1.02)
            ax.legend(frameon=False, fontsize=6.5, loc="best")
        fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=INK)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=150)
        plt.close(fig)
        names.append(fname)
    return names


def fig_events(runs, out):
    rs = [r for r in runs if r["T"] and "events" in r["T"]
          and "hist_event" in r["T"]["events"].get(f"trigger_{r['T']['trigger']}", {})]
    if not rs:
        return []
    fig, axes = plt.subplots(1, len(rs), figsize=(3.8 * len(rs), 3.2), squeeze=False)
    for ax, r in zip(axes[0], rs):
        e = r["T"]["events"][f"trigger_{r['T']['trigger']}"]
        lo, hi = np.percentile(e["hist_control"] + e["hist_event"], [1, 99])
        bins = np.linspace(lo, hi, 40)
        ax.hist(e["hist_control"], bins=bins, density=True, color=MUTED, alpha=0.6, label=f"matched controls (n={e['n_controls']})")
        ax.hist(e["hist_event"], bins=bins, density=True, color=C[0], alpha=0.75, label=f"\"{r['T']['trigger']}\" (n={e['n_events']})")
        _axes(ax, f"{r['short']}: d = {e['cohens_d']:.2f}", "one-step change of probe logit", "density")
        ax.legend(frameon=False, fontsize=6.5)
    fig.suptitle("Probe step at the retraction token vs matched controls (paper Fig. 11 analogue)", x=0.01, ha="left",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out / "fig_events.png", dpi=150)
    plt.close(fig)
    return ["fig_events.png"]


def event_example(r):
    T = r["T"]
    if not T:
        return []
    ex = (T.get("events", {}).get(f"trigger_{T.get('trigger')}", {}) or {}).get("examples") or []
    if not ex:
        return []
    e = ex[0]
    L = [f"*{r['short']}, `{e['inj_id']}` ({e['outcome']}). Row ▶ is the trigger token.*", "",
         "| t | Token | P(FOLLOW) |", "|---|---|---|"]
    for i, (tk, p) in enumerate(zip(e["tokens"], e["p_follow"])):
        tk = tk.replace("\n", "⏎").replace("|", "\\|")
        L.append(f"| {'▶ ' if i == e['event_index'] else ''}{i - e['event_index']:+d} | `{tk}` | {p:.3f} |")
    return L + [""]


# ============================================================================ worked examples

def worked_example(r):
    """Figure-1 analogue: one held-out injection where the error-token probe is right and the
    black-box predictor is wrong, with every predictor's P(FOLLOW)."""
    p = r["rd"] / "scores_pooled.csv"
    if not p.exists():
        return []
    s = pd.read_csv(p)
    s = s[s.split == "test"]
    good = s[((s.y == 0) & (s.p_error_token < 0.3) & (s.p_blackbox > 0.6))]
    if not len(good):
        good = s.assign(m=(s.p_error_token - s.y).abs() - (s.p_blackbox - s.y).abs()).sort_values("m").head(1)
    row = good.iloc[0]
    inj = next(i for i in read_jsonl(r["rd"] / "injections.jsonl") if i["inj_id"] == row.inj_id)
    cont = next(c for c in read_jsonl(r["rd"] / "continuations.jsonl") if c["inj_id"] == row.inj_id)
    head = inj["prefix_corrupt"][-260:].replace("\n", " ⏎ ")
    tail = cont["greedy_text"][:260].replace("\n", " ⏎ ")
    truth = "FOLLOW (1)" if row.y == 1 else "RECOVER (0)"
    return [f"*{r['short']}, `{row.inj_id}` ({NAME.get(row.dataset, row.dataset)}), operator `{row.kind}`: "
            f"{inj['true_text']} → **{inj['new_text']}**. Gold {inj['gold']:g}, answer if followed "
            f"{inj['propagated_answer']:g}, model's answer {cont['greedy']['answer']}. Truth: {truth}.*", "",
            f"> …{head} **‖** {tail}…", "",
            "| Predictor | Reads | P(FOLLOW) |", "|---|---|---|",
            f"| Constant baseline | nothing | {_f(row.p_constant, 2)} |",
            f"| Black-box predictor | operator, error size, position, surprisal | {_f(row.p_blackbox, 2)} |",
            f"| Prompt-only probe | residual stream at the last prompt token | {_f(row.p_prompt_only, 2)} |",
            f"| Error-token probe | residual stream at the injected number | **{_f(row.p_error_token, 2)}** |", ""]


def token_table(r):
    T = r["T"]
    if not T or not T["examples"]:
        return []
    ex = next((e for e in T["examples"] if e["outcome"] == "RECOVER"), T["examples"][0])
    L = [f"*{r['short']}, `{ex['inj_id']}`, {ex['kind']} {ex['true']} → {ex['new']}, outcome {ex['outcome']}. "
         f"Row marked ▶ is the last token of the injected number (where the probe was trained).*", "",
         "| t | Token | P(FOLLOW) |", "|---|---|---|"]
    for i, (tk, p) in enumerate(zip(ex["tokens"], ex["p_follow"])):
        mark = "▶ " if i == ex["inj_index"] else ""
        tk = tk.replace("\n", "⏎").replace("|", "\\|")
        L.append(f"| {mark}{i - ex['inj_index']:+d} | `{tk}` | {p:.3f} |")
    return L + [""]


# ============================================================================ main

def write_paper(cfg):
    runs = load(cfg)
    if not runs:
        print("[paper] nothing to report yet")
        return
    out = Path(cfg.get("report_dir", "reports")) / cfg["run_prefix"]
    out.mkdir(parents=True, exist_ok=True)
    cols = _cols(runs)
    ic = implied_claims(runs, cols)
    figs_traj = fig_traj(runs, out)
    figs_ev = fig_events(runs, out)
    fig_layers(runs, out)
    pooled = [(r["short"], _m(r["R"]["cells"].get("pooled"), "error_token"),
               _m(r["R"]["cells"].get("pooled"), "blackbox"), _m(r["R"]["cells"].get("pooled"), "prompt_only"))
              for r in runs]
    ets = [p[1] for p in pooled if p[1] is not None] or [float("nan")]
    bbs = [p[2] for p in pooled if p[2] is not None] or [float("nan")]
    pos = [p[3] for p in pooled if p[3] is not None] or [float("nan")]
    n_ds = len([c for c in cols if c != "pooled"])
    asym_L, asym_p, asym_tot = asymmetry(runs)
    trig = next((r["T"]["trigger"] for r in runs if r["T"]), "Wait")
    ev_lines = []
    for r in runs:
        e = ((r["T"] or {}).get("events") or {}).get(f"trigger_{trig}", {})
        if "event_mean" in e:
            ev_lines.append(f"{r['short']}: {e['event_mean']:+.2f} vs {e['control_mean']:+.2f}, d = {e['cohens_d']:.2f}, "
                            f"n = {e['n_events']}")
    syn_names = ", ".join(NAME.get(d, d) for d in cfg["data"].get("synthetic", []))

    L = ["# Do LLMs Linearly Encode Whether They Will Follow an Error?",
         "### A level-1 recreation of *How Much is Left?* (Merzouk et al. 2026) for error propagation in chain of thought",
         "", "*Initial result. Probing only: every claim below is about what is linearly decodable, not about "
         "what the model causally uses. No steering is reported.*", "",
         "## Abstract", "",
         f"When a wrong value enters a model's chain of thought, the model either carries it to a wrong answer "
         f"(FOLLOW) or reaches the right answer anyway (RECOVER). We ask whether the residual stream at the moment "
         f"the wrong value is written already encodes which of the two will happen. We inject deterministic errors into "
         f"each model's own solutions (a wrong computed value, or for TriviaQA a wrong recalled fact), label every "
         f"continuation exactly (no human or LLM judge), and train minimal-capacity linear probes on frozen hidden states "
         f"of {len(runs)} open-weight 7–8B models across {n_ds} datasets, mirroring the paper's probe family, statistical "
         f"floors, cross-dataset matrices and aggregate event analysis. Pooled over datasets, the error-token probe "
         f"reaches AUROC {_f(min(ets), 2)}–{_f(max(ets), 2)} across models, against {_f(min(bbs), 2)}–{_f(max(bbs), 2)} "
         f"for a black-box predictor that sees the error and the model's surprisal at it and {_f(min(pos), 2)}–"
         f"{_f(max(pos), 2)} for a probe at the last prompt token. Over {ic['cells']} (model, dataset) cells it beats the "
         f"black-box predictor in {ic['et_gt_blackbox']}, the prompt-only probe in {ic['et_gt_prompt']}, and the "
         f"prompt-only probe combined with all error features in {ic['et_gt_prompt_struct']}. The prompt-only probe beats "
         f"the constant floor in {ic['po_beats_const_brier']} of {ic['cells']} cells: part of a problem's susceptibility "
         f"is readable before any reasoning is written."
         + (f" Probes trained on natural datasets beat the eval dataset's own floor on {asym_tot['natural'][0]} of "
            f"{asym_tot['natural'][1]} off-diagonal transfer cells, synthetic-trained probes on {asym_tot['synthetic'][0]} of "
            f"{asym_tot['synthetic'][1]} (one-sided Fisher p = {asym_p:.1e})." if asym_p is not None else "")
         + (f" At the pre-specified retraction token \"{trig}\" the probe's logit moves by (event vs matched controls) "
            + "; ".join(ev_lines) + "." if ev_lines else ""), "",
         "## 1 Setup", "",
         "**Target** (paper §3.1 analogue). At a site in a model's own correct solution we replace the value with a wrong "
         "one and let the model continue greedily. *y* = 1 (FOLLOW) if the final answer equals the answer implied by the "
         "wrong value (recomputed symbolically from the model's own equations; for a recalled fact, the wrong value "
         "itself), *y* = 0 (RECOVER) if it equals the gold answer. Other answers, ambiguous labels and set-up failures "
         "are excluded (Appendix B), as the paper excludes non-terminated generations. Errors: +1, swap of the last two "
         "digits, ×10, ×f with f log-uniform in [0.2, 5], and a random number with the same digit count.", "",
         "**Predictor family** (paper §3.3):", "",
         "| Ours | Paper | Reads |", "|---|---|---|",
         "| Constant baseline | Statistical baseline (eval median) | the dev-split FOLLOW rate |",
         "| Black-box predictor | — (observer baseline) | operator, relative error, position, steps downstream, trace "
         "length, dataset, surprisal of the injected digits |",
         "| Prompt-only + structure | Exact countdown | the prompt-only estimate plus everything the black-box sees: the "
         "pre-reasoning estimate updated only by known structure |",
         "| Prompt-only probe | Completion Length Probe | residual stream at the last prompt token, before any reasoning |",
         "| Error-token probe | Remaining Count Probe | residual stream at the last token of the injected value |", "",
         "Probes are L2-regularised logistic regressions on one layer, chosen by grouped 3-fold CV on the dev split; "
         "splits are by problem. Every number is the mean over "
         f"{len(cfg['probes']['split_seeds'])} independent dev/test splits (paper: 3 seeds). Metrics: AUROC (headline), "
         "and Brier score and log-loss divided by the constant baseline's (the paper's MAE / floor ratio: < 1 beats the "
         "floor); Cohen's κ in Appendix G (paper App. A.14).", "",
         "**Models.** " + ", ".join(f"{r['name']}" for r in runs) + ". The paper's Llama-3.1-8B and Mistral-7B; "
         "Qwen2.5-7B (our pilot model); and DeepSeek-R1-Distill-Qwen-7B, a reasoning model that retracts often and so "
         "plays the role of the paper's Qwen3-4B in the event analysis (§2.3).", "",
         "**Datasets** (paper §3.7: 2 synthetic + 5 standard), restricted to what this target needs, a numeric "
         "answer that can be checked exactly:", "",
         "| Dataset | Paper | Our constraint |", "|---|---|---|",
         "| Chains (synthetic) | Count | generated dependency chains, depth 3–8: structure fully controlled |",
         "| BBH-Arith (synthetic) | Countdown | BIG-Bench Hard multistep-arithmetic format |",
         "| GSM8K | GSM8K | test set |",
         "| MATH | MATH | test set, 7 subjects, \\boxed answer is a single number |",
         "| MMLU-Pro | MMLU-Pro | numeric-option questions only; options withheld, the model answers the number; "
         "2% tolerance (options are rounded) |",
         "| OpenThoughts-1k | OpenThoughts-1k | 1k sample of OpenThoughts-114k math problems with numeric answers |",
         "| TriviaQA | TriviaQA | numeric answers (years, counts); the injected error is a wrong recalled fact at the "
         "first place the model states it |", "",
         "**Worked example** (Figure 1 analogue).", ""]
    L += worked_example(runs[0])
    L += ["## 2 Results", "",
          "### 2.1 The outcome is partly decodable before any reasoning (paper §4.1, Tables 4–5)", "",
          "Brier score of the prompt-only probe relative to the constant baseline (< 1 beats the floor; bold is the "
          "winner per cell). The prompt-only probe sees the problem but not the error, so it can only estimate how "
          "error-prone the problem is.", ""]
    L += table_metric(runs, cols, ["constant", "prompt_only"], "rel_brier", higher=False)
    L += ["", "Controls (paper App. A.12–A.13): a TF-IDF + length predictor on the prompt text (Appendix E), and the "
          "sampling-noise floor, i.e. how predictable the outcome of a *sampled* continuation can be at all (Appendix F).", "",
          "### 2.2 Probe family (paper §4.2, Tables 1 and 6)", "",
          "Test AUROC, mean over splits; bold is the best predictor per cell.", ""]
    L += table_metric(runs, cols, ["constant", "blackbox", "prompt_only", "prompt_struct", "error_token"], "auroc", higher=True)
    L += ["", f"**Implied claims** (paper §4.2) over {ic['cells']} (model, dataset) cells: (i) the error-token probe beats "
          f"the constant in {ic['et_gt_constant']}; (ii) it beats the prompt-only estimate plus all error features "
          f"(the exact-countdown analogue) in {ic['et_gt_prompt_struct']}, i.e. the state at the error adds information "
          f"beyond the pre-reasoning estimate and what is known about the error; (iii) it beats the black-box predictor "
          f"in {ic['et_gt_blackbox']}. Relative Brier and log-loss of the full family: Appendix D.", "",
          "### 2.3 Dynamic re-estimation: the probe at retraction tokens (paper §4.3)", "",
          f"The error-token probe is read at every token of the continuation. Following the paper, the trigger token "
          f"\"{trig}\" is fixed in advance and every occurrence in held-out continuations is compared with matched "
          f"non-trigger controls (same dataset, same 50-token position bucket, other continuations, ±10 tokens around any "
          f"trigger excluded). Prediction: P(FOLLOW) *falls* when the model starts to retract. Two further pre-specified "
          f"events: the first token where the model re-writes the true value (RECOVER) or re-uses the injected one (FOLLOW).", ""]
    L += table_events(runs)
    L += ["", "*Δ = one-step change of the probe's logit. p sign-flip: cluster-safe test over per-continuation paired "
          "differences; p label shuffle: each event against its own matched controls.*", ""]
    L += table_event_breakdown(runs)
    L += ["", "*Discovery scan (paper App. A.16) on a disjoint half of the continuations: rank of the trigger among all "
          "token types (≥ 10 occurrences) by mean Δ, most negative first.*", ""]
    L += [f"![{f}]({f})" for f in figs_ev] + [""]
    for r in runs:
        L += discovery_lists(r)
    ex_run = next((r for r in runs if event_example(r)), None)
    if ex_run:
        L += ["**A retraction, token by token** (Figure 3 analogue).", ""] + event_example(ex_run)
    L += ["Mean P(FOLLOW) around the injection and aligned at the value events (qualitative; the probe was trained at "
          "one position):", ""] + [f"![{f}]({f})" for f in figs_traj] + [""]
    if runs[0]["T"]:
        L += token_table(runs[0])
    L += ["### 2.4 Robustness of the pooled result", "",
          "*Within-problem AUROC* compares only injections into the same problem. *Gain* is the AUROC of black-box + "
          "probe minus the black-box alone. The permutation null refits the probe on shuffled dev labels. *Catchable*: "
          "errors of at least 50%. *Horizon*: number of later equations using a wrong value (Appendix H).", ""]
    L += table_robust(runs)
    L += ["", "**Within one operator** (error-token / black-box AUROC; F = FOLLOW rate).", ""]
    L += table_operator(runs)
    L += ["", "## 3 Cross-dataset generalization (paper §5, Tables 3 and 13)", "",
          f"Error-token probe trained on one dataset's dev split, evaluated on another dataset's test split. Entries are "
          f"Brier score divided by the **eval dataset's own floor** (its constant predictor), the bar the paper uses: "
          f"bold off-diagonal entries (< 1) carry usable information into a dataset they never saw; the diagonal is in "
          f"italics. Synthetic training sets: {syn_names}. {runs[0]['short']} here; other models in Appendix C.", ""]
    L += table_transfer(runs[0])
    L += ["", "**Transfer asymmetry** (paper App. A.10): off-diagonal cells beating the eval floor, by training set type.", ""]
    L += asym_L
    if asym_p is not None:
        L += ["", f"One-sided Fisher exact test, natural > synthetic: p = {asym_p:.2e} (cells that share a training row "
              f"are not independent; the per-model rows are the replication)."]
    L += ["", "## 4 Limitations", "",
          "- **Decodability, not use.** Linear probes show the information is available, not that the model uses this "
          "direction. No steering or patching is reported here by design.",
          "- **Injected, not natural, errors.** Value edits at verified sites; the continuation is forced from a "
          "truncated prefix (Lanham et al. 2023).",
          "- **Dataset constraints.** Each standard set is restricted to exactly checkable numeric answers; MMLU-Pro is "
          "asked without its options and graded at 2% tolerance; TriviaQA errors are recalled facts, not computations.",
          "- **Greedy labels.** Appendix F measures how much the label moves under sampling.",
          "- **One training position.** §2.3 reads the probe off-distribution; only the matched-control statistics carry "
          "evidential weight, as in the paper.",
          "- **Reasoning model budgets.** R1-Distill traces are long; continuations that hit the token limit are OTHER "
          "and excluded (Appendix B).", "",
          "## Appendix", "", "### A Split-to-split standard deviation of test AUROC", ""]
    L += table_sd(runs, cols)
    L += ["", "### B Datasets, outcome rates and exclusions (paper App. A.7)", ""]
    L += table_setup(runs)
    L += ["", "| Model | Dataset | No final answer (truncated) | Site type |", "|---|---|---|---|"]
    for r in runs:
        for d in ORDER:
            x = r["R"]["datasets"].get(d)
            if x and "no_final_answer" in x:
                L.append(f"| {r['short']} | {NAME[d]} | {100 * x['no_final_answer']:.1f}% | {x.get('site', 'equation')} |")
    L += ["", "| Model | Injections | Clean prefix not solved | Injected number extended | Operand re-use | "
          "Rounds to gold | Kept |", "|---|---|---|---|---|---|---|"]
    for r in runs:
        e = r["R"]["exclusions"]
        L.append(f"| {r['short']} | {e['injections']} | {e['clean_prefix_failed']} | {e['glued']} | "
                 f"{e['operand_reuse']} | {e['rounds_to_gold']} | {e['kept']} |")
    L += ["", "### C Cross-dataset transfer, other models, and AUROC versions", ""]
    for r in runs[1:]:
        L += [f"**{r['short']}** (Brier / eval floor)", ""] + table_transfer(r) + [""]
    for r in runs:
        L += [f"**{r['short']}** (AUROC; black-box predictor under each row)", ""] + table_transfer(r, "probe", bb=True) + [""]
    L += ["### D Relative Brier and log-loss of the full family (paper Table 6 analogue)", ""]
    fam = ["constant", "blackbox", "prompt_only", "prompt_struct", "error_token"]
    L += table_metric(runs, cols, fam, "rel_brier", higher=False)
    L += [""] + table_metric(runs, cols, fam, "rel_logloss", higher=False)
    L += ["", "### E Controls: surface features and shuffled labels (paper App. A.12)", "",
          "Pooled cell, AUROC / relative Brier. *Surface (prompt)* is the paper's surface-feature control for the "
          "prompt-end probe (TF-IDF 1–2-grams, 5000 terms, plus lengths); *surface (prompt + context)* adds the 300 "
          "characters before and including the injected value, the comparable control for the error-token probe. "
          "*Global* shuffle permutes dev labels; *within-problem* shuffle permutes them only inside each problem, keeping "
          "each problem's FOLLOW rate (paper: within-completion shuffle), so error-token minus within-shuffle is the "
          "injection-level signal.", ""]
    L += table_controls(runs)
    L += ["", "### F Sampling-noise floor (paper App. A.13)", "",
          f"A random subset of on-path injections was continued {cfg['gen'].get('noise_k', 0)} more times at the "
          f"paper's decoding (T = {cfg['gen'].get('noise_temperature', 0.7)}, top-p = {cfg['gen'].get('noise_top_p', 0.8)}); "
          "only kept injections with ≥ 2 FOLLOW/RECOVER samples enter. The floor scores each sample against the mean of the "
          "others (leave-one-out, as in the paper), rescaled by (k−1)/k so it estimates the per-injection outcome variance "
          "p(1−p): no predictor that sees only the prefix can beat it in expectation. Probe scores are out-of-fold. "
          "*Captured* = (constant − probe) / (constant − floor). The probes were trained on greedy labels, so this is a "
          "conservative comparison.", ""]
    L += table_noise(runs)
    L += ["", "### G Cohen's κ at P = 0.5 (paper App. A.14 analogue)", ""]
    L += table_kappa(runs, cols)
    L += ["", "### H Contamination horizon: MAE against the constant-median floor (the paper's exact metric)", "",
          "Regression target: number of later equations in the continuation that use a wrong value (ridge probe on "
          "log(1 + horizon), error-token position). Floor: median of the eval split.", ""]
    L += table_horizon(runs)
    L += ["", "### I Layer sweep (paper App. A.2)", "", "![fig_layers.png](fig_layers.png)", "",
          "| Model | Layers | Hidden size | Error-token layer (pooled) | Prompt-only layer (pooled) |",
          "|---|---|---|---|---|"]
    for r in runs:
        s0 = r["R"].get("pooled_seed0", {})
        L.append(f"| {r['short']} | {r['R']['n_layers']} | {r['R']['d_model']} | {s0.get('error_token_layer')} | "
                 f"{s0.get('prompt_only_layer')} |")
    pc, g = cfg["probes"], cfg["gen"]
    from .data import RECALL_INSTRUCTION
    L += ["", "### J Hyperparameters and prompts (paper App. A.6, A.8)", "",
          "| Setting | Value |", "|---|---|",
          "| Probe | L2 logistic regression on standardised activations, one layer |",
          f"| Regularisation grid C | {pc['C_grid']} |",
          f"| Layers searched | every {pc['layer_step']} (+ last) |",
          f"| Layer / C selection | grouped {pc['folds']}-fold CV on dev |",
          f"| Test fraction | {pc['test_frac']} of problem groups |",
          f"| Splits | {pc['split_seeds']} |",
          f"| Permutation null | {pc['n_perm']} shuffles |",
          "| Black-box predictor | L2 logistic regression, C = 1 |",
          f"| Generation | greedy; clean ≤ {g['clean_max_tokens']} tokens, continuation ≤ {g['cont_max_tokens']} tokens "
          f"(R1-Distill: see config overrides) |",
          "| Hidden states | frozen model, bf16, one forward pass per prefix |", "",
          f"System prompt: “{SYSTEM}” (folded into the user turn for R1-Distill)  ",
          f"User message: the problem, followed by “{INSTRUCTION}”  ",
          f"TriviaQA instruction: “{RECALL_INSTRUCTION}”", ""]
    (out / "PAPER.md").write_text("\n".join(L))
    for r in runs:
        for f in ("results_paper.json", "trajectories.json", "scores_pooled.csv"):
            if (r["rd"] / f).exists():
                shutil.copy(r["rd"] / f, out / f"{r['tag']}_{f}")
    print(f"[paper] wrote {out / 'PAPER.md'} ({len(runs)} models)")
