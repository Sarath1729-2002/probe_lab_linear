"""organic_v6 reports: REPORT.md per model (summary, baselines, matched attribution, strength, second stage,
signal, continuous, transfer, datasets), figures, tables.csv; COMPARE.md across models."""
from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np

from .design6 import CONTRASTS, MATCHED, ORDER, SENT, T_DOUBT, T_HEDGED, T_PEERS6
from .report import INK, MUTED, _f, _pct, _style
from .run import root

C_NONE, C_SOC, C_INF = "#8a8984", "#2a78d6", "#eb6834"
KIND = {"none": ("no source", C_NONE), "user": ("Level 1: a person", C_SOC), "expert": ("Level 1: a person", C_SOC),
        "document": ("Level 2: a source", C_INF), "tool": ("Level 2: a source", C_INF), "peer": ("Level 2: a source", C_INF)}
LABEL = {"none": "no source", "user": "user", "expert": "expert", "document": "document", "tool": "tool", "peer": "another AI",
         "user_hedged": "user, hedged", "peers6": "six AIs", "passage": "real passage",
         "c_user": "user, 2nd turn", "c_document": "document, 2nd turn"}


def _pp(x, d=1):
    return "–" if x is None else f"{100 * x:+.{d}f}"


def _ci(ci):
    return "–" if not ci or ci[0] is None else f"[{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]"


def _dig(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _fig_matched(B, out, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cs = [c for c in MATCHED if _dig(B, "by_channel", c, "defer") is not None]
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    for k, c in enumerate(cs):
        r = B["by_channel"][c]
        y = 100 * r["defer"]
        lo, hi = [100 * v if v is not None else y for v in (r.get("defer_ci") or [None, None])]
        ax.bar(k, y, 0.66, color=KIND[c][1])
        ax.errorbar(k, y, yerr=[[max(0, y - lo)], [max(0, hi - y)]], color=INK, lw=1, capsize=3)
        ax.annotate(f"{y:.0f}", (k, hi), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(cs)), [LABEL[c] for c in cs], fontsize=8)
    ax.set_ylabel("defers to the wrong answer (%)", fontsize=8, color=MUTED)
    ax.set_ylim(0, max(10, ax.get_ylim()[1] * 1.14))
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    _style(ax)
    h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (C_NONE, C_SOC, C_INF)]
    ax.legend(h, ["no source", "Level 1: a person", "Level 2: a source"], frameon=False, fontsize=8, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out.name


def _fig_strength(K, out, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [(lab, a, b, K["strength"].get(lab) or {}) for lab, a, b in CONTRASTS]
    rows = [r for r in rows if r[3].get("diff") is not None]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(6.6, 0.62 * len(rows) + 1.2))
    for k, (lab, a, b, r) in enumerate(rows):
        y = len(rows) - 1 - k
        ax.plot([100 * r["b"], 100 * r["a"]], [y, y], color=MUTED, lw=1.2, zorder=1)
        ax.scatter(100 * r["b"], y, s=42, color="#ffffff", edgecolor=INK, lw=1.2, zorder=2)
        ax.scatter(100 * r["a"], y, s=42, color=INK, zorder=3)
        ax.annotate(f"{_pp(r['diff'])} pts {_ci(r['ci'])}", (max(100 * r["a"], 100 * r["b"]), y), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=7, color=INK)
    ax.set_yticks(range(len(rows)), [f"{LABEL[a]}  vs  {LABEL[b]}" for _, a, b, _ in rows][::-1], fontsize=8)
    ax.set_xlabel("defers (%)   ○ matched counterpart   ● variant", fontsize=8, color=MUTED)
    ax.set_xlim(0, 118)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    _style(ax)
    ax.grid(axis="y", lw=0)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out.name


def _fig_transfer(T, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cs = T["channels"]
    M = np.array([[np.nan if x is None else x for x in row] for row in T["matrix_train_row_test_col"]], float)
    cmap = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
    fig, ax = plt.subplots(figsize=(1.6 + 0.62 * len(cs), 1.3 + 0.55 * len(cs)))
    ax.imshow(M, cmap=cmap, vmin=0.5, vmax=1.0)
    for i in range(len(cs)):
        for j in range(len(cs)):
            v = M[i, j]
            ax.text(j, i, "–" if v != v else f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="#ffffff" if v == v and v > 0.8 else INK)
    ax.set_xticks(range(len(cs)), [LABEL.get(c, c) for c in cs], rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(cs)), [LABEL.get(c, c) for c in cs], fontsize=7)
    ax.set_xlabel("tested on (other half of the items)", fontsize=8, color=MUTED)
    ax.set_ylabel("trained on", fontsize=8, color=MUTED)
    ax.set_title(f"'About to defer' probe across channels (AUROC, layer {T['layer']})", fontsize=9, color=INK, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out.name


def _fig_layers(P, out, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    layers = P["layers"]
    groups = {"no source": (["none"], C_NONE), "Level 1: a person": (["user", "expert"], C_SOC),
              "Level 2: a source": (["document", "tool", "peer"], C_INF)}
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    drawn = False
    for name, (cs, col) in groups.items():
        curves = [_dig(P, "per_channel", c, "last_prompt_token", "outcome_curve_cv") for c in cs]
        curves = [np.array([np.nan if v is None else v for v in c], float) for c in curves if c]
        if not curves:
            continue
        m = np.nanmean(np.vstack(curves), 0)
        ax.plot(layers, m, color=col, lw=2, marker="o", ms=3.5, label=name)
        ax.annotate(name.split(":")[0], (layers[-1], m[-1]), xytext=(4, 0), textcoords="offset points", fontsize=7,
                    color=INK, va="center")
        drawn = True
    if not drawn:
        plt.close(fig)
        return None
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("layer", fontsize=8, color=MUTED)
    ax.set_ylabel("defer vs hold, AUROC (grouped CV)", fontsize=8, color=MUTED)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    _style(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out.name


def write_report_v6(cfg: dict) -> Path:
    dj = json.load(open(root(cfg) / "deference.json"))
    out = Path(os.path.expandvars(cfg["report_dir"])) / cfg["run_prefix"]
    out.mkdir(parents=True, exist_ok=True)
    B, P, G, K, A, CV = (dj["behaviour"], dj["probes"], dj.get("regression", {}), dj.get("contrasts", {}),
                         dj.get("anchors", {}), dj.get("coverage", {}))
    name = cfg["model"].split("/")[-1]
    bc = B["by_channel"]
    rows_csv = []
    add = lambda sec, ch, metric, v: rows_csv.append([name, sec, ch, metric, "" if v is None else v])

    # ---------------------------------------------------------------- summary
    u, dc = _dig(bc, "user", "defer"), _dig(bc, "document", "defer")
    ul = K.get("user_level_minus_source_level", {})
    pk = [_dig(P, "per_channel", c, "last_prompt_token", "outcome_auroc") for c in MATCHED]
    pk = [x for x in pk if x is not None]
    bm = _dig(P, "transfer", "block_means") or {}
    S = ["## Summary", ""]
    if u is not None and dc is not None:
        S.append(f"- **Same sentence, different source.** Told \"The answer is X\" (X wrong) on questions it answers correctly "
                 f"alone, {name} defers {_pct(_dig(bc, 'none', 'defer'))} with no source, {_pct(u)} when the user says it, "
                 f"{_pct(_dig(bc, 'expert', 'defer'))} an expert, {_pct(dc)} a retrieved document, {_pct(_dig(bc, 'tool', 'defer'))} "
                 f"a tool and {_pct(_dig(bc, 'peer', 'defer'))} another AI.")
    if ul.get("diff") is not None:
        S.append(f"- **Person vs source, matched content:** {_pp(ul['diff'])} points {_ci(ul['ci'])} (person minus source, "
                 f"n {ul['n']} items). Negative = it defers more to sources than to people.")
    for lab, a, b in CONTRASTS[:3]:
        r = _dig(K, "strength", lab) or {}
        if r.get("diff") is not None:
            S.append(f"- **{lab}:** {_pp(r['diff'])} points {_ci(r['ci'])} ({_pct(r['a'])} vs {_pct(r['b'])}).")
    cap = _dig(bc, "doubt", "capitulation")
    if cap is not None:
        S.append(f"- **After its own correct answer**, a content-free \"Are you sure?\" loses the answer {_pct(cap)} of the time; "
                 f"the user's sentence {_pct(_dig(bc, 'c_user', 'defer'))}, the document's {_pct(_dig(bc, 'c_document', 'defer'))}.")
    if pk:
        S.append(f"- **Readable before the reply:** defer vs hold AUROC {min(pk):.2f}–{max(pk):.2f} across the six matched channels "
                 f"(shuffled-label control ≈ 0.5); person → source transfer {_f(bm.get('social->info'))}, source → person "
                 f"{_f(bm.get('info->social'))} on disjoint items.")
    L = [f"# Deference against knowledge: {cfg['model']}", "",
         "Observation only: fixed prompts from established benchmarks, unedited greedy generation, hidden states and answer "
         "probabilities only read. **Known** = the model answers the bare question correctly; **firm** = 5/5 correct when "
         "re-asked at temperature 0.7. **Defers** = its final answer is the benchmark's wrong answer that was planted.", ""] + S

    # ---------------------------------------------------------------- data
    L += ["", "## Data", "", "| Dataset | Items | Accuracy alone | Known | Firm |", "|---|---|---|---|---|"]
    for b, c in CV.items():
        L.append(f"| {b} | {c['items']} | {_pct(c['accuracy'])} | {c['known']} | {c['firm']} |")
        add("data", b, "known", c["known"])

    # ---------------------------------------------------------------- 1 baselines
    L += ["", "## 1. Published baselines, reproduced", "",
          "The published protocols run on our models. Models and data differ from the papers, so this checks direction and "
          "rough size.", "", "| Baseline | Ours | Published |", "|---|---|---|"]
    s = A.get("sharma")
    if s:
        L.append(f"| Sharma et al., TriviaQA, authors' prompts (n {s['n']}): accuracy with no suggestion / wrong / right / doubt | "
                 f"{_pct(s['accuracy_no_suggestion'])} / {_pct(s['accuracy_wrong_suggestion'])} / {_pct(s['accuracy_right_suggestion'])} / "
                 f"{_pct(s['accuracy_doubt'])} | {s['published']} |")
        for k in ("accuracy_no_suggestion", "accuracy_wrong_suggestion", "accuracy_right_suggestion", "accuracy_doubt"):
            add("baseline", "sharma", k, s[k])
    for key, lab in (("nqswap", "Longpre et al., NQ-Swap real substituted passages"), ("clasheval", "ClashEval modified documents")):
        r = A.get(key)
        if r:
            L.append(f"| {lab} (n {r['n']}): memorisation ratio; follows the passage although it knew (n {r['n_known']}) | "
                     f"{_f(r['memorisation_ratio'])}; {_pct(r['context_bias_known'])} | {r['published']} |")
            add("baseline", key, "memorisation_ratio", r["memorisation_ratio"])
            add("baseline", key, "context_bias_known", r["context_bias_known"])
    r = A.get("benchform")
    if r:
        L.append(f"| BenchForm-style conformity on BIG-Bench Hard, six unanimous assistants (n {r['n_known']}) | "
                 f"{_pct(r['conformity_rate'])} | {r['published']} |")
        add("baseline", "benchform", "conformity_rate", r["conformity_rate"])
    r = A.get("flipflop")
    if r:
        L.append(f"| FlipFlop: flips after \"Are you sure?\" following its own correct answer (n {r['n_known']}) | "
                 f"{_pct(r['flip_rate'])} | {r['published']} |")
        add("baseline", "flipflop", "flip_rate", r["flip_rate"])

    # ---------------------------------------------------------------- 2 matched
    fn = _fig_matched(B, out / "fig_matched.png", f"{name}: the same sentence from six sources (known items, 95% CI)")
    L += ["", "## 2. The same sentence, attributed to different sources", "",
          f"After the question, one line: `<source> {SENT.format(x='X')}`. Only the source changes.", "", f"![matched]({fn})", "",
          "| Source | n | Defers [95% CI] | Holds | Other | Defers (firm, n) | Correct when X is correct | vs user (points) [95% CI] | vs no source |",
          "|---|---|---|---|---|---|---|---|---|"]
    for c in MATCHED:
        r = bc.get(c) or {}
        if r.get("defer") is None:
            continue
        ci = r.get("defer_ci") or [None, None]
        vu, vn = _dig(K, "matched_vs_user", c) or {}, _dig(K, "matched_vs_none", c) or {}
        L.append(f"| {LABEL[c]} | {r['n']} | **{_pct(r['defer'])}** [{_pct(ci[0])}, {_pct(ci[1])}] | {_pct(r['hold'])} | {_pct(r['other'])} | "
                 f"{_pct(r['defer_firm'])} ({r['n_firm']}) | {_pct(r['correct_with_right_content'])} | "
                 f"{_pp(vu.get('diff'))} {_ci(vu.get('ci'))} | {_pp(vn.get('diff'))} {_ci(vn.get('ci'))} |")
        for k in ("defer", "hold", "defer_firm", "correct_with_right_content"):
            add("matched", c, k, r.get(k))
        add("matched", c, "minus_user", vu.get("diff"))
    if ul.get("diff") is not None:
        L += ["", f"**Person level (user, expert) minus source level (document, tool, another AI):** {_pp(ul['diff'])} points "
              f"{_ci(ul['ci'])}, n {ul['n']} items."]
        add("matched", "person_minus_source", "diff", ul["diff"])

    # ---------------------------------------------------------------- 3 strength
    fs = _fig_strength(K, out / "fig_strength.png", f"{name}: what makes the same claim stronger")
    L += ["", "## 3. Strength of evidence, one factor at a time", ""]
    if fs:
        L += [f"![strength]({fs})", ""]
    L += [f"Hedged user = Sharma's `{T_HEDGED.format(x='X')}`; six AIs = `{T_PEERS6.format(x='X')}`; real passage = the dataset's "
          "own substituted passage (NQ-Swap, ClashEval).", "",
          "| Contrast | n | Variant | Matched counterpart | Difference (points) [95% CI] |", "|---|---|---|---|---|"]
    for lab, a, b in CONTRASTS:
        r = _dig(K, "strength", lab) or {}
        if r.get("diff") is None:
            continue
        L.append(f"| {lab} | {r['n']} | {_pct(r['a'])} | {_pct(r['b'])} | **{_pp(r['diff'])}** {_ci(r['ci'])} |")
        add("strength", lab, "diff", r["diff"])

    # ---------------------------------------------------------------- 4 second stage
    L += ["", "## 4. After its own correct answer", "",
          f"The model's actual reply to the bare question is kept as its turn; then one line follows. Doubt = `{T_DOUBT}`.", "",
          "| Second turn | n | Loses the correct answer | Adopts X | Holds |", "|---|---|---|---|---|"]
    d = bc.get("doubt") or {}
    if d.get("n"):
        L.append(f"| \"Are you sure?\" (no content) | {d['n']} | **{_pct(d.get('capitulation'))}** | – | {_pct(d.get('hold'))} |")
        add("second_stage", "doubt", "capitulation", d.get("capitulation"))
    for c in ("c_user", "c_document"):
        r = bc.get(c) or {}
        if r.get("defer") is not None:
            L.append(f"| {LABEL[c]} | {r['n']} | {_pct(1 - r['hold'])} | **{_pct(r['defer'])}** | {_pct(r['hold'])} |")
            add("second_stage", c, "defer", r["defer"])

    # ---------------------------------------------------------------- 5 signal
    fl = _fig_layers(P, out / "fig_layers.png", f"{name}: where the signal appears")
    L += ["", "## 5. Is deference readable before the reply?", "",
          "Mass-mean probe on the last prompt token, defer vs hold, grouped out-of-fold. **Shuffled** = control task on shuffled "
          "labels. **Confidence only** = knowledge strength alone. **Firm** = items it gets right 5/5.", ""]
    if fl:
        L += [f"![layers]({fl})", ""]
    L += ["| Channel | Defer / hold (n) | AUROC (layer) | Shuffled | At end of content | Confidence only | Firm items (n defer) | Detection |",
          "|---|---|---|---|---|---|---|---|"]
    for c in ORDER:
        r = P["per_channel"].get(c) or {}
        a, e = r.get("last_prompt_token", {}), r.get("info_end", {})
        if "outcome_auroc" not in a:
            continue
        L.append(f"| {LABEL[c]} | {r['n_defer']} / {r['n_hold']} | **{_f(a['outcome_auroc'], 3)}** ({a['outcome_layer']}) | "
                 f"{_f(a.get('control_shuffled_auroc'), 3)} | {_f(e.get('outcome_auroc'), 3)} | {_f(a.get('confidence_only_auroc'), 3)} | "
                 f"{_f(a.get('outcome_auroc_firm'), 3)} ({a.get('n_firm_defer', '–')}) | {_f(a.get('detection_auroc'), 3)} |")
        for k in ("outcome_auroc", "control_shuffled_auroc", "confidence_only_auroc", "outcome_auroc_firm"):
            add("probe", c, k, a.get(k))

    # ---------------------------------------------------------------- 6 continuous
    if "per_channel" in G:
        L += ["", "## 6. How strongly it still prefers the correct answer (continuous, read-only)", "",
              "**Posterior margin** = log-odds of the correct over the planted answer with the content in context (teacher-forced; "
              "below 0 = it now prefers X). **Prior** = the same on the bare question. Ridge regression probe from the last prompt "
              "token, grouped out-of-fold.", "",
              "| Channel | n | Posterior < 0 | Posterior → defers | Prior → defers | Probe R² (layer) | Spearman | Prior-only R² | Beyond prior R² | Shuffled R² |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for c in ORDER:
            r = G["per_channel"].get(c) or {}
            if "r2" not in r:
                continue
            L.append(f"| {LABEL[c]} | {r['n']} | {_pct(r.get('posterior_below_zero'))} | {_f(r.get('posterior_predicts_defer_auroc'), 3)} | "
                     f"{_f(r.get('prior_predicts_defer_auroc'), 3)} | **{_f(r['r2'], 3)}** ({r['layer']}) | {_f(r['spearman'], 3)} | "
                     f"{_f(r['prior_only_r2'], 3)} | {_f(r['partial_r2_beyond_prior'], 3)} | {_f(r['control_shuffled_r2'], 3)} |")
            for k in ("r2", "prior_only_r2", "partial_r2_beyond_prior", "control_shuffled_r2", "posterior_predicts_defer_auroc"):
                add("regression", c, k, r.get(k))

    # ---------------------------------------------------------------- 7 transfer
    T, TR = P.get("transfer", {}), G.get("transfer", {})
    if "matrix_train_row_test_col" in T:
        ft = _fig_transfer(T, out / "fig_transfer.png")
        L += ["", "## 7. One signal for people and for sources?", "",
              "A probe trained on one channel is tested on another, on disjoint halves of the items.", "", f"![transfer]({ft})", "",
              "| Block | Defer vs hold (AUROC) | Posterior regression (Spearman) |", "|---|---|---|"]
        for k, lab in (("social->social", "person → person"), ("info->info", "source → source"),
                       ("social->info", "person → source"), ("info->social", "source → person")):
            L.append(f"| {lab} | {_f(_dig(T, 'block_means', k), 3)} | {_f(_dig(TR, 'block_means', k), 3)} |")
            add("transfer", k, "auroc", _dig(T, "block_means", k))
        L += ["", "| Held-out channel | Trained on all other channels |", "|---|---|"]
        L += [f"| {LABEL.get(c, c)} | {_f(v, 3)} |" for c, v in (T.get("leave_one_channel_out") or {}).items()]

    # ---------------------------------------------------------------- 8 datasets
    bs = sorted({b for c in MATCHED for b in B["by_bucket"].get(c, {})})
    L += ["", "## 8. By dataset (defers, matched channels)", "", "| Source | " + " | ".join(bs) + " |", "|---|" + "---|" * len(bs)]
    for c in MATCHED:
        bb = B["by_bucket"].get(c, {})
        L.append(f"| {LABEL[c]} | " + " | ".join(f"{_pct(bb[b]['defer'])} (n {bb[b]['n']})" if b in bb else "–" for b in bs) + " |")
        for b in bs:
            if b in bb:
                add("by_dataset", f"{c}/{b}", "defer", bb[b]["defer"])
    dm = K.get("by_bucket_document_minus_user", {})
    L.append("| **document − user (points)** | " + " | ".join(f"{_pp(_dig(dm, b, 'diff'))} {_ci(_dig(dm, b, 'ci'))}" for b in bs) + " |")

    L += ["", "## Method notes", "",
          "- Items come from established benchmarks and carry the benchmark's own wrong answer (distractor, substituted entity, "
          "modified document, or another model's wrong final answer). Nothing is selected using the model's outputs except the "
          "known / firm filters.",
          "- All confidence intervals are item-level bootstraps; contrasts are paired by item.",
          "- Probe layers are chosen by grouped cross-validation on the same rows (mildly optimistic); transfer uses disjoint items.",
          "- The second-stage prompts contain the model's own earlier reply verbatim; no assistant text is written by us.",
          "- Correlational: no activations are modified anywhere in this study.",
          "- Every number in this report is also in `tables.csv`; per-reply data is in `<out_dir>/<run_prefix>/<dataset>/gen.jsonl` "
          "and `score.jsonl`.", ""]
    (out / "REPORT.md").write_text("\n".join(L))
    with open(out / "tables.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "section", "row", "metric", "value"])
        w.writerows(rows_csv)
    shutil.copy(root(cfg) / "deference.json", out / "deference.json")
    print(f"[report] {out / 'REPORT.md'}")
    return out / "REPORT.md"


def write_compare_v6(cfg: dict, tags: list[str]) -> Path:
    base = Path(os.path.expandvars(cfg["report_dir"])) / cfg["run_prefix"]
    R = {t: json.load(open(base / t / "deference.json")) for t in tags if (base / t / "deference.json").exists()}
    ts = list(R)
    hdr = ["| | " + " | ".join(ts) + " |", "|---|" + "---|" * len(ts)]
    L = ["# Deference against knowledge: across models", "", "## The same sentence from six sources (defers, known items)", ""] + hdr
    for c in MATCHED:
        L.append(f"| {LABEL[c]} | " + " | ".join(_pct(_dig(R[t], "behaviour", "by_channel", c, "defer")) for t in ts) + " |")
    L.append("| **person − source (points) [95% CI]** | " + " | ".join(
        f"{_pp(_dig(R[t], 'contrasts', 'user_level_minus_source_level', 'diff'))} "
        f"{_ci(_dig(R[t], 'contrasts', 'user_level_minus_source_level', 'ci'))}" for t in ts) + " |")
    L += ["", "## Strength of evidence (points) [95% CI]", ""] + hdr
    for lab, a, b in CONTRASTS:
        L.append(f"| {lab} | " + " | ".join(f"{_pp(_dig(R[t], 'contrasts', 'strength', lab, 'diff'))} "
                                            f"{_ci(_dig(R[t], 'contrasts', 'strength', lab, 'ci'))}" for t in ts) + " |")
    L += ["", "## Published baselines", ""] + hdr
    for lab, ks in (("Sharma TriviaQA accuracy: none → wrong suggestion", None),
                    ("NQ-Swap memorisation ratio", ("anchors", "nqswap", "memorisation_ratio")),
                    ("ClashEval context bias (known)", ("anchors", "clasheval", "context_bias_known")),
                    ("BenchForm conformity on BBH", ("anchors", "benchform", "conformity_rate")),
                    ("FlipFlop flip rate", ("anchors", "flipflop", "flip_rate"))):
        if ks is None:
            L.append(f"| {lab} | " + " | ".join(f"{_pct(_dig(R[t], 'anchors', 'sharma', 'accuracy_no_suggestion'))} → "
                                                f"{_pct(_dig(R[t], 'anchors', 'sharma', 'accuracy_wrong_suggestion'))}" for t in ts) + " |")
        else:
            L.append(f"| {lab} | " + " | ".join(_f(_dig(R[t], *ks)) for t in ts) + " |")
    L += ["", "## Readable before the reply (defer vs hold AUROC; shuffled control)", ""] + hdr
    for c in ORDER:
        L.append(f"| {LABEL[c]} | " + " | ".join(
            f"{_f(_dig(R[t], 'probes', 'per_channel', c, 'last_prompt_token', 'outcome_auroc'), 3)}; "
            f"{_f(_dig(R[t], 'probes', 'per_channel', c, 'last_prompt_token', 'control_shuffled_auroc'), 3)}" for t in ts) + " |")
    L += ["", "## Transfer between people and sources (AUROC, disjoint items)", ""] + hdr
    for k, lab in (("social->social", "person → person"), ("info->info", "source → source"),
                   ("social->info", "person → source"), ("info->social", "source → person")):
        L.append(f"| {lab} | " + " | ".join(_f(_dig(R[t], "probes", "transfer", "block_means", k), 3) for t in ts) + " |")
    L += ["", "## Continuous probe on the posterior margin (R²; beyond prior)", ""] + hdr
    for c in MATCHED:
        L.append(f"| {LABEL[c]} | " + " | ".join(f"{_f(_dig(R[t], 'regression', 'per_channel', c, 'r2'), 3)}; "
                                                 f"{_f(_dig(R[t], 'regression', 'per_channel', c, 'partial_r2_beyond_prior'), 3)}" for t in ts) + " |")
    (base / "COMPARE.md").write_text("\n".join(L) + "\n")
    rows = []
    for t in ts:
        p = base / t / "tables.csv"
        if p.exists():
            rows += list(csv.reader(open(p)))[1:]
    with open(base / "tables_all_models.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "section", "row", "metric", "value"])
        w.writerows(rows)
    print(f"[compare] {base / 'COMPARE.md'}")
    return base / "COMPARE.md"
