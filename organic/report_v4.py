"""organic_v4 reports: REPORT.md per model and COMPARE.md across models."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np

from .channels import CHANNELS, INFO, SOCIAL, _TEXT, DOUBT, FOLLOWUP, PUSHBACK
from .report import INK, MUTED, _f, _pct, _style
from .run import root

COL = {"social": "#2a78d6", "info": "#eb6834"}          # reference categorical slots 1-2
DESC = dict(_TEXT, pushback="(model already answered correctly) \"" + PUSHBACK + "\"",
            self_prior="(the model's own previous reply said `Answer: {x}`) \"" + FOLLOWUP + "\"")


def _channels_fig(B, out, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cs = [c for c in CHANNELS if B["by_channel"].get(c, {}).get("defer") is not None]
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    for k, c in enumerate(cs):
        r = B["by_channel"][c]
        kind = "social" if c in SOCIAL else "info"
        y = 100 * r["defer"]
        lo, hi = [100 * v if v is not None else y for v in r.get("defer_ci", [None, None])]
        ax.bar(k, y, 0.7, color=COL[kind], label=kind if kind not in [h.get_label() for h in ax.patches[:k]] else None)
        ax.errorbar(k, y, yerr=[[y - lo], [hi - y]], color=INK, lw=1, capsize=3)
        ax.annotate(f"{y:.0f}", (k, hi), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7, color=INK)
    ax.set_xticks(range(len(cs)), [c.replace("_", "\n") for c in cs], fontsize=7)
    ax.set_ylabel("defers to the wrong answer (%)", fontsize=8, color=MUTED)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.set_ylim(0, max(10, ax.get_ylim()[1] * 1.1))
    _style(ax)
    h = [plt.Rectangle((0, 0), 1, 1, color=COL["social"]), plt.Rectangle((0, 0), 1, 1, color=COL["info"])]
    ax.legend(h, ["social (user pressure)", "informational (no pressure)"], frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out.name


def _transfer_fig(T, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cs, M = T["channels"], np.array([[np.nan if x is None else x for x in row] for row in T["matrix_train_row_test_col"]], float)
    cmap = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
    fig, ax = plt.subplots(figsize=(1.2 + 0.75 * len(cs), 1.0 + 0.65 * len(cs)))
    ax.imshow(M, cmap=cmap, vmin=0.5, vmax=1.0)
    for i in range(len(cs)):
        for j in range(len(cs)):
            v = M[i, j]
            ax.text(j, i, "–" if v != v else f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="#ffffff" if v == v and v > 0.8 else INK)
    ax.set_xticks(range(len(cs)), cs, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(cs)), cs, fontsize=7)
    k = len([c for c in cs if c in SOCIAL])
    if 0 < k < len(cs):
        ax.axhline(k - 0.5, color="#ffffff", lw=2)
        ax.axvline(k - 0.5, color="#ffffff", lw=2)
    ax.set_xlabel("tested on (other half of the items)", fontsize=8, color=MUTED)
    ax.set_ylabel("trained on", fontsize=8, color=MUTED)
    ax.set_title(f"'About to defer' probe across channels (AUROC, layer {T['layer']})", fontsize=9, color=INK, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out.name


def write_report_v4(cfg: dict) -> Path:
    dj = json.load(open(root(cfg) / "deference.json"))
    out = Path(os.path.expandvars(cfg["report_dir"])) / cfg["run_prefix"]
    out.mkdir(parents=True, exist_ok=True)
    B, P = dj["behaviour"], dj["probes"]
    name = cfg["model"].split("/")[-1]
    L = [f"# Deference against knowledge: {cfg['model']}", "",
         "**Question.** When the model knows the answer, does it say what it knows (candor) or go along with what it is "
         "told? Is going along driven by social pressure from the user (sycophancy) or by information from any source "
         "(credulity)? Is there one internal signal for it?", "",
         "Every item is one the model answers correctly on its own (**known**; **firm** = 5/5 when re-asked at "
         "temperature 0.7). The same content, the dataset's wrong answer or (control) the correct one, is placed right "
         "after the question; only the channel delivering it changes.", "",
         "| Channel | Kind | How the content arrives (`{x}` = the content) |", "|---|---|---|"]
    for c in CHANNELS:
        L.append(f"| {c} | {'social' if c in SOCIAL else 'informational'} | {DESC[c].replace(chr(10), ' ')} |")
    L.append(f"| doubt_pushback | capitulation | (model already answered correctly) \"{DOUBT}\" |")

    # ---------------------------------------------------------------- behaviour
    fn = _channels_fig(B, out / "fig_channels.png", f"{name}: deference by channel, known items, 95% CI")
    L += ["", "## How often it defers", "", f"![channels]({fn})", "",
          "| Channel | n | Defers [95% CI] | Holds | Other | Defers (firm, n) | Correct when the content is correct |",
          "|---|---|---|---|---|---|---|"]
    for c in CHANNELS:
        r = B["by_channel"].get(c)
        if not r or r["defer"] is None:
            continue
        ci = r.get("defer_ci") or [None, None]
        L.append(f"| {c} | {r['n']} | **{_pct(r['defer'])}** [{_pct(ci[0])}, {_pct(ci[1])}] | {_pct(r['hold'])} | "
                 f"{_pct(r['other'])} | {_pct(r['defer_firm'])} ({r['n_firm']}) | {_pct(r['correct_with_right_content'])} |")
    d = B["by_channel"].get("doubt_pushback", {})
    sm = B["social_minus_info"]
    L += ["", f"**Capitulation** to a content-free \"{DOUBT}\": the correct answer is lost **{_pct(d.get('capitulation'))}** "
          f"of the time (n {d.get('n')}).",
          f"**Social minus informational** deference, per item: **{_f(100 * sm['mean'] if sm['mean'] is not None else None, 1)} points** "
          f"[{_f(100 * sm['ci'][0] if sm['ci'][0] is not None else None, 1)}, {_f(100 * sm['ci'][1] if sm['ci'][1] is not None else None, 1)}] "
          f"(n {sm['n_items']} items). Positive = the user's say-so moves it more than the same content from a source.", "",
          "### By dataset (defer rate)", "", "| Channel | " + " | ".join(sorted({b for c in B["by_bucket"].values() for b in c})) + " |",
          "|---|" + "---|" * len({b for c in B["by_bucket"].values() for b in c})]
    bs = sorted({b for c in B["by_bucket"].values() for b in c})
    for c in CHANNELS + ["doubt_pushback"]:
        bb = B["by_bucket"].get(c, {})
        key = "hold" if c.startswith("doubt") else "defer"
        L.append(f"| {c}{' (holds)' if c.startswith('doubt') else ''} | " + " | ".join(
            f"{_pct(bb[b][key])} (n {bb[b]['n']})" if b in bb else "–" for b in bs) + " |")

    # ---------------------------------------------------------------- probes
    L += ["", "## Can deference be read before the reply?", "",
          "Mass-mean probes, known items, grouped out-of-fold. **Outcome**: defer vs hold on wrong content. "
          "**Detection**: wrong vs correct content. Confidence only = predicting deference from knowledge strength alone.", "",
          "| Channel | Outcome @ last prompt token (layer) | Outcome @ end of content | Confidence only | Outcome, firm items (n defer) | Detection @ last prompt token |",
          "|---|---|---|---|---|---|"]
    for c in CHANNELS:
        r = P["per_channel"].get(c, {})
        a, e = r.get("last_prompt_token", {}), r.get("info_end", {})
        L.append(f"| {c} | {_f(a.get('outcome_auroc'), 3)} ({a.get('outcome_layer', '–')}) | {_f(e.get('outcome_auroc'), 3)} | "
                 f"{_f(a.get('confidence_only_auroc'), 3)} | {_f(a.get('outcome_auroc_firm'), 3)} ({a.get('n_firm_defer', '–')}) | "
                 f"{_f(a.get('detection_auroc'), 3)} |")
    T = P.get("transfer", {})
    if "matrix_train_row_test_col" in T:
        fn2 = _transfer_fig(T, out / "fig_transfer.png")
        bm = T["block_means"]
        L += ["", "## One signal across channels?", "",
              "Outcome probe trained on one channel, tested on another, at one layer, **on disjoint halves of the items** "
              "(so an item's own tendency cannot carry over).", "", f"![transfer]({fn2})", "",
              "| Block | Mean AUROC |", "|---|---|",
              f"| social → social | {_f(bm['social->social'], 3)} |", f"| informational → informational | {_f(bm['info->info'], 3)} |",
              f"| social → informational | {_f(bm['social->info'], 3)} |", f"| informational → social | {_f(bm['info->social'], 3)} |",
              "", "| Held-out channel | Trained on all other channels |", "|---|---|"]
        L += [f"| {c} | {_f(v, 3)} |" for c, v in T["leave_one_channel_out"].items()]

    # ---------------------------------------------------------------- steering
    sp = root(cfg) / "steer_multi.json"
    if sp.exists():
        S = json.load(open(sp))
        L += ["", "## Causal test: one direction, every channel", "",
              f"Direction = mean(hold) − mean(defer) at the last prompt token, layer {S['layer']}, pooled over "
              f"{', '.join(S['fit_channels'])} on half the items ({S['n_fit_hold']} hold / {S['n_fit_defer']} defer); tested on "
              f"the other half in {', '.join(S['test_channels'])}. alpha = 1 is the full mean difference (norm {S['norm']:.1f}; "
              f"typical state norm {S['mean_state_norm']:.0f}). Random = a random direction of the same norm.", "",
              "| Condition | Channel | Defers | Holds | Other | Correct when content is correct | Reply chars |",
              "|---|---|---|---|---|---|---|"]
        for c in S["conditions"]:
            L.append(f"| {c['cond']} α={c['alpha']} | {c['channel']} | **{_pct(c['defer'])}** | {_pct(c['hold'])} | {_pct(c['other'])} | "
                     f"{_pct(c['correct_on_right'])} | {_f(c['mean_chars'], 0)} |")
    L += ["", "## Notes", "",
          "- Outcome layers are chosen by grouped CV on the same rows (mildly optimistic); transfer uses disjoint items.",
          "- Grading is deterministic; per-reply data in `<out_dir>/<run_prefix>/<dataset>/gen.jsonl`, steering in `steer_multi.jsonl`.", ""]
    (out / "REPORT.md").write_text("\n".join(L))
    shutil.copy(root(cfg) / "deference.json", out / "deference.json")
    if sp.exists():
        shutil.copy(sp, out / "steer_multi.json")
    print(f"[report] {out / 'REPORT.md'}")
    return out / "REPORT.md"


def write_compare_v4(cfg: dict, tags: list[str]) -> Path:
    base = Path(os.path.expandvars(cfg["report_dir"])) / cfg["run_prefix"]
    R = {t: json.load(open(base / t / "deference.json")) for t in tags if (base / t / "deference.json").exists()}
    S = {t: json.load(open(base / t / "steer_multi.json")) for t in tags if (base / t / "steer_multi.json").exists()}
    ts = list(R)
    hdr = ["| | " + " | ".join(ts) + " |", "|---|" + "---|" * len(ts)]
    L = ["# Deference against knowledge: across models", "", "## Defers to the wrong answer (known items)", ""] + hdr
    for c in CHANNELS:
        L.append(f"| {c} | " + " | ".join(_pct(R[t]["behaviour"]["by_channel"].get(c, {}).get("defer")) for t in ts) + " |")
    L.append("| capitulation (\"are you sure?\") | " + " | ".join(
        _pct(R[t]["behaviour"]["by_channel"].get("doubt_pushback", {}).get("capitulation")) for t in ts) + " |")
    L.append("| social − informational (points) | " + " | ".join(
        _f(100 * R[t]["behaviour"]["social_minus_info"]["mean"], 1) if R[t]["behaviour"]["social_minus_info"]["mean"] is not None else "–"
        for t in ts) + " |")
    L += ["", "## Correct kept when the content is correct (harm control)", ""] + hdr
    for c in CHANNELS:
        L.append(f"| {c} | " + " | ".join(_pct(R[t]["behaviour"]["by_channel"].get(c, {}).get("correct_with_right_content")) for t in ts) + " |")
    L += ["", "## Outcome probe at the last prompt token (AUROC; confidence only)", ""] + hdr
    for c in CHANNELS:
        L.append(f"| {c} | " + " | ".join(
            f"{_f(R[t]['probes']['per_channel'].get(c, {}).get('last_prompt_token', {}).get('outcome_auroc'), 3)}; "
            f"{_f(R[t]['probes']['per_channel'].get(c, {}).get('last_prompt_token', {}).get('confidence_only_auroc'), 3)}" for t in ts) + " |")
    L += ["", "## Transfer across channels (disjoint items)", ""] + hdr
    for k in ("social->social", "info->info", "social->info", "info->social"):
        L.append(f"| {k} | " + " | ".join(_f((R[t]["probes"].get("transfer", {}).get("block_means") or {}).get(k), 3) for t in ts) + " |")
    L += ["", "## Steering, all test channels pooled (defer; correct kept)", ""] + hdr
    conds = []
    for t in S:
        for c in S[t]["conditions"]:
            if c["channel"] == "ALL" and (c["cond"], c["alpha"]) not in conds:
                conds.append((c["cond"], c["alpha"]))
    for cn, a in conds:
        cells = []
        for t in ts:
            c = next((c for c in S.get(t, {}).get("conditions", []) if c["channel"] == "ALL" and c["cond"] == cn and c["alpha"] == a), None)
            cells.append(f"{_pct(c['defer'])}; {_pct(c['correct_on_right'])}" if c else "–")
        L.append(f"| {cn} α={a} | " + " | ".join(cells) + " |")
    (base / "COMPARE.md").write_text("\n".join(L) + "\n")
    print(f"[compare] {base / 'COMPARE.md'}")
    return base / "COMPARE.md"
