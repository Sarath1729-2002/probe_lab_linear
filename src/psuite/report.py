"""REPORT.md + scorecard.csv + figures from the analysis pickles.  The scorecard applies the pre-declared STRICT rules of docs/PROTOCOL.md mechanically."""
import json
import math
import pickle
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import core  # noqa: E402
from .trace import CK_NAMES  # noqa: E402

MIN_R2 = 0.10          # a SUPPORTED signal must explain at least 10% of the variance, not merely beat a baseline
SIGNALS = [  # id, label, block, prefix, target, what it is for
    ("finish", "Will it run away? (hits the token cap)", "P", "P[pooled]", "runaway", "compute budget / overthinking risk"),
    ("finish_soft", "Runaway rate across samples (0..1)", "P", "P[pooled]", "runaway_rate", "soft version of the above"),
    ("length", "Output length (log tokens)", "P", "P[pooled]", "logT", "resource scheduling; baseline replication"),
    ("spread", "Length variability across its own samples", "P", "P[pooled]", "len_sd", "does the model know how unsure its length is"),
    ("deliver", "Will it fail to give a parsable answer?", "P", "P[pooled]", "noans", "silent failure to deliver"),
    ("correct", "Will the answer be correct? (reference)", "P", "P[pooled]", "correct", "reliability gating (known signal; reference point)"),
    ("answer_size", "Answer magnitude (before any reasoning)", "P", "P[pooled]", "ans_logmag", "pre-computation of the result"),
    ("comply", "Will it obey the output constraint?", "C", "C[pooled]", "comply", "controllability: predict an instruction violation"),
    ("overshoot", "How far over the word limit?", "C", "C[pooled]", "over", "length-awareness under a constraint"),
    ("c_runaway", "Runaway despite a constraint", "C", "C[pooled]", "c_runaway", "controllability failure"),
    ("settled", "Is the answer already settled? (own final answer)", "K", "K", "settled_p", "early stopping; does the visible chain match the internal state"),
    ("ready", "Is the CORRECT answer already available?", "K", "K", "ready_p", "answer-readiness (overthinking)"),
    ("remaining", "Tokens still to come", "K", "K", "log_rem", "replication of 'How Much is Left?' at checkpoints"),
]


def _load(d: Path, name: str):
    f = d / f"{name}.pkl"
    return pickle.load(open(f, "rb")) if f.exists() else None


def _rows(res, prefix, target, regime=None):
    return [r for r in res if r["prefix"].startswith(prefix) and r["target"] == target and (regime is None or r["regime"] == regime)]


def _v(r, which="obs"):
    return core.verdict(r[f"delta_vs_{'obs' if which == 'obs' else 'oracle'}"], r["obs_lo" if which == "obs" else "oracle_lo"], r["probe_test_r2"])


def _status(res, prefix, target, ck=None):
    """strict rule: SUPPORTED = iid PASS vs the observable ladder with R2 >= MIN_R2 AND the same holds in at least half of the leave-dataset-out tests."""
    pre = f"K[{ck}]" if ck else prefix
    iid = _rows(res, pre, target, "iid")
    if not iid:
        return {"status": "no data"}
    iid = iid[0]
    lod = _rows(res, pre, target, None)
    lod = [r for r in lod if r["regime"].startswith("lodo:") and not np.isnan(r["probe_test_r2"])]
    npass = sum(_v(r) == "PASS" for r in lod)
    ok = _v(iid) == "PASS" and iid["probe_test_r2"] >= MIN_R2
    gen = bool(lod) and npass >= math.ceil(len(lod) / 2)
    st = "SUPPORTED" if (ok and gen) else ("PARTIAL (does not generalise)" if ok else ("WEAK (beats baseline, R2 small)" if _v(iid) == "PASS" else "NOT SUPPORTED"))
    return {"status": st, "r2": iid["probe_test_r2"], "layer": iid["probe_layer"], "obs": iid["best_observable"], "obs_r2": iid["obs_r2"], "d_obs": iid["delta_vs_obs"],
            "d_obs_lo": iid["obs_lo"], "d_obs_hi": iid["obs_hi"], "v_oracle": _v(iid, "oracle"), "lodo": f"{npass}/{len(lod)}",
            "auroc": iid.get("auroc_probe", np.nan), "auroc_obs": iid.get("auroc_obs", np.nan), "base_rate": iid.get("base_rate", np.nan)}


def scorecard(P, C, K):
    rows = []
    for sid, label, blk, prefix, target, use in SIGNALS:
        src = {"P": P, "C": C, "K": K}[blk]
        if src is None:
            rows.append({"signal": sid, "what": label, "status": "not run"})
            continue
        res = src["res"]
        if blk == "K":
            per = {ck: _status(res, "K", target, ck) for ck in CK_NAMES}
            npass = sum(v.get("status", "").startswith(("SUPPORTED", "PARTIAL", "WEAK")) for v in per.values())
            mid = per["o48"]
            rows.append({"signal": sid, "what": label, "checkpoints_beating_text": f"{npass}/{len(CK_NAMES)}", **mid, "use": use})
        else:
            rows.append({"signal": sid, "what": label, **_status(res, prefix, target), "use": use})
    return pd.DataFrame(rows)


def _fmt(df, cols=None, fmt="{:.3f}"):
    return core.md(df[[c for c in (cols or df.columns) if c in df.columns]], fmt)


DETAIL = ["prefix", "target", "regime", "n_test", "probe_layer", "probe_test_r2", "obs_r2", "best_observable", "delta_vs_obs", "obs_lo", "obs_hi", "verdict_obs", "delta_vs_oracle",
          "oracle_lo", "verdict_oracle", "base_rate", "auroc_probe", "auroc_obs", "auroc_oracle"]


def behaviour(df):
    g = df.groupby("ds")
    t = pd.DataFrame({"n": g.size(), "correct": g.correct.mean(), "noans": g.noans.mean(), "runaway": g.runaway.mean(), "runaway_rate": g.runaway_rate.mean(),
                      "median_T": g.T.median(), "len_sd": g.len_sd.mean(), "comply": g.comply.mean(), "c_runaway": g.c_runaway.mean()}).reset_index()
    c = df.groupby("ctype").agg(n=("comply", "size"), comply=("comply", "mean"), c_runaway=("c_runaway", "mean"), c_correct=("c_correct", "mean")).reset_index()
    return t, c


def figures(a: Path, P, C, K, ct):
    done = []
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for axi, (cur, tg, ttl) in zip(ax, [(P, ["runaway", "logT", "noans", "correct"], "plain prompt-end"), (C, ["comply", "over", "c_logT"], "constrained prompt-end")]):
        if cur is None:
            continue
        c = cur["cur"]
        c = c[(c.scope == "pooled") & (c.regime == "iid")]
        for t in tg:
            s = c[c.target == t].sort_values("layer")
            axi.plot(s.layer, s.test_r2, marker="o", ms=3, label=t)
        axi.set_title(f"{ttl}: probe R2 by layer (pooled, iid)")
        axi.set_xlabel("layer")
        axi.set_ylabel("test R2")
        axi.legend(fontsize=7)
    if K is not None:
        res = K["res"]
        for tg, col in (("settled_p", "C0"), ("ready_p", "C1"), ("log_rem", "C2")):
            pr = [next((r["probe_test_r2"] for r in res if r["ck"] == ck and r["target"] == tg and r["regime"] == "iid"), np.nan) for ck in CK_NAMES]
            ob = [next((r["obs_r2"] for r in res if r["ck"] == ck and r["target"] == tg and r["regime"] == "iid"), np.nan) for ck in CK_NAMES]
            ax[2].plot(CK_NAMES, pr, color=col, marker="o", label=f"{tg} probe")
            ax[2].plot(CK_NAMES, ob, color=col, ls="--", label=f"{tg} best observable")
        ax[2].set_title("checkpoints along the solution (pooled, iid)")
        ax[2].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(a / "fig1_layers_and_checkpoints.png", dpi=130)
    plt.close(fig)
    done.append("fig1_layers_and_checkpoints.png")
    if ct is not None and len(ct) and "auroc_probe" in ct:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        x = np.arange(len(ct))
        for i, (c, lab) in enumerate((("auroc_probe", "probe"), ("auroc_obs", "observable"), ("auroc_oracle", "plain-behaviour oracle"))):
            ax.bar(x + 0.27 * (i - 1), ct[c], 0.27, label=lab)
        ax.set_xticks(x)
        ax.set_xticklabels(ct.ctype, rotation=20)
        ax.axhline(0.5, color="k", lw=0.5)
        ax.set_ylabel("AUROC (comply)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(a / "fig2_comply_by_constraint.png", dpi=130)
        plt.close(fig)
        done.append("fig2_comply_by_constraint.png")
    return done


def write(run_dir, log=print) -> Path:
    run_dir = Path(run_dir)
    a = run_dir / "analysis"
    P, C, K, T = (_load(a, n) for n in ("P", "C", "K", "tables"))
    sc = scorecard(P, C, K)
    sc.to_csv(a / "scorecard.csv", index=False)
    figs = figures(a, P, C, K, C["ctype"] if C else None)
    out = ["# Probe suite: what does the residual stream know about its own continuation? (frozen LLM, linear probes)\n",
           "Everything is observational on a frozen model. Probes are closed-form ridge regressions on one layer's frozen hidden state; lambda and layer are chosen on validation problems; "
           "test problems are touched once; CIs are paired bootstraps over test problems. Every label is a deterministic function of the stored text (greedy decoding), "
           "so labels carry no annotation noise. `verdict_obs` compares against the strongest baseline that uses only what is observable at that point (text features, dataset identity, "
           "the model's own early layers); `verdict_oracle` against the strongest baseline that also sees behaviour the model could not know yet. PASS = CI lower bound > 0 and probe R2 > 0.05.\n"]
    af = run_dir / "ground_truth_audit.json"
    if af.exists():
        au = pd.DataFrame(json.load(open(af)))
        out += ["## 1. Ground-truth audit (a dataset is admitted only if every check passes on >= 99.5% of problems)\n", core.md(au[[c for c in ("dataset", "tier", "n_in", "n_kept", "pass_rate", "admitted", "fails") if c in au.columns]]), "\n"]
    out += ["## 2. What the model does (greedy)\n"]
    if T is not None:
        bt, bc = behaviour(T["df"])
        out += [core.md(bt), "\nCompliance by constraint type:\n\n", core.md(bc), "\n"]
    out += ["## 3. Scorecard (strict, mechanical; rules in docs/PROTOCOL.md)\n",
            f"SUPPORTED = iid PASS vs the observable ladder, probe R2 >= {MIN_R2}, AND the same PASS in at least half of the leave-dataset-out tests. "
            "PARTIAL = passes iid but does not generalise across datasets. WEAK = beats the baseline but explains little. For checkpoint signals the row shows the checkpoint 48 tokens into the solution (o48) "
            "and how many of the 6 checkpoints beat the text baseline.\n\n",
            _fmt(sc, ["signal", "what", "status", "r2", "layer", "obs", "obs_r2", "d_obs", "d_obs_lo", "d_obs_hi", "v_oracle", "lodo", "auroc", "auroc_obs", "base_rate", "checkpoints_beating_text"]), "\n"]
    for title, blk, key in (("4. Plain prompt-end state (before any output)", P, "P"), ("5. Constrained prompt-end state", C, "C")):
        if blk is None:
            continue
        s = core.strip(blk["res"])
        out += [f"## {title}\n", "Pooled scope, all regimes (iid, easy->hard shift, leave-one-dataset-out):\n\n", _fmt(s[s.prefix == f"{key}[pooled]"], DETAIL), "\n",
                "Per dataset (iid and shift):\n\n", _fmt(s[(s.prefix != f"{key}[pooled]")], DETAIL), "\n"]
    if C is not None and len(C["ctype"]):
        out += ["### Compliance AUROC within each constraint type (pooled iid test problems)\n", core.md(C["ctype"]), "\n"]
    if K is not None:
        s = core.strip(K["res"])
        out += ["## 6. Checkpoints along the solution\n", "Probe at the natural state after that fraction of the solution; `Bpos` = position only, `Btext` = position + visible solution-so-far + question; "
                "the oracle sees whether the answer string already appears in the text and the total length.\n\n",
                _fmt(s[s.regime.isin(["iid", "shift"])], ["ck", "target", "regime", "n_test", "probe_layer", "probe_test_r2", "obs_r2", "best_observable", "delta_vs_obs", "obs_lo", "obs_hi", "verdict_obs",
                                                          "verdict_oracle", "auroc_probe", "auroc_obs"]), "\n",
                "### Silent settledness: only rows whose visible text does not yet contain the model's final answer\n\n", core.md(K["silent"]), "\n",
                "### Leave-one-dataset-out at the mid checkpoint\n\n", _fmt(s[(s.ck == "o48") & s.regime.str.startswith("lodo")], ["target", "regime", "probe_test_r2", "obs_r2", "delta_vs_obs", "obs_lo", "verdict_obs"]), "\n"]
    if figs:
        out += ["## Figures\n"] + [f"![{f}]({f})\n" for f in figs]
    out += ["\n## Limitations and how to read this\n",
            "- One model (Qwen2.5-7B-Instruct) and one decoding mode (greedy for labels). A second model is a one-line change (`MODEL_ID`) and is the first replication to run.\n",
            "- Observational: a probe reading a quantity does not show that the model USES it. Steering along a probe direction is the deferred next study.\n",
            "- `runaway` means 'hit the token cap' (512 tokens; 1024 for MATH), a budget-relative notion; right-censored lengths are masked, never imputed.\n",
            "- Forced-answer labels (settled/ready) insert a fixed cue after the visible prefix; they measure what the model would answer NOW, not what it will end up answering for any other reason.\n",
            "- MATH is restricted to integer answers; ARC is expert-keyed (tier B). Constraint types are a small, hand-picked, deterministic set, not a general instruction-following benchmark.\n",
            "- R2 below ~0.05, or CIs that include 0, are reported as such; do not read WEAK/NOT SUPPORTED rows as evidence of absence at larger scale.\n"]
    p = a / "REPORT.md"
    p.write_text("\n".join(out))
    log(f"wrote {p}")
    return p
