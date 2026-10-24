"""Steering aggregation + the written report (REPORT.md and figures) for the main study."""
import glob
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from . import analysis as A
from .steer import ALPHAS


def load_steer(run_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(Path(run_dir) / "steer" / "*.jsonl"))):
        rows += [json.loads(l) for l in open(f)]
    return pd.DataFrame(rows)


def steer_summary(df: pd.DataFrame, B=1000, seed=0):
    """-> (per-arm table, slope table).  slope = d(adoption rate)/d(alpha); CI by bootstrap over problems."""
    if df is None or not len(df):
        return pd.DataFrame(), pd.DataFrame()
    df = df.assign(rate=df.n_follow / df.n, acc=df.n_correct / df.n, fail=df.n_none / df.n)
    df["fam"] = df.arm.where(~df.arm.str.startswith("rand"), "random")
    rng = np.random.RandomState(seed)
    arms, slopes = [], []
    for f, g in df.groupby("framing"):
        base = g[g.arm == "none"].set_index("id")
        ids = base.index.values
        tables = {}
        for fam in ("probe", "random"):
            gg = g[g.fam == fam]
            piv = gg.groupby(["id", "alpha"])[["rate", "acc", "fail"]].mean().unstack("alpha")
            piv = piv.reindex(ids)
            tables[fam] = piv
            for a in sorted(gg.alpha.unique()):
                arms.append({"framing": f, "arm": fam, "alpha": a, "adoption_rate": piv["rate"][a].mean(), "accuracy": piv["acc"][a].mean(),
                             "parse_fail": piv["fail"][a].mean(), "n_problems": int(piv["rate"][a].notna().sum())})
        arms.append({"framing": f, "arm": "none", "alpha": 0.0, "adoption_rate": base.rate.mean(), "accuracy": base.acc.mean(),
                     "parse_fail": base.fail.mean(), "n_problems": len(ids)})

        def slope(idx, fam):
            piv = tables[fam]["rate"].iloc[idx]
            al = np.array([0.0] + list(piv.columns))
            ys = np.array([base.rate.values[idx].mean()] + [piv[a].mean() for a in piv.columns])
            return np.polyfit(al, ys, 1)[0]
        full = np.arange(len(ids))
        sp, sr = slope(full, "probe"), slope(full, "random")
        bs = []
        for _ in range(B):
            i = rng.randint(0, len(ids), len(ids))
            bs.append((slope(i, "probe"), slope(i, "random")))
        bs = np.array(bs)
        d = bs[:, 0] - bs[:, 1]
        slopes.append({"framing": f, "probe_slope": sp, "probe_lo": np.percentile(bs[:, 0], 2.5), "probe_hi": np.percentile(bs[:, 0], 97.5),
                       "random_slope": sr, "diff": sp - sr, "diff_lo": np.percentile(d, 2.5), "diff_hi": np.percentile(d, 97.5),
                       "supported": bool(np.percentile(d, 2.5) > 0)})
    return pd.DataFrame(arms), pd.DataFrame(slopes)


def _fmt(df, cols=None, fmt="{:.3f}"):
    return A.md(df[cols] if cols else df, fmt)


def write(run_dir: Path, log=print) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run_dir = Path(run_dir)
    out = run_dir / "analysis_v3"
    P = pickle.load(open(out / "main_results.pkl", "rb"))
    a, b, lo, dose, ceil = P["a"], P["b"], P["lofo"], P["dose"], P["ceil"]
    for t in (a, b):
        t["verdict"] = t.apply(A.verdict, axis=1)
    steer = load_steer(run_dir)
    arms, slopes = steer_summary(steer)
    fr = P["framings"]

    # ---------------- figures
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for k, reg in enumerate(("iid", "shift")):
        c = P["curves_a"][reg]
        ax[k].plot(c.layer, c.r2, color="C0"), ax[k].fill_between(c.layer, c.lo, c.hi, color="C0", alpha=0.25)
        ax[k].set_title(f"S (susceptibility) from the CLEAN prompt-end state [{reg}]"), ax[k].set_xlabel("layer"), ax[k].set_ylabel("test R2")
        ax[k].axhline(0, color="k", lw=0.5)
    fig.tight_layout(), fig.savefig(out / "fig1_layer_sweep_S.png", dpi=130), plt.close(fig)

    cb = P["cur_b"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for k, reg in enumerate(("iid", "shift")):
        for f in fr:
            h = cb[(cb.framing == f) & (cb.regime == reg) & (cb.target == f"follow_{f}")]
            ax[k].plot(h.layer, h.test_r2, label=f)
        ax[k].set_title(f"adoption rate from the HINTED prompt-end state [{reg}]"), ax[k].set_xlabel("layer"), ax[k].set_ylabel("test R2")
        ax[k].axhline(0, color="k", lw=0.5)
    ax[0].legend(fontsize=7)
    fig.tight_layout(), fig.savefig(out / "fig2_layer_sweep_framings.png", dpi=130), plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.bar(dose.framing, dose.follow_rate, yerr=[dose.follow_rate - dose.lo, dose.hi - dose.follow_rate], color="C1")
    ax.set_ylabel("adoption rate (planted wrong answer)"), ax.set_title("dose-response across framings")
    fig.tight_layout(), fig.savefig(out / "fig3_dose_response.png", dpi=130), plt.close(fig)

    if len(lo):
        fig, ax = plt.subplots(figsize=(7, 3.6))
        x = np.arange(len(lo))
        ax.bar(x - 0.2, lo.probe_r2, 0.4, label="probe (unseen framing)"), ax.bar(x + 0.2, lo.baseline_r2, 0.4, label="best baseline")
        ax.set_xticks(x), ax.set_xticklabels(lo.held_out, rotation=30), ax.set_ylabel("test R2"), ax.legend(fontsize=7)
        ax.set_title("leave-one-framing-out")
        fig.tight_layout(), fig.savefig(out / "fig4_lofo.png", dpi=130), plt.close(fig)

    if len(P["cal"]):
        fig, ax = plt.subplots(figsize=(4.2, 4))
        ax.plot(P["cal"].pred_mean, P["cal"].observed_mean, "o-"), ax.plot([0, P["cal"].pred_mean.max()], [0, P["cal"].pred_mean.max()], "k--", lw=0.6)
        ax.set_xlabel("probe-predicted S (decile mean)"), ax.set_ylabel("observed S"), ax.set_title("calibration (held-out problems)")
        fig.tight_layout(), fig.savefig(out / "fig5_calibration.png", dpi=130), plt.close(fig)

    if len(arms):
        fs = arms.framing.unique()
        fig, ax = plt.subplots(1, len(fs), figsize=(5.2 * len(fs), 3.6), squeeze=False)
        for k, f in enumerate(fs):
            g = arms[arms.framing == f]
            base = g[g.arm == "none"].adoption_rate.iloc[0]
            for arm, col in (("probe", "C3"), ("random", "C7")):
                h = g[g.arm == arm].sort_values("alpha")
                xs = list(h.alpha.values)
                ys = list(h.adoption_rate.values)
                xs.insert(int(np.searchsorted(xs, 0)), 0.0), ys.insert(int(np.searchsorted(h.alpha.values, 0)), base)
                ax[0][k].plot(xs, ys, "o-", color=col, label=arm)
            ax[0][k].set_title(f"steering [{f}]"), ax[0][k].set_xlabel("alpha (fraction of residual norm)"), ax[0][k].set_ylabel("adoption rate"), ax[0][k].legend(fontsize=7)
        fig.tight_layout(), fig.savefig(out / "fig6_steering.png", dpi=130), plt.close(fig)

    # ---------------- headline numbers
    def pick(t, **kw):
        m = np.ones(len(t), bool)
        for k, v in kw.items():
            m &= (t[k] == v).values
        return t[m]
    hl = []
    for reg in ("iid", "shift"):
        r = pick(a, target="S", regime=reg)
        if len(r):
            r = r.iloc[0]
            hl.append(f"- **Latent susceptibility `S` from the clean prompt-end state ({reg})**: R2 {r.probe_test_r2:.3f}, Spearman {r.probe_spearman:.3f}, "
                      f"layer {int(r.probe_layer)}; +{r.delta_vs_best:.3f} R2 over the best baseline ({r.best_baseline}), 95% CI [{r.delta_lo:.3f}, {r.delta_hi:.3f}] -> **{r.verdict}**.")
    sup = pick(b, regime="iid"); sup = sup[sup.target.str.startswith("follow")]
    hl.append(f"- **Per-framing adoption from the hinted state (iid)**: supported in {int((sup.verdict == 'PASS').sum())}/{len(sup)} framings; "
              f"under the step shift {int((pick(b, regime='shift')[lambda t: t.target.str.startswith('follow')].verdict == 'PASS').sum())}/{len(sup)}.")
    if len(lo):
        hl.append(f"- **Leave-one-framing-out**: probe R2 {lo.probe_r2.mean():.3f} on average over unseen framings; beats the best baseline with CI>0 in "
                  f"{int((lo.delta_lo > 0).sum())}/{len(lo)} folds.")
    if len(slopes):
        hl.append(f"- **Steering**: probe-direction slope differs from random directions (CI excludes 0) in {int(slopes.supported.sum())}/{len(slopes)} framings tested.")
    else:
        hl.append("- **Steering**: not run / no results found.")

    L = ["# Pre-generation readout of hint adoption in Qwen2.5-7B-Instruct (GSM8K)\n",
         f"n = {P['n']} problems, {len(fr)} framings, 16 sampled solutions per (problem, framing).  All numbers are on held-out problems.\n",
         "## Headline\n", "\n".join(hl) + "\n",
         "## Question and metrics\n",
         "Does the residual stream at the last prompt token, *before the model writes anything*, linearly encode how likely it is to adopt a planted wrong answer, "
         "beyond problem difficulty and beyond what the prompt text reveals?\n\n"
         "* **M1** best-layer ridge probe R2 / Spearman (layer and lambda chosen on validation problems only; test touched once).\n"
         "* **M2** delta-R2 versus the strongest baseline (chosen on TEST, so conservative), paired bootstrap 95% CI over test problems. "
         "**PASS** = CI lower bound > 0 and probe R2 > 0.05.\n"
         "* **M3** AUROC for 'adopts' (adoption rate >= 0.25).  **M4** leave-one-framing-out R2.  **M5** steering dose-response vs random directions.\n"
         "* Splits: `iid` (by problem) and `shift` (train on <=3-step problems, test on >=5-step).\n"
         "* Baselines: question length/numbers, hashed words, reference-solution step count (oracle), measured clean pass-rate / entropy / forced log P(gold) (oracle), "
         "hint-number features, hint+gold features (oracle), the model's own layer-4/8 state of the same prompt, and the adoption rate in the other framings (oracle).\n",
         "## 1. Label diagnostics\n", "### Dose-response across framings\n", _fmt(dose),
         "![dose](fig3_dose_response.png)\n",
         "### Noise ceiling of the labels (binomial sampling noise, n=16)\n", _fmt(ceil),
         "R2 can't exceed `r2_ceiling` even for a perfect predictor.\n",
         "## 2. Susceptibility from the clean prompt-end state\n",
         "Targets: `S` = mean adoption over framings, `Sgap` = mean forced log P(hint) - log P(gold), plus the knowledge-boundary labels "
         "(`p_clean` pass rate, `ent_clean` answer entropy, `lpg_clean` forced log P(gold)).\n",
         _fmt(a, ["target", "regime", "probe_layer", "probe_test_r2", "probe_spearman", "best_baseline", "delta_vs_best", "delta_lo", "delta_hi", "verdict"]),
         "![sweep](fig1_layer_sweep_S.png)\n", "### Calibration (iid)\n", _fmt(P["cal"]), "![cal](fig5_calibration.png)\n",
         f"Permutation control (labels shuffled within the training set, best layer): test R2 = {P['perm'][0]:.4f} +- {P['perm'][1]:.4f}.\n",
         "## 3. Adoption from the hinted prompt-end state, per framing\n",
         _fmt(b[b.target.str.startswith("follow")], ["framing", "regime", "probe_layer", "probe_test_r2", "probe_spearman", "auroc_adopt", "base_rate_adopt",
                                                     "best_baseline", "delta_vs_best", "delta_lo", "delta_hi", "verdict"]),
         "![frames](fig2_layer_sweep_framings.png)\n",
         "### Forced-answer gap (no reasoning): log P(hint) - log P(gold)\n",
         _fmt(b[b.target.str.startswith("gap")], ["framing", "regime", "probe_layer", "probe_test_r2", "best_baseline", "delta_vs_best", "delta_lo", "delta_hi", "verdict"]),
         "## 4. Leave-one-framing-out\n", _fmt(lo, ["held_out", "probe_layer", "probe_r2", "spearman", "auroc_adopt", "base_rate", "best_baseline", "baseline_r2",
                                                 "delta", "delta_lo", "delta_hi"]), "![lofo](fig4_lofo.png)\n",
         "## 5. Causal test: steering\n"]
    if len(slopes):
        L += ["Slope = change in adoption rate per unit alpha (alpha in fractions of the typical residual norm at the probe layer). "
              "Random = mean of random unit directions of equal norm.\n", _fmt(slopes), _fmt(arms), "![steer](fig6_steering.png)\n"]
    else:
        L.append("No steering results found.\n")
    ap = A.analyse_introspect(run_dir)
    if ap:
        L += ["## Appendix: injected-concept readout (E3, separate small experiment)\n", "Probe layer chosen on validation concepts. n=12 injected test trials per cell: treat as a pilot.\n", A.md(ap["table"])]
    L += ["## Limitations (what not to claim)\n",
          "* One model, one dataset (GSM8K), which Qwen2.5-7B mostly solves; hint adoption is mostly a property of the hard minority.\n"
          "* Labels are sampling estimates (n=16); see ceilings.  Planted hints are wrong by a small offset (+-1..10), which may be more or less believable than other errors.\n"
          "* The forced-answer gap is a no-reasoning proxy and is not the same behaviour as adopting the answer after reasoning.\n"
          "* Linear readout is correlational; only the steering experiment bears on causality, and steering magnitudes were not tuned.\n"
          "* `oracle` baselines use information not available at the prompt end; beating them is the strong claim but they are not a ceiling on what text alone can predict.\n"]
    p = out / "REPORT.md"
    p.write_text("\n".join(L))
    (out / "headline.json").write_text(json.dumps({"headline": hl}, indent=1))
    steer_csv = out / "steering_slopes.csv"
    if len(slopes):
        slopes.to_csv(steer_csv, index=False), arms.to_csv(out / "steering_arms.csv", index=False)
    log(f"report -> {p}")
    return p
