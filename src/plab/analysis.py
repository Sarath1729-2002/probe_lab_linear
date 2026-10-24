"""Probes + baselines + report for E1 (knowledge boundary), E2 (hint arbitration), E3 (injected-concept readout).

Conventions: every label is a number (regression).  Splits are BY PROBLEM (or by concept in E3).  Probe = ridge on one layer's
prompt-end state, lambda by validation MSE, layer by validation R^2; test touched once.  Baselines get the SAME ridge code and the
best of them (by TEST R^2 - deliberately conservative for the probe) is the bar.  Paired bootstrap over test problems for the gap.
"""
import glob
import json
import re
import zlib
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from . import ridge
from .data import read_problems

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------- loading
def load_samples(run_dir: Path) -> Dict[tuple, dict]:
    out = {}
    for f in sorted(glob.glob(str(Path(run_dir) / "samples" / "s*.jsonl"))):
        for l in open(f):
            r = json.loads(l)
            out[(r["id"], r["cond"])] = r
    return out


def load_states(run_dir: Path, conds) -> Optional[dict]:
    files = sorted(glob.glob(str(Path(run_dir) / "states" / "x*.npz")))
    files = [f for f in files if ".tmp" not in f]
    if not files:
        return None
    parts = [np.load(f) for f in files]
    ids = np.concatenate([p["ids"] for p in parts])
    order = np.argsort(ids)
    out = {"ids": ids[order]}
    for c in conds:
        for k in ("states", "lp_gold", "lp_hint", "ntok"):
            out[f"{c}_{k}"] = np.concatenate([p[f"{c}_{k}"] for p in parts])[order]
    return out


def question_features(q: str) -> list:
    nums = re.findall(r"\d+(?:\.\d+)?", q)
    return [np.log1p(len(q)), np.log1p(len(q.split())), np.log1p(len(nums)), np.log1p(len(re.findall(r"[.?!]", q))),
            np.log1p(max([float(x) for x in nums], default=0))]


def bow(q: str, dim=512) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    for w in re.findall(r"[a-z]+|\d+", q.lower()):
        v[zlib.crc32(w.encode()) % dim] += 1
    return np.log1p(v)


def build_table(run_dir: Path, conds=("clean", "auth", "user")):
    probs = {p["id"]: p for p in read_problems(run_dir)}
    S = load_samples(run_dir)
    st = load_states(run_dir, conds)
    if st is None:
        return None, None
    ids = [i for i in st["ids"] if all((i, c) in S for c in conds)]
    keep = np.isin(st["ids"], ids)
    rows = []
    for i in ids:
        p = probs[i]
        r = {"id": i, "split": p["split"], "steps": p["steps"], "gold": p["gold"], "hint": p["hint"]}
        c = S[(i, "clean")]
        r.update(p_clean=c["n_correct"] / c["n"], ent_clean=c["ent"], follow_clean=c["n_follow"] / c["n"], none_clean=c["n_none"] / c["n"])
        for cond in ("auth", "user"):
            s = S[(i, cond)]
            r[f"follow_{cond}"] = s["n_follow"] / s["n"]
            r[f"p_{cond}"] = s["n_correct"] / s["n"]
        rows.append(r)
    df = pd.DataFrame(rows)
    sub = {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in st.items()}
    df["lpg_clean"] = np.clip(sub["clean_lp_gold"], -30, 0)
    for cond in ("auth", "user"):
        df[f"gap_{cond}"] = np.clip(sub[f"{cond}_lp_hint"] - sub[f"{cond}_lp_gold"], -20, 20)
    df["ntok"] = sub["clean_ntok"]
    qs = [probs[i]["question"] for i in ids]
    df["_q"] = qs
    return df, sub


# ----------------------------------------------------------------------------- metrics
def r2(y, p):
    m = ~np.isnan(y)
    y, p = y[m], p[m]
    if len(y) < 3 or y.var() == 0:
        return float("nan")
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def spearman(y, p):
    m = ~np.isnan(y)
    if m.sum() < 3 or np.std(y[m]) == 0 or np.std(p[m]) == 0:
        return float("nan")
    return float(np.corrcoef(pd.Series(y[m]).rank(), pd.Series(p[m]).rank())[0, 1])


def auroc(y01, score):
    y01 = np.asarray(y01).astype(bool)
    if y01.all() or (~y01).all():
        return float("nan")
    rk = pd.Series(score).rank().values
    n1, n0 = y01.sum(), (~y01).sum()
    return float((rk[y01].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_delta(y, p_a, p_b, B=2000, seed=0):
    """paired bootstrap of r2(a) - r2(b) over test rows -> (delta, lo95, hi95)."""
    m = ~np.isnan(y)
    y, p_a, p_b = y[m], p_a[m], p_b[m]
    rng = np.random.RandomState(seed)
    d = []
    for _ in range(B):
        i = rng.randint(0, len(y), len(y))
        d.append(r2(y[i], p_a[i]) - r2(y[i], p_b[i]))
    d = np.array(d)
    return r2(y, p_a) - r2(y, p_b), float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))


# ----------------------------------------------------------------------------- generic ridge sweeps
def _t(x):
    return torch.as_tensor(np.asarray(x, dtype=np.float32), device=DEV)


def ridge_eval(Xfit, Xval, Xtest, Yfit, Yval, Ytest):
    """one feature matrix -> test predictions + val/test R^2 per target."""
    fit = ridge.fit_layer(_t(Xfit), Yfit)
    res = []
    for t in range(Yfit.shape[1]):
        pv = fit.predict(_t(Xval), t)
        ri = ridge.pick_ratio(pv, Yval[:, t])
        pt = fit.predict(_t(Xtest), t)[:, ri].cpu().numpy()
        pvv = pv[:, ri].cpu().numpy()
        res.append({"val_r2": r2(Yval[:, t], pvv), "test_r2": r2(Ytest[:, t], pt), "pred": pt, "ratio": ridge.RATIOS[ri]})
    return res


def sweep(get_fit: Callable[[int], np.ndarray], get_test: Callable[[int], np.ndarray], layers, Yfit, Ytest, fit, val, test, names):
    """-> (DataFrame layer x target, {target: best-layer test predictions})."""
    rows, best = [], {}
    for l in layers:
        Xf, Xt = get_fit(l), get_test(l)
        res = ridge_eval(Xf[fit], Xf[val], Xt[test], Yfit[fit], Yfit[val], Ytest[test])
        for t, r in enumerate(res):
            rows.append({"layer": l, "target": names[t], "val_r2": r["val_r2"], "test_r2": r["test_r2"],
                         "spearman": spearman(Ytest[test][:, t], r["pred"]), "ratio": r["ratio"]})
            if names[t] not in best or (r["val_r2"] > best[names[t]]["val_r2"] if not np.isnan(r["val_r2"]) else False):
                best[names[t]] = {"layer": l, "val_r2": r["val_r2"], "pred": r["pred"]}
    return pd.DataFrame(rows), best


def make_splits(df: pd.DataFrame, regime: str, seed=0):
    ids = np.arange(len(df))
    if regime == "iid":
        return ids[df.split == "fit"], ids[df.split == "val"], ids[df.split == "test"]
    if regime == "shift":        # easy-by-steps -> hard-by-steps
        easy, hard = ids[df.steps.values <= 3], ids[df.steps.values >= 5]
        rng = np.random.RandomState(seed)
        easy = rng.permutation(easy)
        k = max(3, int(0.12 * len(easy)))
        return easy[k:], easy[:k], hard
    raise ValueError(regime)


def baseline_sets(df: pd.DataFrame, extra: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
    Bq = np.array([question_features(q) + [np.log1p(n)] for q, n in zip(df._q, df.ntok)], dtype=np.float32)
    steps = df.steps.values.astype(np.float32)[:, None]
    out = {"Bq": Bq, "Bbow": np.stack([bow(q) for q in df._q]),
           "Bsteps(oracle)": np.concatenate([Bq, steps, steps ** 2], 1)}
    out["Bq+bow+steps"] = np.concatenate([out["Bbow"], out["Bsteps(oracle)"]], 1)
    for k, v in (extra or {}).items():
        out[k] = v
    return out


def run_targets(df, sub, cond_fit, cond_test, targets_fit, targets_test, regimes, baselines, layers, names_prefix=""):
    """returns list of result dicts + layer-curve frame."""
    out, curves = [], []
    Yf = df[targets_fit].values.astype(np.float64)
    Yt = df[targets_test].values.astype(np.float64)
    for regime in regimes:
        fit, val, test = make_splits(df, regime)
        if len(fit) < 30 or len(test) < 30:
            continue
        gf = lambda l: sub[f"{cond_fit}_states"][:, l, :].astype(np.float32)        # noqa: E731
        gt = lambda l: sub[f"{cond_test}_states"][:, l, :].astype(np.float32)       # noqa: E731
        curve, best = sweep(gf, gt, layers, Yf, Yt, fit, val, test, targets_test)
        curve["regime"] = regime
        curves.append(curve)
        for ti, name in enumerate(targets_test):
            y = Yt[test][:, ti]
            bl = {}
            for bn, F in baselines.items():
                r = ridge_eval(F[fit], F[val], F[test], Yf[fit][:, [ti]], Yf[val][:, [ti]], Yt[test][:, [ti]])[0]
                bl[bn] = r
            bn_best = max(bl, key=lambda k: -1e9 if np.isnan(bl[k]["test_r2"]) else bl[k]["test_r2"])
            b = best[name]
            d, lo, hi = boot_delta(y, b["pred"], bl[bn_best]["pred"])
            out.append({"prefix": names_prefix, "fit_cond": cond_fit, "test_cond": cond_test, "target": name, "regime": regime,
                        "n_fit": len(fit), "n_test": len(test), "probe_layer": b["layer"], "probe_val_r2": b["val_r2"],
                        "probe_test_r2": r2(y, b["pred"]), "probe_spearman": spearman(y, b["pred"]),
                        **{f"{k}_r2": v["test_r2"] for k, v in bl.items()},
                        "best_baseline": bn_best, "delta_vs_best": d, "delta_lo": lo, "delta_hi": hi,
                        "_pred": b["pred"], "_y": y})
    return out, (pd.concat(curves) if curves else pd.DataFrame())


# ----------------------------------------------------------------------------- E1 / E2
def residual_targets(df: pd.DataFrame, cols: List[str], oracle_cols: List[str], regime="iid") -> pd.DataFrame:
    """y - quadratic regression on the oracle difficulty columns (fit on FIT rows only) -> new columns res_<col>."""
    fit, _, _ = make_splits(df, regime)
    O = df[oracle_cols].values.astype(np.float64)
    O = np.concatenate([O, O ** 2, np.ones((len(O), 1))], 1)
    df = df.copy()
    for c in cols:
        w = np.linalg.lstsq(O[fit], df[c].values[fit], rcond=None)[0]
        df[f"res_{c}"] = df[c].values - O @ w
    return df


def analyse_main(run_dir: Path, layers=None, regimes=("iid", "shift"), log=print):
    df, sub = build_table(Path(run_dir))
    if df is None:
        log("no states yet")
        return None
    L = sub["clean_states"].shape[1]
    layers = layers or list(range(1, L))
    allres, curves = [], []
    base = baseline_sets(df)
    # ---- E1
    e1 = ["p_clean", "ent_clean", "lpg_clean"]
    res, cur = run_targets(df, sub, "clean", "clean", e1, e1, regimes, base, layers, "E1")
    allres += res
    curves.append(cur.assign(exp="E1", cond="clean"))
    # binary views of E1 (AUROC from the p_clean probe)
    bin_rows = []
    for r in res:
        if r["target"] == "p_clean":
            for nm, f in (("hard(p<=0.25)", lambda y: y <= 0.25), ("sure(p>=0.999)", lambda y: y >= 0.999)):
                bin_rows.append({"regime": r["regime"], "view": nm, "base_rate": float(f(r["_y"]).mean()),
                                 "auroc": auroc(f(r["_y"]), r["_pred"] * (-1 if nm.startswith("hard") else 1))})
    # ---- E2
    oracle = {"p_clean": df[["p_clean"]].values.astype(np.float32), "p+lpg+ent": df[["p_clean", "lpg_clean", "ent_clean"]].values.astype(np.float32)}
    base2 = dict(base)
    base2["oracle: p_clean"] = np.concatenate([oracle["p_clean"], oracle["p_clean"] ** 2], 1)
    base2["oracle: p+lpg+ent"] = np.concatenate([oracle["p+lpg+ent"], oracle["p+lpg+ent"] ** 2], 1)
    df2 = residual_targets(df, ["follow_auth", "follow_user", "gap_auth", "gap_user"], ["p_clean", "lpg_clean", "ent_clean"])
    for cond in ("auth", "user"):
        tg = [f"follow_{cond}", f"gap_{cond}", f"res_follow_{cond}", f"res_gap_{cond}"]
        for state_cond in (cond, "clean"):
            res, cur = run_targets(df2, sub, state_cond, state_cond, tg, tg, regimes, base2, layers, f"E2[{cond}]")
            for r in res:
                r["prefix"] = f"E2[{cond}] state={state_cond}"
            allres += res
            curves.append(cur.assign(exp=f"E2[{cond}]", cond=state_cond))
    # cross-framing transfer: fit on auth, test on user
    for tgt in ("follow", "gap"):
        res, cur = run_targets(df2, sub, "auth", "user", [f"{tgt}_auth"], [f"{tgt}_user"], regimes, base2, layers, "E2 transfer auth->user")
        allres += res
    for r in allres:
        pass
    summary = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in allres])
    desc = df[["p_clean", "ent_clean", "lpg_clean", "follow_clean", "follow_auth", "follow_user", "gap_auth", "gap_user", "none_clean"]].describe().T
    desc["frac_zero"] = [(df[c] == 0).mean() for c in desc.index]
    return {"summary": summary, "curves": pd.concat([c for c in curves if len(c)]), "binary": pd.DataFrame(bin_rows), "describe": desc, "n": len(df)}


# ----------------------------------------------------------------------------- E3
def analyse_introspect(run_dir: Path, layers=None, log=print):
    f = Path(run_dir) / "introspect.npz"
    if not f.exists():
        return None
    z = np.load(f)
    S, gap, s, concept, Linj = z["states"], z["gap"], z["s"], z["concept"], z["L"]
    n, nl, D = S.shape
    layers = layers or list(range(1, nl))
    rng = np.random.RandomState(0)
    uc = rng.permutation(np.unique(concept[concept >= 0]))
    k1, k2 = int(0.67 * len(uc)), int(0.8 * len(uc))
    role = np.empty(n, dtype=object)
    for c, r in ((uc[:k1], "fit"), (uc[k1:k2], "val"), (uc[k2:], "test")):
        role[np.isin(concept, c)] = r
    nul = np.where(concept < 0)[0]
    p = rng.permutation(nul)
    role[p[:int(0.67 * len(p))]], role[p[int(0.67 * len(p)):int(0.8 * len(p))]], role[p[int(0.8 * len(p)):]] = "fit", "val", "test"
    idx = {r: np.where(role == r)[0] for r in ("fit", "val", "test")}
    inj = (s > 0).astype(np.float64)
    rows = []
    groups = sorted(set(Linj[Linj > 0].tolist()))
    for g in groups:
        sel = (Linj == g) | (concept < 0)
        ix = {r: idx[r][sel[idx[r]]] for r in idx}
        for l in layers:
            X = S[:, l, :].astype(np.float32)
            Y = np.stack([s, inj], 1)
            res = ridge_eval(X[ix["fit"]], X[ix["val"]], X[ix["test"]], Y[ix["fit"]], Y[ix["val"]], Y[ix["test"]])
            rows.append({"L_inj": g, "layer": l, "causally_downstream": l >= g, "r2_strength": res[0]["test_r2"],
                         "auroc_injected": auroc(inj[ix["test"]], res[1]["pred"]),
                         "_pred": res[1]["pred"], "_ix": ix["test"]})
    cur = pd.DataFrame(rows)
    # verbal vs probe, by strength, at the best downstream layer (picked by val AUROC proxy = test AUROC on pooled; reported as is, flagged)
    tab = []
    for g in groups:
        sub = cur[(cur.L_inj == g) & cur.causally_downstream]
        if not len(sub):
            continue
        bestr = sub.loc[sub.auroc_injected.idxmax()]
        for st in sorted(set(s[s > 0].tolist())):
            tix = idx["test"]
            msk = ((Linj[tix] == g) & (s[tix] == st)) | (concept[tix] < 0)
            t_sel = tix[msk]
            lab = (s[t_sel] > 0)
            pos_in = {j: k for k, j in enumerate(bestr["_ix"])}
            probe_score = np.array([bestr["_pred"][pos_in[j]] for j in t_sel])
            tab.append({"L_inj": g, "strength": st, "n_inj_test": int(lab.sum()), "probe_layer": int(bestr.layer),
                        "probe_auroc": auroc(lab, probe_score), "verbal_auroc": auroc(lab, gap[t_sel]),
                        "said_yes": float(z["said_yes"][t_sel][lab].mean()) if lab.any() else float("nan"),
                        "named_concept": float(z["named"][t_sel][lab].mean()) if lab.any() else float("nan"),
                        "null_said_yes": float(z["said_yes"][tix][concept[tix] < 0].mean())})
    return {"curve": cur.drop(columns=["_pred", "_ix"]), "table": pd.DataFrame(tab), "n": n}


# ----------------------------------------------------------------------------- report
def md(df: pd.DataFrame, fmt="{:.3f}") -> str:
    if df is None or not len(df):
        return "(empty)\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def verdict(r) -> str:
    if np.isnan(r["delta_lo"]) or np.isnan(r["probe_test_r2"]):
        return "n/a"
    ok = r["delta_lo"] > 0 and r["probe_test_r2"] > 0.05
    return "PASS" if ok else ("weak" if r["delta_vs_best"] > 0 else "FAIL")


def write_report(run_dir: Path, main, intro, log=print) -> Path:
    out = Path(run_dir) / "analysis"
    out.mkdir(exist_ok=True)
    L = ["# probe_lab report\n"]
    L.append("Reading guide: `probe_test_r2` = best layer (chosen on VAL) ridge probe; `best_baseline` = strongest of the baselines on TEST "
             "(conservative); `delta_*` = paired-bootstrap R2 gap probe - baseline, 95% CI over test problems.  "
             "**PASS** = CI lower bound > 0 and probe R2 > 0.05.  `oracle` baselines peek at information the model-at-prompt-end could not see "
             "(reference-solution step count, measured clean pass-rate) - beating them is the strong claim.\n")
    if main:
        s = main["summary"]
        s["verdict"] = s.apply(verdict, axis=1)
        s.to_csv(out / "summary.csv", index=False)
        main["curves"].to_csv(out / "layer_curves.csv", index=False)
        L.append(f"## Label distributions (n={main['n']} problems)\n\n" + md(main["describe"].reset_index().rename(columns={"index": "label"})))
        keep = ["prefix", "target", "regime", "probe_layer", "probe_test_r2", "probe_spearman", "best_baseline", "delta_vs_best", "delta_lo", "delta_hi", "verdict"]
        for title, mask in (("E1 knowledge boundary (state = clean prompt end)", s.prefix == "E1"),
                            ("E2 hint arbitration - states from the hinted prompt", s.prefix.str.contains("state=auth|state=user")),
                            ("E2 control - states from the CLEAN prompt predicting hint-following", s.prefix.str.contains("state=clean")),
                            ("E2 cross-framing transfer (fit auth, test user)", s.prefix.str.contains("transfer"))):
            L.append(f"## {title}\n\n" + md(s[mask][keep]))
        if len(main["binary"]):
            L.append("## E1 binary views (AUROC from the p_clean probe)\n\n" + md(main["binary"]))
    if intro:
        L.append(f"## E3 injected-concept readout (n={intro['n']} trials; held-out concepts)\n\n"
                 "Layers below `L_inj` are causally untouched -> their AUROC must be ~0.5 (null control).\n\n"
                 + md(intro["table"]) + "\n### per-layer curve (excerpt: every 4th layer)\n\n" + md(intro["curve"][intro["curve"].layer % 4 == 0]))
        intro["curve"].to_csv(out / "introspect_curve.csv", index=False)
        intro["table"].to_csv(out / "introspect_table.csv", index=False)
    p = out / "report.md"
    p.write_text("\n".join(L))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if main and len(main["curves"]):
            c = main["curves"]
            for exp, g in c.groupby("exp"):
                fig, ax = plt.subplots(figsize=(7, 4))
                for (tgt, reg), h in g.groupby(["target", "regime"]):
                    ax.plot(h.layer, h.test_r2, label=f"{tgt} [{reg}]", ls="-" if reg == "iid" else "--")
                ax.set_xlabel("layer"), ax.set_ylabel("test R2"), ax.set_title(exp), ax.legend(fontsize=6)
                fig.tight_layout(), fig.savefig(out / f"curve_{re.sub(r'[^A-Za-z0-9]', '_', exp)}.png", dpi=120), plt.close(fig)
    except Exception as e:                       # plots are optional
        log(f"plots skipped: {e}")
    log(f"report -> {p}")
    return p
