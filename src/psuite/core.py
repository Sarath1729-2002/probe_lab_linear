"""Shared machinery: metrics, split regimes, ridge sweeps over layers, paired-bootstrap comparison with baselines.

Conventions (identical for every experiment): every label is a number (regression).  Probe = closed-form ridge on ONE layer's frozen state,
lambda chosen on validation MSE, layer chosen on validation R^2, test rows touched once.  Baselines go through the SAME ridge code.
Baselines are split in two families: OBSERVABLE (computable from the prompt text / the model's early layers) and ORACLE (use gold, measured
pass-rate, behaviour in other framings: information the model cannot have at the prompt end).  Verdicts are given against both.
"""
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from . import ridge

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MIN_ROWS = 30


# ----------------------------------------------------------------------------- metrics
def r2(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    m = ~np.isnan(y)
    y, p = y[m], p[m]
    if len(y) < 3 or y.var() == 0:
        return float("nan")
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def spearman(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    m = ~np.isnan(y)
    if m.sum() < 3 or np.std(y[m]) == 0 or np.std(p[m]) == 0:
        return float("nan")
    return float(np.corrcoef(pd.Series(y[m]).rank(), pd.Series(p[m]).rank())[0, 1])


def auroc(y01, score):
    y01 = np.asarray(y01).astype(bool)
    if y01.all() or (~y01).all():
        return float("nan")
    rk = pd.Series(np.asarray(score, dtype=float)).rank().values
    n1, n0 = y01.sum(), (~y01).sum()
    return float((rk[y01].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_delta(y, p_a, p_b, B=1000, seed=0):
    """paired bootstrap of r2(a) - r2(b) over test rows -> (delta, lo95, hi95)."""
    y, p_a, p_b = np.asarray(y, float), np.asarray(p_a, float), np.asarray(p_b, float)
    m = ~np.isnan(y)
    y, p_a, p_b = y[m], p_a[m], p_b[m]
    if len(y) < 5:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    d = np.array([r2(y[i], p_a[i]) - r2(y[i], p_b[i]) for i in (rng.randint(0, len(y), len(y)) for _ in range(B))])
    return r2(y, p_a) - r2(y, p_b), float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))


def boot_ci(x, fn=np.mean, B=1000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    v = np.array([fn(x[rng.randint(0, len(x), len(x))]) for _ in range(B)])
    return float(fn(x)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def boot_stat(idx_pairs_fn: Callable, n: int, B=1000, seed=0):
    """generic bootstrap of a statistic computed from a resampled index vector -> (value, lo, hi)."""
    rng = np.random.RandomState(seed)
    base = idx_pairs_fn(np.arange(n))
    v = np.array([idx_pairs_fn(rng.randint(0, n, n)) for _ in range(B)], dtype=float)
    return float(base), float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))


# ----------------------------------------------------------------------------- ridge plumbing
def _t(x):
    return torch.as_tensor(np.asarray(x, dtype=np.float32), device=DEV)


def ridge_eval(Xfit, Xval, Xtest, Yfit, Yval, Ytest):
    fit = ridge.fit_layer(_t(Xfit), Yfit)
    res = []
    for t in range(Yfit.shape[1]):
        pv = fit.predict(_t(Xval), t)
        ri = ridge.pick_ratio(pv, Yval[:, t])
        pt = fit.predict(_t(Xtest), t)[:, ri].cpu().numpy()
        pvv = pv[:, ri].cpu().numpy()
        res.append({"val_r2": r2(Yval[:, t], pvv), "test_r2": r2(Ytest[:, t], pt), "pred": pt, "pred_val": pvv, "ratio": ridge.RATIOS[ri]})
    return res


def sweep(get_fit: Callable[[int], np.ndarray], get_test: Callable[[int], np.ndarray], layers, Yfit, Ytest, fit, val, test, names, allpred=None):
    """-> (DataFrame layer x target, {target: best-layer (chosen on VAL) predictions})."""
    rows, best = [], {}
    for l in layers:
        Xf, Xt = get_fit(l), get_test(l)
        res = ridge_eval(Xf[fit], Xf[val], Xt[test], Yfit[fit], Yfit[val], Ytest[test])
        for t, r in enumerate(res):
            if allpred is not None:
                allpred[(l, names[t])] = r["pred"]
            rows.append({"layer": l, "target": names[t], "val_r2": r["val_r2"], "test_r2": r["test_r2"],
                         "spearman": spearman(Ytest[test][:, t], r["pred"]), "ratio": r["ratio"]})
            ok = not np.isnan(r["val_r2"])
            if names[t] not in best or (ok and (np.isnan(best[names[t]]["val_r2"]) or r["val_r2"] > best[names[t]]["val_r2"])):
                best[names[t]] = {"layer": l, "val_r2": r["val_r2"], "pred": r["pred"], "pred_val": r["pred_val"]}
    return pd.DataFrame(rows), best


# ----------------------------------------------------------------------------- split regimes
def make_splits(df: pd.DataFrame, regime: str, seed=0):
    """iid   : the stored by-problem fit/val/test split.
    shift  : per dataset, train on the EASY side of the design difficulty d1 (<= 35th pct), test on the HARD side (>= 65th pct).
    lodo:X : leave-dataset-out - fit/val on the other datasets, test on ALL problems of X (never seen: new problems AND new distribution)."""
    ids = np.arange(len(df))
    sp, dsv = df.split.values, df.ds.values
    if regime == "iid":
        return ids[sp == "fit"], ids[sp == "val"], ids[sp == "test"]
    if regime == "shift":
        rng = np.random.RandomState(seed)
        easy_all, hard_all = [], []
        for ds in np.unique(dsv):
            m = dsv == ds
            d1 = df.d1.values[m]
            if len(np.unique(d1)) < 3:
                continue
            lo, hi = np.percentile(d1, 35), np.percentile(d1, 65)
            ix = ids[m]
            easy = ix[d1 <= lo]
            hard = ix[d1 >= hi] if hi > lo else ix[d1 > lo]
            easy_all.append(rng.permutation(easy))
            hard_all.append(hard)
        if not easy_all:
            return ids[:0], ids[:0], ids[:0]
        easy, hard = np.concatenate(easy_all), np.concatenate(hard_all)
        k = max(3, int(0.12 * len(easy)))
        easy = rng.permutation(easy)
        return easy[k:], easy[:k], hard
    if regime.startswith("lodo:"):
        held = regime.split(":", 1)[1]
        other = dsv != held
        return ids[other & (sp == "fit")], ids[other & (sp == "val")], ids[dsv == held]
    raise ValueError(regime)


def take(df: pd.DataFrame, sub: Dict[str, np.ndarray], mask) -> tuple:
    mask = np.asarray(mask)
    return df[mask].reset_index(drop=True), {k: v[mask] for k, v in sub.items()}


# ----------------------------------------------------------------------------- the generic probe-vs-baselines experiment
def run_targets(df, sub, cond_fit, cond_test, targets_fit, targets_test, regimes, baselines: Dict[str, np.ndarray], layers, prefix="",
                exclude_for: Optional[Dict[str, List[str]]] = None, B=1000, ci_for=()):
    """baselines: name -> [N, f] feature matrix (rows aligned with df).  exclude_for[target] = baseline names that contain that target.
    -> (list of result dicts, per-layer curve frame)."""
    out, curves = [], []
    Yf = df[targets_fit].values.astype(np.float64)
    Yt = df[targets_test].values.astype(np.float64)
    gf = lambda l: sub[cond_fit][:, l, :].astype(np.float32)             # noqa: E731
    gt = lambda l: sub[cond_test][:, l, :].astype(np.float32)            # noqa: E731
    for regime in regimes:
        fit, val, test = make_splits(df, regime)
        if len(fit) < MIN_ROWS or len(test) < MIN_ROWS or len(val) < 5:
            continue
        allpred = {} if ci_for else None
        curve, best = sweep(gf, gt, layers, Yf, Yt, fit, val, test, targets_test, allpred=allpred)
        curve["regime"] = regime
        if ci_for:
            curve["r2_lo"], curve["r2_hi"] = np.nan, np.nan
            rng = np.random.RandomState(0)
            for name in ci_for:
                if name not in targets_test:
                    continue
                y = Yt[test][:, targets_test.index(name)]
                bi = [rng.randint(0, len(y), len(y)) for _ in range(200)]
                for l in layers:
                    v = [r2(y[i], allpred[(l, name)][i]) for i in bi]
                    m = (curve.layer == l) & (curve.target == name)
                    curve.loc[m, "r2_lo"], curve.loc[m, "r2_hi"] = np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)
        curves.append(curve)
        blall = {bn: ridge_eval(F[fit], F[val], F[test], Yf[fit], Yf[val], Yt[test]) for bn, F in baselines.items()}      # all targets in ONE fit per baseline
        for ti, name in enumerate(targets_test):
            y = Yt[test][:, ti]
            bl = {bn: r[ti] for bn, r in blall.items() if bn not in (exclude_for or {}).get(name, [])}
            if not bl:
                continue
            key = lambda k: -1e9 if np.isnan(bl[k]["test_r2"]) else bl[k]["test_r2"]       # noqa: E731
            bn_best = max(bl, key=key)                                    # best of ALL baselines, chosen on TEST (conservative for the probe)
            avail = [k for k in bl if "oracle" not in k]
            bn_av = max(avail, key=key) if avail else bn_best
            b = best[name]
            d, lo, hi = boot_delta(y, b["pred"], bl[bn_best]["pred"], B=B)
            da, loa, hia = boot_delta(y, b["pred"], bl[bn_av]["pred"], B=B)
            dsv = df.ds.values[test]
            per_ds = {f"r2[{u}]": r2(y[dsv == u], b["pred"][dsv == u]) for u in np.unique(dsv)} if len(np.unique(dsv)) > 1 else {}
            out.append({"prefix": prefix, "fit_cond": cond_fit, "test_cond": cond_test, "target": name, "regime": regime,
                        "n_fit": len(fit), "n_test": len(test), "probe_layer": b["layer"], "probe_val_r2": b["val_r2"],
                        "probe_test_r2": r2(y, b["pred"]), "probe_spearman": spearman(y, b["pred"]),
                        **{f"{k}_r2": v["test_r2"] for k, v in bl.items()},
                        "best_baseline": bn_best, "delta_vs_oracle": d, "oracle_lo": lo, "oracle_hi": hi,
                        "best_observable": bn_av, "obs_r2": bl[bn_av]["test_r2"], "delta_vs_obs": da, "obs_lo": loa, "obs_hi": hia,
                        **per_ds, "_pred": b["pred"], "_y": y, "_test_ids": df.id.values[test], "_pred_obs": bl[bn_av]["pred"],
                        "_pred_oracle": bl[bn_best]["pred"], "_ds": dsv})
    return out, (pd.concat(curves) if curves else pd.DataFrame())


def verdict(delta, lo, r2p, floor=0.05) -> str:
    if np.isnan(lo) or np.isnan(r2p):
        return "n/a"
    return "PASS" if (lo > 0 and r2p > floor) else ("weak" if delta > 0 else "FAIL")


def strip(results) -> pd.DataFrame:
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in results])
    if len(df):
        df["verdict_obs"] = [verdict(a, b, c) for a, b, c in zip(df.delta_vs_obs, df.obs_lo, df.probe_test_r2)]
        df["verdict_oracle"] = [verdict(a, b, c) for a, b, c in zip(df.delta_vs_oracle, df.oracle_lo, df.probe_test_r2)]
    return df


def md(df: pd.DataFrame, fmt="{:.3f}", maxrows=None) -> str:
    if df is None or not len(df):
        return "(empty)\n"
    if maxrows:
        df = df.head(maxrows)
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(fmt.format(v) if isinstance(v, (float, np.floating)) and not np.isnan(v) else ("" if isinstance(v, float) and np.isnan(v) else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
