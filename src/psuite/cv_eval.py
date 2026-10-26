"""Cross-validated linear-probe evaluation: the CORRECTED protocol of the mega run.

Why: in the uncertainty POC the layer was chosen on a ~80-row validation split and landed on layer 2 for three labels (R2 0.06-0.22) while per-dataset fits at layers 20-28
scored 0.11-0.34.  Layer selection on a tiny split is noise.  Here the (layer, lambda) pair is chosen by GROUPED K-FOLD cross-validation over the whole training pool
(folds never split a problem), then the probe is refit on the pool and the test rows are touched once.  Baselines go through the identical code (feature set = one 'layer').
Rows can be many per problem (steps, checkpoints): folds AND bootstraps are over problems (`groups`).
"""
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from . import ridge
from .core import DEV, r2, spearman, verdict


def _t(x):
    return torch.as_tensor(np.asarray(x, dtype=np.float32), device=DEV)


def group_folds(groups: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    ug = np.unique(groups)
    perm = np.random.RandomState(seed).permutation(len(ug))
    fold_of = {g: i % k for i, g in enumerate(ug[perm])}
    return np.array([fold_of[g] for g in groups], dtype=int)


def cv_probe(get_X: Callable[[int], np.ndarray], layers: Sequence[int], Y: np.ndarray, pool: np.ndarray, test: np.ndarray, groups: np.ndarray, k: int = 5, seed: int = 0) -> List[dict]:
    """-> one dict per target: layer, ratio, cv_r2 (of the chosen cell), curve {layer: best cv_r2 over lambda}, pred (test rows).  Y [n, T] float, NaN allowed."""
    Y = np.asarray(Y, dtype=np.float64)
    T, R = Y.shape[1], len(ridge.RATIOS)
    folds = group_folds(groups[pool], min(k, max(2, len(np.unique(groups[pool])))), seed)
    nf = int(folds.max()) + 1
    sse = np.zeros((len(layers), T, R))
    preds = {}
    tss = np.array([np.nansum((Y[pool, t] - np.nanmean(Y[pool, t])) ** 2) if np.isfinite(Y[pool, t]).sum() > 2 else np.nan for t in range(T)])
    for li, l in enumerate(layers):
        X = np.asarray(get_X(l), dtype=np.float32)
        for f in range(nf):
            tr, va = pool[folds != f], pool[folds == f]
            if len(tr) < 10 or len(va) == 0:
                continue
            fit = ridge.fit_layer(_t(X[tr]), Y[tr])
            for t in range(T):
                pv = fit.predict(_t(X[va]), t).cpu().numpy()
                y = Y[va, t]
                m = ~np.isnan(y)
                if m.any():
                    sse[li, t] += ((pv[m] - y[m, None]) ** 2).sum(0)
        fit = ridge.fit_layer(_t(X[pool]), Y[pool])
        for t in range(T):
            preds[(li, t)] = fit.predict(_t(X[test]), t).cpu().numpy()
    out = []
    for t in range(T):
        cv = 1 - sse[:, t, :] / tss[t] if np.isfinite(tss[t]) and tss[t] > 0 else np.full((len(layers), R), np.nan)
        if np.isnan(cv).all():
            out.append({"layer": layers[0], "ratio": ridge.RATIOS[0], "cv_r2": float("nan"), "curve": {}, "pred": np.full(len(test), np.nan)})
            continue
        li, ri = np.unravel_index(np.nanargmax(cv), cv.shape)
        out.append({"layer": layers[li], "ratio": ridge.RATIOS[ri], "cv_r2": float(cv[li, ri]), "curve": {layers[i]: float(np.nanmax(cv[i])) for i in range(len(layers))}, "pred": preds[(li, t)][:, ri]})
    return out


def boot_delta_cluster(y, pa, pb, groups, B: int = 500, seed: int = 0):
    """paired bootstrap of r2(a) - r2(b), resampling PROBLEMS (groups) with replacement.  -> (delta, lo95, hi95)."""
    y, pa, pb, groups = (np.asarray(x) for x in (y, pa, pb, groups))
    m = ~np.isnan(y)
    y, pa, pb, groups = y[m].astype(float), pa[m].astype(float), pb[m].astype(float), groups[m]
    if len(y) < 10:
        return float("nan"), float("nan"), float("nan")
    ug, inv = np.unique(groups, return_inverse=True)
    rows = [np.where(inv == i)[0] for i in range(len(ug))]
    rng = np.random.RandomState(seed)
    d = np.empty(B)
    for b in range(B):
        ix = np.concatenate([rows[i] for i in rng.randint(0, len(ug), len(ug))])
        d[b] = r2(y[ix], pa[ix]) - r2(y[ix], pb[ix])
    return r2(y, pa) - r2(y, pb), float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))


def compare(get_X, layers, baselines: Dict[str, np.ndarray], Y, names: Sequence[str], pool, test, groups, regime: str, prefix: str = "", B: int = 500, k: int = 5) -> List[dict]:
    """probe (CV layer/lambda) vs every baseline (CV lambda), per target.  Verdicts: against the best OBSERVABLE baseline (names without 'oracle'; chosen on TEST, conservative for the probe)
    and against the best baseline overall."""
    Y = np.asarray(Y, dtype=np.float64)
    pr = cv_probe(get_X, layers, Y, pool, test, groups, k)
    bl = {bn: cv_probe(lambda _l, F=F: F, [0], Y, pool, test, groups, k) for bn, F in baselines.items()}
    rows = []
    for t, name in enumerate(names):
        y = Y[test, t]
        if np.isfinite(y).sum() < 10 or not np.isfinite(pr[t]["pred"]).all():
            continue
        r2b = {bn: r2(y, v[t]["pred"]) for bn, v in bl.items()}
        key = lambda n: -1e9 if np.isnan(r2b[n]) else r2b[n]                    # noqa: E731
        best = max(r2b, key=key)
        avail = [n for n in r2b if "oracle" not in n]
        obs = max(avail, key=key) if avail else best
        p = pr[t]["pred"]
        dA, loA, hiA = boot_delta_cluster(y, p, bl[best][t]["pred"], groups[test], B)
        dO, loO, hiO = boot_delta_cluster(y, p, bl[obs][t]["pred"], groups[test], B)
        rows.append({"prefix": prefix, "target": name, "regime": regime, "n_pool": len(pool), "n_test": len(test), "n_test_groups": len(np.unique(groups[test])),
                     "probe_layer": pr[t]["layer"], "probe_cv_r2": pr[t]["cv_r2"], "probe_test_r2": r2(y, p), "probe_spearman": spearman(y, p),
                     "best_observable": obs, "obs_r2": r2b[obs], "delta_vs_obs": dO, "obs_lo": loO, "obs_hi": hiO,
                     "best_baseline": best, "oracle_r2": r2b[best], "delta_vs_oracle": dA, "oracle_lo": loA, "oracle_hi": hiA,
                     **{f"b[{bn}]": v for bn, v in r2b.items()}, "curve": pr[t]["curve"],
                     "verdict_obs": verdict(dO, loO, r2(y, p)), "verdict_oracle": verdict(dA, loA, r2(y, p))})
    return rows


# ----------------------------------------------------------------------------- splits over rows (many rows per problem allowed)
def split_indices(split: np.ndarray, d1: np.ndarray, regime: str, seed: int = 0):
    """iid: pool = rows of problems with split in (fit, val), test = split == 'test'.  shift: pool = the easiest 35% by design difficulty d1, test = the hardest 35%."""
    ids = np.arange(len(split))
    if regime == "iid":
        return ids[np.isin(split, ["fit", "val"])], ids[split == "test"]
    if regime == "shift":
        if len(np.unique(d1)) < 3:
            return ids[:0], ids[:0]
        lo, hi = np.percentile(d1, 35), np.percentile(d1, 65)
        return ids[d1 <= lo], ids[d1 >= hi] if hi > lo else ids[d1 > lo]
    raise ValueError(regime)


def run_block(frame_split, frame_d1, groups, get_X, layers, baselines, Y, names, regimes=("iid", "shift"), prefix="", log=print, B=500) -> List[dict]:
    rows = []
    for rg in regimes:
        pool, test = split_indices(np.asarray(frame_split), np.asarray(frame_d1), rg)
        if len(pool) < 60 or len(test) < 30:
            log(f"  {prefix} {rg}: skipped (pool {len(pool)}, test {len(test)})")
            continue
        rows += compare(get_X, layers, baselines, Y, names, pool, test, np.asarray(groups), rg, prefix, B)
        log(f"  {prefix} {rg}: done (pool {len(pool)}, test {len(test)})")
    return rows
