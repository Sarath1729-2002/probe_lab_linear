"""Linear probes, layer sweeps and the statistics that go with them.

Protocol for every probe:
  1. Problems are split once into dev (80%) and test (20%). All injections of a problem stay
     on the same side, so no problem is seen at test time.
  2. On dev, grouped k-fold CV picks the layer and the regularisation strength.
  3. The probe is refit on all of dev at that layer and scored once on test.
Reported numbers are test numbers; the per-layer curves are dev cross-validation.
"""
from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ----------------------------------------------------------------------------- metrics

def safe_auc(y, s) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def within_auc(s, y, groups) -> tuple[float, int]:
    """AUROC computed only over (positive, negative) pairs from the SAME problem.
    Anything that is constant within a problem (difficulty, topic, length of the question)
    cannot help here. Returns (auc, number of problems with both classes)."""
    s, y, groups = np.asarray(s, float), np.asarray(y), np.asarray(groups)
    num = den = 0.0
    n_mixed = 0
    for g in np.unique(groups):
        m = groups == g
        pos, neg = s[m & (y == 1)], s[m & (y == 0)]
        if len(pos) and len(neg):
            n_mixed += 1
            diff = pos[:, None] - neg[None, :]
            num += (diff > 0).sum() + 0.5 * (diff == 0).sum()
            den += diff.size
    return (num / den if den else float("nan")), n_mixed


def spearman(y, p) -> float:
    if len(y) < 3 or np.std(y) == 0 or np.std(p) == 0:
        return float("nan")
    return float(spearmanr(y, p).correlation)


def within_spearman(y, p, groups) -> float:
    """Spearman after removing each problem's mean from both target and prediction."""
    y, p, groups = np.asarray(y, float), np.asarray(p, float), np.asarray(groups)
    yd, pd_ = y.copy(), p.copy()
    keep = np.zeros(len(y), bool)
    for g in np.unique(groups):
        m = groups == g
        if m.sum() >= 2:
            yd[m] -= y[m].mean()
            pd_[m] -= p[m].mean()
            keep |= m
    return spearman(yd[keep], pd_[keep])


def cluster_bootstrap(fn, groups, n_boot=1000, seed=0):
    """95% CI of fn(idx) under resampling of whole problems."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rows_of = {g: np.where(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([rows_of[g] for g in pick])
        v = fn(idx)
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return [float("nan"), float("nan")]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


# ----------------------------------------------------------------------------- models

def clf(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))


def reg(alpha):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def split_problems(groups, test_frac, seed):
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    test_g = set(rng.choice(uniq, max(1, int(round(test_frac * len(uniq)))), replace=False))
    test = np.array([g in test_g for g in groups])
    return ~test, test


def _layer_job(acts_path, rows, layer, y, groups, grid, folds, kind):
    X = np.load(acts_path, mmap_mode="r")[rows, layer, :].astype(np.float32)
    nf = min(folds, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=nf)
    best = None
    for h in grid:
        oof = np.full(len(y), np.nan)
        for tr, va in gkf.split(X, y, groups):
            if kind == "clf":
                if len(np.unique(y[tr])) < 2:
                    continue
                m = clf(h).fit(X[tr], y[tr])
                oof[va] = m.decision_function(X[va])
            else:
                m = reg(h).fit(X[tr], y[tr])
                oof[va] = m.predict(X[va])
        ok = ~np.isnan(oof)
        if kind == "clf":
            score = safe_auc(y[ok], oof[ok])
            wscore = within_auc(oof[ok], y[ok], groups[ok])[0]
        else:
            score = spearman(y[ok], oof[ok])
            wscore = within_spearman(y[ok], oof[ok], groups[ok])
        if best is None or (not np.isnan(score) and (np.isnan(best["score"]) or score > best["score"])):
            best = {"layer": layer, "h": h, "score": score, "within": wscore, "oof": oof}
    return best


def sweep_and_test(acts_path, rows, y, groups, dev, test, layers, grid, folds, kind, n_jobs=-1):
    """Layer sweep on dev (grouped CV), refit best layer on dev, score test.

    rows: indices into the activation file for every example (same order as y/groups/dev/test).
    """
    y, groups, rows = np.asarray(y), np.asarray(groups), np.asarray(rows)
    jobs = Parallel(n_jobs=n_jobs)(
        delayed(_layer_job)(acts_path, rows[dev], l, y[dev], groups[dev], grid, folds, kind)
        for l in layers)
    curve = [{"layer": j["layer"], "h": j["h"], "dev_cv": j["score"], "dev_within": j["within"]} for j in jobs]
    valid = [j for j in jobs if not np.isnan(j["score"])]
    if not valid:
        return {"curve": curve, "best": None}
    best = max(valid, key=lambda j: j["score"])
    X = np.load(acts_path, mmap_mode="r")[rows, best["layer"], :].astype(np.float32)
    m = (clf if kind == "clf" else reg)(best["h"]).fit(X[dev], y[dev])
    pred = m.decision_function(X) if kind == "clf" else m.predict(X)
    # OOF predictions on dev at the chosen layer (for stacking with covariates)
    oof_dev = best["oof"]
    out = {"curve": curve, "best_layer": int(best["layer"]), "best_h": best["h"],
           "pred_all": pred, "oof_dev": oof_dev, "model": m}
    return out


def classification_report(scores, y, groups, n_boot, seed=0):
    scores, y, groups = np.asarray(scores), np.asarray(y), np.asarray(groups)
    auc = safe_auc(y, scores)
    w, n_mixed = within_auc(scores, y, groups)
    return {
        "n": int(len(y)), "pos_rate": float(np.mean(y)) if len(y) else float("nan"),
        "auroc": auc,
        "auroc_ci": cluster_bootstrap(lambda i: safe_auc(y[i], scores[i]), groups, n_boot, seed),
        "within_auroc": w, "n_mixed_problems": n_mixed,
        "within_auroc_ci": cluster_bootstrap(lambda i: within_auc(scores[i], y[i], groups[i])[0], groups, n_boot, seed),
    }


def regression_report(pred, y, groups, n_boot, seed=0):
    pred, y, groups = np.asarray(pred), np.asarray(y), np.asarray(groups)
    return {
        "n": int(len(y)),
        "spearman": spearman(y, pred),
        "spearman_ci": cluster_bootstrap(lambda i: spearman(y[i], pred[i]), groups, n_boot, seed),
        "within_spearman": within_spearman(y, pred, groups),
    }


# ----------------------------------------------------------------------------- covariate baselines

def covariate_clf(F, y, dev, test, C=1.0):
    m = clf(C).fit(F[dev], y[dev])
    return m.decision_function(F)


def covariate_reg(F, y, dev, alpha=1.0):
    m = reg(alpha).fit(F[dev], y[dev])
    return m.predict(F)


def oof_covariate_clf(F, y, groups, dev, folds=5, C=1.0):
    """Out-of-fold covariate scores on dev (for a fair stacked model)."""
    idx = np.where(dev)[0]
    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=min(folds, len(np.unique(groups[idx]))))
    for tr, va in gkf.split(idx, y[idx], groups[idx]):
        tr, va = idx[tr], idx[va]
        if len(np.unique(y[tr])) < 2:
            continue
        oof[va] = clf(C).fit(F[tr], y[tr]).decision_function(F[va])
    return oof
