"""Level-1 recreation of "How Much is Left?" (Merzouk et al. 2026) for error propagation.

Target. For an injected wrong value at step k of the model's own chain of thought:
    y = 1 (FOLLOW)  final answer == answer recomputed symbolically with the wrong value
    y = 0 (RECOVER) final answer == gold
OTHER outcomes, ambiguous rows and setup failures are excluded (see stage_fit).

Predictor family (mirrors the paper's §3.3, one row each in Table 2):
    constant baseline   the dev-split FOLLOW rate at every injection            (paper: constant median)
    black-box predictor logistic regression on what an observer can see without the residual
                        stream: error operator, relative error size, position, steps downstream,
                        trace length, dataset, and the model's surprisal at the injected digits
                        (token log-probs)                                        (paper: exact countdown)
    prompt-only probe   linear probe on the LAST PROMPT TOKEN, before any reasoning is written;
                        it can only know how error-prone the problem is      (paper: completion length probe)
    error-token probe   linear probe on the LAST TOKEN OF THE INJECTED NUMBER (paper: remaining count probe)

Controls and secondary rows (paper §4.1-4.2, App. A.12-A.14):
    prompt+structure    prompt-only score + the black-box features: the model's pre-reasoning estimate
                        plus everything known about the error. Closest analogue of the paper's
                        exact countdown (prompt-end estimate + position). error-token > this means
                        the state AT the error adds information (paper's case ii).
    surface (prompt)    TF-IDF + length logistic on the prompt text      (paper: surface-feature ridge)
    surface (context)   same on prompt + the 300 characters before and including the injected number
    shuffled (global / within-problem)   probe trained on permuted labels  (paper: shuffled-label heads)
Metrics: AUROC (headline), Brier and log-loss relative to the constant (paper: MAE / floor),
within-problem AUROC, Cohen's kappa at p = 0.5 (paper: classification ablation).

Implied claims (paper §4.2): error-token > constant -> the state encodes the outcome; > prompt-only
-> the state at the error adds information beyond the pre-reasoning estimate; > black-box -> beyond
what the error itself and the model's surprise reveal.

Every number is a mean over `split_seeds` group-level dev/test splits (paper: 3 seeds).
"""
from __future__ import annotations

import os

DEVICE = os.environ.get("ERRPROP_DEVICE", "cuda")   # set to cpu for local smoke tests

import json
import math

import numpy as np
import pandas as pd

from .analysis import build_table, rounds_to_gold, log
from .io import run_dir, read_jsonl
from .probes import (sweep_and_test, clf, reg, safe_auc, within_auc, cluster_bootstrap, split_problems,
                     spearman)

KINDS = ("plus_one", "digit_swap", "times_ten", "random_scale", "random_same_digits")
PRED = ("constant", "blackbox", "prompt_only", "error_token")
EXTRA = ("stacked", "prompt_struct", "surface_prompt", "surface_context", "shuffle_global", "shuffle_within")


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _feat(t: pd.DataFrame, datasets) -> np.ndarray:
    cols = ["step_frac", "n_downstream", "n_eq", "n_tokens", "log_rel_err", "abs_log_ratio",
            "surprise_corrupt", "d_surprise", "level"] + [f"kind_{k}" for k in KINDS] + [f"ds_{d}" for d in datasets]
    return t[cols].astype(float).fillna(0.0).to_numpy()


# ============================================================================ table

def build(rd) -> tuple[pd.DataFrame, dict]:
    df = build_table(rd)
    cont = read_jsonl(rd / "continuations.jsonl")
    df["operand_reuse"] = [bool(c["greedy"].get("operand_reuse", False)) for c in cont]
    for k in KINDS:
        df[f"kind_{k}"] = (df.kind == k).astype(float)
    datasets = sorted(df.dataset.unique())
    for d in datasets:
        df[f"ds_{d}"] = (df.dataset == d).astype(float)
    tol = df["tol"].fillna(1e-4).to_numpy() if "tol" in df else np.full(len(df), 1e-4)
    amb = df.on_path.to_numpy() & np.array([rounds_to_gold(p, g, t_) for p, g, t_ in
                                            zip(df.propagated_answer, df.gold, tol)])
    df["ambiguous"] = amb
    df["valid"] = (df.clean_cont_correct.astype(bool) & ~df.glued.fillna(False).astype(bool)
                   & ~df.operand_reuse & ~df.ambiguous)
    df["kept"] = df.valid & df.on_path & df.outcome.isin(["FOLLOW", "RECOVER"]) & df.kind.isin(KINDS)
    df["y"] = (df.outcome == "FOLLOW").astype(int)
    injs = read_jsonl(rd / "injections.jsonl")
    df["surf_prompt"] = [i["prompt"] for i in injs]
    df["surf_context"] = [i["prompt"] + " ||| " + i["prefix_corrupt"][-300:] for i in injs]
    df["noise"] = [c.get("noise") for c in cont]
    pids = [r["problem_id"] for r in read_jsonl(rd / "prompt_ids.jsonl")]
    pidx = {p: i for i, p in enumerate(pids)}
    df["prow"] = df.problem_id.map(pidx).fillna(0).astype(int)

    # dataset table (paper Table 7 + outcome rates)
    clean = pd.DataFrame(read_jsonl(rd / "clean.jsonl"))
    dt = {}
    for d in datasets:
        c = clean[clean.dataset == d]
        x = df[df.dataset == d]
        on = x[x.valid & x.on_path & x.kind.isin(KINDS)]
        n_on = max(1, len(on))
        nf = (~c.has_final.astype(bool)).sum() if "has_final" in c else 0
        dt[d] = {"problems": int(len(c)), "accuracy": float(c.correct.mean()) if len(c) else None,
                 "no_final_answer": float(nf / max(1, len(c))),           # paper App. A.7 (truncation)
                 "site": str(x.site.iloc[0]) if "site" in x and len(x) else "equation",
                 "problems_injected": int(x.problem_id.nunique()), "injections": int(len(x)),
                 "kept": int(x.kept.sum()),
                 "follow": float((on.outcome == "FOLLOW").sum() / n_on),
                 "recover": float((on.outcome == "RECOVER").sum() / n_on),
                 "other": float((on.outcome == "OTHER").sum() / n_on)}
    excl = {"injections": int(len(df)), "clean_prefix_failed": int((~df.clean_cont_correct.astype(bool)).sum()),
            "glued": int(df.glued.fillna(False).sum()), "operand_reuse": int(df.operand_reuse.sum()),
            "rounds_to_gold": int(amb.sum()), "kept": int(df.kept.sum())}
    return df, {"datasets": dt, "exclusions": excl, "dataset_list": datasets}


# ============================================================================ one cell

def _surface(text, y, grp, dv, te, folds):
    """TF-IDF (1-2 grams, 5000 terms) + log length features; C chosen by grouped CV on dev."""
    from scipy.sparse import hstack, csr_matrix
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    text = np.asarray(text, dtype=object)
    lens = np.log1p(np.c_[[len(x) for x in text], [len(x.split()) for x in text]].astype(float))
    lens = (lens - lens[dv].mean(0)) / (lens[dv].std(0) + 1e-9)

    def feats(fit_idx):
        v = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
        v.fit(text[fit_idx])
        return hstack([v.transform(text), csr_matrix(lens)]).tocsr()

    idx_dv = np.where(dv)[0]
    best, best_auc = 1.0, -1.0
    gk = GroupKFold(n_splits=min(folds, len(np.unique(grp[dv]))))
    for C in (0.1, 1.0, 10.0):
        oof = np.zeros(len(idx_dv))
        for tr, va in gk.split(idx_dv, y[idx_dv], grp[idx_dv]):
            X = feats(idx_dv[tr])
            if len(np.unique(y[idx_dv[tr]])) < 2:
                continue
            oof[va] = LogisticRegression(C=C, max_iter=3000).fit(X[idx_dv[tr]], y[idx_dv[tr]]).decision_function(X[idx_dv[va]])
        a = safe_auc(y[idx_dv], oof)
        if a == a and a > best_auc:
            best, best_auc = C, a
    X = feats(idx_dv)
    return LogisticRegression(C=best, max_iter=3000).fit(X[dv], y[dv]).decision_function(X)


def _within_perm(y, grp, rng):
    y = y.copy()
    for g in np.unique(grp):
        m = np.where(grp == g)[0]
        y[m] = y[rng.permutation(m)]
    return y


def cell(t: pd.DataFrame, seed: int, pc: dict, layers, paths, nj, datasets, nb=0, keep=False, controls=True):
    """All predictors on table t with one group-level split (paper §3.3 family + App. A.12 controls)."""
    from sklearn.metrics import cohen_kappa_score
    t = t.reset_index(drop=True)
    y = t.y.to_numpy(int)
    grp = t.group.to_numpy()
    pid = t.problem_id.to_numpy()
    dv, te = split_problems(grp, pc["test_frac"], seed)
    out = {"n_dev": int(dv.sum()), "n_test": int(te.sum()), "follow_rate": float(y.mean()) if len(y) else None}
    if len(t) < 60 or len(np.unique(y[dv])) < 2 or min(np.bincount(y[te], minlength=2)) < 5:
        out["skipped"] = "too few rows or fewer than 5 of a class in test"
        return out
    yt, pt_ = y[te], pid[te]
    p0 = float(y[dv].mean())
    b_const = float(np.mean((yt - p0) ** 2))
    ll_const = float(-np.mean(yt * np.log(p0) + (1 - yt) * np.log(1 - p0)))

    def metrics(dec):
        p = np.clip(_sig(dec[te]), 1e-6, 1 - 1e-6)
        b = float(np.mean((yt - p) ** 2))
        ll = float(-np.mean(yt * np.log(p) + (1 - yt) * np.log(1 - p)))
        w, nm = within_auc(dec[te], yt, pt_)
        return {"auroc": safe_auc(yt, dec[te]), "within": w, "n_mixed": nm, "brier": b, "rel_brier": b / b_const,
                "rel_logloss": ll / ll_const, "kappa": float(cohen_kappa_score(yt, (p > 0.5).astype(int)))}

    res = {"constant": {"auroc": 0.5, "within": 0.5, "brier": b_const, "rel_brier": 1.0, "rel_logloss": 1.0,
                        "kappa": 0.0}}
    F = _feat(t, datasets)
    bb = clf(1.0).fit(F[dv], y[dv])
    dec_bb = bb.decision_function(F)
    res["blackbox"] = metrics(dec_bb)
    po = sweep_and_test(paths["prompt"], t.prow.to_numpy(), y, grp, dv, te, layers, pc["C_grid"], pc["folds"], "clf", nj)
    et = sweep_and_test(paths["corrupt"], t.row.to_numpy(), y, grp, dv, te, layers, pc["C_grid"], pc["folds"], "clf", nj)
    if "pred_all" not in po or "pred_all" not in et:
        out["skipped"] = "layer sweep failed"
        return out
    res["prompt_only"] = metrics(po["pred_all"]) | {"layer": po["best_layer"]}
    res["error_token"] = metrics(et["pred_all"]) | {"layer": et["best_layer"], "C": et["best_h"]}
    z = et["pred_all"].copy()
    z[dv] = np.nan_to_num(et["oof_dev"])          # dev rows get out-of-fold probe scores
    stk = clf(1.0).fit(np.c_[F, z][dv], y[dv]).decision_function(np.c_[F, z])
    res["stacked"] = metrics(stk)
    zp = po["pred_all"].copy()
    zp[dv] = np.nan_to_num(po["oof_dev"])
    dec_ps = clf(1.0).fit(np.c_[F, zp][dv], y[dv]).decision_function(np.c_[F, zp])
    res["prompt_struct"] = metrics(dec_ps)
    if controls:
        res["surface_prompt"] = metrics(_surface(t.surf_prompt.to_numpy(), y, grp, dv, te, pc["folds"]))
        res["surface_context"] = metrics(_surface(t.surf_context.to_numpy(), y, grp, dv, te, pc["folds"]))
        X = np.load(paths["corrupt"], mmap_mode="r")[t.row.to_numpy(), et["best_layer"], :].astype(np.float32)
        rng = np.random.default_rng(seed + 77)
        for name, yy in (("shuffle_global", rng.permutation(y[dv])),
                         ("shuffle_within", _within_perm(y[dv], grp[dv], rng))):
            if len(np.unique(yy)) == 2:
                res[name] = metrics(clf(et["best_h"]).fit(X[dv], yy).decision_function(X))
    out.update(res)
    out["delta_over_blackbox"] = res["stacked"]["auroc"] - res["blackbox"]["auroc"]
    out["dv"], out["te"] = dv, te
    if nb:
        out["ci"] = {
            "error_token_auroc": cluster_bootstrap(lambda i: safe_auc(yt[i], et["pred_all"][te][i]), pt_, nb, seed),
            "delta_over_blackbox": cluster_bootstrap(
                lambda i: safe_auc(yt[i], stk[te][i]) - safe_auc(yt[i], dec_bb[te][i]), pt_, nb, seed),
            "error_token_minus_prompt_struct": cluster_bootstrap(
                lambda i: safe_auc(yt[i], et["pred_all"][te][i]) - safe_auc(yt[i], dec_ps[te][i]), pt_, nb, seed)}
    if keep:
        out["_et"], out["_po"], out["_bb_dec"], out["_po_dec"], out["_et_dec"] = et, po, dec_bb, po["pred_all"], et["pred_all"]
        out["_et_oof"], out["_po_oof"], out["_ps_dec"] = z, zp, dec_ps
        out["_p0"] = p0
    return out


def _agg(runs: list[dict]) -> dict:
    """Mean and sd over splits for every predictor metric; skipped runs are ignored."""
    ok = [r for r in runs if "skipped" not in r]
    out = {"n_splits": len(ok), "n_dev": [r["n_dev"] for r in runs], "n_test": [r["n_test"] for r in runs],
           "follow_rate": runs[0]["follow_rate"] if runs else None}
    if not ok:
        out["skipped"] = runs[0].get("skipped") if runs else "no data"
        return out
    for p in PRED + EXTRA:
        out[p] = {}
        for m in ("auroc", "within", "rel_brier", "rel_logloss", "kappa", "n_mixed", "layer"):
            vals = [r[p][m] for r in ok if p in r and m in r[p] and r[p][m] is not None and r[p][m] == r[p][m]]
            if vals:
                out[p][m] = [float(np.mean(vals)), float(np.std(vals))]
    d = [r["delta_over_blackbox"] for r in ok]
    out["delta_over_blackbox"] = [float(np.mean(d)), float(np.std(d))]
    return out


# ============================================================================ fit stage

def stage_fit(cfg: dict) -> None:
    rd = run_dir(cfg)
    pc = cfg["probes"]
    nj = pc["n_jobs"]
    seeds = pc["split_seeds"]
    df, meta = build(rd)
    datasets = meta["dataset_list"]
    paths = {"corrupt": str(rd / "acts_corrupt.npy"), "prompt": str(rd / "acts_prompt.npy")}
    shape = np.load(paths["corrupt"], mmap_mode="r").shape
    L = shape[1]
    layers = sorted(set(range(0, L, pc["layer_step"])) | {L - 1})
    s = df[df.kept].reset_index(drop=True)
    log(f"{cfg['model']}: {len(s)} kept injections, FOLLOW rate {s.y.mean():.3f}, layers {len(layers)}/{L}")
    R = {"model": cfg["model"], "tag": cfg.get("model_tag", cfg["run_name"]), "n_layers": int(L),
         "d_model": int(shape[2]), "split_seeds": seeds, **meta, "cells": {}}

    # ---------------------------------------------------------------- pooled + per dataset
    pooled_runs = []
    for k, sd in enumerate(seeds):
        log(f"pooled, split {sd}")
        pooled_runs.append(cell(s, sd, pc, layers, paths, nj, datasets, nb=pc["n_boot"] if k == 0 else 0, keep=k == 0))
    R["cells"]["pooled"] = _agg(pooled_runs)
    per_ds_runs = {}
    for d in datasets:
        t = s[s.dataset == d]
        per_ds_runs[d] = []
        for sd in seeds:
            log(f"{d}, split {sd}")
            per_ds_runs[d].append(cell(t, sd, pc, layers, paths, nj, datasets, keep=True))
        R["cells"][d] = _agg(per_ds_runs[d])

    p0 = pooled_runs[0]
    if "skipped" in p0:
        _write(R, rd)
        return
    best = p0["error_token"]["layer"]
    R["pooled_seed0"] = {"error_token_layer": best, "prompt_only_layer": p0["prompt_only"]["layer"],
                         "ci": p0.get("ci"), "error_token_auroc": p0["error_token"]["auroc"],
                         "delta_over_blackbox": p0["delta_over_blackbox"],
                         "curves": {"error_token": p0["_et"]["curve"], "prompt_only": p0["_po"]["curve"]}}

    # ---------------------------------------------------------------- permutation null (pooled, seed 0)
    log("permutation null")
    from joblib import Parallel, delayed
    from .analysis import _perm_one
    y = s.y.to_numpy(int)
    X = np.load(paths["corrupt"], mmap_mode="r")[s.row.to_numpy(), best, :].astype(np.float32)
    dv, te = p0["dv"], p0["te"]
    null = Parallel(n_jobs=nj)(delayed(_perm_one)(1000 + k, X[dv], y[dv], X[te], y[te], p0["error_token"]["C"])
                               for k in range(pc["n_perm"]))
    obs = p0["error_token"]["auroc"]
    R["pooled_seed0"]["permutation"] = {"n": len(null), "mean": float(np.mean(null)), "max": float(np.max(null)),
                                        "p": float((1 + sum(v >= obs for v in null)) / (1 + len(null)))}

    # ---------------------------------------------------------------- sampling-noise floor (paper App. A.13)
    nz = [(i, [1 if o == "FOLLOW" else 0 for o in lab if o in ("FOLLOW", "RECOVER")])
          for i, lab in enumerate(s.noise) if lab]
    nz = [(i, v) for i, v in nz if len(v) >= 2]
    if len(nz) >= 30:
        pe, pp = _sig(p0["_et_oof"]), _sig(p0["_po_oof"])     # out-of-fold on dev, held-out on test
        cst, prb, prm, flo, agree = [], [], [], [], []
        for i, v in nz:
            v = np.array(v, float)
            kk = len(v)
            for j in range(kk):
                loo = np.delete(v, j).mean()
                # E[(y - LOO mean)^2] = p(1-p) * k/(k-1); rescale so the floor estimates p(1-p) itself
                flo.append((v[j] - loo) ** 2 * (kk - 1) / kk)
                cst.append((v[j] - p0["_p0"]) ** 2)
                prb.append((v[j] - pe[i]) ** 2)
                prm.append((v[j] - pp[i]) ** 2)
            agree.append(float(np.mean(v == s.y.iloc[i])))
        c_, f_ = float(np.mean(cst)), float(np.mean(flo))
        cap = lambda b: float((c_ - b) / (c_ - f_)) if c_ > f_ else None   # noqa: E731
        R["noise_floor"] = {"n_injections": len(nz), "n_samples": int(len(flo)),
                            "brier_constant": c_, "brier_loo_floor": f_,
                            "brier_error_token": float(np.mean(prb)), "brier_prompt_only": float(np.mean(prm)),
                            "captured_error_token": cap(float(np.mean(prb))), "captured_prompt_only": cap(float(np.mean(prm))),
                            "greedy_agreement": float(np.mean(agree)),
                            "frac_mixed": float(np.mean([0 < np.mean(v) < 1 for _, v in nz]))}

    # save the probe for the trajectory stage
    m = p0["_et"]["model"]
    sc, lr = m.named_steps["standardscaler"], m.named_steps["logisticregression"]
    np.savez(rd / "probe_error_token.npz", layer=best, mean=sc.mean_, scale=sc.scale_,
             coef=lr.coef_[0], intercept=lr.intercept_[0])

    # per-injection scores of every predictor (seed 0, pooled) for the worked example
    sc_tab = s[["inj_id", "problem_id", "dataset", "kind", "true_text", "new_text", "gold",
                "propagated_answer", "outcome", "y"]].copy()
    sc_tab["split"] = np.where(te, "test", "dev")
    sc_tab["p_constant"] = p0["_p0"]
    sc_tab["p_blackbox"] = _sig(p0["_bb_dec"])
    sc_tab["p_prompt_only"] = _sig(p0["_po_dec"])
    sc_tab["p_error_token"] = _sig(p0["_et_dec"])
    sc_tab.to_csv(rd / "scores_pooled.csv", index=False)

    # ---------------------------------------------------------------- robustness (pooled)
    win = sorted({l for l in range(best - 2, best + 3) if 0 <= l < L})
    rob = {"by_operator": {}}
    for k in KINDS:
        t = s[s.kind == k]
        log(f"operator {k}: {len(t)} rows")
        rob["by_operator"][k] = _agg([cell(t, sd, pc, win, paths, nj, datasets, controls=False) for sd in seeds])
    cat = s[s.rel_err >= 0.5]
    log(f"catchable subset: {len(cat)} rows")
    rob["catchable"] = _agg([cell(cat, sd, pc, win, paths, nj, datasets, controls=False) for sd in seeds])
    R["robustness"] = rob

    # ---------------------------------------------------------------- cross-dataset transfer (paper §5)
    log("cross-dataset transfer")
    A = np.load(paths["corrupt"], mmap_mode="r")
    mats = {"probe": [], "blackbox": [], "probe_rel": [], "floor": []}
    for k, sd in enumerate(seeds):
        P = np.full((len(datasets), len(datasets)), np.nan)
        B = np.full((len(datasets), len(datasets)), np.nan)
        Q = np.full((len(datasets), len(datasets)), np.nan)      # Brier / eval dataset's own floor
        FL = np.full(len(datasets), np.nan)
        for i, a in enumerate(datasets):
            ra = per_ds_runs[a][k]
            if "skipped" in ra:
                continue
            ta = s[s.dataset == a].reset_index(drop=True)
            la, ca = ra["error_token"]["layer"], ra["error_token"]["C"]
            Xa = A[ta.row.to_numpy()[ra["dv"]], la, :].astype(np.float32)
            ya = ta.y.to_numpy(int)[ra["dv"]]
            mp = clf(ca).fit(Xa, ya)
            mb = clf(1.0).fit(_feat(ta, datasets)[ra["dv"]], ya)
            for j, b in enumerate(datasets):
                rb = per_ds_runs[b][k]
                if "skipped" in rb:
                    continue
                tb = s[s.dataset == b].reset_index(drop=True)
                rows = tb.row.to_numpy()[rb["te"]]
                yb = tb.y.to_numpy(int)[rb["te"]]
                dec = mp.decision_function(A[rows, la, :].astype(np.float32))
                P[i, j] = safe_auc(yb, dec)
                B[i, j] = safe_auc(yb, mb.decision_function(_feat(tb, datasets)[rb["te"]]))
                floor = float(np.mean((yb - yb.mean()) ** 2))      # eval split's own constant (paper: eval-median floor)
                FL[j] = floor
                Q[i, j] = float(np.mean((yb - _sig(dec)) ** 2)) / max(floor, 1e-9)
        mats["probe"].append(P)
        mats["blackbox"].append(B)
        mats["probe_rel"].append(Q)
        mats["floor"].append(FL)
    syn = set(cfg["data"].get("synthetic", ["chains", "bbh_arith"]))
    Qm, Pm = np.nanmean(mats["probe_rel"], 0), np.nanmean(mats["probe"], 0)
    asym = {}
    for crit, M, good in (("rel_brier_below_1", Qm, lambda v: v < 1), ("auroc_above_0.5", Pm, lambda v: v > 0.5)):
        c = {"synthetic": [0, 0], "natural": [0, 0]}
        for i, a in enumerate(datasets):
            for j in range(len(datasets)):
                if i == j or M[i, j] != M[i, j]:
                    continue
                key = "synthetic" if a in syn else "natural"
                c[key][1] += 1
                c[key][0] += int(good(M[i, j]))
        asym[crit] = c
    R["transfer"] = {"datasets": datasets, "synthetic": sorted(syn & set(datasets)),
                     "probe": Pm.tolist(), "probe_sd": np.nanstd(mats["probe"], 0).tolist(),
                     "blackbox": np.nanmean(mats["blackbox"], 0).tolist(),
                     "probe_rel_brier": Qm.tolist(), "probe_rel_brier_sd": np.nanstd(mats["probe_rel"], 0).tolist(),
                     "floor_brier": np.nanmean(mats["floor"], 0).tolist(), "asymmetry": asym}

    # ---------------------------------------------------------------- secondary target: contamination horizon
    h = df[df.valid & df.on_path & df.kind.isin(KINDS)].reset_index(drop=True)
    yh = np.log1p(h.horizon.fillna(0).to_numpy(float))
    if len(h) > 100 and yh.std() > 0:
        log("contamination horizon (ridge)")
        dvh, teh = split_problems(h.group.to_numpy(), pc["test_frac"], seeds[0])
        r = sweep_and_test(paths["corrupt"], h.row.to_numpy(), yh, h.group.to_numpy(), dvh, teh, layers,
                           pc["alpha_grid"], pc["folds"], "reg", nj)
        if "best_layer" in r:
            Fh = _feat(h, datasets)
            bbh = reg(1.0).fit(Fh[dvh], yh[dvh]).predict(Fh)
            raw = h.horizon.fillna(0).to_numpy(float)
            med = float(np.median(raw[teh]))                       # paper: median of the eval split
            mae = lambda pr: float(np.mean(np.abs(np.clip(np.expm1(pr[teh]), 0, None) - raw[teh])))  # noqa: E731
            R["horizon"] = {"n": int(len(h)), "layer": r["best_layer"], "mean_horizon": float(h.horizon.mean()),
                            "probe_spearman": spearman(yh[teh], r["pred_all"][teh]),
                            "blackbox_spearman": spearman(yh[teh], bbh[teh]),
                            "mae_constant_median": float(np.mean(np.abs(raw[teh] - med))),
                            "mae_blackbox": mae(bbh), "mae_probe": mae(r["pred_all"])}
    _write(R, rd)
    log("fit done")


def _write(R, rd):
    from .analysis import _jsonable
    with open(rd / "results_paper.json", "w") as f:
        json.dump(R, f, indent=2, default=_jsonable)


# ============================================================================ trajectory stage

def _event_stats(ev, ctrls, items, n_flip, seed):
    """Paper §4.3 / App. A.16 statistics for one pre-specified event type.
    ev: list of (item_index, delta); ctrls: list of arrays of matched control deltas (one per event)."""
    if len(ev) < 3:
        return {"n_events": len(ev)}
    d_ev = np.array([d for _, d in ev])
    d_ct = np.concatenate(ctrls)
    sd = np.sqrt((d_ev.var(ddof=1) * (len(d_ev) - 1) + d_ct.var(ddof=1) * (len(d_ct) - 1)) / (len(d_ev) + len(d_ct) - 2))
    paired = d_ev - np.array([c.mean() for c in ctrls])
    # cluster-safe sign flip over per-continuation paired differences
    per_item = {}
    for (it, _), pd_ in zip(ev, paired):
        per_item.setdefault(it, []).append(pd_)
    m = np.array([np.mean(v) for v in per_item.values()])
    rng = np.random.default_rng(seed)
    flips = np.abs((rng.choice([-1.0, 1.0], size=(n_flip, len(m))) * m).mean(1))
    p_flip = float((1 + np.sum(flips >= abs(m.mean()))) / (1 + n_flip))
    # stratified label shuffle: each event competes with its own matched controls
    obs = paired.mean()
    n_sh = 2000
    acc = np.zeros(n_sh)
    for d, c in zip(d_ev, ctrls):
        pool = np.r_[d, c]
        pick = pool[rng.integers(len(pool), size=n_sh)]
        acc += pick - (pool.sum() - pick) / (len(pool) - 1)
    null = acc / len(d_ev)
    p_shuf = float((1 + np.sum(np.abs(null) >= abs(obs))) / (1 + len(null)))
    return {"n_events": int(len(ev)), "n_continuations": int(len(per_item)), "n_controls": int(len(d_ct)),
            "event_mean": float(d_ev.mean()), "event_median": float(np.median(d_ev)),
            "control_mean": float(d_ct.mean()), "control_median": float(np.median(d_ct)),
            "event_frac_down": float(np.mean(d_ev < 0)), "control_frac_down": float(np.mean(d_ct < 0)),
            "cohens_d": float((d_ev.mean() - d_ct.mean()) / max(sd, 1e-9)),
            "p_signflip": p_flip, "p_label_shuffle": p_shuf,
            "hist_event": d_ev.tolist(), "hist_control": rng.choice(d_ct, min(len(d_ct), 5000), replace=False).tolist()}


def stage_trajectory(cfg: dict) -> None:
    """Read the trained error-token probe at every token of the continuation (paper §4.3 analogue).

    The probe was trained at one position (the injected number), so readings elsewhere are
    off-distribution. Like the paper we therefore (1) show curves/examples as qualitative evidence and
    (2) put the evidential weight on an AGGREGATE, PRE-SPECIFIED event test with matched controls:
      - trigger token "Wait" (the paper's retraction trigger), and a secondary correction-word set;
      - the first token where the model re-writes the TRUE value (RECOVER) or re-uses the INJECTED
        value (FOLLOW).
    For each event: one-step change of the probe logit at that token vs controls drawn from other
    continuations in the same dataset and 50-token position bucket, excluding +-10 tokens around any
    trigger; Cohen's d; cluster-safe sign-flip test over per-continuation paired differences;
    stratified label shuffle. A discovery scan ranking every token type by mean change is run on a
    disjoint half of the continuations, as in the paper's App. A.16.
    Prediction: the FOLLOW-probability falls at retraction tokens (the model is about to repair)."""
    import torch
    import zlib
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .arith import NUM_RE, close, to_float
    import re

    rd = run_dir(cfg)
    tc = cfg["trajectory"]
    W = np.load(rd / "probe_error_token.npz")
    L = int(W["layer"])
    sc = pd.read_csv(rd / "scores_pooled.csv")
    pool = sc[sc.split == "test"]
    n_each = tc["n_items"] // 2
    pick = pd.concat([g.sample(n=min(n_each, len(g)), random_state=cfg["seed"]) for _, g in pool.groupby("outcome")])
    injs = {i["inj_id"]: i for i in read_jsonl(rd / "injections.jsonl")}
    conts = {c["inj_id"]: c for c in read_jsonl(rd / "continuations.jsonl")}

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.bfloat16, device_map=DEVICE).eval()
    mean = torch.tensor(W["mean"], device=DEVICE, dtype=torch.float32)
    scale = torch.tensor(W["scale"], device=DEVICE, dtype=torch.float32)
    coef, b = torch.tensor(W["coef"], device=DEVICE, dtype=torch.float32), float(W["intercept"])
    before, after, half = tc["before"], tc["after"], tc["align_half"]
    max_tok = tc.get("max_tokens", 1536)
    trig = tc.get("trigger", "Wait")
    secondary = {"wait", "actually", "hmm", "oops", "mistake", "incorrect", "recalculate", "recheck", "correction"}

    def enc(x):
        return tok(x, add_special_tokens=False)["input_ids"]

    @torch.no_grad()
    def read(prefix: str, cont: str, upto: int):
        a, c = enc(prefix), enc(cont)
        ids = (a + c)[: len(a) + upto]
        o = model(torch.tensor([ids], device=DEVICE), output_hidden_states=True)
        h = o.hidden_states[L][0].float()
        z = (((h - mean) / scale) @ coef + b).cpu().numpy()
        return z, len(a) - 1, ids

    def event_tok(cont, value):
        for m in re.finditer(NUM_RE, cont):
            if close(to_float(m.group(0)), value):
                return len(enc(cont[: m.start()]))
        return None

    rel = list(range(-before, after + 1))
    curves = {"FOLLOW": [], "RECOVER": [], "CLEAN": []}
    aligned = {"RECOVER": [], "FOLLOW": []}
    examples, items = [], []
    for n, (_, r) in enumerate(pick.iterrows()):
        inj, c = injs[r.inj_id], conts[r.inj_id]
        z, k, ids = read(inj["prompt"] + inj["prefix_corrupt"], c["greedy_text"], max_tok)
        p = _sig(z)
        curves[r.outcome].append([p[k + d] if 0 <= k + d < len(p) else np.nan for d in rel])
        ev = event_tok(c["greedy_text"], inj["true_value"] if r.outcome == "RECOVER" else inj["new_value"])
        ev_pos = None
        if ev is not None and k + 1 + ev < len(p):
            ev_pos = k + 1 + ev
            aligned[r.outcome].append([p[ev_pos + d] if 0 <= ev_pos + d < len(p) else np.nan for d in range(-half, half + 1)])
        toks = [tok.decode([t]) for t in ids[k:]]          # position 0 = injected number's last token
        items.append({"inj_id": r.inj_id, "outcome": r.outcome, "dataset": r.dataset, "z": z[k:], "toks": toks,
                      "value_event": None if ev_pos is None else ev_pos - k,
                      "half": zlib.crc32(r.inj_id.encode()) % 2})
        if len(curves["CLEAN"]) < len(pick):
            zc, kc, _ = read(inj["prompt"] + inj["prefix_clean"], c["clean_text"], after + 1)
            pc_ = _sig(zc)
            curves["CLEAN"].append([pc_[kc + d] if 0 <= kc + d < len(pc_) else np.nan for d in rel])
        if sum(e_["outcome"] == r.outcome for e_ in examples) < tc["n_examples"]:
            lo, hi = max(0, k - before), min(len(ids), k + 24)
            examples.append({"inj_id": r.inj_id, "outcome": r.outcome, "kind": inj["kind"],
                             "true": inj["true_text"], "new": inj["new_text"], "inj_index": k - lo,
                             "tokens": [tok.decode([t]) for t in ids[lo:hi]],
                             "p_follow": [float(x) for x in p[lo:hi]]})
        if n % 100 == 0:
            log(f"trajectory {n}/{len(pick)}")

    # ------------------------------------------------------------ aggregate event analysis
    win, bucket, nc = tc.get("control_window", 10), tc.get("position_bucket", 50), tc.get("n_controls", 100)
    is_trig = lambda s_: s_.strip() == trig                                     # noqa: E731
    is_sec = lambda s_: s_.strip().lower().strip(",.:;!") in secondary          # noqa: E731
    ctrl_pool = {}
    for ii, it in enumerate(items):
        dz = np.diff(it["z"])                       # dz[t-1] = z_t - z_{t-1}, t >= 1 (continuation tokens)
        it["dz"] = dz
        bad = np.zeros(len(it["toks"]), bool)
        for t, s_ in enumerate(it["toks"]):
            if t >= 1 and (is_trig(s_) or is_sec(s_)):
                bad[max(0, t - win): t + win + 1] = True
        if it["value_event"] is not None:
            v = it["value_event"]
            bad[max(0, v - win): v + win + 1] = True
        for t in range(1, len(it["toks"])):
            if not bad[t]:
                ctrl_pool.setdefault((it["dataset"], (t - 1) // bucket), []).append((ii, dz[t - 1]))
    rng = np.random.default_rng(cfg["seed"])

    def controls_for(ii, t):
        cand = [d for j, d in ctrl_pool.get((items[ii]["dataset"], (t - 1) // bucket), []) if j != ii]
        if not cand:
            return None
        return np.asarray(cand)[rng.integers(len(cand), size=nc)]

    def collect(select, which_half=None):
        ev, cts, ex = [], [], []
        for ii, it in enumerate(items):
            if which_half is not None and it["half"] != which_half:
                continue
            for t in select(it):
                if 1 <= t < len(it["toks"]):
                    c_ = controls_for(ii, t)
                    if c_ is None:
                        continue
                    ev.append((ii, float(it["dz"][t - 1])))
                    cts.append(c_)
                    if len(ex) < 3:
                        lo, hi = max(0, t - 12), min(len(it["toks"]), t + 13)
                        ex.append({"inj_id": it["inj_id"], "outcome": it["outcome"], "event_index": t - lo,
                                   "tokens": it["toks"][lo:hi], "p_follow": _sig(it["z"][lo:hi]).tolist()})
        return ev, cts, ex

    nf = tc.get("n_signflip", 10000)
    events = {}
    specs = {
        f"trigger_{trig}": lambda it: [t for t, s_ in enumerate(it["toks"]) if t >= 1 and is_trig(s_)],
        "correction_words": lambda it: [t for t, s_ in enumerate(it["toks"]) if t >= 1 and is_sec(s_)],
        "true_value_rewritten (RECOVER)": lambda it: [it["value_event"]] if it["outcome"] == "RECOVER" and it["value_event"] else [],
        "injected_value_reused (FOLLOW)": lambda it: [it["value_event"]] if it["outcome"] == "FOLLOW" and it["value_event"] else [],
    }
    for name, sel in specs.items():
        ev, cts, ex = collect(sel)
        events[name] = _event_stats(ev, cts, items, nf, cfg["seed"]) | {"examples": ex}
        if name.startswith("trigger"):
            for o in ("FOLLOW", "RECOVER"):
                sub = [(e, c_) for e, c_ in zip(ev, cts) if items[e[0]]["outcome"] == o]
                events[name][f"by_outcome_{o}"] = {k_: v for k_, v in _event_stats(
                    [e for e, _ in sub], [c_ for _, c_ in sub], items, nf, cfg["seed"]).items() if not k_.startswith("hist")}
            evc, ctc, _ = collect(sel, which_half=1)
            events[name]["confirmation_half"] = {k_: v for k_, v in _event_stats(evc, ctc, items, nf, cfg["seed"]).items()
                                                 if not k_.startswith("hist")}
    # discovery scan on the other half: mean one-step change by token type
    by_type = {}
    for it in items:
        if it["half"] != 0:
            continue
        for t in range(1, len(it["toks"])):
            by_type.setdefault(it["toks"][t].strip(), []).append(float(it["dz"][t - 1]))
    drift = float(np.mean([d for v in by_type.values() for d in v])) if by_type else None
    ranked = sorted(((k_, float(np.mean(v)), len(v)) for k_, v in by_type.items() if len(v) >= 10 and k_), key=lambda x: x[1])
    rank_of = next((i + 1 for i, x in enumerate(ranked) if x[0] == trig), None)
    discovery = {"n_token_types": len(ranked), "global_drift": drift, "most_negative": ranked[:12],
                 "most_positive": ranked[-12:][::-1], "trigger_rank_most_negative": rank_of}

    def summarise(rows):
        a = np.array(rows, dtype=float)
        return {"n": int(len(a)), "mean": np.nanmean(a, 0).tolist() if len(a) else [],
                "se": (np.nanstd(a, 0) / np.sqrt(np.maximum(1, np.sum(~np.isnan(a), 0)))).tolist() if len(a) else []}

    out = {"layer": L, "rel": rel, "curves": {k_: summarise(v) for k_, v in curves.items()},
           "aligned_rel": list(range(-half, half + 1)),
           "aligned": {"RECOVER_at_true_value_rewritten": summarise(aligned["RECOVER"]),
                       "FOLLOW_at_injected_value_reused": summarise(aligned["FOLLOW"])},
           "examples": examples, "events": events, "discovery": discovery,
           "n_items": len(items), "trigger": trig,
           "n_items_with_trigger": int(sum(any(is_trig(s_) for s_ in it["toks"][1:]) for it in items))}
    from .analysis import _jsonable
    with open(rd / "trajectories.json", "w") as f:
        json.dump(out, f, indent=1, default=_jsonable)
    log(f"trajectory done: {out['n_items_with_trigger']}/{len(items)} continuations contain '{trig}'")
