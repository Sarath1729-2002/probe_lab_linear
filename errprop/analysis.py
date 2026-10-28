"""Stage `probes`: assemble the table, run every probe, control and the intervention simulation,
write results.json (+ csv tables and the steering direction).

Layout of the analysis (all numbers on held-out problems unless marked 'dev'):
  P0   corruption detection (sanity)
  P1   FOLLOW vs RECOVER at the injection token
         P1_all        every valid on-path injection
         P1_catchable  pre-registered headline: relative error >= 50% (errors the model sometimes catches)
         robustness    by operator / dataset / error size, 5 group-level splits, 50-shuffle null
  P1r  follow rate over greedy + samples (regression)
  P2   value tracking (does the residual stream still carry the true digit?)
  P3   silent vs visible recovery
  P4   contamination horizon
  P5   prompt-time susceptibility
  W    probe-gated "Wait" intervention, simulated offline from the two continuation arms
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd

from .io import run_dir, read_jsonl
from .probes import (sweep_and_test, classification_report, regression_report, covariate_clf,
                     covariate_reg, split_problems, safe_auc, within_auc, cluster_bootstrap, clf)

_T0 = time.time()


def log(msg):
    print(f"[probes {time.time() - _T0:6.0f}s] {msg}", flush=True)


KINDS = ("plus_one", "digit_swap", "times_ten", "random_scale", "random_same_digits", "copy_operand")
DATASETS = ("gsm8k", "gsmsym_p1", "gsmsym_p2", "math", "chains", "bbh_arith")
DIFFICULTY = ["pass_rate", "level"] + [f"ds_{d}" for d in DATASETS]
STRUCTURE = ["step_frac", "n_downstream", "n_eq", "n_tokens", "log_rel_err", "abs_log_ratio"] + \
            [f"kind_{k}" for k in KINDS]
SURPRISE = ["surprise_corrupt", "d_surprise"]
ALL_COV = DIFFICULTY + STRUCTURE + SURPRISE
CATCHABLE_MIN_REL_ERR = 0.5           # pre-registered (DIRECTION.md)
ERR_BINS = [(-99, -1.0, "<10%"), (-1.0, math.log10(0.5), "10-50%"),
            (math.log10(0.5), math.log10(2.0), "50%-2x"), (math.log10(2.0), 99, ">2x")]


# ============================================================================ table

def build_table(rd) -> pd.DataFrame:
    inj = pd.DataFrame(read_jsonl(rd / "injections.jsonl"))
    cont = read_jsonl(rd / "continuations.jsonl")
    meta = pd.DataFrame(read_jsonl(rd / "extract_meta.jsonl"))
    assert list(inj.inj_id) == list(meta.inj_id) == [c["inj_id"] for c in cont], "stage files out of order"
    df = inj.drop(columns=["prompt", "prefix_corrupt", "prefix_clean"]).copy()
    if "group" not in df:
        df["group"] = df.problem_id
    df["row"] = np.arange(len(df))
    for k in ("outcome", "horizon", "uptake", "marker", "recovery_type", "true_restated_char", "n_chars", "glued"):
        df[k] = [c["greedy"].get(k) for c in cont]
    df["follow_rate"] = [c["follow_rate"] for c in cont]
    df["clean_cont_correct"] = [c["clean_cont_correct"] for c in cont]
    df["wait_outcome"] = [c.get("wait", {}).get("outcome") for c in cont]
    df["wait_clean_correct"] = [c.get("wait_clean_correct") for c in cont]
    df = df.join(meta.drop(columns=["inj_id"]))
    df["d_surprise"] = df.surprise_corrupt - df.surprise_clean
    for d in DATASETS:
        df[f"ds_{d}"] = (df.dataset == d).astype(float)
    for k in KINDS:
        df[f"kind_{k}"] = (df.kind == k).astype(float)
    ratio = (df.new_value.abs() + 1e-9) / (df.true_value.abs() + 1e-9)
    df["abs_log_ratio"] = np.abs(np.log10(ratio))
    df["rel_err"] = (df.new_value - df.true_value).abs() / df.true_value.abs().clip(lower=1e-9)
    df["err_bin"] = pd.cut(df.log_rel_err, [b[0] for b in ERR_BINS] + [ERR_BINS[-1][1]],
                           labels=[b[2] for b in ERR_BINS]).astype(str)
    df["catchable"] = df.rel_err >= CATCHABLE_MIN_REL_ERR
    df["true_last"] = [int(abs(round(v))) % 10 for v in df.true_value]
    df["new_last"] = [int(abs(round(v))) % 10 for v in df.new_value]
    df["is_int"] = [abs(v - round(v)) < 1e-9 and abs(n - round(n)) < 1e-9 for v, n in zip(df.true_value, df.new_value)]
    df["correct_no_int"] = df.outcome.isin(["RECOVER", "NA_RECOVER"])
    df["correct_wait"] = df.wait_outcome.isin(["RECOVER", "NA_RECOVER"])
    return df


def rounds_to_gold(prop: float, gold: float) -> bool:
    """True when the propagated answer is indistinguishable from gold after ordinary rounding."""
    if prop is None or gold is None:
        return False
    if abs(gold - round(gold)) < 1e-9:
        return abs(prop - gold) < 0.5
    return abs(prop - gold) <= 0.01 * max(1.0, abs(gold))


def _f(df, cols):
    cols = [c for c in cols if c in df]
    return df[cols].astype(float).fillna(0.0).to_numpy()


def _window(best, n_layers, half=3):
    return list(range(max(0, best - half), min(n_layers, best + half + 1)))


# ============================================================================ P1 helpers

def _p1_core(s, path_c, layers, pc, seed, nb, nj, split):
    """FOLLOW vs RECOVER: probe, all-covariate baseline, gain of probe over covariates."""
    y = (s.outcome == "FOLLOW").astype(int).to_numpy()
    g = s.problem_id.to_numpy()
    dv, te = split == "dev", split == "test"
    out = {"n_dev": int(dv.sum()), "n_test": int(te.sum()),
           "follow_rate": float(y.mean()) if len(y) else None}
    if len(s) < 40 or len(np.unique(y[dv])) < 2 or len(np.unique(y[te])) < 2 or min(np.bincount(y[te])) < 5:
        out["skipped"] = "too few rows, or fewer than 5 of one class in test"
        return out
    r = sweep_and_test(path_c, s.row.to_numpy(), y, g, dv, te, layers, pc["C_grid"], pc["folds"], "clf", nj)
    probe = r["pred_all"]
    F = _f(s, ALL_COV)
    z_all = probe.copy()
    z_all[dv] = np.nan_to_num(r["oof_dev"])
    cov_s = covariate_clf(F, y, dv, te)
    stk_s = covariate_clf(np.c_[F, z_all], y, dv, te)
    yt, gt = y[te], g[te]
    w, n_mixed = within_auc(probe[te], yt, gt)
    out.update({
        "best_layer": r["best_layer"], "best_h": r["best_h"], "curve": r["curve"],
        "probe_auroc": safe_auc(yt, probe[te]), "probe_within": w, "n_mixed": n_mixed,
        "cov_auroc": safe_auc(yt, cov_s[te]), "cov_within": within_auc(cov_s[te], yt, gt)[0],
        "delta": safe_auc(yt, stk_s[te]) - safe_auc(yt, cov_s[te]),
        "_probe": probe, "_model": r["model"], "_cov": cov_s, "_stack": stk_s,
    })
    if nb:
        out["probe_auroc_ci"] = cluster_bootstrap(lambda i: safe_auc(yt[i], probe[te][i]), gt, nb, seed)
        out["delta_ci"] = cluster_bootstrap(
            lambda i: safe_auc(yt[i], stk_s[te][i]) - safe_auc(yt[i], cov_s[te][i]), gt, nb, seed)
    return out


def _perm_one(seed, Xd, yd, Xt, yt, h):
    ysh = np.random.default_rng(seed).permutation(yd)
    return safe_auc(yt, clf(h).fit(Xd, ysh).decision_function(Xt))


def _p1_full(s, path_c, layers, pc, seed, nb, nj, rd, tag):
    """Main P1 analysis on table s: probe, 4 covariate baselines, stacked gain, permutation null."""
    split = s.split.to_numpy()
    core = _p1_core(s, path_c, layers, pc, seed, nb, nj, split)
    if "skipped" in core:
        return core, None
    y = (s.outcome == "FOLLOW").astype(int).to_numpy()
    g = s.problem_id.to_numpy()
    dv, te = split == "dev", split == "test"
    yt, gt = y[te], g[te]
    out = {k: v for k, v in core.items() if not k.startswith("_")}
    out["probe"] = classification_report(core["_probe"][te], yt, gt, nb, seed)
    out["baselines"] = {}
    for name, cols in [("difficulty", DIFFICULTY), ("structure", STRUCTURE),
                       ("surprise", SURPRISE), ("all_covariates", ALL_COV)]:
        sc = covariate_clf(_f(s, cols), y, dv, te)
        out["baselines"][name] = classification_report(sc[te], yt, gt, nb, seed)
    out["stacked"] = classification_report(core["_stack"][te], yt, gt, nb, seed)

    # permutation null: refit the probe at the chosen layer/C on shuffled dev labels
    X = np.load(path_c, mmap_mode="r")[s.row.to_numpy(), core["best_layer"], :].astype(np.float32)
    from joblib import Parallel, delayed
    Xd, Xt = X[dv], X[te]
    null = Parallel(n_jobs=nj)(delayed(_perm_one)(seed * 1000 + k, Xd, y[dv], Xt, yt, core["best_h"])
                               for k in range(pc.get("n_perm", 50)))
    obs = out["probe"]["auroc"]
    out["permutation_null"] = {"n": len(null), "mean": float(np.mean(null)), "sd": float(np.std(null)),
                               "max": float(np.max(null)),
                               "p_value": float((1 + sum(v >= obs for v in null)) / (1 + len(null)))}
    s.assign(probe_score=core["_probe"])[["inj_id", "problem_id", "group", "split", "dataset", "kind",
                                         "err_bin", "outcome", "probe_score"] + [c for c in ALL_COV if c in s]] \
        .to_csv(rd / f"p1_scores_{tag}.csv", index=False)
    return out, core


# ============================================================================ main

def run_probes(cfg: dict) -> dict:
    rd = run_dir(cfg)
    pc = cfg["probes"]
    nb, nj, seed = pc["n_boot"], pc["n_jobs"], cfg["seed"]
    df = build_table(rd)
    path_c = str(rd / "acts_corrupt.npy")
    n_layers = np.load(path_c, mmap_mode="r").shape[1]
    layers_all = list(range(0, n_layers, pc["layer_stride"]))
    if layers_all[-1] != n_layers - 1:
        layers_all.append(n_layers - 1)
    layers_coarse = sorted(set(range(0, n_layers, 2)) | {n_layers - 1})

    # split by GROUP (GSM-Symbolic instances of one template stay together)
    dev_g, test_g = split_problems(df.group.to_numpy(), pc["test_frac"], seed)
    df["split"] = np.where(test_g, "test", "dev")
    R: dict = {"counts": {}, "probes": {}}

    # ------------------------------------------------------------ validity + counts
    valid = df.clean_cont_correct.to_numpy(bool) & ~df.glued.fillna(False).to_numpy(bool)
    amb = df.on_path.to_numpy() & np.array([rounds_to_gold(p, g) for p, g in zip(df.propagated_answer, df.gold)])
    valid &= ~amb
    df["valid"] = valid
    on = df.on_path.to_numpy()
    C = R["counts"]
    C["injections"], C["problems"] = int(len(df)), int(df.problem_id.nunique())
    C["valid_rate"], C["glued_rate"] = float(valid.mean()), float(df.glued.fillna(False).mean())
    C["ambiguous_label_n"] = int(amb.sum())
    C["clean_cont_correct_rate"] = float(df.clean_cont_correct.mean())
    v_on = df[valid & on]
    C["outcomes_on_path"] = v_on.outcome.value_counts().to_dict()
    C["outcomes_off_path"] = df[valid & ~on].outcome.value_counts().to_dict()
    for key, col in [("by_dataset", "dataset"), ("by_kind", "kind"), ("by_err_bin", "err_bin")]:
        C[key] = v_on.groupby(col).outcome.value_counts().unstack(fill_value=0).to_dict(orient="index")
    C["recovery_types"] = v_on[v_on.outcome == "RECOVER"].recovery_type.value_counts().to_dict()
    C["uptake_rate_on_path"] = float(v_on.uptake.mean())
    C["catchable_rate_on_path"] = float(v_on.catchable.mean())
    if "chains" in set(df.dataset):
        ch = v_on[v_on.dataset == "chains"]
        C["chains_follow_by_depth"] = (ch.assign(f=ch.outcome == "FOLLOW").groupby("level").f.mean()
                                       .round(3).to_dict())
        C["chains_follow_by_downstream"] = (ch.assign(f=ch.outcome == "FOLLOW").groupby("n_downstream").f.mean()
                                            .round(3).to_dict())
    log(f"table: {len(df)} injections, valid {valid.mean():.3f}, on-path outcomes {C['outcomes_on_path']}")

    # ------------------------------------------------------------ P0
    log("P0 corruption detection")
    sub = df[valid].sample(n=min(3000, int(valid.sum())), random_state=seed)
    acts0 = rd / "acts_p0.npy"
    A_c = np.load(rd / "acts_corrupt.npy", mmap_mode="r")
    A_k = np.load(rd / "acts_clean.npy", mmap_mode="r")
    rows = np.sort(sub.row.to_numpy())
    out = np.lib.format.open_memmap(acts0, "w+", np.float16, (2 * len(rows),) + A_c.shape[1:])
    out[: len(rows)] = A_k[rows]
    out[len(rows):] = A_c[rows]
    out.flush()
    del out
    sub = df.iloc[rows]
    y0 = np.r_[np.zeros(len(rows)), np.ones(len(rows))].astype(int)
    g0 = np.r_[sub.problem_id, sub.problem_id]
    d0 = np.r_[sub.split == "dev", sub.split == "dev"]
    r0 = sweep_and_test(str(acts0), np.arange(len(y0)), y0, g0, d0, ~d0, layers_coarse, pc["C_grid"][-1:],
                        pc["folds"], "clf", nj)
    R["probes"]["P0_detect"] = {"curve": r0["curve"], "best_layer": r0.get("best_layer"),
                                "test": classification_report(r0["pred_all"][~d0], y0[~d0], g0[~d0], 200, seed)}
    acts0.unlink(missing_ok=True)

    # ------------------------------------------------------------ P1 all + catchable
    fr = df.outcome.isin(["FOLLOW", "RECOVER"]).to_numpy()
    s_all = df[valid & on & fr].reset_index(drop=True)
    s_cat = df[valid & on & fr & df.catchable.to_numpy()].reset_index(drop=True)
    log(f"P1 all on-path ({len(s_all)} rows)")
    P1a, core_a = _p1_full(s_all, path_c, layers_all, pc, seed, nb, nj, rd, "all")
    R["probes"]["P1_all"] = P1a
    log(f"P1 catchable ({len(s_cat)} rows)")
    P1c, core_c = _p1_full(s_cat, path_c, layers_all, pc, seed, nb, nj, rd, "catchable")
    R["probes"]["P1_catchable"] = P1c

    ref = core_c if core_c is not None else core_a
    if ref is None:
        R["verdict"] = verdict(R)
        _save(R, df, rd)
        return R
    win = _window(ref["best_layer"], n_layers)

    # off-path control with the all-injections probe
    if core_a is not None:
        off = df[valid & ~on]
        Xa = np.load(path_c, mmap_mode="r")
        if len(off):
            p_off = core_a["_model"].predict_proba(Xa[off.row.to_numpy(), core_a["best_layer"], :].astype(np.float32))[:, 1]
            te = (s_all.split == "test").to_numpy()
            p_on = core_a["_model"].predict_proba(
                Xa[s_all.row.to_numpy()[te], core_a["best_layer"], :].astype(np.float32))[:, 1]
            yt = (s_all.outcome == "FOLLOW").to_numpy()[te]
            P1a["off_path_control"] = {"n_off": int(len(off)), "mean_p_follow_off_path": float(p_off.mean()),
                                       "mean_p_follow_test_FOLLOW": float(p_on[yt].mean()),
                                       "mean_p_follow_test_RECOVER": float(p_on[~yt].mean())}

    # ------------------------------------------------------------ robustness on the layer window
    rob = {}
    for name, col, values in [("by_operator", "kind", KINDS), ("by_dataset", "dataset", DATASETS),
                              ("by_err_bin", "err_bin", [b[2] for b in ERR_BINS])]:
        log(f"P1 {name}")
        rob[name] = {}
        for v in values:
            sv = s_all[s_all[col] == v].reset_index(drop=True)
            if len(sv) == 0:
                continue
            r = _p1_core(sv, path_c, win, pc, seed, nb, nj, sv.split.to_numpy())
            rob[name][v] = {k: x for k, x in r.items() if not k.startswith("_") and k != "curve"}
    log("P1 split stability")
    runs = []
    base = s_cat if core_c is not None else s_all
    for s_ in pc.get("split_seeds", [0, 1, 2, 3, 4]):
        d_, t_ = split_problems(base.group.to_numpy(), pc["test_frac"], s_)
        r_ = _p1_core(base, path_c, win, pc, s_, 0, nj, np.where(t_, "test", "dev"))
        if "probe_auroc" in r_:
            runs.append({"seed": s_, **{k: r_[k] for k in ("best_layer", "probe_auroc", "probe_within",
                                                          "n_mixed", "cov_auroc", "delta")}})
    if runs:
        rob["split_stability"] = {"subset": "catchable" if core_c is not None else "all", "runs": runs,
                                  "mean_sd": {k: [float(np.nanmean([r[k] for r in runs])),
                                                  float(np.nanstd([r[k] for r in runs]))]
                                              for k in ("probe_auroc", "probe_within", "cov_auroc", "delta")}}
    R["probes"]["P1_robustness"] = rob

    # steering direction for the `steer` stage (difference of class means on dev, catchable set)
    sd_tab = s_cat if core_c is not None else s_all
    dv = (sd_tab.split == "dev").to_numpy()
    yy = (sd_tab.outcome == "FOLLOW").to_numpy()
    X = np.load(path_c, mmap_mode="r")[sd_tab.row.to_numpy()[dv], ref["best_layer"], :].astype(np.float32)
    v = X[yy[dv]].mean(0) - X[~yy[dv]].mean(0)
    unit = v / (np.linalg.norm(v) + 1e-9)
    proj = X @ unit
    np.savez(rd / "direction.npz", unit=unit, layer=ref["best_layer"], proj_sd=float(proj.std()),
             mean_norm=float(np.linalg.norm(X, axis=1).mean()))
    R["probes"]["direction"] = {"layer": int(ref["best_layer"]), "proj_sd": float(proj.std()),
                                "class_gap_in_sd": float((proj[yy[dv]].mean() - proj[~yy[dv]].mean()) / proj.std())}

    # ------------------------------------------------------------ W: probe-gated Wait intervention
    if core_a is not None and df.wait_outcome.notna().any():
        log("W wait-intervention policy simulation")
        R["probes"]["W_wait_policy"] = _wait_policy(df, s_all, core_a, path_c, rd, seed)

    # ------------------------------------------------------------ P1r / P4 regression
    for key, mask, target in [
        ("P1r_follow_rate", valid & on & df.follow_rate.notna().to_numpy(), lambda s: s.follow_rate.to_numpy(float)),
        ("P4_horizon", valid & on, lambda s: np.log1p(s.horizon.to_numpy(float))),
    ]:
        s = df[mask].reset_index(drop=True)
        y = target(s)
        if len(s) > 40 and np.std(y) > 0:
            log(key)
            g = s.problem_id.to_numpy()
            dv, te = (s.split == "dev").to_numpy(), (s.split == "test").to_numpy()
            r = sweep_and_test(path_c, s.row.to_numpy(), y, g, dv, te, layers_coarse, pc["alpha_grid"],
                               pc["folds"], "reg", nj)
            cov = covariate_reg(_f(s, ALL_COV), y, dv)
            R["probes"][key] = {"curve": r["curve"], "best_layer": r["best_layer"],
                                "probe": regression_report(r["pred_all"][te], y[te], g[te], nb, seed),
                                "all_covariates": regression_report(cov[te], y[te], g[te], nb, seed)}

    # ------------------------------------------------------------ P2 value tracking
    m2 = valid & df.is_int.to_numpy() & (df.true_last != df.new_last).to_numpy()
    s2 = df[m2]
    if len(s2) > 4000:
        s2 = s2.sample(n=4000, random_state=seed)
    s2 = s2.reset_index(drop=True)
    if len(s2) > 50:
        log("P2 value tracking")
        R["probes"]["P2_value_tracking"] = _value_tracking(rd, s2, layers_coarse)

    # ------------------------------------------------------------ P3 silent recovery
    s3 = df[valid & on & (df.outcome == "RECOVER").to_numpy()].reset_index(drop=True)
    y3 = (s3.recovery_type == "silent").astype(int).to_numpy()
    P3 = {"n": int(len(s3)), "silent_rate": float(y3.mean()) if len(y3) else None}
    if len(s3) >= 60:
        g3 = s3.problem_id.to_numpy()
        dv, te = (s3.split == "dev").to_numpy(), (s3.split == "test").to_numpy()
        if len(np.unique(y3[dv])) == 2 and len(np.unique(y3[te])) == 2:
            log("P3 silent recovery")
            r = sweep_and_test(path_c, s3.row.to_numpy(), y3, g3, dv, te, layers_coarse, pc["C_grid"],
                               pc["folds"], "clf", nj)
            sc = covariate_clf(_f(s3, ALL_COV), y3, dv, te)
            P3.update({"curve": r["curve"], "best_layer": r["best_layer"],
                       "probe": classification_report(r["pred_all"][te], y3[te], g3[te], nb, seed),
                       "all_covariates": classification_report(sc[te], y3[te], g3[te], nb, seed)})
    R["probes"]["P3_silent_recovery"] = P3

    # ------------------------------------------------------------ P5 prompt-time susceptibility
    pids = [r["problem_id"] for r in read_jsonl(rd / "prompt_ids.jsonl")]
    pidx = {p: i for i, p in enumerate(pids)}
    o5 = df[valid & on & df.follow_rate.notna().to_numpy()]
    per = o5.groupby("problem_id").agg(follow=("follow_rate", "mean"), pass_rate=("pass_rate", "first"),
                                       level=("level", "first"), split=("split", "first"),
                                       **{f"ds_{d}": (f"ds_{d}", "first") for d in DATASETS}).reset_index()
    per["row"] = per.problem_id.map(pidx)
    if len(per) > 60 and per.follow.std() > 0:
        log("P5 susceptibility")
        y = per.follow.to_numpy(float)
        g = per.problem_id.to_numpy()
        dv, te = (per.split == "dev").to_numpy(), (per.split == "test").to_numpy()
        r = sweep_and_test(str(rd / "acts_prompt.npy"), per.row.to_numpy(), y, g, dv, te, layers_coarse,
                           pc["alpha_grid"], pc["folds"], "reg", nj)
        cov = covariate_reg(_f(per, DIFFICULTY), y, dv)
        R["probes"]["P5_susceptibility"] = {
            "curve": r["curve"], "best_layer": r["best_layer"], "n_problems": int(len(per)),
            "probe": regression_report(r["pred_all"][te], y[te], g[te], nb, seed),
            "difficulty_covariates": regression_report(cov[te], y[te], g[te], nb, seed)}

    R["verdict"] = verdict(R)
    _save(R, df, rd)
    log("done")
    return R


# ============================================================================ wait policy

def _wait_policy(df, s_all, core_a, path_c, rd, seed):
    """Offline simulation of a monitor that may append 'Wait' at a step, for a fixed budget.

    Pool A (corrupted only): every valid on-path TEST injection. Without intervention the item is
      correct iff the greedy continuation recovered; if chosen, iff the Wait continuation is correct.
    Pool B (deployment-like): pool A plus the clean versions of the same steps. Intervening on a clean
      step can break it (Wait-induced second-guessing), so false alarms now cost accuracy.
    Every policy ranks items by a score and intervenes on the top `budget` fraction.
    """
    X = np.load(path_c, mmap_mode="r")
    K = np.load(rd / "acts_clean.npy", mmap_mode="r")
    L = core_a["best_layer"]
    m = core_a["_model"]
    pool = df[df.valid & df.on_path & (df.split == "test") & df.wait_outcome.notna()].reset_index(drop=True)
    pool = pool[pool.outcome.isin(["FOLLOW", "RECOVER", "OTHER"])].reset_index(drop=True)
    if len(pool) < 30:
        return {"skipped": "too few test items"}
    p_corrupt = m.predict_proba(X[pool.row.to_numpy(), L, :].astype(np.float32))[:, 1]
    p_clean = m.predict_proba(K[pool.row.to_numpy(), L, :].astype(np.float32))[:, 1]

    # covariate scorer fit on dev on-path rows (predicting FOLLOW), as a non-probe competitor
    dev = s_all[s_all.split == "dev"]
    from .probes import clf as _clf
    cm = _clf(1.0).fit(_f(dev, ALL_COV), (dev.outcome == "FOLLOW").astype(int))
    cov_corrupt = cm.decision_function(_f(pool, ALL_COV))

    budgets = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    rng = np.random.default_rng(seed)

    def curve(base_ok, wait_ok, score):
        n = len(base_ok)
        order = np.argsort(-score, kind="stable")
        out = []
        for b in budgets:
            k = int(round(b * n))
            chosen = np.zeros(n, bool)
            chosen[order[:k]] = True
            acc = np.where(chosen, wait_ok, base_ok).mean()
            out.append({"budget": b, "accuracy": float(acc),
                        "rescued": int((chosen & ~base_ok & wait_ok).sum()),
                        "broken": int((chosen & base_ok & ~wait_ok).sum())})
        return out

    def random_curve(base_ok, wait_ok, reps=200):
        accs = np.array([[r["accuracy"] for r in curve(base_ok, wait_ok, rng.random(len(base_ok)))]
                         for _ in range(reps)])
        return [{"budget": b, "accuracy": float(a)} for b, a in zip(budgets, accs.mean(0))]

    base_ok = pool.correct_no_int.to_numpy(bool)
    wait_ok = pool.correct_wait.to_numpy(bool)
    oracle = (wait_ok & ~base_ok).astype(float) - (base_ok & ~wait_ok).astype(float)
    A = {"n": int(len(pool)), "no_intervention_acc": float(base_ok.mean()), "all_wait_acc": float(wait_ok.mean()),
         "probe": curve(base_ok, wait_ok, p_corrupt),
         "surprise": curve(base_ok, wait_ok, pool.surprise_corrupt.to_numpy(float)),
         "covariates": curve(base_ok, wait_ok, cov_corrupt),
         "random": random_curve(base_ok, wait_ok),
         "oracle": curve(base_ok, wait_ok, oracle)}

    # pool B: clean twins added
    base_B = np.r_[base_ok, pool.clean_cont_correct.to_numpy(bool)]
    wait_B = np.r_[wait_ok, pool.wait_clean_correct.fillna(True).to_numpy(bool)]
    probe_B = np.r_[p_corrupt, p_clean]
    surpr_B = np.r_[pool.surprise_corrupt.to_numpy(float), pool.surprise_clean.to_numpy(float)]
    B = {"n": int(len(base_B)), "no_intervention_acc": float(base_B.mean()), "all_wait_acc": float(wait_B.mean()),
         "probe": curve(base_B, wait_B, probe_B), "surprise": curve(base_B, wait_B, surpr_B),
         "random": random_curve(base_B, wait_B),
         "clean_broken_by_wait_rate": float((pool.clean_cont_correct & ~pool.wait_clean_correct.fillna(True)).mean())}
    return {"layer": int(L), "corrupted_pool": A, "mixed_pool": B}


# ============================================================================ value tracking

def _value_tracking(rd, s2, layers):
    """10-way last-digit probe trained on CLEAN runs (written digit == true digit), read on CORRUPTED
    runs: does it report the true digit (model still carries the right value) or the injected one?"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    A_k = np.load(rd / "acts_clean.npy", mmap_mode="r")
    A_c = np.load(rd / "acts_corrupt.npy", mmap_mode="r")
    rows = s2.row.to_numpy()
    order = np.argsort(rows)
    rows, s2 = rows[order], s2.iloc[order].reset_index(drop=True)
    dv, te = (s2.split == "dev").to_numpy(), (s2.split == "test").to_numpy()
    yt, yn = s2.true_last.to_numpy(), s2.new_last.to_numpy()
    outc = s2.outcome.to_numpy()[te]
    curve = []
    for l in layers:
        Xk = A_k[rows, l, :].astype(np.float32)
        Xc = A_c[rows, l, :].astype(np.float32)
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000)).fit(Xk[dv], yt[dv])
        classes = list(m.classes_)
        P = m.predict_proba(Xc[te])
        it = np.array([classes.index(d) if d in classes else -1 for d in yt[te]])
        inn = np.array([classes.index(d) if d in classes else -1 for d in yn[te]])
        ok = (it >= 0) & (inn >= 0)
        pt = np.where(ok, P[np.arange(len(P)), np.clip(it, 0, None)], np.nan)
        pn = np.where(ok, P[np.arange(len(P)), np.clip(inn, 0, None)], np.nan)
        pred = np.array(classes)[P.argmax(1)]
        margin = np.log(pt + 1e-9) - np.log(pn + 1e-9)
        rec = {"layer": l, "clean_test_acc": float((m.predict(Xk[te]) == yt[te]).mean()),
               "corrupt_reads_true": float((pred == yt[te]).mean()),
               "corrupt_reads_injected": float((pred == yn[te]).mean())}
        for o in ("FOLLOW", "RECOVER"):
            mm = (outc == o) & ok
            rec[f"reads_true_{o}"] = float((pred[mm] == yt[te][mm]).mean()) if mm.any() else None
        mm = np.isin(outc, ["FOLLOW", "RECOVER"]) & ok
        if mm.sum() > 10:
            rec["margin_auroc_recover"] = safe_auc((outc[mm] == "RECOVER").astype(int), margin[mm])
        curve.append(rec)
    return {"n": int(len(s2)), "n_test": int(te.sum()), "curve": curve}


# ============================================================================ verdict + save

def verdict(R: dict) -> dict:
    """Pre-registered rule (DIRECTION.md), applied to the catchable headline and to all injections."""
    out = {}
    for key in ("P1_catchable", "P1_all"):
        p1 = R["probes"].get(key, {})
        if "probe" not in p1:
            out[key] = {"summary": "not run (too few FOLLOW or RECOVER)"}
            continue
        a, w = p1["probe"]["auroc"], p1["probe"]["within_auroc"]
        d, ci = p1["delta"], p1.get("delta_ci", [float("nan")] * 2)
        pv = p1["permutation_null"]["p_value"]
        h1 = a >= 0.70 and pv < 0.05
        h2 = (w >= 0.65 or w != w) and d >= 0.05 and ci[0] > 0
        s = ("Signal beyond difficulty, structure and surprisal" if h1 and h2 else
             "Signal exists but is mostly explained by covariates" if h1 else "No usable signal")
        out[key] = {"H1_auroc_ge_0.70_and_perm_p_lt_0.05": h1, "H2_within_ge_0.65_and_delta_ge_0.05": h2,
                    "auroc": a, "within_auroc": w, "n_mixed_problems": p1["probe"]["n_mixed_problems"],
                    "delta": d, "delta_ci": ci, "perm_p": pv, "summary": s}
    return out


def _save(R, df, rd):
    with open(rd / "results.json", "w") as f:
        json.dump(R, f, indent=2, default=_jsonable)
    df.to_csv(rd / "table.csv", index=False)
    print(json.dumps(R.get("verdict", {}), indent=2, default=_jsonable), flush=True)


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)
