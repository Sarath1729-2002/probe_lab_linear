"""Stage `probes`: assemble the table, run P0-P5 and every control, write results.json."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .io import run_dir, read_jsonl
from .probes import (sweep_and_test, classification_report, regression_report, covariate_clf,
                     covariate_reg, split_problems, safe_auc, within_auc, cluster_bootstrap, clf)

DIFFICULTY = ["pass_rate", "is_math", "level"]
STRUCTURE = ["step_frac", "n_downstream", "n_eq", "n_tokens", "log_rel_err",
             "kind_plus_one", "kind_times_ten", "kind_digit_swap"]
SURPRISE = ["surprise_corrupt", "d_surprise"]
ALL_COV = DIFFICULTY + STRUCTURE + SURPRISE


def build_table(rd) -> pd.DataFrame:
    inj = pd.DataFrame(read_jsonl(rd / "injections.jsonl"))
    cont = read_jsonl(rd / "continuations.jsonl")
    meta = pd.DataFrame(read_jsonl(rd / "extract_meta.jsonl"))
    assert list(inj.inj_id) == list(meta.inj_id) == [c["inj_id"] for c in cont], "stage files out of order"
    df = inj.drop(columns=["prompt", "prefix_corrupt", "prefix_clean"]).copy()
    df["row"] = np.arange(len(df))
    for k in ("outcome", "horizon", "uptake", "marker", "recovery_type", "true_restated_char", "n_chars", "glued"):
        df[k] = [c["greedy"][k] for c in cont]
    df["follow_rate"] = [c["follow_rate"] for c in cont]
    df["clean_cont_correct"] = [c["clean_cont_correct"] for c in cont]
    df["sample_outcomes"] = [[s["outcome"] for s in c["samples"]] for c in cont]
    df = df.join(meta.drop(columns=["inj_id"]))
    df["d_surprise"] = df.surprise_corrupt - df.surprise_clean
    df["is_math"] = (df.dataset == "math").astype(float)
    for k in ("plus_one", "times_ten", "digit_swap"):
        df[f"kind_{k}"] = (df.kind == k).astype(float)
    df["true_last"] = [int(abs(round(v))) % 10 for v in df.true_value]
    df["new_last"] = [int(abs(round(v))) % 10 for v in df.new_value]
    df["is_int"] = [abs(v - round(v)) < 1e-9 and abs(n - round(n)) < 1e-9 for v, n in zip(df.true_value, df.new_value)]
    return df


def _f(df, cols):
    return df[cols].astype(float).fillna(0.0).to_numpy()


def rounds_to_gold(prop: float, gold: float) -> bool:
    """True when the propagated answer is indistinguishable from gold after ordinary rounding."""
    if prop is None or gold is None:
        return False
    if abs(gold - round(gold)) < 1e-9:                 # integer gold: model would round to it
        return abs(prop - gold) < 0.5
    return abs(prop - gold) <= 0.01 * max(1.0, abs(gold))


def _p1_core(s, path_c, layers, pc, seed, nb, nj, split):
    """FOLLOW vs RECOVER on table `s` with a given dev/test split: probe, all-covariate baseline,
    and the AUROC gain from adding the probe to the covariates. nb=0 skips bootstrap CIs."""
    y = (s.outcome == "FOLLOW").astype(int).to_numpy()
    g = s.problem_id.to_numpy()
    dv, te = split == "dev", split == "test"
    out = {"n_dev": int(dv.sum()), "n_test": int(te.sum()),
           "follow_rate": float(y.mean()) if len(y) else None}
    if len(s) < 40 or len(np.unique(y[dv])) < 2 or len(np.unique(y[te])) < 2:
        out["skipped"] = "too few rows or only one class in dev/test"
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
        "best_layer": r["best_layer"],
        "probe_auroc": safe_auc(yt, probe[te]), "probe_within": w, "n_mixed": n_mixed,
        "cov_auroc": safe_auc(yt, cov_s[te]), "cov_within": within_auc(cov_s[te], yt, gt)[0],
        "delta": safe_auc(yt, stk_s[te]) - safe_auc(yt, cov_s[te]),
    })
    if nb:
        out["probe_auroc_ci"] = cluster_bootstrap(lambda i: safe_auc(yt[i], probe[te][i]), gt, nb, seed)
        out["delta_ci"] = cluster_bootstrap(
            lambda i: safe_auc(yt[i], stk_s[te][i]) - safe_auc(yt[i], cov_s[te][i]), gt, nb, seed)
    return out


def run_probes(cfg: dict) -> dict:
    rd = run_dir(cfg)
    pc = cfg["probes"]
    df = build_table(rd)
    n_layers = np.load(rd / "acts_corrupt.npy", mmap_mode="r").shape[1]
    layers = list(range(0, n_layers, pc["layer_stride"]))
    if layers[-1] != n_layers - 1:
        layers.append(n_layers - 1)
    nb, nj, seed = pc["n_boot"], pc["n_jobs"], cfg["seed"]
    dev_all, test_all = split_problems(df.problem_id.to_numpy(), pc["test_frac"], seed)
    df["split"] = np.where(test_all, "test", "dev")
    R: dict = {"counts": {}, "probes": {}}

    # ---------------------------------------------------------------- counts / label table
    R["counts"]["injections"] = int(len(df))
    R["counts"]["problems"] = int(df.problem_id.nunique())
    R["counts"]["greedy_outcomes_on_path"] = df[df.on_path].outcome.value_counts().to_dict()
    R["counts"]["greedy_outcomes_off_path"] = df[~df.on_path].outcome.value_counts().to_dict()
    R["counts"]["by_dataset"] = (df[df.on_path].groupby("dataset").outcome.value_counts()
                                 .unstack(fill_value=0).to_dict(orient="index"))
    R["counts"]["by_kind"] = (df[df.on_path].groupby("kind").outcome.value_counts()
                              .unstack(fill_value=0).to_dict(orient="index"))
    R["counts"]["clean_cont_correct_rate"] = float(df.clean_cont_correct.mean())
    R["counts"]["uptake_rate_on_path"] = float(df[df.on_path].uptake.mean())
    rec = df[df.on_path & (df.outcome == "RECOVER")]
    R["counts"]["recovery_types"] = rec.recovery_type.value_counts().to_dict()

    # setup sanity: the clean truncated prefix still solves it, and the model did not extend the
    # injected number with more digits
    valid = df.clean_cont_correct.to_numpy() & ~df.glued.to_numpy(bool)
    # label ambiguity: if the propagated answer rounds to gold (e.g. 59.97 vs 60), a model that
    # followed the error and rounded is indistinguishable from one that recovered
    amb = df.on_path.to_numpy() & np.array([rounds_to_gold(p, g) for p, g in zip(df.propagated_answer, df.gold)])
    df["ambiguous"] = amb
    valid &= ~amb
    R["counts"]["glued_rate"] = float(df.glued.mean())
    R["counts"]["ambiguous_label_rate"] = float(amb.mean())
    R["counts"]["ambiguous_label_n"] = int(amb.sum())
    R["counts"]["valid_rate"] = float(valid.mean())
    path_c = str(rd / "acts_corrupt.npy")

    # ---------------------------------------------------------------- P0: corruption detection
    sub = df[valid]
    acts0 = rd / "acts_p0.npy"            # [clean ; corrupted] stacked, rebuilt every run
    A_c = np.load(rd / "acts_corrupt.npy", mmap_mode="r")
    A_k = np.load(rd / "acts_clean.npy", mmap_mode="r")
    out = np.lib.format.open_memmap(acts0, "w+", np.float16, (2 * len(sub),) + A_c.shape[1:])
    out[: len(sub)] = A_k[sub.row.to_numpy()]
    out[len(sub):] = A_c[sub.row.to_numpy()]
    out.flush()
    del out
    y0 = np.r_[np.zeros(len(sub)), np.ones(len(sub))].astype(int)
    g0 = np.r_[sub.problem_id, sub.problem_id]
    d0 = np.r_[sub.split == "dev", sub.split == "dev"]
    r0 = sweep_and_test(str(acts0), np.arange(len(y0)), y0, g0, d0, ~d0, layers, pc["C_grid"],
                        pc["folds"], "clf", nj)
    R["probes"]["P0_detect"] = {"curve": r0["curve"], "best_layer": r0.get("best_layer"),
                                "test": classification_report(r0["pred_all"][~d0], y0[~d0], g0[~d0], nb, seed)}

    # ---------------------------------------------------------------- P1: FOLLOW vs RECOVER
    m1 = valid & df.on_path.to_numpy() & df.outcome.isin(["FOLLOW", "RECOVER"]).to_numpy()
    s1 = df[m1].reset_index(drop=True)
    y1 = (s1.outcome == "FOLLOW").astype(int).to_numpy()
    g1 = s1.problem_id.to_numpy()
    dv1, te1 = (s1.split == "dev").to_numpy(), (s1.split == "test").to_numpy()
    P1 = {"n_dev": int(dv1.sum()), "n_test": int(te1.sum()), "follow_rate": float(y1.mean()) if len(y1) else None}
    if len(np.unique(y1[dv1])) == 2 and len(np.unique(y1[te1])) == 2:
        r1 = sweep_and_test(path_c, s1.row.to_numpy(), y1, g1, dv1, te1, layers, pc["C_grid"], pc["folds"], "clf", nj)
        probe = r1["pred_all"]
        P1["curve"], P1["best_layer"] = r1["curve"], r1["best_layer"]
        P1["probe"] = classification_report(probe[te1], y1[te1], g1[te1], nb, seed)

        # covariate baselines (fit on dev, scored on test)
        base = {}
        for name, cols in [("difficulty", DIFFICULTY), ("structure", STRUCTURE),
                           ("surprise", SURPRISE), ("all_covariates", ALL_COV)]:
            sc = covariate_clf(_f(s1, cols), y1, dv1, te1)
            base[name] = classification_report(sc[te1], y1[te1], g1[te1], nb, seed)
        P1["baselines"] = base

        # stacked: all covariates + probe score. Probe enters as its dev out-of-fold score.
        F = _f(s1, ALL_COV)
        z = r1["oof_dev"].copy()
        z_all = probe.copy()
        z_all[dv1] = np.nan_to_num(z)
        F_stack = np.c_[F, z_all]
        cov_s = covariate_clf(F, y1, dv1, te1)
        stk_s = covariate_clf(F_stack, y1, dv1, te1)
        yt, gt = y1[te1], g1[te1]
        P1["stacked"] = classification_report(stk_s[te1], yt, gt, nb, seed)
        P1["delta_auroc_probe_over_covariates"] = {
            "value": safe_auc(yt, stk_s[te1]) - safe_auc(yt, cov_s[te1]),
            "ci": cluster_bootstrap(lambda i: safe_auc(yt[i], stk_s[te1][i]) - safe_auc(yt[i], cov_s[te1][i]),
                                    gt, nb, seed)}

        # control 1: shuffled labels on dev, same layer/C
        rng = np.random.default_rng(seed)
        ysh = y1.copy()
        ysh[dv1] = rng.permutation(ysh[dv1])
        X = np.load(path_c, mmap_mode="r")[s1.row.to_numpy(), r1["best_layer"], :].astype(np.float32)
        msh = clf(r1["best_h"]).fit(X[dv1], ysh[dv1])
        P1["shuffled_label_test_auroc"] = safe_auc(y1[te1], msh.decision_function(X[te1]))

        # control 2: off-path injections (corrupted, but cannot change the answer)
        off = df[valid & ~df.on_path.to_numpy()]
        if len(off):
            Xo = np.load(path_c, mmap_mode="r")[off.row.to_numpy(), r1["best_layer"], :].astype(np.float32)
            p_off = r1["model"].predict_proba(Xo)[:, 1]
            p_on = r1["model"].predict_proba(X[te1])[:, 1]
            P1["off_path_control"] = {
                "mean_p_follow_off_path": float(p_off.mean()), "n_off": int(len(off)),
                "mean_p_follow_test_FOLLOW": float(p_on[y1[te1] == 1].mean()),
                "mean_p_follow_test_RECOVER": float(p_on[y1[te1] == 0].mean())}
        s1 = s1.assign(probe_score=probe)
        s1[["inj_id", "problem_id", "split", "outcome", "probe_score"] + ALL_COV].to_csv(rd / "p1_scores.csv", index=False)
    else:
        P1["skipped"] = "need both FOLLOW and RECOVER in dev and test"
    R["probes"]["P1_follow_vs_recover"] = P1

    # ---------------------------------------------------------------- P1 within each operator
    # Outcome depends strongly on the operator (+1 mostly followed, x10 mostly caught). Inside one
    # operator that shortcut is gone: does the probe still beat the covariates?
    if "probe" in P1:
        per_op = {}
        for kind in sorted(s1.kind.unique()):
            sk = s1[s1.kind == kind].reset_index(drop=True)
            per_op[kind] = _p1_core(sk, path_c, layers, pc, seed, nb, nj, split=sk.split.to_numpy())
        R["probes"]["P1_by_operator"] = per_op

        # ------------------------------------------------------------ P1 split stability
        seeds = pc.get("split_seeds", [0, 1, 2, 3, 4])
        runs = []
        for s_ in seeds:
            d_, t_ = split_problems(s1.problem_id.to_numpy(), pc["test_frac"], s_)
            r_ = _p1_core(s1, path_c, layers, pc, s_, 0, nj, split=np.where(t_, "test", "dev"))
            if "probe_auroc" in r_:
                runs.append({"seed": s_, **{k: r_[k] for k in ("best_layer", "probe_auroc", "probe_within",
                                                              "n_mixed", "cov_auroc", "delta")}})
        if runs:
            agg = {k: [float(np.nanmean([r[k] for r in runs])), float(np.nanstd([r[k] for r in runs]))]
                   for k in ("probe_auroc", "probe_within", "cov_auroc", "delta")}
            R["probes"]["P1_split_stability"] = {"runs": runs, "mean_sd": agg}

    # ---------------------------------------------------------------- P1r: follow rate (regression)
    m1r = valid & df.on_path.to_numpy() & df.follow_rate.notna().to_numpy()
    s = df[m1r].reset_index(drop=True)
    if len(s) > 20 and s.follow_rate.std() > 0:
        y = s.follow_rate.to_numpy(float)
        g = s.problem_id.to_numpy()
        dv, te = (s.split == "dev").to_numpy(), (s.split == "test").to_numpy()
        r = sweep_and_test(path_c, s.row.to_numpy(), y, g, dv, te, layers, pc["alpha_grid"], pc["folds"], "reg", nj)
        cov = covariate_reg(_f(s, ALL_COV), y, dv)
        R["probes"]["P1r_follow_rate"] = {
            "curve": r["curve"], "best_layer": r["best_layer"],
            "probe": regression_report(r["pred_all"][te], y[te], g[te], nb, seed),
            "all_covariates": regression_report(cov[te], y[te], g[te], nb, seed)}

    # ---------------------------------------------------------------- P2: value tracking
    m2 = valid & df.is_int.to_numpy() & (df.true_last != df.new_last).to_numpy()
    s2 = df[m2].reset_index(drop=True)
    if len(s2) > 30:
        R["probes"]["P2_value_tracking"] = _value_tracking(rd, s2, layers, seed)

    # ---------------------------------------------------------------- P3: silent vs visible recovery
    m3 = valid & df.on_path.to_numpy() & (df.outcome == "RECOVER").to_numpy()
    s3 = df[m3].reset_index(drop=True)
    y3 = (s3.recovery_type == "silent").astype(int).to_numpy()
    P3 = {"n": int(len(s3)), "silent_rate": float(y3.mean()) if len(y3) else None}
    if len(s3) >= 30:
        g3 = s3.problem_id.to_numpy()
        dv, te = (s3.split == "dev").to_numpy(), (s3.split == "test").to_numpy()
        if len(np.unique(y3[dv])) == 2 and len(np.unique(y3[te])) == 2:
            r = sweep_and_test(path_c, s3.row.to_numpy(), y3, g3, dv, te, layers, pc["C_grid"], pc["folds"], "clf", nj)
            sc = covariate_clf(_f(s3, ALL_COV), y3, dv, te)
            P3.update({"curve": r["curve"], "best_layer": r["best_layer"],
                       "probe": classification_report(r["pred_all"][te], y3[te], g3[te], nb, seed),
                       "all_covariates": classification_report(sc[te], y3[te], g3[te], nb, seed)})
    R["probes"]["P3_silent_recovery"] = P3

    # ---------------------------------------------------------------- P4: contamination horizon
    m4 = valid & df.on_path.to_numpy()
    s4 = df[m4].reset_index(drop=True)
    y4 = np.log1p(s4.horizon.to_numpy(float))
    if len(s4) > 20 and y4.std() > 0:
        g4 = s4.problem_id.to_numpy()
        dv, te = (s4.split == "dev").to_numpy(), (s4.split == "test").to_numpy()
        r = sweep_and_test(path_c, s4.row.to_numpy(), y4, g4, dv, te, layers, pc["alpha_grid"], pc["folds"], "reg", nj)
        cov = covariate_reg(_f(s4, ALL_COV), y4, dv)
        R["probes"]["P4_horizon"] = {
            "curve": r["curve"], "best_layer": r["best_layer"], "mean_horizon": float(s4.horizon.mean()),
            "probe": regression_report(r["pred_all"][te], y4[te], g4[te], nb, seed),
            "all_covariates": regression_report(cov[te], y4[te], g4[te], nb, seed)}

    # ---------------------------------------------------------------- P5: prompt-time susceptibility
    pids = [r["problem_id"] for r in read_jsonl(rd / "prompt_ids.jsonl")]
    on = df[valid & df.on_path.to_numpy() & df.follow_rate.notna().to_numpy()]
    per = on.groupby("problem_id").agg(follow=("follow_rate", "mean"), pass_rate=("pass_rate", "first"),
                                       is_math=("is_math", "first"), level=("level", "first"),
                                       split=("split", "first")).reset_index()
    per["row"] = [pids.index(p) for p in per.problem_id]
    if len(per) > 30 and per.follow.std() > 0:
        y = per.follow.to_numpy(float)
        g = per.problem_id.to_numpy()
        dv, te = (per.split == "dev").to_numpy(), (per.split == "test").to_numpy()
        r = sweep_and_test(str(rd / "acts_prompt.npy"), per.row.to_numpy(), y, g, dv, te, layers,
                           pc["alpha_grid"], pc["folds"], "reg", nj)
        cov = covariate_reg(_f(per, DIFFICULTY), y, dv)
        R["probes"]["P5_susceptibility"] = {
            "curve": r["curve"], "best_layer": r["best_layer"], "n_problems": int(len(per)),
            "probe": regression_report(r["pred_all"][te], y[te], g[te], nb, seed),
            "difficulty_covariates": regression_report(cov[te], y[te], g[te], nb, seed)}

    R["verdict"] = verdict(R)
    with open(rd / "results.json", "w") as f:
        json.dump(R, f, indent=2, default=_jsonable)
    df.drop(columns=["sample_outcomes"]).to_csv(rd / "table.csv", index=False)
    print(json.dumps(R["verdict"], indent=2))
    return R


def _value_tracking(rd, s2, layers, seed):
    """Train a 10-way last-digit probe on CLEAN runs (written digit == true digit), then ask what it
    reads on CORRUPTED runs: the true digit (model still carries the right value) or the injected
    digit (model took the error on board). Layer 0 trivially reads the written (injected) digit, so
    the interesting part is where, if anywhere, the true digit takes over."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    A_k = np.load(rd / "acts_clean.npy", mmap_mode="r")
    A_c = np.load(rd / "acts_corrupt.npy", mmap_mode="r")
    rows = s2.row.to_numpy()
    dv, te = (s2.split == "dev").to_numpy(), (s2.split == "test").to_numpy()
    yt, yn = s2.true_last.to_numpy(), s2.new_last.to_numpy()
    out_curve = []
    for l in layers:
        Xk = A_k[rows, l, :].astype(np.float32)
        Xc = A_c[rows, l, :].astype(np.float32)
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000)).fit(Xk[dv], yt[dv])
        classes = list(m.classes_)
        P = m.predict_proba(Xc[te])
        idx_t = np.array([classes.index(d) if d in classes else -1 for d in yt[te]])
        idx_n = np.array([classes.index(d) if d in classes else -1 for d in yn[te]])
        ok = (idx_t >= 0) & (idx_n >= 0)
        pt = np.where(ok, P[np.arange(len(P)), np.clip(idx_t, 0, None)], np.nan)
        pn = np.where(ok, P[np.arange(len(P)), np.clip(idx_n, 0, None)], np.nan)
        pred = np.array(classes)[P.argmax(1)]
        margin = np.log(pt + 1e-9) - np.log(pn + 1e-9)
        outc = s2.outcome.to_numpy()[te]
        rec = {"layer": l,
               "clean_test_acc": float((m.predict(Xk[te]) == yt[te]).mean()),
               "corrupt_reads_true": float((pred == yt[te]).mean()),
               "corrupt_reads_injected": float((pred == yn[te]).mean())}
        for o in ("FOLLOW", "RECOVER"):
            mm = (outc == o) & ok
            rec[f"reads_true_{o}"] = float((pred[mm] == yt[te][mm]).mean()) if mm.any() else None
            rec[f"margin_{o}"] = float(np.nanmean(margin[mm])) if mm.any() else None
        mm = np.isin(outc, ["FOLLOW", "RECOVER"]) & ok
        if mm.sum() > 5:
            # does "the model still carries the true value" predict recovery?
            rec["margin_auroc_recover"] = safe_auc((outc[mm] == "RECOVER").astype(int), margin[mm])
        out_curve.append(rec)
    return {"n": int(len(s2)), "n_test": int(te.sum()), "curve": out_curve}


def verdict(R: dict) -> dict:
    """Apply the pre-registered decision rule from DIRECTION.md."""
    p1 = R["probes"].get("P1_follow_vs_recover", {})
    if "probe" not in p1:
        return {"H1": None, "H2": None, "summary": "P1 not run (label distribution too thin). See counts."}
    a = p1["probe"]["auroc"]
    w = p1["probe"]["within_auroc"]
    d = p1["delta_auroc_probe_over_covariates"]
    h1 = a >= 0.70
    h2 = (w >= 0.65) and (d["value"] >= 0.05) and (d["ci"][0] > 0)
    if h1 and h2:
        s = "Signal beyond difficulty and surprisal (main result)."
    elif h1:
        s = "Signal exists but does not survive difficulty controls (difficulty-probe finding)."
    else:
        s = "No usable signal at the injection point."
    return {"H1_auroc_ge_0.70": h1, "H2_within_and_delta": h2, "auroc": a, "within_auroc": w,
            "n_mixed_problems": p1["probe"]["n_mixed_problems"], "delta_auroc": d["value"],
            "delta_ci": d["ci"], "summary": s}


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
