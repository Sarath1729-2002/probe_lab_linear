"""Main study: PRE-GENERATION READOUT OF HINT ADOPTION (sycophancy susceptibility) in Qwen2.5-7B-Instruct on GSM8K.

Question: does the residual stream at the last prompt token (before any output) linearly encode how likely the model is to adopt a
planted WRONG answer, beyond problem difficulty and beyond what is visible in the prompt text?  Seven graded framings of the same
wrong hint; labels from 16 sampled solutions per (problem, framing).

Metrics (declared before the run):
  M1  R2 and Spearman of the best-layer ridge probe on held-out problems           (layer + lambda chosen on VALIDATION only)
  M2  delta-R2 vs the best baseline, paired bootstrap 95% CI over test problems     (baseline ladder below, best-on-TEST: conservative)
  M3  AUROC for 'adopts' (follow rate >= 0.25)
  M4  leave-one-framing-out R2 (train on 6 framings, test on the 7th)
  M5  causal: dose-response of the adoption rate to steering along the probe direction vs random directions
  Splits: iid (by problem) and shift (train <=3-step problems, test >=5-step).
  A target is 'supported' if M2's CI lower bound > 0 on BOTH splits and M1 R2 > 0.05.
Baseline ladder: B0 mean | Bq question length/numbers | Bbow hashed words | Bsteps(oracle) reference-solution step count |
  Bdiff(oracle) + measured clean pass-rate / answer entropy / forced log P(gold) | Bhint(visible) hint-number features |
  Bhint+gold(oracle) | H4,H8 the model's own early-layer state of the same prompt | Bother(oracle) follow rate in the OTHER framings.
"""
import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from . import analysis as A
from . import ridge
from .data import read_problems
from .prompts import CONDS, FRAMINGS

ADOPT = 0.25


# ----------------------------------------------------------------------------- data
def load_states_multi(run_dir: Path, conds) -> Dict[str, dict]:
    import glob
    per: Dict[str, list] = {}
    for f in sorted(glob.glob(str(Path(run_dir) / "states" / "x*.npz"))):
        if ".tmp" in f:
            continue
        z = np.load(f)
        for key in z.files:
            if key.endswith("_states"):
                c = key[:-7]
                if c in conds:
                    per.setdefault(c, []).append((z["ids"], {k: z[f"{c}_{k}"] for k in ("states", "lp_gold", "lp_hint", "ntok")}))
    out = {}
    for c, lst in per.items():
        ids = np.concatenate([i for i, _ in lst])
        o = np.argsort(ids)
        out[c] = {"ids": ids[o], **{k: np.concatenate([d[k] for _, d in lst])[o] for k in ("states", "lp_gold", "lp_hint", "ntok")}}
    return out


def build(run_dir: Path, framings=FRAMINGS):
    conds = ("clean",) + tuple(framings)
    probs = {p["id"]: p for p in read_problems(run_dir)}
    S = A.load_samples(run_dir)
    st = load_states_multi(run_dir, conds)
    missing = [c for c in conds if c not in st]
    if missing:
        raise RuntimeError(f"no states for {missing}; run the states stage for these conditions first")
    ids = set(st["clean"]["ids"].tolist())
    for c in conds:
        ids &= set(st[c]["ids"].tolist())
        ids &= {i for (i, cc) in S if cc == c}
    ids = np.array(sorted(ids))
    rows, sel = [], {}
    for c in conds:
        sel[c] = np.searchsorted(st[c]["ids"], ids)
    for k, i in enumerate(ids):
        p = probs[int(i)]
        r = {"id": int(i), "split": p["split"], "steps": p["steps"], "gold": p["gold"], "hint": p["hint"]}
        c0 = S[(int(i), "clean")]
        r.update(p_clean=c0["n_correct"] / c0["n"], ent_clean=c0["ent"], n_samples=c0["n"])
        for f in framings:
            s = S[(int(i), f)]
            r[f"follow_{f}"] = s["n_follow"] / s["n"]
            r[f"acc_{f}"] = s["n_correct"] / s["n"]
            r[f"none_{f}"] = s["n_none"] / s["n"]
        rows.append(r)
    df = pd.DataFrame(rows)
    df["lpg_clean"] = np.clip(st["clean"]["lp_gold"][sel["clean"]], -30, 0)
    for f in framings:
        df[f"gap_{f}"] = np.clip(st[f]["lp_hint"][sel[f]] - st[f]["lp_gold"][sel[f]], -20, 20)
    df["S"] = df[[f"follow_{f}" for f in framings]].mean(axis=1)
    df["Sgap"] = df[[f"gap_{f}" for f in framings]].mean(axis=1)
    df["ntok"] = st["clean"]["ntok"][sel["clean"]]
    df["_q"] = [probs[int(i)]["question"] for i in ids]
    sub = {f"{c}_states": st[c]["states"][sel[c]] for c in conds}
    return df, sub


def ceilings(df: pd.DataFrame, framings) -> pd.DataFrame:
    """R2 ceiling implied by binomial sampling noise of the labels (n samples per problem)."""
    n = df.n_samples.values.astype(float)
    rows = []
    for f in framings:
        p = df[f"follow_{f}"].values
        noise = np.mean(p * (1 - p) / np.maximum(n - 1, 1))
        rows.append({"label": f"follow_{f}", "mean": p.mean(), "var": p.var(), "noise_var": noise, "r2_ceiling": 1 - noise / max(p.var(), 1e-12)})
    P = np.stack([df[f"follow_{f}"].values for f in framings], 1)
    noise = np.mean((P * (1 - P) / np.maximum(n[:, None] - 1, 1)).sum(1)) / len(framings) ** 2
    rows.append({"label": "S (mean over framings)", "mean": df.S.mean(), "var": df.S.var(), "noise_var": noise, "r2_ceiling": 1 - noise / max(df.S.var(), 1e-12)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- baselines
def ladder(df: pd.DataFrame, sub: Dict[str, np.ndarray], cond: str, with_hint: bool, early=(4, 8), other: np.ndarray = None) -> Dict[str, np.ndarray]:
    b = A.baseline_sets(df)
    b["Bdiff(oracle)"] = np.concatenate([b["Bsteps(oracle)"], df[["p_clean", "ent_clean", "lpg_clean"]].values.astype(np.float32)], 1)
    if with_hint:
        b.update(A.hint_baselines(df))
        if other is not None:
            b["Bother(oracle)"] = np.concatenate([other, other ** 2, b["Bsteps(oracle)"]], 1)
    for l in early:
        if l < sub[f"{cond}_states"].shape[1]:
            b[f"H{l}(early-layer)"] = sub[f"{cond}_states"][:, l, :].astype(np.float32)
    return b


def other_framing_rate(df: pd.DataFrame, f: str, framings) -> np.ndarray:
    return df[[f"follow_{g}" for g in framings if g != f]].mean(axis=1).values.astype(np.float32)[:, None]


# ----------------------------------------------------------------------------- analyses
def layer_ci(y, allpred, target, layers, B=200, seed=0):
    rng = np.random.RandomState(seed)
    idx = [rng.randint(0, len(y), len(y)) for _ in range(B)]
    rows = []
    for l in layers:
        p = allpred[(l, target)]
        vals = [A.r2(y[i], p[i]) for i in idx]
        rows.append({"layer": l, "r2": A.r2(y, p), "lo": float(np.nanpercentile(vals, 2.5)), "hi": float(np.nanpercentile(vals, 97.5))})
    return pd.DataFrame(rows)


def susceptibility(df, sub, layers, regimes, log):
    """A: from the CLEAN prompt-end state (no hint yet): latent susceptibility S, plus the E1 knowledge-boundary labels."""
    targets = ["S", "Sgap", "p_clean", "ent_clean", "lpg_clean"]
    base = ladder(df, sub, "clean", with_hint=False)
    res, cur = A.run_targets(df, sub, "clean", "clean", targets, targets, regimes, base, layers, "A clean-state")
    curves = {}
    for regime in regimes:
        fit, val, test = A.make_splits(df, regime)
        ap = {}
        A.sweep(lambda l: sub["clean_states"][:, l, :].astype(np.float32), lambda l: sub["clean_states"][:, l, :].astype(np.float32),
                layers, df[["S"]].values.astype(float), df[["S"]].values.astype(float), fit, val, test, ["S"], allpred=ap)
        curves[regime] = layer_ci(df.S.values[test], ap, "S", layers)
    return res, cur, curves


def per_framing(df, sub, framings, layers, regimes, log):
    out, curves = [], []
    for f in framings:
        oth = other_framing_rate(df, f, framings)
        base = ladder(df, sub, f, with_hint=True, other=oth)
        tg = [f"follow_{f}", f"gap_{f}"]
        res, cur = A.run_targets(df, sub, f, f, tg, tg, regimes, base, layers, f"B framing={f}")
        for r in res:
            r["framing"] = f
            if r["target"].startswith("follow"):
                y, p = r["_y"], r["_pred"]
                r["base_rate_adopt"] = float((y >= ADOPT).mean())
                r["auroc_adopt"] = A.auroc(y >= ADOPT, p) if 0.01 < r["base_rate_adopt"] < 0.99 else float("nan")
        out += res
        curves.append(cur.assign(framing=f))
        log(f"[per-framing] {f} done")
    return out, pd.concat(curves)


def lofo(df, sub, framings, layers, log, regime="iid"):
    """C: train on the hinted states of 6 framings (pooled rows), test on the 7th, on held-out problems."""
    fit, val, test = A.make_splits(df, regime)
    steps = df.steps.values.astype(np.float32)
    hb = A.hint_baselines(df)
    bsteps = A.baseline_sets(df)["Bsteps(oracle)"]
    rows = []
    for f in framings:
        tr = [g for g in framings if g != f]
        def pool(idx, getter):
            return np.concatenate([getter(g)[idx] for g in tr], 0)
        yfit = np.concatenate([df[f"follow_{g}"].values[fit] for g in tr])[:, None]
        yval = np.concatenate([df[f"follow_{g}"].values[val] for g in tr])[:, None]
        ytest = df[f"follow_{f}"].values[test][:, None]
        best = None
        for l in layers:
            X = lambda g: sub[f"{g}_states"][:, l, :].astype(np.float32)       # noqa: E731
            Xf, Xv = pool(fit, X), pool(val, X)
            Xt = X(f)[test]
            r = A.ridge_eval(Xf, Xv, Xt, yfit, yval, ytest)[0]
            if best is None or r["val_r2"] > best["val_r2"]:
                best = {**r, "layer": l}
        # baselines on the same pooled rows
        def feats(g, others):
            oth = df[[f"follow_{h}" for h in others]].mean(axis=1).values.astype(np.float32)[:, None]
            h8 = sub[f"{g}_states"][:, min(8, sub[f"{g}_states"].shape[1] - 1), :].astype(np.float32)
            hv = hb["Bhint(visible)"]
            return {"Bhint(visible)": hv, "Bother(oracle)": np.concatenate([oth, oth ** 2, bsteps], 1), "H8(early-layer)": h8,
                    "Bhint+gold+difficulty(oracle)": hb["Bhint+gold+difficulty(oracle)"]}
        Ftr = {g: feats(g, [h for h in tr if h != g]) for g in tr}
        Fte = feats(f, tr)
        bl = {}
        for name in Fte:
            Xf = np.concatenate([Ftr[g][name][fit] for g in tr], 0)
            Xv = np.concatenate([Ftr[g][name][val] for g in tr], 0)
            bl[name] = A.ridge_eval(Xf, Xv, Fte[name][test], yfit, yval, ytest)[0]
        bn = max(bl, key=lambda k: -1e9 if np.isnan(bl[k]["test_r2"]) else bl[k]["test_r2"])
        y = ytest[:, 0]
        d, lo, hi = A.boot_delta(y, best["pred"], bl[bn]["pred"])
        rate = float((y >= ADOPT).mean())
        rows.append({"held_out": f, "probe_layer": best["layer"], "probe_r2": A.r2(y, best["pred"]), "spearman": A.spearman(y, best["pred"]),
                     "auroc_adopt": A.auroc(y >= ADOPT, best["pred"]) if 0.01 < rate < 0.99 else float("nan"), "base_rate": rate,
                     "best_baseline": bn, "baseline_r2": bl[bn]["test_r2"], "delta": d, "delta_lo": lo, "delta_hi": hi, **{f"{k}_r2": v["test_r2"] for k, v in bl.items()}})
        log(f"[lofo] held out {f}: r2={rows[-1]['probe_r2']:.3f} layer={best['layer']}")
    return pd.DataFrame(rows)


def dose_response(df, framings, B=1000, seed=0):
    rng = np.random.RandomState(seed)
    n = len(df)
    rows = []
    for f in framings:
        fo, ac = df[f"follow_{f}"].values, df[f"acc_{f}"].values
        bs = [fo[rng.randint(0, n, n)].mean() for _ in range(B)]
        rows.append({"framing": f, "follow_rate": fo.mean(), "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5),
                     "frac_problems_ever_adopt": float((fo > 0).mean()), "accuracy_under_hint": ac.mean(), "accuracy_clean": df.p_clean.mean(),
                     "parse_fail": df[f"none_{f}"].mean(), "mean_gap_nats": df[f"gap_{f}"].mean()})
    return pd.DataFrame(rows)


def calibration(res) -> pd.DataFrame:
    for r in res:
        if r["target"] == "S" and r["regime"] == "iid":
            y, p = r["_y"], r["_pred"]
            q = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False)
            return pd.DataFrame({"decile": range(10), "pred_mean": [p[q == k].mean() for k in range(10)], "observed_mean": [y[q == k].mean() for k in range(10)]})
    return pd.DataFrame()


def permutation_control(df, sub, layer, reps=10, regime="iid"):
    fit, val, test = A.make_splits(df, regime)
    X = sub["clean_states"][:, layer, :].astype(np.float32)
    y = df.S.values.astype(float)
    rng = np.random.RandomState(0)
    vals = []
    for _ in range(reps):
        yp = y.copy()
        yp[fit] = rng.permutation(y[fit])
        vals.append(A.ridge_eval(X[fit], X[val], X[test], yp[fit, None], y[val, None], y[test, None])[0]["test_r2"])
    return float(np.mean(vals)), float(np.std(vals))


def export_directions(df, sub, framings, layers, res_b, out: Path, top=2, regime="iid"):
    """unit directions (raw hidden space) of the val-best ridge probe for the `top` framings with the highest adoption rate."""
    rates = {f: df[f"follow_{f}"].mean() for f in framings}
    chosen = sorted(rates, key=rates.get, reverse=True)[:top]
    if "auth" in framings and "auth" not in chosen:
        chosen = chosen[:-1] + ["auth"] if top > 1 else ["auth"]
    fit, val, _ = A.make_splits(df, regime)
    pack, meta = {}, {}
    for f in chosen:
        layer = next(r["probe_layer"] for r in res_b if r["framing"] == f and r["target"] == f"follow_{f}" and r["regime"] == regime)
        X = A._t(sub[f"{f}_states"][:, layer, :].astype(np.float32))
        y = df[f"follow_{f}"].values.astype(float)[:, None]
        fitm = ridge.fit_layer(X[fit], y[fit])
        pv = fitm.predict(X[val], 0)
        ri = ridge.pick_ratio(pv, y[val, 0])
        w = (fitm.W[0][ri] / fitm.sd).cpu().numpy()
        nrm = float(np.linalg.norm(w))
        degenerate = not np.isfinite(nrm) or nrm < 1e-10          # e.g. a framing nobody adopts -> no direction to steer along
        if degenerate:
            w = np.random.RandomState(0).randn(len(w)).astype(np.float32)
            nrm = float(np.linalg.norm(w))
        pack[f] = (w / nrm).astype(np.float32)
        meta[f] = {"layer": int(layer), "follow_rate": float(rates[f]), "degenerate": bool(degenerate)}
    np.savez(out / "directions.npz", **pack)
    (out / "directions.json").write_text(json.dumps(meta, indent=1))
    return meta


# ----------------------------------------------------------------------------- driver
def compute(run_dir: Path, layers=None, regimes=("iid", "shift"), framings=FRAMINGS, log=print):
    run_dir = Path(run_dir)
    out = run_dir / "analysis_v3"
    out.mkdir(exist_ok=True)
    df, sub = build(run_dir, framings)
    L = sub["clean_states"].shape[1]
    layers = layers or list(range(1, L))
    log(f"n={len(df)} problems, layers={layers[0]}..{layers[-1]}")
    res_a, cur_a, curves_a = susceptibility(df, sub, layers, regimes, log)
    res_b, cur_b = per_framing(df, sub, framings, layers, regimes, log)
    lo = lofo(df, sub, framings, layers, log)
    dose = dose_response(df, framings)
    ceil = ceilings(df, framings)
    cal = calibration(res_a)
    bestS = next((r["probe_layer"] for r in res_a if r["target"] == "S" and r["regime"] == "iid"), layers[-1])
    perm = permutation_control(df, sub, bestS)
    meta = export_directions(df, sub, framings, layers, res_b, out)
    strip = lambda rs: pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rs])      # noqa: E731
    pack = {"a": strip(res_a), "b": strip(res_b), "lofo": lo, "dose": dose, "ceil": ceil, "cal": cal, "perm": perm, "curves_a": curves_a,
            "cur_a": cur_a, "cur_b": cur_b, "n": len(df), "directions": meta, "framings": list(framings)}
    with open(out / "main_results.pkl", "wb") as fh:
        pickle.dump(pack, fh)
    for k in ("a", "b", "lofo", "dose", "ceil", "cal"):
        pack[k].to_csv(out / f"{k}.csv", index=False)
    cur_a.to_csv(out / "layer_curve_a.csv", index=False)
    cur_b.to_csv(out / "layer_curve_b.csv", index=False)
    log(f"saved {out}")
    return pack
