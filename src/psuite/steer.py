"""THREAD 3 - is the length direction CAUSAL?  Reading (probe) vs writing (intervention).

prep     fit the prompt-end total-length probe (label logT, finished solutions, GSM8K, cross-validated layer) on the fit+val problems; its weight vector (back in raw space) is the PROBE direction.
         Controls, all norm-matched: a MEAN-DIFFERENCE direction (long minus short solutions), and random Gaussian directions.
run      for every condition add  alpha * s * unit_direction  to the residual stream at the chosen layer (all positions, prompt and generation), greedy-decode the TEST problems and record
         tokens, finished, answer, correctness.   s = std of the probe projection over problems, so alpha is in 'standard deviations of the probe'.  Resumable per (condition, shard).
analyze  dose-response: slope of per-problem log tokens against alpha (bootstrap over problems) for the probe direction vs the controls; accuracy as the damage check.
"""
import json
import zlib
from pathlib import Path
from typing import Dict

import numpy as np

from .hf_util import generate_ids
from .prompts import chat_prompt, plain_text
from .sched import Budget

ALPHAS = (-4.0, -2.0, -1.0, 1.0, 2.0, 4.0)
RAND_ALPHAS = (-4.0, -2.0, 2.0, 4.0)
N_RAND = 8


def conditions():
    c = [("base", None, 0.0)]
    for fam in ("probe", "diff"):
        c += [(f"{fam}_a{a:+g}", fam, a) for a in ALPHAS]
    for j in range(N_RAND):
        c += [(f"rand{j}_a{a:+g}", f"rand{j}", a) for a in RAND_ALPHAS]
    return c


class Steer:
    """forward hook adding `vec` to the output of decoder layer `layer` (= hidden_states[layer])."""

    def __init__(self, model, layer: int, vec):
        import torch
        self.h = None
        self.vec = None if vec is None else torch.as_tensor(vec, dtype=next(model.parameters()).dtype, device=next(model.parameters()).device)
        self.model, self.layer = model, layer

    def __enter__(self):
        if self.vec is None:
            return self
        blk = getattr(self.model, "model", self.model).layers[self.layer - 1]

        def hook(_m, _i, out):
            if isinstance(out, tuple):
                return (out[0] + self.vec,) + tuple(out[1:])
            return out + self.vec
        self.h = blk.register_forward_hook(hook)
        return self

    def __exit__(self, *a):
        if self.h is not None:
            self.h.remove()


def prep(run_dir: Path, out_dir: Path, log=print) -> dict:
    from . import cv_eval, table
    from .probes import prompt_layers
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df, sub = table.build(run_dir, ["gsm8k"], need_states=True, names=("plain",), log=log)
    X = sub["plain"]
    L = X.shape[1]
    layers = [l for l in prompt_layers(L) if l < L - 1]          # the last hidden state is post-final-norm: a decoder-layer hook cannot write there
    y = df.logT.values.astype(np.float64)[:, None]
    pool = np.where(np.isin(df.split.values, ["fit", "val"]) & ~np.isnan(df.logT.values))[0]
    test = np.where((df.split.values == "test") & ~np.isnan(df.logT.values))[0]
    if len(pool) < 50 or len(test) < 10:
        raise RuntimeError(f"steer prep: only {len(pool)} fit/val and {len(test)} test GSM8K problems with a finished solution - cannot fit a length probe")
    res = cv_eval.cv_probe(lambda l: X[:, l, :].astype(np.float32), layers, y, pool, test, df.id.values)[0]
    layer = int(res["layer"])
    Xl = X[:, layer, :].astype(np.float32)
    from . import ridge
    fit = ridge.fit_layer(cv_eval._t(Xl[pool]), y[pool])
    ri = ridge.RATIOS.index(res["ratio"])
    w = (fit.W[0][ri] / fit.sd).cpu().numpy().astype(np.float64)               # prediction = ((x - mu)/sd) @ W + b = x @ (W/sd) + const
    probe = w / np.linalg.norm(w)
    s = float((Xl[pool] @ probe).std())
    q = np.nanpercentile(y[pool, 0], [25, 75])
    hi, lo = pool[y[pool, 0] >= q[1]], pool[y[pool, 0] <= q[0]]
    diff = Xl[hi].mean(0) - Xl[lo].mean(0)
    diff = diff / np.linalg.norm(diff)
    rng = np.random.RandomState(0)
    rand = rng.randn(N_RAND, Xl.shape[1])
    rand /= np.linalg.norm(rand, axis=1, keepdims=True)
    from .core import r2
    info = {"layer": layer, "cv_r2": res["cv_r2"], "test_r2": r2(y[test, 0], res["pred"]), "scale_s": s, "cos_probe_diff": float(probe @ diff),
            "cos_probe_rand_max": float(np.abs(rand @ probe).max()), "n_pool": int(len(pool)), "resid_norm_mean": float(np.linalg.norm(Xl[pool], axis=1).mean())}
    np.savez(out_dir / "dirs.npz", probe=probe, diff=diff, rand=rand, layer=layer, s=s)
    (out_dir / "prep.json").write_text(json.dumps(info, indent=1))
    log(f"steer prep: layer {layer}, probe CV R2 {res['cv_r2']:.3f}, test R2 {info['test_r2']:.3f}, s={s:.2f}, cos(probe,diff)={info['cos_probe_diff']:.3f}")
    return info


def run(problems, model, tok, out_dir: Path, shard=0, nshards=1, max_new=512, max_seqs=96, budget_s=0, log=print) -> int:
    out_dir = Path(out_dir)
    z = np.load(out_dir / "dirs.npz")
    layer, s = int(z["layer"]), float(z["s"])
    dirs = {"probe": z["probe"], "diff": z["diff"], **{f"rand{j}": z["rand"][j] for j in range(len(z["rand"]))}}
    mine = [p for p in problems if p["id"] % nshards == shard]
    prompts = [chat_prompt(tok, plain_text(p)) for p in mine]
    bud = Budget(budget_s)
    done = 0
    for name, fam, a in conditions():
        f = out_dir / f"r{shard:02d}_{name}.jsonl"
        if f.exists():
            continue
        if bud.spent():
            log(f"[steer shard {shard}] time budget reached - stopping cleanly (resubmit to continue)")
            break
        vec = None if fam is None else a * s * dirs[fam]
        with Steer(model, layer, vec):
            res = generate_ids(model, tok, prompts, max_new, 1, 0.0, 0.95, max_seqs)
        from .textutil import parse_answer
        tmp = f.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            for p, r in zip(mine, res):
                ids, fin = r[0]
                ans = parse_answer(tok.decode(ids, skip_special_tokens=True), p["kind"])
                fh.write(json.dumps({"id": p["id"], "cond": name, "ntok": len(ids), "fin": bool(fin), "ans": ans, "correct": float(ans == p["gold"])}) + "\n")
        tmp.rename(f)
        done += 1
        log(f"[steer shard {shard}] {name}: {len(mine)} problems")
    return done


def load(out_dir: Path) -> Dict[str, Dict[int, dict]]:
    out = {}
    for f in sorted(Path(out_dir).glob("r*_*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            out.setdefault(r["cond"], {})[r["id"]] = r
    return out


def analyze(out_dir: Path, B: int = 1000, seed: int = 0) -> dict:
    """-> {'table': DataFrame per condition, 'slopes': {family: (slope, lo, hi)}, 'status': ..., 'why': ...}."""
    import pandas as pd
    R = load(out_dir)
    if "base" not in R:
        return {"status": "n/a", "why": "no baseline condition finished", "table": pd.DataFrame(), "slopes": {}}
    ids = sorted(set.intersection(*[set(v) for v in R.values()]))
    if len(ids) < 30:
        return {"status": "n/a", "why": f"only {len(ids)} problems finished in every condition", "table": pd.DataFrame(), "slopes": {}}
    lt = {c: np.log(np.maximum([R[c][i]["ntok"] for i in ids], 1)) for c in R}
    ac = {c: np.array([R[c][i]["correct"] for i in ids]) for c in R}
    fin = {c: np.array([R[c][i]["fin"] for i in ids], dtype=float) for c in R}
    rows = [{"cond": c, "alpha": float(c.split("_a")[1]) if "_a" in c else 0.0, "family": c.split("_a")[0] if "_a" in c else "base", "mean_logT": lt[c].mean(),
             "d_logT_vs_base": (lt[c] - lt["base"]).mean(), "acc": ac[c].mean(), "d_acc_vs_base": (ac[c] - ac["base"]).mean(), "finished": fin[c].mean()} for c in sorted(R)]
    tab = pd.DataFrame(rows)
    fams = sorted({r["family"] for r in rows if r["family"] != "base"})
    rng = np.random.RandomState(seed)
    boots = [rng.randint(0, len(ids), len(ids)) for _ in range(B)]

    def slope_of(fam, ix=None):
        cs = [c for c in R if c.startswith(fam + "_a")] + ["base"]
        a = np.array([float(c.split("_a")[1]) if "_a" in c else 0.0 for c in cs])
        Y = np.stack([lt[c] for c in cs], 1)                           # [n, conditions]
        Y = Y if ix is None else Y[ix]
        am = a - a.mean()
        return float(((Y - Y.mean(1, keepdims=True)) @ am).mean() / (am @ am))      # OLS slope of per-problem log tokens on alpha

    slopes = {}
    for fam in fams:
        b = np.array([slope_of(fam, ix) for ix in boots])
        slopes[fam] = (slope_of(fam), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))
    rand = [abs(slopes[f][0]) for f in fams if f.startswith("rand")]
    pr = slopes.get("probe")
    if pr is None:
        return {"status": "n/a", "why": "probe family missing", "table": tab, "slopes": slopes}
    max_rand = max(rand) if rand else 0.0
    sel = tab[tab.family == "probe"]
    dmg = float(sel[sel.alpha.abs() <= 2].d_acc_vs_base.abs().max()) if len(sel) else float("nan")
    ok_sign = pr[1] > 0
    big = pr[0] > 1.5 * max_rand
    if ok_sign and big and dmg <= 0.05:
        st, why = "SUPPORTED", f"slope {pr[0]:.3f} log-tokens/sd [{pr[1]:.3f}, {pr[2]:.3f}], > 1.5x the largest random-direction slope ({max_rand:.3f}); accuracy within {dmg:.3f} for |alpha|<=2"
    elif ok_sign and big:
        st, why = "PARTIAL", f"slope {pr[0]:.3f} [{pr[1]:.3f}, {pr[2]:.3f}] beats random ({max_rand:.3f}) but accuracy moves by {dmg:.3f} for |alpha|<=2 (the effect may be damage, not control)"
    elif pr[1] > -0.02 and pr[2] < 0.02:
        st, why = "NEGATIVE", f"slope {pr[0]:.3f} [{pr[1]:.3f}, {pr[2]:.3f}] is inside +-0.02 log-tokens/sd: the probe direction does not move length"
    else:
        st, why = "INCONCLUSIVE", f"slope {pr[0]:.3f} [{pr[1]:.3f}, {pr[2]:.3f}] vs largest random {max_rand:.3f}"
    return {"status": st, "why": why, "table": tab, "slopes": slopes, "n": len(ids), "max_rand": max_rand, "acc_damage": dmg}
