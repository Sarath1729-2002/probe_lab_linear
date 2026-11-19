"""Two kill tests raised by review (Nov 19, 2026).

K1, selectivity off the ceiling (GPU). Correct-information uptake on KNOWN items is already 95-98%, so the gate could not
   lower it much. The fair test is items the model gets WRONG on its own, with the CORRECT answer planted (user and
   document sentences, right role): accepting it is real updating. Conditions: none, gate, add_0.5 (same steering
   directions, layers and thresholds as steering v2; items from its test half). Pass = the gate keeps that uptake.

K2, is the probe real? (CPU, stored activations and scores). On the five matched sources, wrong role, known items:
   - pooled grouped out-of-fold AUROC of the mass-mean probe (replication)
   - WITHIN-ITEM AUROC: for each question, across its sources, does the probe rank the ones that flip above the ones
     that hold? Compared with a source-prior baseline (each source's training-fold defer rate), which ranks sources
     identically for every item: the probe has to beat it
   - split by reply style (bare answer line vs reasoned)
   - against the pressured-margin baseline: the model's own teacher-forced log P(correct) - log P(X) with the pressure
     present, i.e. what its output distribution already says. Reported alone, and as probe beyond margin (logistic on
     both vs margin alone, grouped CV)
   Pass = within-item AUROC clearly above the source prior and above 0.5, and the probe adds beyond the margin.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .common import auc, read_jsonl, write_jsonl


def kdir(cfg) -> Path:
    from .run import root
    d = root(cfg) / "kill"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ----------------------------------------------------------------------------- K1

def _unknown_sel(cfg):
    from .run import load, root
    from .steer2 import sdir
    items, V = load(cfg)
    Vd = {v["vid"]: v for v in V}
    G = {}
    for p in sorted(root(cfg).glob("*/gen.jsonl")):
        for r in read_jsonl(p):
            G[r["vid"]] = r
    S = json.load(open(sdir(cfg) / "dirs.json"))
    tset = set(S["train_items"])
    unk = [i for i in items if i not in tset and G.get(f"{i}|base|base", {}).get("outcome") not in (None, "GOLD", "SKIP")]
    rng = np.random.default_rng(cfg["seed"] + 41)
    n = cfg.get("kill", {}).get("n_per_bucket", 150)
    sel = []
    for b in sorted({items[i]["bucket"] for i in unk}):
        ks = sorted(rng.permutation(sorted(i for i in unk if items[i]["bucket"] == b))[:n].tolist())
        sel += [Vd[f"{i}|{f}|right"] for i in ks for f in ("m_user", "m_document") if f"{i}|{f}|right" in Vd]
    return items, sel


def stage_k1(cfg: dict) -> dict:
    from .common import render
    from .run import _grade
    from .steer2 import _Model, _Steer, _mt, sdir
    S = json.load(open(sdir(cfg) / "dirs.json"))
    Z = np.load(sdir(cfg) / "dirs.npz")
    C = json.load(open(sdir(cfg) / "calib.json"))
    items, sel = _unknown_sel(cfg)
    M = _Model(cfg)
    bs = cfg.get("s2", {}).get("batch", 48)
    for c in ("none", "gate", "add_0.5"):
        out = kdir(cfg) / f"k1_{c}.jsonl"
        old = read_jsonl(out) if out.exists() else []
        have = {r["vid"] for r in old}
        todo = [v for v in sel if v["vid"] not in have]
        hooks = []
        for k, (L, pl) in enumerate(zip(S["layers"], S["per_layer"])):
            if c == "gate":
                hooks.append((L, _Steer(Z["d"][k], "gate", tau=C["per_layer"][k]["tau"])))
            elif c == "add_0.5":
                hooks.append((L, _Steer(Z["d"][k], "add", shift=0.5 * pl["gap"])))
        rows = list(old)
        for b in sorted({v["bucket"] for v in todo}):
            vb = [v for v in todo if v["bucket"] == b]
            outs, _ = M.generate([render(M.tok, v)[0] for v in vb], _mt(cfg, b), bs, hooks)
            for v, t in zip(vb, outs):
                oc, a = _grade(items, v, t)
                rows.append({"vid": v["vid"], "cond": c, "text": t, "outcome": oc})
            write_jsonl(out, rows)
            print(f"[k1] {c} {b}: {len(vb)}", flush=True)
    return analyse_k1(cfg)


def analyse_k1(cfg) -> dict:
    from .honest_defer import _boot
    R = {c: {r["vid"]: r for r in read_jsonl(kdir(cfg) / f"k1_{c}.jsonl")} for c in ("none", "gate", "add_0.5")
         if (kdir(cfg) / f"k1_{c}.jsonl").exists()}
    out = {}
    for fam in ("m_user", "m_document", "both"):
        ks = [k for k in R["none"] if fam == "both" or f"|{fam}|" in k]
        base = np.array([R["none"][k]["outcome"] == "GOLD" for k in ks], float)
        out[fam] = {"n": len(ks), "uptake_none": float(base.mean())}
        for c in ("gate", "add_0.5"):
            if c in R:
                kk = [k for k in ks if k in R[c]]
                a = np.array([R[c][k]["outcome"] == "GOLD" for k in kk], float)
                b0 = np.array([R["none"][k]["outcome"] == "GOLD" for k in kk], float)
                out[fam][c] = {"uptake": float(a.mean()), "minus_none_points": float(100 * (a - b0).mean()),
                               "ci_points": [100 * x for x in _boot(a - b0, cfg["seed"])]}
    json.dump(out, open(kdir(cfg) / "k1.json", "w"), indent=1)
    for fam, m in out.items():
        print(f"[k1] {fam}: n {m['n']}, uptake with no steering {m['uptake_none']:.3f}; " +
              "; ".join(f"{c} {m[c]['uptake']:.3f} ({m[c]['minus_none_points']:+.1f} [{m[c]['ci_points'][0]:+.1f}, {m[c]['ci_points'][1]:+.1f}])"
                        for c in ("gate", "add_0.5") if c in m), flush=True)
    return out


# ----------------------------------------------------------------------------- K2

def _within_item_auc(y, s, g):
    """Share of (defer, hold) pairs from the SAME item ranked correctly (ties count half)."""
    num = den = 0.0
    for i in np.unique(g):
        m = g == i
        a, b = s[m & y], s[m & ~y]
        if len(a) and len(b):
            d = a[:, None] - b[None, :]
            num += (d > 0).sum() + 0.5 * (d == 0).sum()
            den += d.size
    return float(num / den) if den else None, int(den)


def _folds(g, k, seed):
    ug = np.unique(g)
    f = dict(zip(ug, np.random.default_rng(seed).integers(0, k, len(ug))))
    return np.array([f[x] for x in g])


def stage_k2(cfg: dict) -> dict:
    from sklearn.linear_model import LogisticRegression
    from .deference import _X, _load_all, _order, _rows
    from .design61 import is_bare
    from .honest_defer import DEF_FAMS
    from .run import bdir
    seed = cfg["seed"]
    items, D = _load_all(cfg)
    R = []
    for f in DEF_FAMS:
        R += [r + (f,) for r in _rows(D, f, ["wrong"]) if r[4] in ("WRONG", "GOLD")]
    R = _order(D, R)
    y = np.array([r[4] == "WRONG" for r in R])
    g = np.array([r[2] for r in R])
    fam = np.array([r[5] for r in R])
    vids = [f"{r[2]}|{r[5]}|wrong" for r in R]
    G, M = {}, {}
    for b in D:
        G.update(D[b]["G"])
        p = bdir(cfg, b) / "score.jsonl"
        if p.exists():
            for s in read_jsonl(p):
                M[s["vid"]] = s["margin"]
    bare = np.array([is_bare(G[v].get("text")) for v in vids])
    margin = np.array([M.get(v, np.nan) for v in vids], float)
    layers = next(d["acts"]["layers"] for d in D.values() if d["acts"])
    k = 5
    fo = _folds(g, k, seed)
    # source-prior baseline: each source's defer rate in the training folds
    prior = np.zeros(len(R))
    for j in range(k):
        tr, te = fo != j, fo == j
        for f in DEF_FAMS:
            m = fam == f
            prior[te & m] = y[tr & m].mean() if (tr & m).any() else y[tr].mean()
    per = []
    best = None
    for li, L in enumerate(layers):
        X = _X(D, R, 1, li)
        s = np.zeros(len(R))
        for j in range(k):
            tr, te = fo != j, fo == j
            d = X[tr & y].mean(0) - X[tr & ~y].mean(0)
            s[te] = X[te] @ d
        w, npairs = _within_item_auc(y, s, g)
        row = {"layer": L, "pooled": auc(y, s), "within_item": w, "pairs": npairs,
               "bare": auc(y[bare], s[bare]) if y[bare].min() != y[bare].max() else None,
               "reasoned": auc(y[~bare], s[~bare]) if (~bare).sum() and y[~bare].min() != y[~bare].max() else None}
        per.append(row)
        if best is None or row["pooled"] > per[best]["pooled"]:
            best, sbest = li, s.copy()
        print(f"[k2] layer {L}: pooled {row['pooled']:.3f}, within-item {w:.3f} ({npairs} pairs), "
              f"bare {row['bare']}, reasoned {row['reasoned']}", flush=True)
    s = sbest
    ok = ~np.isnan(margin)
    res = {"n_rows": len(R), "defer_rate": float(y.mean()), "share_bare": float(bare.mean()),
           "items_with_both_outcomes": int(sum(1 for i in np.unique(g) if y[g == i].any() and (~y[g == i]).any())),
           "per_layer": per, "best_layer": layers[best],
           "source_prior": {"pooled": auc(y, prior), "within_item": _within_item_auc(y, prior, g)[0]},
           "margin": {"pooled": auc(y[ok], -margin[ok]), "within_item": _within_item_auc(y[ok], -margin[ok], g[ok])[0],
                      "n": int(ok.sum())}}

    def cv_logit(F):
        p = np.zeros(ok.sum())
        yy, gg, ff = y[ok], g[ok], fo[ok]
        for j in range(k):
            tr, te = ff != j, ff == j
            mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-9
            lr = LogisticRegression(C=1.0, max_iter=1000).fit((F[tr] - mu) / sd, yy[tr])
            p[te] = lr.decision_function((F[te] - mu) / sd)
        return auc(yy, p), _within_item_auc(yy, p, gg)[0]
    m_only = cv_logit(margin[ok][:, None])
    both = cv_logit(np.stack([margin[ok], s[ok]], 1))
    res["probe_beyond_margin"] = {"margin_only": {"pooled": m_only[0], "within_item": m_only[1]},
                                  "margin_plus_probe": {"pooled": both[0], "within_item": both[1]}}
    res["at_best_layer"] = per[best]
    json.dump(res, open(kdir(cfg) / "k2.json", "w"), indent=1)
    b = per[best]
    print(f"[k2] best layer {layers[best]}: probe pooled {b['pooled']:.3f}, within-item {b['within_item']:.3f}; "
          f"source prior pooled {res['source_prior']['pooled']:.3f}, within-item {res['source_prior']['within_item']:.3f}; "
          f"margin pooled {res['margin']['pooled']:.3f}, within-item {res['margin']['within_item']:.3f}; "
          f"margin+probe pooled {both[0]:.3f} vs margin {m_only[0]:.3f}; bare {b['bare']}, reasoned {b['reasoned']}", flush=True)
    return res


def write_report_kill(cfg: dict) -> Path:
    rep = Path(os.path.expandvars(cfg["report_dir"])) / cfg["run_prefix"]
    rep.mkdir(parents=True, exist_ok=True)
    L = [f"# Kill tests — {cfg['model']}", ""]
    if (kdir(cfg) / "k1.json").exists():
        K1 = json.load(open(kdir(cfg) / "k1.json"))
        L += ["## K1: uptake of correct information on questions the model gets wrong by itself (held-out half)", "",
              "| Prompt | n | No steering | Gate (points vs none, 95% CI) | Always on, 0.5x gap |", "|---|---|---|---|---|"]
        for f, m in K1.items():
            cell = lambda c: f"{100 * m[c]['uptake']:.0f}% ({m[c]['minus_none_points']:+.1f} [{m[c]['ci_points'][0]:+.1f}, {m[c]['ci_points'][1]:+.1f}])" if c in m else "–"
            L.append(f"| {f} | {m['n']} | {100 * m['uptake_none']:.0f}% | {cell('gate')} | {cell('add_0.5')} |")
        L.append("")
    if (kdir(cfg) / "k2.json").exists():
        K = json.load(open(kdir(cfg) / "k2.json"))
        b = K["at_best_layer"]
        f = lambda x: "–" if x is None else f"{x:.3f}"
        L += ["## K2: is the probe real?", "",
              f"{K['n_rows']} rows (five sources, wrong role, known items), defer rate {K['defer_rate']:.2f}, bare replies {K['share_bare']:.2f}; "
              f"{K['items_with_both_outcomes']} items flip under some sources and hold under others. Best layer {K['best_layer']}.", "",
              "| Score | Pooled AUROC | Within-item AUROC |", "|---|---|---|",
              f"| Probe (mass-mean, grouped out-of-fold) | {f(b['pooled'])} | {f(b['within_item'])} |",
              f"| Source prior (each source's defer rate) | {f(K['source_prior']['pooled'])} | {f(K['source_prior']['within_item'])} |",
              f"| Pressured margin (model's own output preference) | {f(K['margin']['pooled'])} | {f(K['margin']['within_item'])} |",
              f"| Margin alone, logistic | {f(K['probe_beyond_margin']['margin_only']['pooled'])} | {f(K['probe_beyond_margin']['margin_only']['within_item'])} |",
              f"| Margin + probe, logistic | {f(K['probe_beyond_margin']['margin_plus_probe']['pooled'])} | {f(K['probe_beyond_margin']['margin_plus_probe']['within_item'])} |",
              "", f"By reply style at the best layer: bare answer {f(b['bare'])}, reasoned {f(b['reasoned'])}.", "",
              "| Layer | Pooled | Within-item | Bare | Reasoned |", "|---|---|---|---|---|"]
        L += [f"| {p['layer']} | {f(p['pooled'])} | {f(p['within_item'])} | {f(p['bare'])} | {f(p['reasoned'])} |" for p in K["per_layer"]]
    p = rep / "REPORT_kill.md"
    p.write_text("\n".join(L) + "\n")
    print(f"[kill_report] {p}")
    return p
