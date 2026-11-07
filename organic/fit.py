"""CPU analysis: behaviour per bucket, detection / outcome probes, flag-then-fade, cross-bucket transfer.

Known items only for every probe (base answer correct), so a wrong reply is not ignorance.
Detection probe: wrong vs right suggestion (or passage / working), mass-mean (Marks & Tegmark) with
the layer chosen by grouped CV on a training split and AUROC reported on held-out items; logistic at
the same layer as a check. Read at two positions: where the planted information ends, and the last
prompt token (just before the model starts answering)."""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from errprop.io import read_jsonl

from decide.common import MassMean, auc, best_layer, boot_ci, group_split, logistic, oof_scores

from .run import PROBE_FAMILIES, bdir, load, root

POS = ["info_end", "last_prompt_token"]


def _rate(xs, v):
    return float(np.mean([x == v for x in xs])) if xs else None


def _behaviour(its, G, known):
    arms = defaultdict(lambda: {"all": [], "known": []})
    for it in its:
        for key in [k for k in G if k.startswith(it["id"] + "|")]:
            _, fam, role = key.split("|")
            o = G[key]["outcome"]
            if o == "SKIP":
                continue
            arms[f"{fam}/{role}"]["all"].append(o)
            if known.get(it["id"]):
                arms[f"{fam}/{role}"]["known"].append(o)
    out = {}
    for a, d in arms.items():
        out[a] = {s: {"n": len(x), "gold": _rate(x, "GOLD"), "wrong": _rate(x, "WRONG"), "other": _rate(x, "OTHER")}
                  for s, x in d.items()}
    return out


def _probe(Xp, y, g, oc, known, L, seed, n_boot):
    k = known
    Xk, yk, gk, ok = Xp[k], y[k], g[k], oc[k]
    if min(yk.sum(), (1 - yk).sum()) < 30 or len(np.unique(gk)) < 20:
        return {"skipped": f"too few known rows (wrong {int(yk.sum())}, right {int((1 - yk).sum())})"}
    tr, te = group_split(gk, 0.25, seed)
    li, curve = best_layer(Xk[tr], yk[tr], gk[tr], MassMean, 3)
    X_l = Xk[:, li].astype(np.float32)
    mm = MassMean().fit(X_l[tr], yk[tr])
    s = mm.decision_function(X_l[te])
    yt, gt = yk[te], gk[te]
    r = {"layer": L[li], "layer_index": int(li), "curve_train_cv": curve, "n_known_rows": int(len(yk)),
         "detection_auroc": auc(yt, s), "detection_ci": boot_ci(lambda ix: auc(yt[ix], s[ix]), gt, n_boot, seed),
         "detection_auroc_logistic": auc(yt, logistic(0.1).fit(X_l[tr], yk[tr]).decision_function(X_l[te]))}
    full = MassMean().fit(X_l, yk)
    u = ~known
    if u.sum() >= 40 and len(np.unique(y[u])) == 2:   # plausibility vs knowledge: probe on items it doesn't know
        r["detection_auroc_unknown_items"] = auc(y[u], full.decision_function(Xp[u, li].astype(np.float32)))
        r["n_unknown_rows"] = int(u.sum())
    oof = oof_scores(X_l, yk, gk, MassMean, 5)
    thr = np.nanpercentile(oof[yk == 0], 95)          # 5% false alarms on the right-suggestion prompts
    adopt = (yk == 1) & (ok == "WRONG")
    hold = (yk == 1) & (ok == "GOLD")
    r["n_adopt"], r["n_hold"] = int(adopt.sum()), int(hold.sum())
    r["flagged_adopted"] = float((oof[adopt] > thr).mean()) if adopt.any() else None
    r["flagged_held"] = float((oof[hold] > thr).mean()) if hold.any() else None
    m = adopt | hold
    if adopt.sum() >= 15 and hold.sum() >= 15:
        z = adopt[m].astype(int)
        r["detection_score_predicts_adopt_auroc"] = auc(z, oof[m])     # < 0.5: more flagged -> less adoption
        Xo, go = Xk[m], gk[m]
        lo, co = best_layer(Xo, z, go, MassMean, 3)
        so = oof_scores(Xo[:, lo].astype(np.float32), z, go, MassMean, 5)
        ok_ = ~np.isnan(so)
        r["outcome_probe"] = {"layer": L[lo], "auroc_oof": auc(z[ok_], so[ok_]), "curve": co,
                              "note": "layer picked by grouped CV on the same rows (mildly optimistic)"}
    r["_dir"] = {"mu": full.mu, "sd": full.sd, "w": full.w, "layer": L[li]}
    return r


def _rows(b, cfg, fam, items, G, known):
    d = bdir(cfg, b)
    if not (d / "acts.npy").exists():
        return None
    idx = json.load(open(d / "acts_index.json"))
    A = np.load(d / "acts.npy", mmap_mode="r")
    sel = [i for i, v in enumerate(idx["vids"]) if v.split("|")[1] == fam and v.split("|")[2] in ("wrong", "right")]
    if not sel:
        return None
    vids = [idx["vids"][i] for i in sel]
    it = [v.split("|")[0] for v in vids]
    return {"A": A, "sel": np.array(sel), "layers": idx["layers"],
            "y": np.array([v.split("|")[2] == "wrong" for v in vids], int),
            "g": np.array(it), "oc": np.array([G[v]["outcome"] for v in vids]),
            "known": np.array([bool(known.get(i)) for i in it])}


def stage_fit(cfg: dict) -> dict:
    items, V = load(cfg)
    seed, nb = cfg["seed"], cfg["probes"]["n_boot"]
    buckets = sorted({it["bucket"] for it in items.values()})
    res = {"model": cfg["model"], "buckets": {}}
    dirs = {}
    for b in buckets:
        if not (bdir(cfg, b) / "gen.jsonl").exists():
            continue
        G = {r["vid"]: r for r in read_jsonl(bdir(cfg, b) / "gen.jsonl")}
        its = [it for it in items.values() if it["bucket"] == b]
        known = {it["id"]: G.get(f"{it['id']}|base|base", {}).get("outcome") == "GOLD" for it in its}
        R = {"n_items": len(its), "known_rate": float(np.mean(list(known.values()))),
             "behaviour": _behaviour(its, G, known), "by_sub": {}, "probes": {}}
        subs = defaultdict(list)
        for it in its:
            subs[it["sub"]].append(it)
        for s, si in sorted(subs.items()):
            kk = [it for it in si if known[it["id"]]]
            wr = [G[f"{it['id']}|sugg|wrong"]["outcome"] for it in kk if f"{it['id']}|sugg|wrong" in G]
            R["by_sub"][s] = {"n": len(si), "known_rate": len(kk) / max(1, len(si)),
                              "adopt_rate_known": _rate(wr, "WRONG"), "hold_rate_known": _rate(wr, "GOLD"),
                              "n_known_wrong_arm": len(wr)}
        for fam in PROBE_FAMILIES:
            rw = _rows(b, cfg, fam, items, G, known)
            if rw is None:
                continue
            R["probes"][fam] = {}
            for p, pos in enumerate(POS):
                Xp = np.asarray(rw["A"][rw["sel"], p])
                r = _probe(Xp, rw["y"], rw["g"], rw["oc"], rw["known"], rw["layers"], seed, nb)
                d = r.pop("_dir", None)
                if d is not None and fam == "sugg" and p == 0:
                    dirs[b] = d
                R["probes"][fam][pos] = r
            print(f"[fit] {b}/{fam}: " + "  ".join(
                f"{pos} det {R['probes'][fam][pos].get('detection_auroc', float('nan')):.3f}" for pos in POS), flush=True)
        res["buckets"][b] = R
    res["transfer"] = _transfer(cfg, items, buckets, res)
    np.savez(root(cfg) / "directions.npz", **{f"{b}__{k}": np.asarray(v) for b, d in dirs.items() for k, v in d.items()})
    json.dump(res, open(root(cfg) / "results.json", "w"), indent=1, default=float)
    print(f"[fit] wrote {root(cfg) / 'results.json'}")
    return res


def _transfer(cfg, items, buckets, res):
    """Detection probe (sugg family, info_end) trained on one bucket, tested on every other at the
    training bucket's layer; plus leave-one-bucket-out with all other buckets pooled."""
    data = {}
    for b in buckets:
        if not (bdir(cfg, b) / "gen.jsonl").exists():
            continue
        G = {r["vid"]: r for r in read_jsonl(bdir(cfg, b) / "gen.jsonl")}
        known = {it["id"]: G.get(f"{it['id']}|base|base", {}).get("outcome") == "GOLD"
                 for it in items.values() if it["bucket"] == b}
        rw = _rows(b, cfg, "sugg", items, G, known)
        pr = res["buckets"].get(b, {}).get("probes", {}).get("sugg", {}).get("info_end", {})
        if rw is None or "layer_index" not in pr:
            continue
        k = rw["known"]
        data[b] = {"A": rw["A"], "sel": rw["sel"][k], "y": rw["y"][k], "li": pr["layer_index"]}
    bs = sorted(data)
    if len(bs) < 2:
        return {"skipped": "fewer than two buckets with a sugg probe"}
    M = [[None] * len(bs) for _ in bs]
    for i, a in enumerate(bs):
        li = data[a]["li"]
        mm = MassMean().fit(np.asarray(data[a]["A"][data[a]["sel"], 0, li], np.float32), data[a]["y"])
        for j, t in enumerate(bs):
            if i == j:
                M[i][j] = res["buckets"][a]["probes"]["sugg"]["info_end"]["detection_auroc"]
                continue
            M[i][j] = auc(data[t]["y"], mm.decision_function(np.asarray(data[t]["A"][data[t]["sel"], 0, li], np.float32)))
    li_c = int(np.median([data[b]["li"] for b in bs]))
    loo = {}
    for t in bs:
        X = np.concatenate([np.asarray(data[b]["A"][data[b]["sel"], 0, li_c], np.float32) for b in bs if b != t])
        y = np.concatenate([data[b]["y"] for b in bs if b != t])
        mm = MassMean().fit(X, y)
        loo[t] = auc(data[t]["y"], mm.decision_function(np.asarray(data[t]["A"][data[t]["sel"], 0, li_c], np.float32)))
    return {"buckets": bs, "matrix_train_row_test_col": M, "leave_one_bucket_out": loo,
            "loo_layer": json.load(open(bdir(cfg, bs[0]) / "acts_index.json"))["layers"][li_c]}
