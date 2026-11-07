"""Thread A: detection vs propagation.

A1 (CPU, existing paper_v1 runs). On the same injections, at the same token (the injected value):
    detection probe  = corrupted vs clean prefix
    R1 (Goel et al.) = the detection score, reused to predict FOLLOW vs RECOVER
    R2 (paper_v1)    = a probe trained on FOLLOW vs RECOVER itself
  If R1 is near chance while R2 is high, the published disagreement is a readout artefact.

A2 (GPU). The same corrupted working, but placed in the USER's message ("here is my working so far")
  instead of the model's own reply. Compares FOLLOW rates (self vs user attribution) and R1/R2 at the
  value token and at the last prompt token (Goel et al. read the last prompt token).
"""
from __future__ import annotations

import json
import random

import numpy as np

from errprop.io import read_jsonl, write_jsonl
from errprop.labels import extract_answer, outcome

from .common import (DEVICE, MassMean, Reader, auc, best_layer, boot_ci, chat, group_split, layer_list,
                     logistic, make_generator, run_dir)


def _paper_dir(cfg, tag):
    from pathlib import Path
    import os
    return Path(os.path.expandvars(cfg["paper_out_dir"])) / f"{cfg['paper_run_prefix']}-{tag}"


def _raw_dir(model, X_fit=None):
    """Unit direction in raw activation space (comparable across probes fitted on different rows)."""
    if isinstance(model, MassMean):
        w = model.w / model.sd
    else:
        w = model[-1].coef_[0] / model[0].scale_
    return w / (np.linalg.norm(w) + 1e-12)


def _readouts(Xc, Xk, y, groups, seed, layer_idx, C=0.1):
    """Xc / Xk: corrupted / clean states [N, L, D] for the same injections; y = FOLLOW (1) / RECOVER (0).
    Layer chosen by grouped CV of the detection probe (Goel et al.: same layer for both readouts),
    plus the outcome probe's own best layer."""
    dv, te = group_split(groups, 0.2, seed)
    Xd = np.concatenate([Xc, Xk])
    yd = np.r_[np.ones(len(Xc)), np.zeros(len(Xk))].astype(int)
    gd = np.r_[groups, groups]
    dvd = np.r_[dv, dv]
    l_det, curve_det = best_layer(Xd[dvd], yd[dvd], gd[dvd], lambda: logistic(C))
    l_out, curve_out = best_layer(Xc[dv], y[dv], groups[dv], lambda: logistic(C))
    out = {"layers": layer_idx, "curve_detection": curve_det, "curve_outcome": curve_out,
           "n_train": int(dv.sum()), "n_test": int(te.sum()), "follow_rate_test": float(y[te].mean())}
    for name, l in (("at_detection_layer", l_det), ("at_outcome_layer", l_out)):
        det = logistic(C).fit(Xd[dvd, l].astype(np.float32), yd[dvd])
        mm = MassMean().fit(Xd[dvd, l], yd[dvd])
        o = logistic(C).fit(Xc[dv, l].astype(np.float32), y[dv])
        s_det = det.decision_function(np.concatenate([Xc[te, l], Xk[te, l]]).astype(np.float32))
        r1 = det.decision_function(Xc[te, l].astype(np.float32))
        r1m = mm.decision_function(Xc[te, l])
        r2 = o.decision_function(Xc[te, l].astype(np.float32))
        yt, gt = y[te], groups[te]
        out[name] = {
            "layer": layer_idx[l],
            "detection_auroc": auc(np.r_[np.ones(te.sum()), np.zeros(te.sum())], s_det),
            "R1_detection_score_for_outcome": auc(yt, r1),
            "R1_massmean_for_outcome": auc(yt, r1m),
            "R2_outcome_probe": auc(yt, r2),
            "R2_minus_R1": auc(yt, r2) - auc(yt, r1),
            "R2_minus_R1_ci": boot_ci(lambda i: auc(yt[i], r2[i]) - auc(yt[i], r1[i]), gt, 300, seed),
            "cos_detection_outcome": float(abs(_raw_dir(det) @ _raw_dir(o))),
        }
    return out


# ============================================================================ A1

def stage_a1(cfg: dict, tag: str) -> None:
    from errprop.paper import build
    pd_ = _paper_dir(cfg, tag)
    rd = run_dir(cfg, tag)
    if not (pd_ / "acts_corrupt.npy").exists():
        json.dump({"skipped": f"no paper_v1 run at {pd_}"}, open(rd / "a1.json", "w"))
        print(f"[a1] skipped: no paper_v1 run at {pd_}")
        return
    df, _ = build(pd_)
    keep = df.kept.to_numpy()
    rows = np.where(keep)[0]
    Ac = np.load(pd_ / "acts_corrupt.npy", mmap_mode="r")
    Ak = np.load(pd_ / "acts_clean.npy", mmap_mode="r")
    L = layer_list(Ac.shape[1], cfg["probes"]["layer_step"] * 2)
    Xc = np.asarray(Ac[rows][:, L], np.float16)
    Xk = np.asarray(Ak[rows][:, L], np.float16)
    y = df.y.to_numpy(int)[rows]
    groups = df.group.astype(str).to_numpy()[rows]
    res = {"model": tag, "n_kept": int(len(rows)), "follow_rate": float(y.mean())}
    res["pooled"] = _readouts(Xc, Xk, y, groups, cfg["seed"], L)
    by_ds = {}
    for d in sorted(df.dataset[keep].unique()):
        m = (df.dataset.to_numpy()[rows] == d)
        if m.sum() >= 150 and 10 <= y[m].sum() <= m.sum() - 10:
            by_ds[d] = _readouts(Xc[m], Xk[m], y[m], groups[m], cfg["seed"], L)
    res["by_dataset"] = by_ds
    json.dump(res, open(rd / "a1.json", "w"), indent=1)
    p = res["pooled"]["at_detection_layer"]
    print(f"[a1] {tag}: detection {p['detection_auroc']:.3f}  R1 {p['R1_detection_score_for_outcome']:.3f}  "
          f"R2 {p['R2_outcome_probe']:.3f}")


# ============================================================================ A2

USER_TMPL = ("{question}\n\n{instruction}\n\nHere is my working so far:\n\n{working}\n\n"
             "Please continue from exactly where my working stops, and finish with a final line of the form "
             "`#### <number>`.")


def _a2_items(cfg, tag):
    from errprop.data import INSTRUCTION
    from errprop.paper import build
    pd_ = _paper_dir(cfg, tag)
    df, _ = build(pd_)
    injs = read_jsonl(pd_ / "injections.jsonl")
    q_of = {r["id"]: r["question"] for r in read_jsonl(pd_ / "clean.jsonl")}
    kept = [i for i, k in enumerate(df.kept.to_numpy()) if k]
    random.Random(cfg["seed"]).shuffle(kept)
    kept = sorted(kept[: cfg["a2"]["n_items"]])
    items = []
    for i in kept:
        inj = injs[i]
        if "</think>" in inj["prompt"] or "<think>" in inj["prefix_corrupt"]:
            continue                                   # reasoning traces don't fit a user message
        items.append({"row": i, "inj_id": inj["inj_id"], "problem_id": inj["problem_id"],
                      "group": inj.get("group", inj["problem_id"]), "gold": inj["gold"],
                      "propagated_answer": inj["propagated_answer"], "tol": inj.get("tol", 1e-4),
                      "new_text": inj["new_text"], "true_text": inj["true_text"],
                      "self_outcome": df.outcome.iloc[i],
                      "user_corrupt": USER_TMPL.format(question=q_of[inj["problem_id"]], instruction=INSTRUCTION,
                                                       working=inj["prefix_corrupt"]),
                      "user_clean": USER_TMPL.format(question=q_of[inj["problem_id"]], instruction=INSTRUCTION,
                                                     working=inj["prefix_clean"])})
    return items


def stage_a2_gen(cfg: dict, tag: str, model_name: str) -> None:
    rd = run_dir(cfg, tag)
    items = _a2_items(cfg, tag)
    if not items:
        print("[a2] no items")
        write_jsonl(rd / "a2_items.jsonl", [])
        return
    gen = make_generator(cfg, model_name)
    tok = gen.tokenizer
    prompts = [chat(tok, it["user_corrupt"]) for it in items] + [chat(tok, it["user_clean"]) for it in items]
    outs = gen.generate(prompts, 1, 0.0, cfg["a2"]["max_tokens"], cfg["seed"])
    n = len(items)
    for k, it in enumerate(items):
        it["prompt_corrupt"], it["prompt_clean"] = prompts[k], prompts[n + k]
        it["text_corrupt"], it["text_clean"] = outs[k][0], outs[n + k][0]
        ans_c, ans_k = extract_answer(outs[k][0]), extract_answer(outs[n + k][0])
        it["user_outcome"] = outcome(ans_c, it["gold"], it["propagated_answer"], it["tol"])
        it["clean_ok"] = ans_k is not None and abs(ans_k - it["gold"]) <= 1e-4 * max(1, abs(it["gold"]))
    write_jsonl(rd / "a2_items.jsonl", items)
    print(f"[a2] {n} items; user-attributed outcomes:",
          {o: sum(it["user_outcome"] == o for it in items) for o in ("FOLLOW", "RECOVER", "OTHER")})


def stage_a2_extract(cfg: dict, tag: str, model_name: str) -> None:
    rd = run_dir(cfg, tag)
    items = read_jsonl(rd / "a2_items.jsonl")
    if not items:
        return
    R = Reader(model_name)
    L = layer_list(R.n_layers, cfg["probes"]["layer_step"])

    def ends(prompt, value_text, working_marker):
        start = prompt.find("Here is my working so far:")
        stop = prompt.find("Please continue from exactly", start)
        seg = prompt[start:stop]
        v_end = start + seg.rfind(value_text) + len(value_text)
        return [v_end, len(prompt)]

    for side, vt in (("corrupt", "new_text"), ("clean", "true_text")):
        texts = [it[f"prompt_{side}"] for it in items]
        ce = [ends(it[f"prompt_{side}"], it[vt], None) for it in items]
        H, _ = R.read(texts, L, ce, add_special=False, bs=cfg["a2"]["batch"])
        np.save(rd / f"a2_acts_{side}.npy", H)
    json.dump({"layers": L}, open(rd / "a2_layers.json", "w"))
    print(f"[a2] extracted {len(items)} items x 2 positions x {len(L)} layers")


def stage_a2_fit(cfg: dict, tag: str) -> None:
    rd = run_dir(cfg, tag)
    if not (rd / "a2_acts_corrupt.npy").exists():
        return
    items = read_jsonl(rd / "a2_items.jsonl")
    L = json.load(open(rd / "a2_layers.json"))["layers"]
    Hc, Hk = np.load(rd / "a2_acts_corrupt.npy"), np.load(rd / "a2_acts_clean.npy")
    keep = np.array([it["clean_ok"] and it["user_outcome"] in ("FOLLOW", "RECOVER") for it in items])
    self_f = np.array([it["self_outcome"] == "FOLLOW" for it in items])
    user_f = np.array([it["user_outcome"] == "FOLLOW" for it in items])
    both = keep & np.array([it["self_outcome"] in ("FOLLOW", "RECOVER") for it in items])
    res = {"model": tag, "n_items": len(items), "n_valid_user": int(keep.sum()),
           "follow_rate_self": float(self_f[both].mean()) if both.any() else None,
           "follow_rate_user": float(user_f[both].mean()) if both.any() else None,
           "n_paired": int(both.sum()),
           "user_outcomes": {o: int(sum(it["user_outcome"] == o for it in items)) for o in ("FOLLOW", "RECOVER", "OTHER")}}
    if both.any():
        sf, uf = self_f[both].astype(float), user_f[both].astype(float)
        g = np.array([it["group"] for it in items])[both]
        res["follow_diff_user_minus_self"] = float(uf.mean() - sf.mean())
        res["follow_diff_ci"] = boot_ci(lambda i: uf[i].mean() - sf[i].mean(), g, 500, cfg["seed"])
        res["agreement_self_user"] = float((sf == uf).mean())
    y = user_f[keep].astype(int)
    groups = np.array([it["group"] for it in items])[keep]
    if keep.sum() >= 100 and 10 <= y.sum() <= len(y) - 10:
        for p, name in ((0, "value_token"), (1, "last_prompt_token")):
            res[name] = _readouts(Hc[keep, p], Hk[keep, p], y, groups, cfg["seed"], L)
    else:
        res["readouts_skipped"] = "too few FOLLOW or RECOVER under user attribution"
    json.dump(res, open(rd / "a2.json", "w"), indent=1)
    print(f"[a2] {tag}: FOLLOW self {res['follow_rate_self']}  user {res['follow_rate_user']}")
