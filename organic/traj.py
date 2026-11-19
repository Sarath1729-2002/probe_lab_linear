"""From the suggestion to the answer, done properly: a SEPARATE probe at each position.

v1 projected one direction (fitted where the suggestion ends) onto every later token. A direction
fitted at one position does not mean the same thing elsewhere, so the curves collapsed to zero
regardless of what the model did. Here (GPU, read-only) the states of both suggestion arms are stored
at fixed positions: suggestion end, the midpoint of the remaining prompt, the last prompt token, and
reply tokens 1, 2, 4, 8, 16, 32, 64. The analysis (CPU) then fits a wrong-vs-right probe at every
position with grouped out-of-fold scoring, and asks, position by position, how much of the detection
signal is there for items the model later adopted vs held.

Caveat for reply positions: the two arms' replies differ in content, so from reply token 1 on the
probe can also read what the model is writing, not only what it registered."""
from __future__ import annotations

import json
import random

import numpy as np

from .common import MassMean, Reader, auc, oof_scores, read_jsonl, render
from .run import bdir, load, root

REPLY = [1, 2, 4, 8, 16, 32, 64]
POSITIONS = ["suggestion_end", "prompt_mid", "last_prompt_token"] + [f"reply_{t}" for t in REPLY]


def stage_traj(cfg: dict, buckets: list[str]) -> None:
    res = json.load(open(root(cfg) / "results.json")) if (root(cfg) / "results.json").exists() else None
    if res is None:
        print("[traj] run fit first")
        return
    items, V = load(cfg)
    vmap = {v["vid"]: v for v in V}
    n_items = cfg["traj"]["n_items"]
    fam = cfg["traj"].get("family", "sugg")
    R = None
    for b in buckets:
        d = bdir(cfg, b)
        pr = res["buckets"].get(b, {}).get("probes", {}).get(fam, {}).get("info_end", {})
        if "layer" not in pr or (d / "traj_acts.npy").exists():
            continue
        layers = sorted({pr["layer"], res["buckets"][b]["probes"][fam]["last_prompt_token"].get("layer", pr["layer"])})
        G = {r["vid"]: r for r in read_jsonl(d / "gen.jsonl")}
        kn = [i for i, it in items.items() if it["bucket"] == b and G.get(f"{i}|base|base", {}).get("outcome") == "GOLD"]
        has = lambda i, r: bool(G.get(f"{i}|{fam}|{r}", {}).get("text"))
        rng = random.Random(cfg["seed"])
        W = [i for i in kn if has(i, "wrong")]
        rng.shuffle(W)
        # balance followed and recovered among wrong versions (following is usually the rarer outcome)
        ad = [i for i in W if G[f"{i}|{fam}|wrong"]["outcome"] == "WRONG"]
        ho = [i for i in W if G[f"{i}|{fam}|wrong"]["outcome"] == "GOLD"]
        na = min(len(ad), n_items // 2)
        wsel = ad[:na] + ho[: n_items - na]
        # right versions: the same items when the family is paired (sugg), other items otherwise (frames)
        paired = [i for i in wsel if has(i, "right")]
        if paired:
            rsel = paired
        else:
            Rr = [i for i in kn if has(i, "right")]
            rng.shuffle(Rr)
            rsel = Rr[: len(wsel)]
        todo = [(i, "wrong") for i in wsel] + [(i, "right") for i in rsel]
        if not wsel or not rsel:
            print(f"[traj] {b}: no usable items for {fam}")
            continue
        if R is None:
            R = Reader(cfg["model"], cfg["extract"]["dtype"])
        torch = R.torch
        body = getattr(R.model, "model", R.model)
        H = np.full((len(todo), len(POSITIONS), len(layers), R.d), np.nan, np.float16)
        meta = []
        k = 0
        for i, role in todo:
            vid = f"{i}|{fam}|{role}"
            v = vmap[vid]
            p, info = render(R.tok, v)
            enc = R.tok(p + G[vid]["text"], return_tensors="pt", add_special_tokens=False,
                        return_offsets_mapping=True)
            ends = [e for _, e in enc.pop("offset_mapping")[0].tolist()]
            t_info = max([t for t, e in enumerate(ends) if e <= info] or [0])
            t_last = max([t for t, e in enumerate(ends) if e <= len(p)] or [0])
            pos = [t_info, (t_info + t_last) // 2, t_last] + [t_last + r for r in REPLY]
            stop = min(len(ends), t_last + max(REPLY) + 1)
            with torch.no_grad():
                hs = body(input_ids=enc["input_ids"][:, :stop].to(R.model.device), output_hidden_states=True).hidden_states
            for pi, t in enumerate(pos):
                if t < stop:
                    for li, l in enumerate(layers):
                        H[k, pi, li] = hs[l][0, t].float().cpu().numpy()
            meta.append({"vid": vid, "item": i, "role": role, "outcome": G[vid]["outcome"],
                         "n_reply_tokens": int(stop - t_last - 1)})
            k += 1
        np.save(d / "traj_acts.npy", H)
        json.dump({"rows": meta, "positions": POSITIONS, "layers": layers}, open(d / "traj_index.json", "w"))
        print(f"[traj] {b}: {len(meta)} prompts x {len(POSITIONS)} positions at layers {layers}", flush=True)


def analyse_traj(cfg: dict, b: str) -> dict | None:
    """Per position: held-out detection AUROC (wrong vs right), detection score -> adoption AUROC, and
    the share of adopted / held items flagged at a 5% false-alarm threshold."""
    d = bdir(cfg, b)
    if not (d / "traj_acts.npy").exists():
        return None
    H = np.load(d / "traj_acts.npy")
    idx = json.load(open(d / "traj_index.json"))
    rows = idx["rows"]
    if not rows:
        return None
    y = np.array([r["role"] == "wrong" for r in rows], int)
    g = np.array([r["item"] for r in rows])
    oc = np.array([r["outcome"] for r in rows])
    adopt, hold = (y == 1) & (oc == "WRONG"), (y == 1) & (oc == "GOLD")
    out = {"positions": idx["positions"], "layers": idx["layers"], "n_adopt": int(adopt.sum()), "n_hold": int(hold.sum()),
           "per_layer": {}}
    for li, l in enumerate(idx["layers"]):
        curve = []
        for pi, pos in enumerate(idx["positions"]):
            X = H[:, pi, li].astype(np.float32)
            m = ~np.isnan(X).any(1)                          # position exists (reply long enough)
            r = {"position": pos, "n": int(m.sum())}
            if m.sum() >= 40 and len(np.unique(g[m])) >= 20:
                s = np.full(len(y), np.nan)
                s[m] = oof_scores(X[m], y[m], g[m], MassMean, 5)
                good = m & ~np.isnan(s)
                r["detection_auroc"] = auc(y[good], s[good])
                thr = np.nanpercentile(s[good & (y == 0)], 95)
                a, h = adopt & good, hold & good
                r["flagged_adopted"] = float((s[a] > thr).mean()) if a.sum() >= 5 else None
                r["flagged_held"] = float((s[h] > thr).mean()) if h.sum() >= 5 else None
                if a.sum() >= 10 and h.sum() >= 10:
                    z = (a | h) & good
                    r["score_predicts_adopt_auroc"] = auc(a[z].astype(int), s[z])
            curve.append(r)
        out["per_layer"][str(l)] = curve
    return out
