"""Conditional gate: steer against deference only when the model registers a conflict with what it knows.

Why: the deference direction was learned on wrong claims only (defer vs hold), so it encodes "about to adopt the answer
in the context", right or wrong. The plain gate therefore also dampened uptake of CORRECT information on questions the
model does not know (K1: -1.7 points for Qwen). Here a second direction decides WHETHER to steer:

  conflict direction  last-prompt-token state, known items, training half: wrong claim minus correct claim (user and
                      document sentences), at the first steering layer
  decision            at the end of the prompt, per sequence: steer only if the conflict projection exceeds tau_c =
                      the 95th percentile over correct-claim prompts on known training items (at most ~5% false firing)
  steering            the steering-v2 gate (same deference directions, layers, generated-token thresholds), applied
                      only to sequences flagged at the prompt

Stages: cg_dirs (CPU, stored activations: direction, threshold, offline firing rates on the held-out half), cg_steer
(GPU: one condition, on the steering-v2 test prompts and the K1 prompts, compared with the no-steering replies already
on disk), cg_report.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .common import auc, read_jsonl, write_jsonl
from .kill import _unknown_sel, kdir


def _act(D, vid, li):
    b = next((b for b, d in D.items() if d["acts"] and vid in d["acts"]["row"]), None)
    return None if b is None else np.asarray(D[b]["acts"]["A"][D[b]["acts"]["row"][vid], 1, li], np.float32)


def stage_cg_dirs(cfg: dict) -> dict:
    from .deference import _load_all
    from .steer2 import sdir
    items, D = _load_all(cfg)
    S = json.load(open(sdir(cfg) / "dirs.json"))
    layers = next(d["acts"]["layers"] for d in D.values() if d["acts"])
    L1 = S["layers"][0]
    li = layers.index(L1)
    train = set(S["train_items"])
    G, known = {}, set()
    for d in D.values():
        G.update(d["G"])
        known |= d["known"]
    fams = ("m_user", "m_document")

    def collect(ids, role):
        X, meta = [], []
        for i in ids:
            for f in fams:
                v = f"{i}|{f}|{role}"
                a = _act(D, v, li)
                if a is not None and v in G and G[v]["outcome"] != "SKIP":
                    X.append(a)
                    meta.append((i, f, G[v]["outcome"]))
        return (np.stack(X) if X else np.zeros((0, 1))), meta
    k_tr = sorted(i for i in known if i in train)
    k_te = sorted(i for i in known if i not in train)
    u_te = sorted(i for i in items if i not in train and i not in known and G.get(f"{i}|base|base", {}).get("outcome") not in (None, "SKIP"))
    Xw, _ = collect(k_tr, "wrong")
    Xr, _ = collect(k_tr, "right")
    dc = Xw.mean(0) - Xr.mean(0)
    dc = dc / (np.linalg.norm(dc) + 1e-12)
    q = cfg.get("cg", {}).get("quantile", 0.95)
    tau_c = float(np.quantile(Xr @ dc, q))
    Tw, mw = collect(k_te, "wrong")
    Tr, _ = collect(k_te, "right")
    Ur, _ = collect(u_te, "right")
    pw, pr, pu = Tw @ dc, Tr @ dc, Ur @ dc
    deferred = np.array([m[2] == "WRONG" for m in mw])
    out = {"layer": L1, "tau_c": tau_c, "quantile": q, "n_train": {"wrong": len(Xw), "right": len(Xr)},
           "auroc_known_wrong_vs_known_right": auc(np.r_[np.ones(len(pw)), np.zeros(len(pr))], np.r_[pw, pr]),
           "auroc_known_wrong_vs_unknown_right": auc(np.r_[np.ones(len(pw)), np.zeros(len(pu))], np.r_[pw, pu]),
           "fires": {"known items, wrong claim (all)": float(np.mean(pw > tau_c)),
                     "known items, wrong claim, model gave in": float(np.mean(pw[deferred] > tau_c)) if deferred.any() else None,
                     "known items, correct claim": float(np.mean(pr > tau_c)),
                     "unknown items, correct claim": float(np.mean(pu > tau_c))},
           "n_test": {"known_wrong": len(pw), "known_right": len(pr), "unknown_right": len(pu)}}
    np.save(kdir(cfg) / "cg_dir.npy", dc.astype(np.float32))
    json.dump(out, open(kdir(cfg) / "cg_dirs.json", "w"), indent=1)
    print(f"[cg] layer {L1}: AUROC wrong vs right (known) {out['auroc_known_wrong_vs_known_right']:.3f}, "
          f"known wrong vs unknown right {out['auroc_known_wrong_vs_unknown_right']:.3f}; fires: " +
          ", ".join(f"{k} {v:.2f}" for k, v in out["fires"].items() if v is not None), flush=True)
    return out


class _CondGate:
    """Deference gate that acts only on sequences flagged at the end of the prompt. The hook at the first steering layer
    also computes the flag (from its unmodified state) during the prefill pass; the flag persists through decoding."""

    def __init__(self, d, tau, state, dc=None, tau_c=None):
        import torch
        self.d = torch.tensor(d, dtype=torch.float32)
        self.dc = None if dc is None else torch.tensor(dc, dtype=torch.float32)
        self.tau, self.tau_c, self.state, self.rec = tau, tau_c, state, None

    def __call__(self, mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        hf = h.float()
        if self.dc is not None and h.shape[1] > 1:                     # prefill: decide per sequence
            fire = (hf[:, -1] @ self.dc.to(h.device) > self.tau_c).float()
            self.state["fire"] = fire
            self.state["log"].append(fire.detach().cpu())
        fire = self.state.get("fire")
        if fire is None or fire.shape[0] != h.shape[0]:
            return out
        d = self.d.to(h.device)
        ex = (hf @ d - self.tau).clamp(min=0.0) * fire[:, None]
        h2 = (hf - ex[..., None] * d).to(h.dtype)
        return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2


def stage_cg_steer(cfg: dict) -> None:
    from .common import render
    from .run import _grade
    from .steer2 import _Model, _mt, _sel, sdir
    S = json.load(open(sdir(cfg) / "dirs.json"))
    Z = np.load(sdir(cfg) / "dirs.npz")
    C = json.load(open(sdir(cfg) / "calib.json"))
    K = json.load(open(kdir(cfg) / "cg_dirs.json"))
    dc = np.load(kdir(cfg) / "cg_dir.npy")
    items, known_sel = _sel(cfg, train=False)
    _, unk_sel = _unknown_sel(cfg)
    sel = known_sel + unk_sel
    M = _Model(cfg)
    state = {"log": []}
    hooks = [(L, _CondGate(Z["d"][k], C["per_layer"][k]["tau"], state, dc if k == 0 else None, K["tau_c"]))
             for k, L in enumerate(S["layers"])]
    out = kdir(cfg) / "cg_steer.jsonl"
    old = read_jsonl(out) if out.exists() else []
    have = {r["vid"] for r in old}
    todo = [v for v in sel if v["vid"] not in have]
    rows = list(old)
    for b in sorted({v["bucket"] for v in todo}):
        vb = [v for v in todo if v["bucket"] == b]
        outs, _ = M.generate([render(M.tok, v)[0] for v in vb], _mt(cfg, b), cfg.get("s2", {}).get("batch", 48), hooks)
        for v, t in zip(vb, outs):
            oc, _a = _grade(items, v, t)
            rows.append({"vid": v["vid"], "cond": "cgate", "text": t, "outcome": oc})
        write_jsonl(out, rows)
        print(f"[cg_steer] {b}: {len(vb)} replies", flush=True)
    fired = float(np.mean(np.concatenate([x.numpy() for x in state["log"]]))) if state["log"] else None
    if fired is not None:                                   # a resumed run with nothing left keeps the earlier record
        json.dump({"share_of_prompts_flagged": fired}, open(kdir(cfg) / "cg_runtime.json", "w"))
    print(f"[cg_steer] share of prompts flagged at runtime: {fired}", flush=True)


def analyse_cg(cfg) -> dict:
    from .honest_defer import _boot
    from .steer2 import sdir
    P = lambda p: {r["vid"]: r for r in read_jsonl(p)} if p.exists() else {}
    none = {**P(sdir(cfg) / "steer_none.jsonl"), **P(kdir(cfg) / "k1_none.jsonl")}
    conds = {"plain gate": {**P(sdir(cfg) / "steer_gate.jsonl"), **P(kdir(cfg) / "k1_gate.jsonl")},
             "conditional gate": P(kdir(cfg) / "cg_steer.jsonl")}
    _, unk = _unknown_sel(cfg)
    unk = {v["vid"] for v in unk}
    M = [("gives in to user", lambda k: "|m_user|wrong" in k, "WRONG"),
         ("gives in to document", lambda k: "|m_document|wrong" in k, "WRONG"),
         ("gives in to user, 2nd turn", lambda k: "|c_user|wrong" in k, "WRONG"),
         ("flips after \"Are you sure?\"", lambda k: "|c_doubt|doubt" in k, "FLIP"),
         ("accuracy (bare question)", lambda k: k.endswith("|base|base"), "GOLD"),
         ("takes correct info, known items", lambda k: k not in unk and k.endswith("|right"), "GOLD"),
         ("takes correct info, UNKNOWN items", lambda k: k in unk, "GOLD")]
    out = {}
    for c, R in conds.items():
        if not R:
            continue
        m = {}
        for lab, sel, val in M:
            ks = [k for k in none if sel(k) and k in R]
            hit = (lambda r: r["outcome"] != "GOLD") if val == "FLIP" else (lambda r, v=val: r["outcome"] == v)
            a = np.array([hit(R[k]) for k in ks], float)
            b0 = np.array([hit(none[k]) for k in ks], float)
            m[lab] = {"n": len(ks), "none": float(b0.mean()) if len(ks) else None, "rate": float(a.mean()) if len(ks) else None,
                      "points": float(100 * (a - b0).mean()) if len(ks) else None,
                      "ci": [100 * x for x in _boot(a - b0, cfg["seed"])] if len(ks) > 10 else [None, None]}
        out[c] = m
    json.dump(out, open(kdir(cfg) / "cg.json", "w"), indent=1)
    for c, m in out.items():
        print(f"[cg] {c}: " + "; ".join(f"{k} {v['points']:+.1f} [{v['ci'][0]:+.1f}, {v['ci'][1]:+.1f}]" for k, v in m.items()
                                        if v["points"] is not None and v["ci"][0] is not None), flush=True)
    return out


def write_report_cg(cfg: dict) -> Path:
    A = analyse_cg(cfg)
    K = json.load(open(kdir(cfg) / "cg_dirs.json"))
    rt = json.load(open(kdir(cfg) / "cg_runtime.json")) if (kdir(cfg) / "cg_runtime.json").exists() else {}
    rep = Path(os.path.expandvars(cfg["report_dir"])) / cfg["run_prefix"]
    rep.mkdir(parents=True, exist_ok=True)
    L = [f"# Conditional gate — {cfg['model']}", "",
         f"Conflict direction at layer {K['layer']} (wrong claim minus correct claim, known training items); threshold = "
         f"{int(100 * K['quantile'])}th percentile of correct-claim prompts. Held-out half:", "",
         f"- separates wrong from correct claims on known items: AUROC {K['auroc_known_wrong_vs_known_right']:.3f}; "
         f"wrong claims on known items from correct claims on unknown items: {K['auroc_known_wrong_vs_unknown_right']:.3f}",
         "- fires on: " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in K["fires"].items() if v is not None),
         f"- share of steered prompts flagged at run time: {rt.get('share_of_prompts_flagged')}", "",
         "| Metric (rate with no steering) | Plain gate, points [95% CI] | Conditional gate, points [95% CI] |", "|---|---|---|"]
    keys = list(next(iter(A.values())).keys()) if A else []
    for k in keys:
        n0 = A.get("conditional gate", A.get("plain gate"))[k]["none"]
        if n0 is None:
            continue
        cell = lambda c: "–" if c not in A or A[c][k]["points"] is None else (
            f"{A[c][k]['points']:+.1f} [{A[c][k]['ci'][0]:+.1f}, {A[c][k]['ci'][1]:+.1f}]" if A[c][k]["ci"][0] is not None else f"{A[c][k]['points']:+.1f}")
        L.append(f"| {k} ({100 * n0:.0f}%) | {cell('plain gate')} | {cell('conditional gate')} |")
    p = rep / "REPORT_cgate.md"
    p.write_text("\n".join(L) + "\n")
    print(f"[cg_report] {p}")
    return p
