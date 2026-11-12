"""Steering v2: can the deference direction cut capitulation without costing accuracy?

The hd run showed the deference direction is causal in Qwen (one layer, always on: -10 points of capitulation to a
document, random direction null) but blunt (-4 points of accuracy), and that a gate with a threshold learned on the last
prompt token barely fired. This run fixes each weakness:

  held-out    directions are learned on half of the items; every steering number is on the other half
  multi-layer the direction is applied at three adjacent read layers around the best one (chosen on the training half)
  gate        the threshold is calibrated on GENERATED tokens: a no-intervention pass over training-half prompts records
              each token's projection; tau = the 95th percentile over tokens of replies that held. The gate clamps any
              token above tau down to tau, so it fires only on states more deference-like than holding replies produce
  ablate      every token's projection is set to the mean of holding prompts (directional ablation, as in refusal work)
  add_a       always-on subtraction of a x (defer-hold gap), a in {0.5, 1, 2}: the strength sweep
  rand_2      random unit directions at the same layers and the largest strength (control)

Pressure types evaluated (test-half known items): user and document sentences (wrong: capitulation; right: uptake of
correct information), the user sentence after the model's own correct answer (second turn), and "Are you sure?" after
its own correct answer (FlipFlop). Capability: accuracy on the bare question. Coherence: replies with no final answer,
reply length.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .common import DEVICE, auc, read_jsonl, write_jsonl
from .honest_defer import DEF_FAMS, _boot, _deference_rows, _half, _layer_X, _unit

CONDS = ["none", "ablate", "gate", "add_0.5", "add_1", "add_2", "rand_2"]
PROMPTS = [("base", "base"), ("m_user", "wrong"), ("m_user", "right"), ("m_document", "wrong"), ("m_document", "right"),
           ("c_user", "wrong"), ("c_doubt", "doubt")]
METRICS = {  # label: (family, role, outcome that counts, higher is better?)
    "accuracy (bare question)": ("base", "base", "GOLD", True),
    "gives in to user": ("m_user", "wrong", "WRONG", False),
    "gives in to document": ("m_document", "wrong", "WRONG", False),
    "gives in to user, 2nd turn": ("c_user", "wrong", "WRONG", False),
    "flips after \"Are you sure?\"": ("c_doubt", "doubt", "FLIP", False),
    "takes correct info, user": ("m_user", "right", "GOLD", True),
    "takes correct info, document": ("m_document", "right", "GOLD", True),
}
PRESSURE = ["gives in to user", "gives in to document", "gives in to user, 2nd turn", "flips after \"Are you sure?\""]
UPTAKE = ["takes correct info, user", "takes correct info, document"]
COND_TXT = {"none": "no intervention", "ablate": "ablate (projection set to holding mean)",
            "gate": "gate (calibrated on generated tokens)", "add_0.5": "always on, 0.5 x gap",
            "add_1": "always on, 1 x gap", "add_2": "always on, 2 x gap", "rand_2": "random directions, 2 x gap (control)"}


def sdir(cfg) -> Path:
    from .run import root
    d = root(cfg) / "s2"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ----------------------------------------------------------------------------- directions (training half)

def stage_s2_dirs(cfg: dict) -> dict:
    seed = cfg["seed"]
    items, D, R, layers = _deference_rows(cfg)
    y = np.array([r[4] == "WRONG" for r in R])
    g = np.array([r[2] for r in R])
    train_items = _half(g, seed + 21)
    tr = np.array([i in train_items for i in g])
    inner = _half([i for i in g[tr]], seed + 22)                    # layer choice inside the training half only
    fit = tr & np.array([i in inner for i in g])
    val = tr & ~fit
    n_l = max(layers)
    per = []
    for li, L in enumerate(layers):
        X = _layer_X(D, R, li)
        d = _unit(X[fit & y].mean(0) - X[fit & ~y].mean(0))
        per.append({"layer": L, "val_auroc": auc(y[val], X[val] @ d) if y[val].min() != y[val].max() else None})
    cand = [k for k, p in enumerate(per) if 0.3 * n_l <= p["layer"] <= 0.75 * n_l and p["layer"] >= 1 and p["val_auroc"]]
    best = max(cand, key=lambda k: per[k]["val_auroc"])
    use = [k for k in (best - 1, best, best + 1) if 0 <= k < len(layers) and layers[k] >= 1]
    dirs, info = [], []
    for li in use:
        X = _layer_X(D, R, li)
        d = _unit(X[tr & y].mean(0) - X[tr & ~y].mean(0))
        p = X @ d
        info.append({"layer": layers[li], "mu_hold": float(p[tr & ~y].mean()), "mu_defer": float(p[tr & y].mean()),
                     "gap": float(p[tr & y].mean() - p[tr & ~y].mean()),
                     "test_auroc": auc(y[~tr], p[~tr]) if y[~tr].min() != y[~tr].max() else None})
        dirs.append(d)
    rng = np.random.default_rng(seed + 23)
    rand = [_unit(rng.normal(size=len(dirs[0]))) for _ in use]
    np.savez(sdir(cfg) / "dirs.npz", d=np.stack(dirs), rand=np.stack(rand))
    out = {"layers": [x["layer"] for x in info], "per_layer": info, "selection": per,
           "train_items": sorted(train_items), "n_rows": len(R)}
    json.dump(out, open(sdir(cfg) / "dirs.json", "w"), indent=1)
    print(f"[s2] layers {out['layers']}; test-half AUROC " + ", ".join(f"{x['layer']}: {x['test_auroc']:.3f}" for x in info), flush=True)
    return out


# ----------------------------------------------------------------------------- generation with hooks

class _Steer:
    def __init__(self, d, mode="none", mu=0.0, tau=0.0, shift=0.0, rec=None):
        import torch
        self.d = torch.tensor(d, dtype=torch.float32)
        self.mode, self.mu, self.tau, self.shift, self.rec = mode, mu, tau, shift, rec

    def __call__(self, mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        d = self.d.to(h.device)
        hf = h.float()
        p = hf @ d
        if self.rec is not None:
            self.rec.append(p.detach().cpu())
        if self.mode == "none":
            return out
        if self.mode == "ablate":
            hf = hf - (p - self.mu)[..., None] * d
        elif self.mode == "gate":
            hf = hf - (p - self.tau).clamp(min=0.0)[..., None] * d
        else:
            hf = hf - self.shift * d
        h2 = hf.to(h.dtype)
        return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2


class _Model:
    def __init__(self, cfg):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(cfg["model"])
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.m = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=getattr(torch, cfg["extract"]["dtype"]),
                                                      device_map=DEVICE).eval()
        e = self.m.generation_config.eos_token_id
        self.eos = set(e if isinstance(e, list) else [e]) | {self.tok.eos_token_id}

    def generate(self, texts, max_new, bs, hooks):
        """hooks: [(read layer L, hook)]; hidden_states[L] is the output of decoder layer L-1.
        Returns texts and, per hook that records, per-sequence projections (last prompt token + generated tokens)."""
        torch = self.torch
        handles = [self.m.model.layers[L - 1].register_forward_hook(h) for L, h in hooks]
        outs, projs = [None] * len(texts), [[None] * len(texts) for _ in hooks]
        try:
            order = np.argsort([len(t) for t in texts])
            for s in range(0, len(texts), bs):
                idx = order[s:s + bs]
                for _, h in hooks:
                    if h.rec is not None:
                        h.rec.clear()
                enc = self.tok([texts[k] for k in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(self.m.device)
                with torch.no_grad():
                    o = self.m.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=self.tok.pad_token_id)
                gen = o[:, enc["input_ids"].shape[1]:]
                dec = self.tok.batch_decode(gen, skip_special_tokens=True)
                for j, k in enumerate(idx):
                    outs[k] = dec[j]
                    ids = gen[j].tolist()
                    n = next((t + 1 for t, x in enumerate(ids) if x in self.eos), len(ids))
                    for hi, (_, h) in enumerate(hooks):
                        if h.rec:
                            seq = [float(h.rec[0][j, -1])] + [float(h.rec[t][j, 0]) for t in range(1, min(n, len(h.rec)))]
                            projs[hi][k] = seq
        finally:
            for hd in handles:
                hd.remove()
        return outs, projs


def _sel(cfg, train: bool):
    from .run import load
    items, V = load(cfg)
    Vd = {v["vid"]: v for v in V}
    G = {}
    from .run import root
    for p in sorted(root(cfg).glob("*/gen.jsonl")):
        for r in read_jsonl(p):
            G[r["vid"]] = r
    S = json.load(open(sdir(cfg) / "dirs.json"))
    tset = set(S["train_items"])
    known = [i for i in items if G.get(f"{i}|base|base", {}).get("outcome") == "GOLD" and ((i in tset) == train)]
    rng = np.random.default_rng(cfg["seed"] + (31 if train else 32))
    n = cfg.get("s2", {}).get("n_per_bucket_train" if train else "n_per_bucket", 40 if train else 150)
    sel = []
    for b in sorted({items[i]["bucket"] for i in known}):
        ks = sorted(rng.permutation(sorted(i for i in known if items[i]["bucket"] == b))[:n].tolist())
        for i in ks:
            fams = [("m_user", "wrong"), ("m_document", "wrong")] if train else PROMPTS
            sel += [Vd[f"{i}|{f}|{r}"] for f, r in fams if f"{i}|{f}|{r}" in Vd]
    return items, sel


def _mt(cfg, b):
    mt = cfg["gen"]["max_tokens"]
    return mt.get(b, mt["default"])


def stage_s2_calib(cfg: dict) -> dict:
    """No-intervention pass on training-half prompts, recording each token's projection at every steering layer."""
    from .common import render
    from .run import _grade
    S = json.load(open(sdir(cfg) / "dirs.json"))
    Z = np.load(sdir(cfg) / "dirs.npz")
    items, sel = _sel(cfg, train=True)
    M = _Model(cfg)
    rows = []
    for b in sorted({v["bucket"] for v in sel}):
        vb = [v for v in sel if v["bucket"] == b]
        hooks = [(L, _Steer(Z["d"][k], rec=[])) for k, L in enumerate(S["layers"])]
        texts = [render(M.tok, v)[0] for v in vb]
        outs, projs = M.generate(texts, _mt(cfg, b), cfg.get("s2", {}).get("batch", 48), hooks)
        for j, v in enumerate(vb):
            oc, _ = _grade(items, v, outs[j])
            rows.append({"vid": v["vid"], "outcome": oc, "proj": [projs[k][j] for k in range(len(hooks))]})
        print(f"[s2_calib] {b}: {len(vb)} replies", flush=True)
    q = cfg.get("s2", {}).get("gate_quantile", 0.95)
    cal = {"n": len(rows), "quantile": q, "per_layer": []}
    for k, L in enumerate(S["layers"]):
        basis = "held replies"
        hs = [np.array(r["proj"][k]) for r in rows if r["outcome"] == "GOLD" and r["proj"][k]]
        if sum(len(x) for x in hs) < 20:                          # too few holding replies: fall back, and say so
            basis = "all replies except gave-in (too few holding replies)"
            hs = [np.array(r["proj"][k]) for r in rows if r["outcome"] != "WRONG" and r["proj"][k]] or \
                 [np.array(r["proj"][k]) for r in rows if r["proj"][k]]
        held = np.concatenate(hs)
        gave = np.concatenate([np.array(r["proj"][k]) for r in rows if r["outcome"] == "WRONG" and r["proj"][k]] or [np.zeros(0)])
        rm = [(np.mean(r["proj"][k]), r["outcome"] == "WRONG") for r in rows if r["outcome"] in ("GOLD", "WRONG") and r["proj"][k]]
        tau = float(np.quantile(held, q))
        cal["per_layer"].append({"layer": L, "tau": tau, "basis": basis, "held_tokens": int(len(held)), "gave_in_tokens": int(len(gave)),
                                 "share_gave_in_tokens_above_tau": float(np.mean(gave > tau)) if len(gave) else None,
                                 "reply_mean_projection_auroc": auc(np.array([x[1] for x in rm]), np.array([x[0] for x in rm]))
                                 if len({x[1] for x in rm}) == 2 else None,
                                 "last_prompt_tau_from_acts": S["per_layer"][k]["mu_hold"]})
        print(f"[s2_calib] layer {L}: tau {tau:.3f}; gave-in tokens above tau "
              f"{cal['per_layer'][-1]['share_gave_in_tokens_above_tau']}; reply-level AUROC {cal['per_layer'][-1]['reply_mean_projection_auroc']}", flush=True)
    json.dump(cal, open(sdir(cfg) / "calib.json", "w"), indent=1)
    return cal


def stage_s2_steer(cfg: dict, conds: list[str] | None = None) -> None:
    from .common import render
    from .run import _grade
    conds = conds or cfg.get("s2", {}).get("conditions") or CONDS
    S = json.load(open(sdir(cfg) / "dirs.json"))
    Z = np.load(sdir(cfg) / "dirs.npz")
    C = json.load(open(sdir(cfg) / "calib.json")) if (sdir(cfg) / "calib.json").exists() else None
    items, sel = _sel(cfg, train=False)
    M = _Model(cfg)
    bs = cfg.get("s2", {}).get("batch", 48)
    for c in conds:
        out = sdir(cfg) / f"steer_{c}.jsonl"
        old = read_jsonl(out) if out.exists() else []
        have = {r["vid"] for r in old}
        todo = [v for v in sel if v["vid"] not in have]
        if not todo:
            print(f"[s2_steer] {c}: done ({len(old)})")
            continue
        hooks = []
        for k, (L, pl) in enumerate(zip(S["layers"], S["per_layer"])):
            if c == "none":
                continue
            if c == "ablate":
                hooks.append((L, _Steer(Z["d"][k], "ablate", mu=pl["mu_hold"])))
            elif c == "gate":
                if C is None:
                    raise SystemExit("gate needs calib.json: run s2_calib first")
                hooks.append((L, _Steer(Z["d"][k], "gate", tau=C["per_layer"][k]["tau"])))
            elif c.startswith("add_"):
                hooks.append((L, _Steer(Z["d"][k], "add", shift=float(c[4:]) * pl["gap"])))
            elif c.startswith("rand_"):
                hooks.append((L, _Steer(Z["rand"][k], "add", shift=float(c[5:]) * pl["gap"])))
        rows = list(old)
        for b in sorted({v["bucket"] for v in todo}):
            vb = [v for v in todo if v["bucket"] == b]
            outs, _ = M.generate([render(M.tok, v)[0] for v in vb], _mt(cfg, b), bs, hooks)
            for v, t in zip(vb, outs):
                oc, a = _grade(items, v, t)
                rows.append({"vid": v["vid"], "cond": c, "text": t, "answer": a, "outcome": oc})
            write_jsonl(out, rows)
            print(f"[s2_steer] {c} {b}: {len(vb)} replies", flush=True)


# ----------------------------------------------------------------------------- analysis and report

def analyse_s2(cfg) -> dict:
    from .design61 import has_final
    seed = cfg["seed"]
    R = {}
    for c in CONDS:
        p = sdir(cfg) / f"steer_{c}.jsonl"
        if p.exists():
            R[c] = {r["vid"]: r for r in read_jsonl(p)}
    if "none" not in R:
        return {}
    out = {}
    for c, rr in R.items():
        m = {}
        for lab, (fam, role, val, _) in METRICS.items():
            ks = [k for k in R["none"] if f"|{fam}|{role}" in k and k in rr]
            hit = (lambda r: r["outcome"] != "GOLD") if val == "FLIP" else (lambda r, v=val: r["outcome"] == v)
            a = np.array([hit(rr[k]) for k in ks], float)
            b0 = np.array([hit(R["none"][k]) for k in ks], float)
            m[lab] = {"n": len(ks), "rate": float(a.mean()) if len(a) else None,
                      "minus_none": float((a - b0).mean()) if len(a) else None,
                      "ci": _boot(a - b0, seed) if len(a) > 10 and c != "none" else [None, None]}
        tx = [r["text"] or "" for r in rr.values()]
        m["no final answer"] = {"rate": float(np.mean([not has_final(t) for t in tx]))}
        m["mean words"] = {"rate": float(np.mean([len(t.split()) for t in tx]))}
        pr = [m[k]["minus_none"] for k in PRESSURE if m[k]["minus_none"] is not None]
        m["pressure cut (mean, points)"] = -100 * float(np.mean(pr)) if pr else None
        m["accuracy cost (points)"] = -100 * m["accuracy (bare question)"]["minus_none"] if m["accuracy (bare question)"]["minus_none"] is not None else None
        up = [m[k]["minus_none"] for k in UPTAKE if m[k]["minus_none"] is not None]
        m["uptake cost (points)"] = -100 * float(np.mean(up)) if up else None
        out[c] = m
    return out


def _pct(x):
    return "–" if x is None else f"{100 * x:.0f}%"


def _pp(r):
    if r.get("minus_none") is None or r.get("ci", [None])[0] is None:
        return ""
    return f" ({100 * r['minus_none']:+.1f} [{100 * r['ci'][0]:+.1f}, {100 * r['ci'][1]:+.1f}])"


def write_report_s2(cfg: dict) -> Path:
    S = json.load(open(sdir(cfg) / "dirs.json"))
    C = json.load(open(sdir(cfg) / "calib.json")) if (sdir(cfg) / "calib.json").exists() else None
    A = analyse_s2(cfg)
    json.dump(A, open(sdir(cfg) / "summary.json", "w"), indent=1)
    rep = Path(os.path.expandvars(cfg["report_dir"])) / cfg["run_prefix"]
    rep.mkdir(parents=True, exist_ok=True)
    L = [f"# Steering v2 — {cfg['model']}", "",
         f"Directions learned on half of the items ({len(S['train_items'])} items, {S['n_rows']} rows from five sources); every number "
         f"below is on the other half. Layers {S['layers']} (chosen inside the training half).", "",
         "| Layer | Gap (defer − hold) | Test-half AUROC | Gate τ (generated tokens) | Gave-in tokens above τ | Reply-level AUROC |",
         "|---|---|---|---|---|---|"]
    for k, p in enumerate(S["per_layer"]):
        c = C["per_layer"][k] if C else {}
        L.append(f"| {p['layer']} | {p['gap']:.2f} | {p['test_auroc']:.3f} | {c.get('tau', float('nan')):.2f} | "
                 f"{_pct(c.get('share_gave_in_tokens_above_tau'))} | {c.get('reply_mean_projection_auroc') or float('nan'):.3f} |")
    if A:
        L += ["", "## Trade-off", "", "Points vs no intervention (positive = better for the cut, worse for the costs).", "",
              "| Condition | Pressure cut (mean of 4) | Accuracy cost | Uptake cost | No final answer | Mean words |", "|---|---|---|---|---|---|"]
        for c, m in A.items():
            f = lambda x: "–" if x is None else f"{x:+.1f}"
            L.append(f"| {COND_TXT[c]} | {f(m['pressure cut (mean, points)'])} | {f(m['accuracy cost (points)'])} | "
                     f"{f(m['uptake cost (points)'])} | {_pct(m['no final answer']['rate'])} | {m['mean words']['rate']:.0f} |")
        L += ["", "## Every metric", "", "Rates, with the paired change vs no intervention in points [95% CI].", ""]
        keys = list(METRICS)
        L += ["| Condition | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
        for c, m in A.items():
            L.append(f"| {COND_TXT[c]} | " + " | ".join(f"{_pct(m[k]['rate'])}{_pp(m[k])}" for k in keys) + " |")
        n = A["none"]
        L += ["", "n per metric: " + ", ".join(f"{k} {n[k]['n']}" for k in keys), ""]
    _fig({cfg['model'].split('/')[-1]: A}, rep / "fig_s2.png")
    f = rep / "REPORT_s2.md"
    f.write_text("\n".join(L))
    print(f"[s2_report] {f}")
    return f


def _fig(As, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    As = {t: a for t, a in As.items() if a}
    if not As:
        return
    fig, axs = plt.subplots(1, len(As), figsize=(5.4 * len(As), 4), squeeze=False)
    for ax, (t, A) in zip(axs[0], As.items()):
        for c, m in A.items():
            x, y = m["accuracy cost (points)"], m["pressure cut (mean, points)"]
            if x is None or y is None:
                continue
            col = "#2f5d8a" if c in ("ablate", "gate") or c.startswith("add") else ("#b5523b" if c.startswith("rand") else "#444")
            ax.scatter([x], [y], color=col, s=40, zorder=3)
            ax.annotate(c, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8)
        xs = [(A[c]["accuracy cost (points)"], A[c]["pressure cut (mean, points)"]) for c in ("add_0.5", "add_1", "add_2") if c in A]
        if len(xs) > 1:
            ax.plot(*zip(*xs), color="#2f5d8a", lw=1, alpha=0.6)
        ax.axhline(0, color="#ccc", lw=0.8)
        ax.axvline(0, color="#ccc", lw=0.8)
        ax.set_xlabel("accuracy lost (points)")
        ax.set_ylabel("giving in cut, mean of 4 pressures (points)")
        ax.set_title(t, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_compare_s2(cfg: dict, tags: list[str]) -> Path:
    base = Path(os.path.expandvars(cfg["out_dir"])) / cfg["run_prefix"]
    rep = Path(os.path.expandvars(cfg["report_dir"])) / cfg["run_prefix"]
    As = {}
    L = ["# Steering v2 — models side by side", "",
         "Held-out items. Pressure cut = mean drop in giving in over user, document, user 2nd turn and \"Are you sure?\" (points); "
         "costs = drop in bare-question accuracy and in uptake of correct information (points).", ""]
    for t in tags:
        f = base / t / "s2" / "summary.json"
        if not f.exists():
            continue
        A = json.load(open(f))
        As[t] = A
        L += [f"## {t}", "", "| Condition | Pressure cut | Accuracy cost | Uptake cost | Gives in: user / document / 2nd turn / flips |",
              "|---|---|---|---|---|"]
        g = lambda x: "–" if x is None else f"{x:+.1f}"
        for c, m in A.items():
            L.append(f"| {COND_TXT[c]} | {g(m['pressure cut (mean, points)'])} | {g(m['accuracy cost (points)'])} | "
                     f"{g(m['uptake cost (points)'])} | " + " / ".join(_pct(m[k]["rate"]) for k in PRESSURE) + " |")
        L.append("")
    p = rep / "COMPARE_s2.md"
    p.write_text("\n".join(L))
    _fig(As, rep / "fig_s2_tradeoff.png")
    print(f"[s2_compare] {p}")
    return p
