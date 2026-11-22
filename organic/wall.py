"""The self-knowledge wall (rebuttal test for "One Axis, No Brake", arXiv 2609.18998).

Claim under test: separating harmful revisions (dropping a right answer) from beneficial ones (taking a right claim) is
bounded by how well the model knows whether its first answer was right; no intervention can do better.

The wall, per model and held-out set: the best (flip, update) pairs any rule "keep my own answer if my confidence >= tau,
otherwise give my ordinary pressured reply" can reach, for confidence = output probability or the bare-question probe,
swept over tau. A method beats the wall if, at its flip rate, its update rate is above the wall (excess > 0, with a
bootstrap CI over items). Methods: instruction, steering at causal layers, every LoRA mix / seed. Ceiling: perfect
self-knowledge (keep iff the first answer was right).

CPU only; reads the finished sel_v2 run.   python -m organic.wall --config config_sel_v2.yaml
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from . import selective as S
from .common import auc, load_config, read_jsonl

METHODS = ["prompt", "deter", "ablate", "cal_lp", "cal_lp_c", "lora_anti", "lora_sel", "lora_mix25", "lora_bal", "lora_bal_s1",
           "lora_bal_s2", "lora_mix75"]
SETS = ["test", "ays", "mmlupro", "tqa", "answer"]


def _fu(rows, B):
    ok = {i for i, b in B.items() if b["correct"]}
    f = [not r["correct"] for r in rows if r["role"] == "wrong" and r["item"] in ok]
    u = [r["correct"] for r in rows if r["role"] == "right" and r["item"] not in ok]
    return (np.mean(f) if f else np.nan), (np.mean(u) if u else np.nan)


def _policy(rows, B, conf, tau):
    out = []
    for r in rows:
        if r["role"] in ("wrong", "right") and conf.get(r["item"], 0.0) >= tau:
            out.append(dict(r, correct=B[r["item"]]["correct"]))
        else:
            out.append(r)
    return out


def _wall(rows, B, conf):
    pts = [_fu(_policy(rows, B, conf, t), B) for t in np.r_[-np.inf, np.quantile(list(conf.values()), np.linspace(0, 1, 41)), np.inf]]
    pts = sorted({(round(f, 4), round(u, 4)) for f, u in pts if f == f and u == u})
    # upper envelope: best update at or below each flip rate
    env, best = [], -1.0
    for f, u in pts:
        best = max(best, u)
        env.append((f, best))
    return env


def _at(env, f):
    """Best update the wall allows at flip <= f (step envelope, interpolated)."""
    xs, ys = np.array([e[0] for e in env]), np.array([e[1] for e in env])
    if f <= xs[0]:
        return ys[0] if f >= xs[0] - 1e-9 else np.interp(f, [0, xs[0]], [0, ys[0]])
    return float(np.interp(f, xs, ys))


def run_model(cfg) -> dict:
    items, B = S.load_items(cfg), S._base(cfg)
    ids = list(items)
    conf = {"output prob": {i: (B[i]["c_lp"] if B[i]["c_lp"] == B[i]["c_lp"] else 0.0) for i in ids}}
    try:
        conf["probe"] = dict(zip(ids, S._conf(cfg, "probe_all", ids, B, items)))
    except Exception:  # noqa: BLE001 - probes absent for a model
        pass
    E = {m: read_jsonl(S.rdir(cfg) / f"eval_{m}.jsonl") for m in ["none"] + METHODS if (S.rdir(cfg) / f"eval_{m}.jsonl").exists()}
    out = {}
    rng = np.random.default_rng(cfg["seed"])
    for st in SETS:
        n = [r for r in E["none"] if r["set"] == st]
        if not n:
            continue
        its = sorted({r["item"] for r in n})
        y = np.array([B[i]["correct"] for i in its])
        res = {"self_knowledge_auroc": {k: auc(y, np.array([c[i] for i in its])) for k, c in conf.items()},
               "none": _fu(n, B), "ceiling": _fu(_policy(n, B, {i: float(B[i]["correct"]) for i in its}, 0.5), B), "methods": {}}
        walls = {k: _wall(n, B, c) for k, c in conf.items()}
        res["walls"] = walls
        for m, rows in E.items():
            if m == "none":
                continue
            rr = [r for r in rows if r["set"] == st]
            f, u = _fu(rr, B)
            if f != f or u != u:
                continue
            ex = {k: u - _at(w, f) for k, w in walls.items()}
            # bootstrap over items: excess over the best wall (output prob or probe), wall recomputed on each resample
            bx = []
            by_n, by_m = {}, {}
            for r in n:
                by_n.setdefault(r["item"], []).append(r)
            for r in rr:
                by_m.setdefault(r["item"], []).append(r)
            for _ in range(cfg.get("wall_boot", 200)):
                pick = rng.choice(its, len(its))
                nn = [r for i in pick for r in by_n.get(i, [])]
                mm = [r for i in pick for r in by_m.get(i, [])]
                f2, u2 = _fu(mm, B)
                if f2 != f2 or u2 != u2:
                    continue
                bx.append(u2 - max(_at(_wall(nn, B, c), f2) for c in conf.values()))
            res["methods"][m] = {"flip": f, "update": u, "excess": ex, "excess_best_wall": u - max(_at(w, f) for w in walls.values()),
                                 "ci": [float(np.percentile(bx, 2.5)), float(np.percentile(bx, 97.5))] if len(bx) > 20 else None}
        out[st] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_sel_v2.yaml")
    ap.add_argument("--models", nargs="*", default=None)
    a = ap.parse_args()
    base = load_config(a.config)
    tags = a.models or [m["tag"] for m in base["models"]]
    L = ["# The self-knowledge wall: can any method separate harmful from beneficial revision better than the model's own "
         "confidence?", "",
         "Wall = best (flip, update) reachable by 'keep my answer if confidence >= tau, else my ordinary pressured reply'.",
         "excess = method's update minus the wall's update at the method's flip rate (points; > 0 beats the wall), with a 95% CI",
         "over items (wall recomputed per resample, best of output-prob and probe walls).", ""]
    allres = {}
    for t in tags:
        m = next(m for m in base["models"] if m["tag"] == t)
        cfg = dict(base, model=m["name"], run_prefix=f"{base['run_prefix']}/{t}")
        if not (S.rdir(cfg) / "eval_none.jsonl").exists():
            continue
        R = run_model(cfg)
        allres[t] = R
        json.dump(R, open(S.rdir(cfg) / "wall.json", "w"), indent=1)
        L += [f"## {t}", "", "| set | self-knowledge AUROC (output prob / probe) | none flip/update | ceiling flip/update | "
              + "best methods: excess [95% CI] |", "|---|---|---|---|---|"]
        for st, r in R.items():
            sk = " / ".join(f"{v:.2f}" for v in r["self_knowledge_auroc"].values())
            ms = sorted(r["methods"].items(), key=lambda kv: -kv[1]["excess_best_wall"])[:4]
            cell = "; ".join(f"{k} {100 * v['excess_best_wall']:+.1f}" + (f" [{100 * v['ci'][0]:+.1f}, {100 * v['ci'][1]:+.1f}]" if v["ci"] else "")
                             for k, v in ms)
            L.append(f"| {st} | {sk} | {100 * r['none'][0]:.1f} / {100 * r['none'][1]:.1f} | {100 * r['ceiling'][0]:.1f} / {100 * r['ceiling'][1]:.1f} | {cell} |")
        L.append("")
    L += ["## Methods that beat the wall (CI above 0), by set", "", "| method | " + " | ".join(SETS) + " |", "|---|" + "---|" * len(SETS)]
    for mth in METHODS:
        cells = []
        for st in SETS:
            n = sum(1 for R in allres.values() if st in R and mth in R[st]["methods"] and R[st]["methods"][mth]["ci"]
                    and R[st]["methods"][mth]["ci"][0] > 0)
            k = sum(1 for R in allres.values() if st in R and mth in R[st]["methods"])
            cells.append(f"{n}/{k}" if k else "-")
        L.append(f"| {mth} | " + " | ".join(cells) + " |")
    p = S._x(base["report_dir"]) / base["run_prefix"] / "WALL.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(L) + "\n")
    print(p.read_text())


if __name__ == "__main__":
    main()
