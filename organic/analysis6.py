"""v6 analysis pieces that are specific to the design: paired contrasts, replication anchors, coverage."""
from __future__ import annotations

import numpy as np

from .design6 import CONTRASTS, MATCHED

PUBLISHED = {
    "sharma": "Sharma et al. 2023 (ICLR 2024): a wrong suggestion lowers accuracy, a right one raises it",
    "nqswap": "Longpre et al. 2021 (EMNLP): models largely follow the substituted passage (low memorisation ratio)",
    "clasheval": "Wu et al. 2024 (NeurIPS D&B): context bias 61% for GPT-4o, 15.7% for Claude Opus",
    "benchform": "Weng et al. 2025 (ICLR): 23.5% conformity under wrong guidance, mean of 5 models, on BBH",
    "flipflop": "Laban et al. 2023: 46% average flip rate after \"Are you sure?\", 10 models, classification tasks",
}


def _defer_maps(D, sch):
    """channel -> {item: 1/0 deferred} on known items (wrong role)."""
    M = {}
    for c, fam in sch["fam"].items():
        m = {}
        for d in D.values():
            for i in d["known"]:
                g = d["G"].get(f"{i}|{fam}|wrong")
                if g and g["outcome"] != "SKIP":
                    m[i] = int(g["outcome"] == "WRONG")
        M[c] = m
    return M


def _paired(a: dict, b: dict, seed: int, n=2000):
    ks = sorted(set(a) & set(b))
    if len(ks) < 10:
        return {"n": len(ks), "diff": None, "ci": [None, None]}
    d = np.array([a[k] - b[k] for k in ks], float)
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return {"n": len(ks), "a": float(np.mean([a[k] for k in ks])), "b": float(np.mean([b[k] for k in ks])),
            "diff": float(d.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def contrasts6(items, D, sch, seed):
    M = _defer_maps(D, sch)
    out = {"matched_vs_user": {}, "matched_vs_none": {}, "strength": {}}
    for c in MATCHED:
        if c != "user":
            out["matched_vs_user"][c] = _paired(M[c], M["user"], seed)
        if c != "none":
            out["matched_vs_none"][c] = _paired(M[c], M["none"], seed)
    soc = {i: np.mean([M[c][i] for c in sch["social"] if i in M[c]]) for i in M["user"]}
    inf = {i: np.mean([M[c][i] for c in sch["info"] if i in M[c]]) for i in M["document"] if any(i in M[c] for c in sch["info"])}
    out["user_level_minus_source_level"] = _paired(soc, inf, seed)
    for lab, a, b in CONTRASTS:
        out["strength"][lab] = _paired(M.get(a, {}), M.get(b, {}), seed)
    # per dataset: the matched user-vs-document gap
    out["by_bucket_document_minus_user"] = {}
    for bkt, d in D.items():
        ids = d["known"]
        out["by_bucket_document_minus_user"][bkt] = _paired({i: v for i, v in M["document"].items() if i in ids},
                                                            {i: v for i, v in M["user"].items() if i in ids}, seed)
    return out


def _rate(items, D, b, fam, role, val, known_only=False):
    d = D.get(b)
    if d is None:
        return None, 0
    ids = [i for i, it in items.items() if it["bucket"] == b and (not known_only or i in d["known"])]
    o = [d["G"][f"{i}|{fam}|{role}"]["outcome"] for i in ids if f"{i}|{fam}|{role}" in d["G"]]
    o = [x for x in o if x != "SKIP"]
    return (float(np.mean([x == val for x in o])) if o else None), len(o)


def anchors6(items, D, sch):
    """Published metrics recomputed on our runs (all items unless marked known)."""
    A = {}
    if "triviaqa" in D:
        acc = {r: _rate(items, D, "triviaqa", "b_sharma", r, "GOLD") for r in ("base", "wrong", "right", "doubt")}
        if acc["base"][1]:
            A["sharma"] = {"n": acc["base"][1], "accuracy_no_suggestion": acc["base"][0], "accuracy_wrong_suggestion": acc["wrong"][0],
                           "accuracy_right_suggestion": acc["right"][0], "accuracy_doubt": acc["doubt"][0],
                           "adopts_wrong_answer": _rate(items, D, "triviaqa", "b_sharma", "wrong", "WRONG")[0],
                           "published": PUBLISHED["sharma"]}
    for b, key in (("nqswap", "nqswap"), ("clasheval", "clasheval")):
        if b in D:
            po, n = _rate(items, D, b, "s_passage", "wrong", "GOLD")
            ps, _ = _rate(items, D, b, "s_passage", "wrong", "WRONG")
            psk, nk = _rate(items, D, b, "s_passage", "wrong", "WRONG", True)
            if n:
                A[key] = {"n": n, "original_answer": po, "substituted_answer": ps,
                          "memorisation_ratio": po / (po + ps) if (po + ps) > 0 else None,
                          "n_known": nk, "context_bias_known": psk, "published": PUBLISHED[key]}
    if "bbh" in D:
        h, n = _rate(items, D, "bbh", "s_peers6", "wrong", "GOLD", True)
        if n:
            A["benchform"] = {"n_known": n, "conformity_rate": 1 - h, "published": PUBLISHED["benchform"]}
    fl, nf = [], 0
    for b in D:
        h, n = _rate(items, D, b, "c_doubt", "doubt", "GOLD", True)
        if n:
            fl.append((1 - h, n, b))
            nf += n
    if nf:
        A["flipflop"] = {"n_known": nf, "flip_rate": sum(x * n for x, n, _ in fl) / nf,
                         "by_dataset": {b: x for x, _, b in fl}, "published": PUBLISHED["flipflop"]}
    return A


def coverage6(items, D):
    out = {}
    for b, d in D.items():
        ids = [i for i, it in items.items() if it["bucket"] == b]
        firm = [i for i in d["known"] if d["conf"].get(i, 0) >= 0.999]
        out[b] = {"items": len(ids), "known": len(d["known"]), "firm": len(firm),
                  "accuracy": len(d["known"]) / max(1, len(ids))}
    return out
