"""Thread B: is the truth direction a knowledge direction?

Statements (true / false) are crossed with whether THIS model knows the underlying fact:
  known   = answers the fact correctly in both closed-book phrasings (greedy)
  unknown = wrong in both;  borderline items are dropped (Ferrando et al.'s rule)
Families: PopQA-derived statements (wide popularity range -> many unknowns), Burger et al.'s cities
and neg_cities, and arithmetic statements with 1-3 operations (Poulis et al.'s sets when available).

Tests
  B1  PopQA: probe trained on KNOWN statements; AUROC on held-out known vs unknown, with known items
      matched to unknown ones on statement surprisal (so 'unknown' is not just 'less fluent').
  B2  cities: same, known vs unknown cities.
  B3  Poulis transfer: cities-trained probe on arithmetic, split by whether the model can compute it.
Kill criterion: matched gap (known - unknown AUROC) >= 0.10 with CI above 0, in >= 2 of 3 models.
"""
from __future__ import annotations

import csv
import json
import random
import re

import numpy as np

from errprop.io import read_jsonl, write_jsonl

from .common import (MassMean, Reader, auc, best_layer, boot_ci, chat, data_dir, group_split, last_number,
                     layer_list, logistic, make_generator, num_equal, run_dir, text_match)

POPQA_TEMPLATES = {
    "occupation": "{s}'s occupation is {o}.", "place of birth": "{s} was born in {o}.",
    "genre": "The genre of {s} is {o}.", "father": "The father of {s} is {o}.",
    "mother": "The mother of {s} is {o}.", "country": "{s} is located in the country of {o}.",
    "producer": "{s} was produced by {o}.", "director": "{s} was directed by {o}.",
    "capital of": "{s} is the capital of {o}.", "screenwriter": "The screenwriter of {s} is {o}.",
    "composer": "The composer of {s} is {o}.", "color": "The color of {s} is {o}.",
    "religion": "The religion of {s} is {o}.", "sport": "{s} plays the sport of {o}.",
    "author": "The author of {s} is {o}.", "capital": "The capital of {s} is {o}.",
}
COUNTRY_ALIASES = {
    "United States": ["united states", "usa", "u s", "us", "america", "united states of america"],
    "United Kingdom": ["united kingdom", "uk", "britain", "great britain", "england", "scotland", "wales"],
    "Russia": ["russia", "russian federation"], "South Korea": ["south korea", "korea", "republic of korea"],
    "Czech Republic": ["czech republic", "czechia"], "Turkey": ["turkey", "turkiye"],
    "Netherlands": ["netherlands", "holland"], "Myanmar": ["myanmar", "burma"],
    "Ivory Coast": ["ivory coast", "cote d ivoire"], "China": ["china", "people s republic of china"],
    "Democratic Republic of the Congo": ["democratic republic of the congo", "drc", "dr congo", "congo"],
}
ANSWER_ONLY = " Answer with the answer only, no explanation."


# ============================================================================ data (login node)

def build_items(cfg: dict) -> None:
    """Writes <data_dir>/b_items.jsonl. PopQA needs the HF datasets cache (prepare_decide.py)."""
    dd = data_dir(cfg)
    rng = random.Random(cfg["seed"])
    items = []
    # --- PopQA
    pq = dd / "popqa.jsonl"
    if pq.exists():
        rows = read_jsonl(pq)
        by_prop = {}
        for r in rows:
            if r["prop"] in POPQA_TEMPLATES:
                by_prop.setdefault(r["prop"], []).append(r)
        # stratify by popularity decile, then cap
        sel = [r for rs in by_prop.values() for r in rs]
        sel.sort(key=lambda r: r["s_pop"])
        dec = np.array_split(np.arange(len(sel)), 10)
        per = cfg["b"]["popqa_n"] // 10
        pick = [sel[i] for d in dec for i in rng.sample(list(d), min(per, len(d)))]
        for r in pick:
            pool = [x for x in by_prop[r["prop"]] if x["obj"] not in r["answers"] and
                    not any(a.lower() == x["obj"].lower() for a in r["answers"])]
            if not pool:
                continue
            wrong = rng.choice(pool)["obj"]
            t = POPQA_TEMPLATES[r["prop"]]
            prefix = t.split("{o}")[0].format(s=r["subj"])
            g = f"popqa-{r['id']}"
            know = {"questions": [r["question"] + ANSWER_ONLY,
                                  f"Complete the sentence with the missing words only: \"{prefix}...\""],
                    "type": "text", "answers": r["answers"]}
            for lab, o in ((1, r["obj"]), (0, wrong)):
                items.append({"id": f"{g}-{lab}", "family": "popqa", "group": g, "label": lab,
                              "statement": t.format(s=r["subj"], o=o), "pop": r["s_pop"], "know": know})
    # --- cities / neg_cities (Burger et al., MIT licence)
    for fam in ("cities", "neg_cities"):
        f = dd / "truth_is_universal" / f"{fam}.csv"
        if not f.exists():
            continue
        for k, r in enumerate(csv.DictReader(open(f))):
            c = r["correct_country"]
            know = {"questions": [f"In which country is the city of {r['city']}?" + ANSWER_ONLY,
                                  f"{r['city']} is a city in which country?" + ANSWER_ONLY],
                    "type": "text", "answers": COUNTRY_ALIASES.get(c, [c])}
            items.append({"id": f"{fam}-{k}", "family": fam, "group": f"city-{r['city']}",
                          "label": int(r["label"]), "statement": r["statement"], "know": know})
    # --- arithmetic (Poulis et al. sets if downloaded, else our generator in the same format)
    for n_ops in (1, 2, 3):
        f = dd / "poulis" / f"arith_{n_ops}op{'s' if n_ops > 1 else ''}.csv"
        rows = list(csv.DictReader(open(f))) if f.exists() else _gen_arith(n_ops, 1000, cfg["seed"])
        for k, r in enumerate(rows):
            expr = r["statement"].rsplit("=", 1)[0]
            from errprop.arith import safe_eval
            v = safe_eval(expr.replace("*", "*"))
            if v is None:
                continue
            know = {"questions": [f"Compute {expr}." + " Answer with the number only.",
                                  f"What is the value of {expr}? Reply with just the number."],
                    "type": "num", "value": float(v)}
            items.append({"id": f"arith{n_ops}-{k}", "family": f"arith{n_ops}", "group": f"arith-{expr}",
                          "label": int(r["label"]), "statement": r["statement"], "know": know})
    write_jsonl(dd / "b_items.jsonl", items)
    fam = {}
    for it in items:
        fam[it["family"]] = fam.get(it["family"], 0) + 1
    print(f"[b] {len(items)} statements: {fam}")


def _gen_arith(n_ops: int, n: int, seed: int) -> list[dict]:
    """Fallback generator in Poulis et al.'s format ("46-75=-30", label). Half true, half off by a
    small amount."""
    rng = random.Random(seed + n_ops)
    out = []
    while len(out) < n:
        a = [rng.randint(1, 99) for _ in range(n_ops + 1)]
        ops = [rng.choice("+-*") for _ in range(n_ops)]
        if n_ops == 1:
            e = f"{a[0]}{ops[0]}{a[1]}"
        elif n_ops == 2:
            e = f"({a[0]}{ops[0]}{a[1]}){ops[1]}{a[2]}"
        else:
            e = f"({a[0]}{ops[0]}{a[1]}){ops[1]}({a[2]}{ops[2]}{a[3]})"
        v = int(eval(e))                                   # noqa: S307 (digits and +-*() only)
        lab = int(rng.random() < 0.5)
        shown = v if lab else v + rng.choice([-10, -2, -1, 1, 2, 10])
        out.append({"statement": f"{e}={shown}", "label": lab})
    return out


# ============================================================================ knowledge (GPU, vLLM)

def stage_b_know(cfg: dict, tag: str, model_name: str) -> None:
    rd = run_dir(cfg, tag)
    items = read_jsonl(data_dir(cfg) / "b_items.jsonl")
    groups = {}
    for it in items:
        groups.setdefault(it["group"], it["know"])
    keys = sorted(groups)
    gen = make_generator(cfg, model_name)
    prompts, owners = [], []
    for g in keys:
        for q in groups[g]["questions"]:
            prompts.append(chat(gen.tokenizer, q))
            owners.append(g)
    outs = gen.generate(prompts, 1, 0.0, cfg["b"]["know_max_tokens"], cfg["seed"])
    res = {}
    for g, o in zip(owners, outs):
        k = groups[g]
        ans = o[0]
        ok = (num_equal(last_number(ans), k["value"]) if k["type"] == "num" else text_match(ans, k["answers"]))
        res.setdefault(g, []).append(bool(ok))
    know = {g: (1 if all(v) else 0 if not any(v) else -1) for g, v in res.items()}
    json.dump(know, open(rd / "b_know.json", "w"))
    fams = {}
    for it in items:
        fams.setdefault(it["family"], []).append(know[it["group"]])
    print("[b] knowledge per family (known / unknown / borderline):",
          {f: (v.count(1), v.count(0), v.count(-1)) for f, v in fams.items()})


# ============================================================================ states (GPU, HF)

def stage_b_extract(cfg: dict, tag: str, model_name: str) -> None:
    rd = run_dir(cfg, tag)
    items = read_jsonl(data_dir(cfg) / "b_items.jsonl")
    R = Reader(model_name)
    L = layer_list(R.n_layers, cfg["probes"]["layer_step"])
    H, nll = R.read([it["statement"] for it in items], L, None, add_special=True, bs=cfg["b"]["batch"])
    np.save(rd / "b_acts.npy", H[:, 0])
    np.save(rd / "b_nll.npy", nll)
    json.dump({"layers": L}, open(rd / "b_layers.json", "w"))
    print(f"[b] extracted {len(items)} statements x {len(L)} layers")


# ============================================================================ analysis (CPU)

def _matched_gap(s, y, known, nll, groups, seed, n_boot=300):
    """AUROC on unknown items vs on known items matched to them on statement NLL (nearest neighbour,
    within the same truth label)."""
    rng = np.random.default_rng(seed)
    unk = np.where(known == 0)[0]
    kn = np.where(known == 1)[0]
    if len(unk) < 30 or len(kn) < 30 or len(np.unique(y[unk])) < 2:
        return None
    match = []
    for i in unk:
        cand = kn[y[kn] == y[i]]
        match.append(cand[np.argmin(np.abs(nll[cand] - nll[i]))])
    match = np.array(match)
    a_u, a_k = auc(y[unk], s[unk]), auc(y[match], s[match])
    a_all_k = auc(y[kn], s[kn])
    idx_pairs = np.arange(len(unk))
    pg = groups[unk]

    def gap(ix):
        return auc(y[match[ix]], s[match[ix]]) - auc(y[unk[ix]], s[unk[ix]])
    return {"auroc_known_all": a_all_k, "auroc_known_matched": a_k, "auroc_unknown": a_u,
            "gap_matched": a_k - a_u, "gap_ci": boot_ci(gap, pg, n_boot, seed),
            "n_known": int(len(kn)), "n_unknown": int(len(unk)),
            "nll_known_mean": float(nll[kn].mean()), "nll_unknown_mean": float(nll[unk].mean()),
            "nll_matched_mean": float(nll[match].mean())}


def _family_test(H, items, know, nll, fam_train, fam_test, seed, make):
    """Train on KNOWN items of fam_train (train split by group); test on held-out items of fam_test."""
    fam = np.array([it["family"] for it in items])
    y = np.array([it["label"] for it in items])
    g = np.array([it["group"] for it in items])
    k = np.array([know.get(it["group"], -1) for it in items])
    tr_pool = np.isin(fam, fam_train)
    te_pool = np.isin(fam, fam_test)
    same = set(fam_train) & set(fam_test)
    dv, te = group_split(g, 0.3, seed)
    tr = tr_pool & (k == 1) & (dv if same else np.ones(len(y), bool))
    tst = te_pool & (k >= 0) & (te if same else np.ones(len(y), bool))
    if tr.sum() < 50 or len(np.unique(y[tr])) < 2:
        return None
    l, curve = best_layer(H[tr], y[tr], g[tr], make)
    m = make().fit(H[tr, l].astype(np.float32), y[tr])
    s = np.full(len(y), np.nan)
    s[tst] = m.decision_function(H[tst, l].astype(np.float32))
    idx = np.where(tst)[0]
    r = _matched_gap(s[idx], y[idx], k[idx], nll[idx], g[idx], seed)
    return {"train": fam_train, "test": fam_test, "layer_index": l, "n_train": int(tr.sum()),
            "n_test": int(tst.sum()), "curve": curve, "result": r,
            "auroc_test_all": auc(y[idx], s[idx])}


def stage_b_fit(cfg: dict, tag: str) -> None:
    rd = run_dir(cfg, tag)
    if not (rd / "b_acts.npy").exists():
        return
    items = read_jsonl(data_dir(cfg) / "b_items.jsonl")
    know = json.load(open(rd / "b_know.json"))
    H = np.load(rd / "b_acts.npy", mmap_mode="r")
    H = np.asarray(H)
    nll = np.load(rd / "b_nll.npy")
    L = json.load(open(rd / "b_layers.json"))["layers"]
    seed = cfg["seed"]
    fams = sorted({it["family"] for it in items})
    res = {"model": tag, "layers": L, "families": {}}
    for f in fams:
        ks = [know.get(it["group"], -1) for it in items if it["family"] == f]
        res["families"][f] = {"n": len(ks), "known": ks.count(1), "unknown": ks.count(0), "borderline": ks.count(-1)}
    tests = [("B1_popqa", ["popqa"], ["popqa"]),
             ("B2_cities", ["cities", "neg_cities"], ["cities", "neg_cities"]),
             ("B3_cities_to_arith1", ["cities", "neg_cities"], ["arith1"]),
             ("B3_cities_to_arith2", ["cities", "neg_cities"], ["arith2"]),
             ("B3_cities_to_arith3", ["cities", "neg_cities"], ["arith3"]),
             ("B3b_arith1_to_arith23", ["arith1"], ["arith2", "arith3"]),
             ("B4_cities_to_popqa", ["cities", "neg_cities"], ["popqa"])]
    for name, a, b in tests:
        if not (set(a) <= set(fams) and set(b) <= set(fams)):
            continue
        res[name] = {"logistic": _family_test(H, items, know, nll, a, b, seed, lambda: logistic(0.1)),
                     "massmean": _family_test(H, items, know, nll, a, b, seed, MassMean)}
    json.dump(res, open(rd / "b.json", "w"), indent=1, default=float)
    r = (res.get("B1_popqa") or {}).get("logistic") or {}
    rr = r.get("result") or {}
    print(f"[b] {tag}: PopQA known(matched) {rr.get('auroc_known_matched')}  unknown {rr.get('auroc_unknown')}")
