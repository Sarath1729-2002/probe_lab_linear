"""Causal test (GPU): does pushing the residual stream along the 'recover, not follow' direction reduce
false verification, and is a random direction of the same size harmless?

Direction (CPU, from stored activations): on known ProcessBench items with a WRONG solution in the steering
frame, mean state at the last prompt token for RECOVER minus mean state for FOLLOW, at the frame's
outcome-probe layer. It is fitted on half the items (grouped by item) and tested on the other half, so no
test item contributed to the direction.

Intervention: alpha * direction added to the output of that decoder layer at every position (prompt and
generated tokens), greedy decoding, same prompts. alpha = 1 moves the state by the full FOLLOW -> RECOVER
mean difference; negative alpha pushes toward following. A random unit direction scaled to the same norm
is the specificity control. Correct solutions in the same frame check the side effect: a direction
that just makes the model reject everything would also make it 'correct' right solutions into wrong ones.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from .common import DEVICE, read_jsonl, render, write_jsonl
from .frames import certifies
from .grade import grade
from .run import bdir, load, root


def _direction(cfg, b, fam, items, G, known):
    d = bdir(cfg, b)
    idx = json.load(open(d / "acts_index.json"))
    res = json.load(open(root(cfg) / "results.json"))
    pl = res["buckets"][b]["probes"][fam]["last_prompt_token"]
    op = pl.get("outcome_probe") or {}
    layer = op.get("layer") or pl["layer"]
    # never steer in the last quarter of the network: there the readout layer is close to the logits and
    # adding a mean difference breaks generation (Llama, organic_v3 run 1: layer 32 -> 92% unparseable replies)
    cap = cfg["steer"].get("max_layer_frac", 0.75) * max(idx["layers"])
    if layer > cap:
        curve = op.get("curve") or pl.get("curve_train_cv")
        ok = [(c, l) for c, l in zip(curve, idx["layers"]) if l <= cap and c == c] if curve else []
        layer = max(ok)[1] if ok else max(l for l in idx["layers"] if l <= cap)
    li = idx["layers"].index(layer)
    A = np.load(d / "acts.npy", mmap_mode="r")
    rows_all = [(k, v.split("|")[0], G[v]["outcome"]) for k, v in enumerate(idx["vids"])
                if v.split("|")[1] == fam and v.split("|")[2] == "wrong" and known.get(v.split("|")[0])
                and G[v]["outcome"] != "SKIP"]
    rows = [r for r in rows_all if r[2] in ("WRONG", "GOLD")]
    ids = sorted({i for _, i, _ in rows_all})
    rng = np.random.default_rng(cfg["seed"])
    fit_ids = set(rng.permutation(ids)[: len(ids) // 2].tolist())
    fr = [(k, o) for k, i, o in rows if i in fit_ids]
    X = np.asarray(A[[k for k, _ in fr], 1, li], np.float32)
    o = np.array([o for _, o in fr])
    contrast = "recover - follow"
    away = o == "GOLD"
    if away.sum() < 10:                    # e.g. Mistral almost never recovers: contrast follow vs everything else
        X = np.asarray(A[[k for k, i, _ in rows_all if i in fit_ids], 1, li], np.float32)
        o = np.array([oc for _, i, oc in rows_all if i in fit_ids])
        away = o != "WRONG"
        contrast = "not follow (recover or other) - follow"
    if (o == "WRONG").sum() < 10 or away.sum() < 10:
        raise SystemExit(f"[steer] too few items to fit a direction (follow {(o == 'WRONG').sum()}, other side {away.sum()})")
    v = X[away].mean(0) - X[o == "WRONG"].mean(0)
    return v, layer, fit_ids, {"contrast": contrast, "n_fit_follow": int((o == "WRONG").sum()), "n_fit_recover": int(away.sum()),
                               "norm": float(np.linalg.norm(v)),
                               "mean_state_norm": float(np.linalg.norm(X, axis=1).mean())}


def stage_steer(cfg: dict) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sc = cfg["steer"]
    b, fam = sc.get("bucket", "processbench"), sc["family"]
    d = bdir(cfg, b)
    out = d / "steer.jsonl"
    items, V = load(cfg)
    vmap = {v["vid"]: v for v in V}
    G = {r["vid"]: r for r in read_jsonl(d / "gen.jsonl")}
    known = {i: G.get(f"{i}|base|base", {}).get("outcome") == "GOLD" for i, it in items.items() if it["bucket"] == b}
    vec, layer, fit_ids, info = _direction(cfg, b, fam, items, G, known)
    test = [i for i in known if known[i] and i not in fit_ids]
    rng = np.random.default_rng(cfg["seed"] + 7)
    rng.shuffle(test)
    wrong = [f"{i}|{fam}|wrong" for i in test if f"{i}|{fam}|wrong" in G][: sc["n_wrong"]]
    right = [f"{i}|{fam}|right" for i in test if f"{i}|{fam}|right" in G][: sc["n_right"]]
    rnd = rng.normal(size=vec.shape).astype(np.float32)
    rnd *= np.linalg.norm(vec) / np.linalg.norm(rnd)
    conds = [("direction", a) for a in sc["alphas"]] + [("random", a) for a in sc["random_alphas"]]
    done = {(r["cond"], r["alpha"], r["vid"]) for r in read_jsonl(out)} if out.exists() else set()
    rows = read_jsonl(out) if out.exists() else []

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.bfloat16, device_map=DEVICE).eval()
    block = model.model.layers[layer - 1]                 # hidden_states[layer] is the output of block layer-1
    state = {"add": None}

    def hook(_m, _inp, output):
        if state["add"] is None:
            return output
        if isinstance(output, tuple):
            return (output[0] + state["add"],) + tuple(output[1:])
        return output + state["add"]

    h = block.register_forward_hook(hook)
    try:
        for cname, alpha in conds:
            base = vec if cname == "direction" else rnd
            state["add"] = None if alpha == 0 else torch.tensor(alpha * base, dtype=torch.bfloat16, device=model.device)
            todo = [v for v in wrong + right if (cname, alpha, v) not in done]
            for s in range(0, len(todo), sc["batch"]):
                chunk = todo[s:s + sc["batch"]]
                prompts = [render(tok, vmap[v])[0] for v in chunk]
                enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
                with torch.no_grad():
                    gen = model.generate(**enc, max_new_tokens=sc["max_tokens"], do_sample=False,
                                         pad_token_id=tok.pad_token_id)
                texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                for v, t in zip(chunk, texts):
                    oc, a = grade(items[v.split("|")[0]], t)
                    rows.append({"cond": cname, "alpha": alpha, "vid": v, "role": v.split("|")[2], "outcome": oc,
                                 "answer": a, "certifies": certifies(t), "n_chars": len(t), "text": t})
                write_jsonl(out, rows)
            print(f"[steer] {cname} alpha={alpha}: {len(todo)} prompts", flush=True)
    finally:
        h.remove()
    summ = summarize(rows)
    json.dump({"family": fam, "layer": layer, **info, "n_test_wrong": len(wrong), "n_test_right": len(right),
               "conditions": summ}, open(d / "steer.json", "w"), indent=1)
    print(f"[steer] wrote {d / 'steer.json'}")


def summarize(rows):
    by = defaultdict(list)
    for r in rows:
        by[(r["cond"], r["alpha"])].append(r)
    out = []
    for (c, a), rs in sorted(by.items(), key=lambda x: (x[0][0] != "direction", x[0][1])):
        w = [r for r in rs if r["role"] == "wrong"]
        g = [r for r in rs if r["role"] == "right"]
        f = lambda xs, k, v: float(np.mean([x[k] == v for x in xs])) if xs else None
        out.append({"cond": c, "alpha": a, "n_wrong": len(w), "follow": f(w, "outcome", "WRONG"),
                    "recover": f(w, "outcome", "GOLD"), "other": f(w, "outcome", "OTHER"),
                    "certify_wrong": f(w, "certifies", True), "n_right": len(g),
                    "correct_on_right": f(g, "outcome", "GOLD"), "mean_chars": float(np.mean([r["n_chars"] for r in rs]))})
    return out


# ---------------------------------------------------------------------------------------------- v4

def stage_steer_multi(cfg: dict) -> None:
    """v4: ONE direction, mean(hold) - mean(defer) at the last prompt token, pooled over datasets and over
    `fit_families`, fitted on half the items; tested on the other half in every `test_families` channel.
    Layer = median of the channels' outcome-probe layers (deference.json), capped at max_layer_frac."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .deference import _X, _load_all, _rows
    sc = cfg["steer"]
    out_p = root(cfg) / "steer_multi.jsonl"
    items, D = _load_all(cfg)
    _, V = load(cfg)
    vmap = {v["vid"]: v for v in V}
    dj = json.load(open(root(cfg) / "deference.json"))
    layers = dj["probes"]["layers"]
    ls = [dj["probes"]["per_channel"][c]["last_prompt_token"].get("outcome_layer") for c in sc["fit_channels"]]
    ls = [l for l in ls if l is not None]
    cap = sc.get("max_layer_frac", 0.75) * max(layers)
    layer = min(int(np.median(ls)) if ls else layers[len(layers) // 2], int(cap))
    layer = max(l for l in layers if l <= layer)
    li = layers.index(layer)
    allk = sorted({i for d in D.values() for i in d["known"]})
    rng = np.random.default_rng(cfg["seed"])
    fit_ids = set(rng.permutation(allk)[: len(allk) // 2].tolist())
    R = [r for c in sc["fit_channels"] for r in _rows(D, f"ch_{c}", ["wrong"])
         if r[2] in fit_ids and r[4] in ("WRONG", "GOLD")]
    X = _X(D, R, 1, li)
    R = [r for b in D for r in R if r[0] == b]
    z = np.array([r[4] == "WRONG" for r in R])
    if z.sum() < 20 or (~z).sum() < 20:
        raise SystemExit(f"[steer] too few defer/hold rows to fit ({z.sum()}/{(~z).sum()})")
    vec = X[~z].mean(0) - X[z].mean(0)
    test = [i for i in allk if i not in fit_ids]
    rng.shuffle(test)
    todo_v = []
    for c in sc["test_channels"]:
        w = [f"{i}|ch_{c}|wrong" for i in test if any(f"{i}|ch_{c}|wrong" in d["G"] for d in D.values())][: sc["n_wrong"]]
        r_ = [f"{i}|ch_{c}|right" for i in test if any(f"{i}|ch_{c}|right" in d["G"] for d in D.values())][: sc["n_right"]]
        todo_v += w + r_
    rnd = rng.normal(size=vec.shape).astype(np.float32)
    rnd *= np.linalg.norm(vec) / np.linalg.norm(rnd)
    conds = [("direction", a) for a in sc["alphas"]] + [("random", a) for a in sc["random_alphas"]]
    rows = read_jsonl(out_p) if out_p.exists() else []
    done = {(r["cond"], r["alpha"], r["vid"]) for r in rows}
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.bfloat16, device_map=DEVICE).eval()
    state = {"add": None}

    def hook(_m, _i, output):
        if state["add"] is None:
            return output
        return (output[0] + state["add"],) + tuple(output[1:]) if isinstance(output, tuple) else output + state["add"]

    h = model.model.layers[layer - 1].register_forward_hook(hook)
    try:
        for cname, alpha in conds:
            base = vec if cname == "direction" else rnd
            state["add"] = None if alpha == 0 else torch.tensor(alpha * base, dtype=torch.bfloat16, device=model.device)
            todo = [v for v in todo_v if (cname, alpha, v) not in done]
            for s in range(0, len(todo), sc["batch"]):
                chunk = todo[s:s + sc["batch"]]
                enc = tok([render(tok, vmap[v])[0] for v in chunk], return_tensors="pt", padding=True,
                          add_special_tokens=False).to(model.device)
                with torch.no_grad():
                    gen = model.generate(**enc, max_new_tokens=sc["max_tokens"], do_sample=False, pad_token_id=tok.pad_token_id)
                for v, t in zip(chunk, tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
                    oc, a = grade(items[v.split("|")[0]], t)
                    rows.append({"cond": cname, "alpha": alpha, "vid": v, "channel": v.split("|")[1][3:],
                                 "role": v.split("|")[2], "outcome": oc, "n_chars": len(t), "text": t})
                write_jsonl(out_p, rows)
            print(f"[steer] {cname} alpha={alpha}: {len(todo)} prompts", flush=True)
    finally:
        h.remove()
    summ = []
    for (cname, alpha) in conds:
        for c in sc["test_channels"] + ["ALL"]:
            rs = [r for r in rows if r["cond"] == cname and r["alpha"] == alpha and (c == "ALL" or r["channel"] == c)]
            w = [r for r in rs if r["role"] == "wrong"]
            g = [r for r in rs if r["role"] == "right"]
            f = lambda xs, v: float(np.mean([x["outcome"] == v for x in xs])) if xs else None
            summ.append({"cond": cname, "alpha": alpha, "channel": c, "n_wrong": len(w), "defer": f(w, "WRONG"),
                         "hold": f(w, "GOLD"), "other": f(w, "OTHER"), "n_right": len(g), "correct_on_right": f(g, "GOLD"),
                         "mean_chars": float(np.mean([r["n_chars"] for r in rs])) if rs else None})
    json.dump({"layer": layer, "fit_channels": sc["fit_channels"], "test_channels": sc["test_channels"],
               "n_fit_defer": int(z.sum()), "n_fit_hold": int((~z).sum()), "norm": float(np.linalg.norm(vec)),
               "mean_state_norm": float(np.linalg.norm(X, axis=1).mean()), "conditions": summ},
              open(root(cfg) / "steer_multi.json", "w"), indent=1)
    print(f"[steer] wrote {root(cfg) / 'steer_multi.json'}")
