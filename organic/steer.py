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
    op = (res["buckets"][b]["probes"][fam]["last_prompt_token"].get("outcome_probe") or {})
    layer = op.get("layer") or res["buckets"][b]["probes"][fam]["last_prompt_token"]["layer"]
    li = idx["layers"].index(layer)
    A = np.load(d / "acts.npy", mmap_mode="r")
    rows = [(k, v.split("|")[0], G[v]["outcome"]) for k, v in enumerate(idx["vids"])
            if v.split("|")[1] == fam and v.split("|")[2] == "wrong" and known.get(v.split("|")[0])
            and G[v]["outcome"] in ("WRONG", "GOLD")]
    ids = sorted({i for _, i, _ in rows})
    rng = np.random.default_rng(cfg["seed"])
    fit_ids = set(rng.permutation(ids)[: len(ids) // 2].tolist())
    fr = [(k, o) for k, i, o in rows if i in fit_ids]
    X = np.asarray(A[[k for k, _ in fr], 1, li], np.float32)
    o = np.array([o for _, o in fr])
    if (o == "WRONG").sum() < 10 or (o == "GOLD").sum() < 10:
        raise SystemExit(f"[steer] too few follow/recover items to fit a direction ({(o == 'WRONG').sum()}/{(o == 'GOLD').sum()})")
    v = X[o == "GOLD"].mean(0) - X[o == "WRONG"].mean(0)
    return v, layer, fit_ids, {"n_fit_follow": int((o == "WRONG").sum()), "n_fit_recover": int((o == "GOLD").sum()),
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
