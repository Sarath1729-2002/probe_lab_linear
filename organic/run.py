"""GPU stages. Nothing is edited: greedy generation from the frozen prompts, then a read-only forward
pass to store residual-stream states at two positions per prompt."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .common import read_jsonl, write_jsonl

from .common import Reader, chat, layer_list, make_generator

from .grade import grade

PROBE_FAMILIES = ("sugg", "doc", "work_full", "work_trunc")


def root(cfg) -> Path:
    d = Path(os.path.expandvars(cfg["out_dir"])) / cfg["run_prefix"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def ddir(cfg) -> Path:
    return Path(os.path.expandvars(cfg["data_dir"]))


def bdir(cfg, b) -> Path:
    d = root(cfg) / b
    d.mkdir(parents=True, exist_ok=True)
    return d


def load(cfg):
    items = {it["id"]: it for it in read_jsonl(ddir(cfg) / "items.jsonl")}
    V = read_jsonl(ddir(cfg) / "variants.jsonl")
    return items, V


def stage_gen(cfg: dict, buckets: list[str]) -> None:
    items, V = load(cfg)
    gen = None
    for b in buckets:
        out = bdir(cfg, b) / "gen.jsonl"
        if out.exists():
            print(f"[gen] {b}: done already")
            continue
        vb = [v for v in V if v["bucket"] == b]
        if not vb:
            continue
        if gen is None:
            gen = make_generator(cfg, cfg["model"])
        tok = gen.tokenizer
        mt = cfg["gen"]["max_tokens"].get(b, cfg["gen"]["max_tokens"]["default"])
        prompts = [chat(tok, v["user"]) for v in vb]
        lim = cfg["gen"]["max_model_len"] - mt
        ok = [len(tok(p, add_special_tokens=False)["input_ids"]) <= lim for p in prompts]
        idx = [i for i, o in enumerate(ok) if o]
        texts = [None] * len(vb)
        if idx:
            res = gen.generate([prompts[i] for i in idx], 1, 0.0, mt, cfg["seed"])
            for i, r in zip(idx, res):
                texts[i] = r[0]
        rows = []
        for v, t, o in zip(vb, texts, ok):
            oc, a = grade(items[v["item"]], t) if o else ("SKIP", None)
            rows.append({"vid": v["vid"], "text": t, "answer": a, "outcome": oc})
        write_jsonl(out, rows)
        c = {}
        for r in rows:
            c[r["outcome"]] = c.get(r["outcome"], 0) + 1
        print(f"[gen] {b}: {len(rows)} prompts {c}", flush=True)


def stage_extract(cfg: dict, buckets: list[str]) -> None:
    items, V = load(cfg)
    R = None
    for b in buckets:
        d = bdir(cfg, b)
        if (d / "acts.npy").exists():
            print(f"[extract] {b}: done already")
            continue
        if not (d / "gen.jsonl").exists():
            print(f"[extract] {b}: no gen.jsonl, skipped")
            continue
        G = {r["vid"]: r for r in read_jsonl(d / "gen.jsonl")}
        vb = [v for v in V if v["bucket"] == b and v["family"] in PROBE_FAMILIES and G[v["vid"]]["outcome"] != "SKIP"]
        if R is None:
            R = Reader(cfg["model"], cfg["extract"]["dtype"])
            L = layer_list(R.n_layers, cfg["probes"]["layer_step"])
        texts, ce = [], []
        for v in vb:
            p = chat(R.tok, v["user"])
            off = p.rfind(v["user"])
            texts.append(p)
            ce.append([off + v["info_end"], len(p)])
        order = np.argsort([len(t) for t in texts])          # length-sorted batches, restored below
        H, _ = R.read([texts[i] for i in order], L, [ce[i] for i in order], add_special=False,
                      bs=cfg["extract"]["batch"])
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        np.save(d / "acts.npy", H[inv])
        json.dump({"vids": [v["vid"] for v in vb], "layers": L, "positions": ["info_end", "last_prompt_token"]},
                  open(d / "acts_index.json", "w"))
        print(f"[extract] {b}: {len(vb)} prompts x 2 positions x {len(L)} layers", flush=True)
