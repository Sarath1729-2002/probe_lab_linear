"""Token-by-token reading (GPU, read-only): the bucket's detection direction (sugg family, fitted on
known items where the planted information ends) is projected onto every token from that point,
through the rest of the prompt, and over the first `max_tokens` tokens of the model's own reply.
Nothing is changed in the forward pass; the reply is the one generated in `gen`."""
from __future__ import annotations

import random

import numpy as np

from .common import read_jsonl, write_jsonl

from .common import Reader, chat

from .run import bdir, load, root


def stage_traj(cfg: dict, buckets: list[str]) -> None:
    p = root(cfg) / "directions.npz"
    if not p.exists():
        print("[traj] run fit first")
        return
    D = np.load(p)
    items, V = load(cfg)
    vmap = {v["vid"]: v for v in V}
    T, n_items = cfg["traj"]["max_tokens"], cfg["traj"]["n_items"]
    R = None
    for b in buckets:
        if f"{b}__w" not in D.files or (bdir(cfg, b) / "traj.jsonl").exists():
            continue
        mu, sd, w, layer = D[f"{b}__mu"], D[f"{b}__sd"], D[f"{b}__w"], int(D[f"{b}__layer"])
        G = {r["vid"]: r for r in read_jsonl(bdir(cfg, b) / "gen.jsonl")}
        ids = [i for i, it in items.items() if it["bucket"] == b and G.get(f"{i}|base|base", {}).get("outcome") == "GOLD"
               and f"{i}|sugg|wrong" in G and G[f"{i}|sugg|wrong"]["text"] is not None]
        random.Random(cfg["seed"]).shuffle(ids)
        # balance adopted and held (adoption is usually the rarer outcome)
        ad = [i for i in ids if G[f"{i}|sugg|wrong"]["outcome"] == "WRONG"]
        ho = [i for i in ids if G[f"{i}|sugg|wrong"]["outcome"] == "GOLD"]
        na = min(len(ad), n_items // 2)
        ids = ad[:na] + ho[: n_items - na]
        if R is None:
            R = Reader(cfg["model"], cfg["extract"]["dtype"])
        torch = R.torch
        rows = []
        for i in ids:
            for role in ("wrong", "right"):
                vid = f"{i}|sugg|{role}"
                if vid not in G or G[vid]["text"] is None:
                    continue
                v = vmap[vid]
                pr = chat(R.tok, v["user"])
                info = pr.rfind(v["user"]) + v["info_end"]
                full = pr + G[vid]["text"]
                enc = R.tok(full, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
                off = enc.pop("offset_mapping")[0].tolist()
                ends = [e for _, e in off]
                t_info = max([t for t, e in enumerate(ends) if e <= info] or [0])
                t_last = max([t for t, e in enumerate(ends) if e <= len(pr)] or [0])
                stop = min(len(ends), t_last + 1 + T)
                with torch.no_grad():
                    hs = R.model(input_ids=enc["input_ids"][:, :stop].to(R.model.device),
                                 output_hidden_states=True).hidden_states[layer][0].float().cpu().numpy()
                s = ((hs - mu) / sd) @ w
                rows.append({"vid": vid, "role": role, "outcome": G[vid]["outcome"], "layer": layer,
                             "i_info": int(t_info), "i_last": int(t_last), "scores": s[t_info:].round(4).tolist()})
        write_jsonl(bdir(cfg, b) / "traj.jsonl", rows)
        print(f"[traj] {b}: {len(rows)} prompts at layer {layer}", flush=True)
