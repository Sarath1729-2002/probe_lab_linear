"""GPU stages. Nothing is edited: greedy generation from the frozen prompts, sampled re-answers of the
bare question (knowledge strength), and read-only forward passes that store residual-stream states.

Every stage is incremental: rerunning only computes prompts that are new since the last run (e.g. a
variant family added to build.py), and keeps everything already on disk."""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

from .common import Reader, chat, layer_list, make_generator, read_jsonl, render, write_jsonl
from .grade import grade

NON_PROBE = ("base", "doubt")          # every other family has wrong / right versions and is probed


def is_probe_family(f: str) -> bool:
    # v6.1 additions (controls, reason-first) are behavioural only: no activations or scores
    return (f not in NON_PROBE and not f.startswith("doubt") and not f.endswith("_doubt") and not f.startswith("b_")
            and not f.startswith("r_") and f not in ("m_bare", "m_key"))


def max_tokens(cfg, b, fam) -> int:
    """Token budget for one prompt: per dataset; reason-first prompts (v6.1) get the larger reasoning budget."""
    mt = cfg["gen"]["max_tokens"]
    if fam.startswith("r_") and "max_tokens_reason" in cfg["gen"]:
        mt = cfg["gen"]["max_tokens_reason"]
    return mt.get(b, mt["default"])


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
    for p in sorted(root(cfg).glob("*/variants2.jsonl")):     # v6: this model's second-stage prompts
        V += read_jsonl(p)
    return items, V


def _grade(items, v, text):
    if v.get("free"):                                         # published baseline prompts: free-form reply
        from .design6 import grade_free
        return grade_free(items[v["item"]], text)
    return grade(items[v["item"]], text)


def _check_items(cfg, b, items):
    """A rebuilt items.jsonl must match the one the existing replies were graded against."""
    p = bdir(cfg, b) / "items_fingerprint.json"
    fp = {i: [it["question"][:80], str(it["gold"])[:40]] for i, it in items.items() if it["bucket"] == b}
    if p.exists():
        old = json.load(open(p))
        bad = [i for i in old if i in fp and old[i] != fp[i]]
        if bad:
            raise SystemExit(f"[{b}] items.jsonl changed for {len(bad)} existing ids (e.g. {bad[0]}); "
                             "use a new run_prefix instead of mixing item sets")
    json.dump(fp, open(p, "w"))


def stage_gen(cfg: dict, buckets: list[str]) -> None:
    items, V = load(cfg)
    gen = None
    for b in buckets:
        _check_items(cfg, b, items)
        out = bdir(cfg, b) / "gen.jsonl"
        old = read_jsonl(out) if out.exists() else []
        have = {r["vid"] for r in old}
        vb = [v for v in V if v["bucket"] == b and v["vid"] not in have]
        if not vb:
            print(f"[gen] {b}: nothing new ({len(old)} prompts on disk)")
            continue
        if gen is None:
            gen = make_generator(cfg, cfg["model"])
        tok = gen.tokenizer
        mts = [max_tokens(cfg, b, v["family"]) for v in vb]
        prompts = [render(tok, v)[0] for v in vb]
        ok = [len(tok(p, add_special_tokens=False)["input_ids"]) <= cfg["gen"]["max_model_len"] - m for p, m in zip(prompts, mts)]
        # generate and save in chunks, so a job that hits its time limit keeps what it finished
        chunk = cfg["gen"].get("chunk", 2000)
        rows, done = [], old
        for c0 in range(0, len(vb), chunk):
            sl = range(c0, min(len(vb), c0 + chunk))
            texts = {}
            for m in sorted({mts[i] for i in sl}):             # one call per token budget
                idx = [i for i in sl if ok[i] and mts[i] == m]
                if idx:
                    for i, r in zip(idx, gen.generate([prompts[i] for i in idx], 1, 0.0, m, cfg["seed"])):
                        texts[i] = r[0]
            for i in sl:
                oc, a = _grade(items, vb[i], texts[i]) if ok[i] else ("SKIP", None)
                rows.append({"vid": vb[i]["vid"], "text": texts.get(i), "answer": a, "outcome": oc})
            done = old + rows
            write_jsonl(out, done)
            print(f"[gen] {b}: {len(rows)}/{len(vb)} saved", flush=True)
        print(f"[gen] {b}: {len(rows)} new prompts {dict(Counter(r['outcome'] for r in rows))}", flush=True)


def stage_extend(cfg: dict, buckets: list[str]) -> None:
    """v6.1: regenerate, with a larger budget, every stored reply that used its whole token budget (any family, same
    rule for all). Plain greedy generation of the same prompt; the original is kept as text_v0 / outcome_v0e."""
    items, V = load(cfg)
    Vd = {v["vid"]: v for v in V}
    ext = cfg["gen"]["extend_to"]
    gen = None
    for b in buckets:
        p = bdir(cfg, b) / "gen.jsonl"
        if not p.exists():
            continue
        rows = read_jsonl(p)
        if gen is None:
            gen = make_generator(cfg, cfg["model"])
        tok = gen.tokenizer
        todo = []
        for k, r in enumerate(rows):
            v = Vd.get(r["vid"])
            if v is None or r.get("extended") or r["outcome"] == "SKIP" or not r.get("text"):
                continue
            m = max_tokens(cfg, b, v["family"])
            target = ext.get(b, ext["default"])
            if target > m and len(tok(r["text"], add_special_tokens=False)["input_ids"]) >= m - 3:
                todo.append((k, v, target))
        if not todo:
            print(f"[extend] {b}: no cut-off replies")
            continue
        prompts = [render(tok, v)[0] for _, v, _ in todo]
        moved = Counter()
        for m in sorted({x[2] for x in todo}):
            sel = [j for j, x in enumerate(todo) if x[2] == m]
            sel = [j for j in sel if len(tok(prompts[j], add_special_tokens=False)["input_ids"]) <= cfg["gen"]["max_model_len"] - m]
            for j, res in zip(sel, gen.generate([prompts[j] for j in sel], 1, 0.0, m, cfg["seed"])):
                k, v, _ = todo[j]
                r = rows[k]
                oc, a = _grade(items, v, res[0])
                r.update({"text_v0": r["text"], "outcome_v0e": r["outcome"], "text": res[0], "outcome": oc, "answer": a,
                          "extended": m})
                moved[f"{r['outcome_v0e']}->{oc}"] += 1
        write_jsonl(p, rows)
        print(f"[extend] {b}: {len(todo)} cut-off replies regenerated {dict(moved)}", flush=True)


def stage_gen2(cfg: dict, buckets: list[str]) -> None:
    """v6 second stage: challenges that follow the model's OWN reply to the bare question (only where that reply
    was correct). The prompts are built here, per model, then generated like any other."""
    from .design6 import stage2_variants
    items, _ = load(cfg)
    for b in buckets:
        d = bdir(cfg, b)
        if not (d / "gen.jsonl").exists():
            print(f"[gen2] {b}: no gen.jsonl yet")
            continue
        if not (d / "variants2.jsonl").exists():
            G = {r["vid"]: r for r in read_jsonl(d / "gen.jsonl")}
            V2 = []
            for i, it in items.items():
                g = G.get(f"{i}|base|base")
                if it["bucket"] == b and g and g["outcome"] == "GOLD" and g.get("text"):
                    V2 += stage2_variants(it, g["text"])
            write_jsonl(d / "variants2.tmp", V2)              # atomic: another GPU's process may be loading prompts
            (d / "variants2.tmp").replace(d / "variants2.jsonl")
            print(f"[gen2] {b}: {len(V2)} second-stage prompts from {len(V2) // 3} correct first replies")
    stage_gen(cfg, buckets)


def stage_regrade(cfg: dict, buckets: list[str]) -> None:
    """Re-grade every stored reply with the current grader. The first grade is kept as outcome_v0."""
    items, _ = load(cfg)
    for b in buckets:
        p = bdir(cfg, b) / "gen.jsonl"
        if not p.exists():
            continue
        rows = read_jsonl(p)
        moved = Counter()
        for r in rows:
            r.setdefault("outcome_v0", r["outcome"])
            if r["outcome"] == "SKIP":
                continue
            oc, a = grade(items[r["vid"].split("|")[0]], r["text"])
            moved[(r["outcome"], oc)] += r["outcome"] != oc
            r["outcome"], r["answer"] = oc, a
        write_jsonl(p, rows)
        ch = {f"{a}->{c}": n for (a, c), n in moved.items() if n}
        print(f"[regrade] {b}: {len(rows)} replies; changed {sum(moved.values())}: {ch}", flush=True)


def stage_confidence(cfg: dict, buckets: list[str]) -> None:
    """Knowledge strength: the bare question answered k more times at temperature T. conf = share correct."""
    items, V = load(cfg)
    k, T = cfg["confidence"]["n_samples"], cfg["confidence"]["temperature"]
    gen = None
    for b in buckets:
        out = bdir(cfg, b) / "confidence.jsonl"
        old = read_jsonl(out) if out.exists() else []
        have = {r["item"] for r in old}
        vb = [v for v in V if v["bucket"] == b and v["family"] == "base" and v["item"] not in have]
        if not vb:
            print(f"[confidence] {b}: nothing new")
            continue
        if gen is None:
            gen = make_generator(cfg, cfg["model"])
        tok = gen.tokenizer
        mt = cfg["gen"]["max_tokens"].get(b, cfg["gen"]["max_tokens"]["default"])
        prompts = [chat(tok, v["user"]) for v in vb]
        lim = cfg["gen"]["max_model_len"] - mt
        keep = [i for i, p in enumerate(prompts) if len(tok(p, add_special_tokens=False)["input_ids"]) <= lim]
        res = gen.generate([prompts[i] for i in keep], k, T, mt, cfg["seed"] + 1)
        rows = []
        for i, outs in zip(keep, res):
            it = items[vb[i]["item"]]
            g = [grade(it, t)[0] for t in outs]
            rows.append({"item": it["id"], "conf": sum(x == "GOLD" for x in g) / len(g),
                         "wrong_rate": sum(x == "WRONG" for x in g) / len(g)})
        write_jsonl(out, old + rows)
        print(f"[confidence] {b}: {len(rows)} items, mean conf {np.mean([r['conf'] for r in rows]):.2f}", flush=True)


def stage_extract(cfg: dict, buckets: list[str]) -> None:
    items, V = load(cfg)
    R = None
    for b in buckets:
        d = bdir(cfg, b)
        if not (d / "gen.jsonl").exists():
            print(f"[extract] {b}: no gen.jsonl, skipped")
            continue
        G = {r["vid"]: r for r in read_jsonl(d / "gen.jsonl")}
        old_v = json.load(open(d / "acts_index.json"))["vids"] if (d / "acts.npy").exists() else []
        have = set(old_v)
        vb = [v for v in V if v["bucket"] == b and is_probe_family(v["family"]) and v["vid"] in G
              and G[v["vid"]]["outcome"] != "SKIP" and v["vid"] not in have]
        if not vb:
            print(f"[extract] {b}: nothing new ({len(old_v)} prompts on disk)")
            continue
        if R is None:
            R = Reader(cfg["model"], cfg["extract"]["dtype"])
            L = layer_list(R.n_layers, cfg["probes"]["layer_step"])
        texts, ce = [], []
        for v in vb:
            p, info = render(R.tok, v)
            texts.append(p)
            ce.append([info, len(p)])
        order = np.argsort([len(t) for t in texts])          # length-sorted batches, restored below
        H, _ = R.read([texts[i] for i in order], L, [ce[i] for i in order], add_special=False,
                      bs=cfg["extract"]["batch"])
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        H = H[inv]
        if old_v:
            H = np.concatenate([np.load(d / "acts.npy"), H])
        np.save(d / "acts.npy", H)
        json.dump({"vids": old_v + [v["vid"] for v in vb], "layers": L, "positions": ["info_end", "last_prompt_token"]},
                  open(d / "acts_index.json", "w"))
        print(f"[extract] {b}: {len(vb)} new prompts x 2 positions x {len(L)} layers (total {len(H)})", flush=True)
