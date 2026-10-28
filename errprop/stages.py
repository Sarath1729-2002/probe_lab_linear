"""Pipeline stages. Each reads the previous stage's file from runs/<run_name>/ and writes its own.

  clean     (GPU)  greedy CoT + sampled pass rate on the original problems
  inject    (CPU)  parse equations, build the graph, choose injections, build prefixes
  continue  (GPU)  continue from each corrupted (and clean) prefix, label every continuation
  extract   (GPU)  residual-stream activations at the injection point + token surprisal
  probes    (CPU)  all probes, see probes.py
  report    (CPU)  markdown report + figures
"""
from __future__ import annotations

import math
import zlib
from collections import Counter

from .arith import parse_equations, candidate_injections, select_injections, numbers_in, close
from .io import run_dir, write_jsonl, read_jsonl
from .labels import extract_answer, label_continuation


# ----------------------------------------------------------------------------- clean

def stage_clean(cfg: dict, limit: int | None = None) -> None:
    from .data import load_all, render_prompt
    from .gen import Generator

    g, seed = cfg["gen"], cfg["seed"]
    probs = load_all(cfg)
    if limit:
        # keep a quick test representative: an equal share of every dataset
        by_ds: dict[str, list] = {}
        for p in probs:
            by_ds.setdefault(p["dataset"], []).append(p)
        per = max(1, limit // len(by_ds))
        probs = [p for ps in by_ds.values() for p in ps[:per]]
    print(f"[clean] problems per dataset: {dict(Counter(p['dataset'] for p in probs))}", flush=True)

    gen = Generator(cfg)
    prompts = [render_prompt(gen.tokenizer, p["question"]) for p in probs]
    greedy = gen.generate(prompts, n=1, temperature=0.0, max_tokens=g["clean_max_tokens"], seed=seed)
    samples = gen.generate(prompts, n=g["n_samples"], temperature=g["temperature"],
                           max_tokens=g["clean_max_tokens"], seed=seed)

    rows = []
    for p, pr, gr, sm in zip(probs, prompts, greedy, samples):
        trace = gr[0]
        ans = extract_answer(trace)
        n_ok = sum(close(extract_answer(s), p["gold"], rel=1e-4) for s in sm)
        rows.append({**p, "prompt": pr, "trace": trace, "greedy_answer": ans,
                     "correct": close(ans, p["gold"], rel=1e-4),
                     "pass_rate": n_ok / max(1, len(sm)),
                     "n_eq": len(parse_equations(trace))})
    write_jsonl(run_dir(cfg) / "clean.jsonl", rows)
    acc = sum(r["correct"] for r in rows) / len(rows)
    print(f"[clean] {len(rows)} problems, greedy accuracy {acc:.3f}")
    for ds in sorted({r["dataset"] for r in rows}):
        rr = [r for r in rows if r["dataset"] == ds]
        print(f"[clean]   {ds:12s} n={len(rr):5d}  acc={sum(r['correct'] for r in rr) / len(rr):.3f}  "
              f"mean verified equations={sum(r['n_eq'] for r in rr) / len(rr):.1f}")


# ----------------------------------------------------------------------------- inject

def stage_inject(cfg: dict) -> None:
    rd = run_dir(cfg)
    rows = read_jsonl(rd / "clean.jsonl")
    ic = cfg["inject"]
    out, stats = [], Counter()
    for r in rows:
        if not r["correct"]:
            stats["skip_wrong"] += 1
            continue
        trace = r["trace"]
        if cfg.get("reasoning_model") and "</think>" in trace:
            # only inject inside the thinking block
            trace_for_parse = trace.split("</think>")[0]
        else:
            trace_for_parse = trace
        eqs = parse_equations(trace_for_parse)
        pseed = cfg["seed"] + zlib.crc32(r["id"].encode()) % 100_000
        cands = candidate_injections(eqs, r["gold"], numbers_in(r["question"]),
                                     kinds=tuple(ic["kinds"]), seed=pseed)
        sel = select_injections(cands, ic["max_on"], ic["max_off"], seed=pseed)
        if not sel:
            stats["skip_no_injection"] += 1
            continue
        stats["problems_used"] += 1
        for j, inj in enumerate(sel):
            e = eqs[inj.eq_idx]
            prefix_corrupt = trace[: e.rhs_start] + inj.new_text
            prefix_clean = trace[: e.rhs_end]
            rec = {
                "inj_id": f"{r['id']}#{j}", "problem_id": r["id"], "group": r.get("group", r["id"]),
                "dataset": r["dataset"],
                "level": r["level"], "gold": r["gold"], "pass_rate": r["pass_rate"],
                "prompt": r["prompt"], "prefix_corrupt": prefix_corrupt, "prefix_clean": prefix_clean,
                "n_eq": len(eqs), "base_chars": e.rhs_start, "trace_chars": len(trace),
                **{k: v for k, v in inj.__dict__.items()},
            }
            rec["log_rel_err"] = math.log10(abs(inj.new_value - inj.true_value) / max(abs(inj.true_value), 1e-9))
            out.append(rec)
            stats["on_path" if inj.on_path else "off_path"] += 1
    write_jsonl(rd / "injections.jsonl", out)
    print(f"[inject] {dict(stats)}")
    print(f"[inject] by operator: {dict(Counter(o['kind'] for o in out))}")
    print(f"[inject] by dataset:  {dict(Counter(o['dataset'] for o in out))}")


# ----------------------------------------------------------------------------- continue

def stage_continue(cfg: dict) -> None:
    from .gen import Generator
    rd = run_dir(cfg)
    injs = read_jsonl(rd / "injections.jsonl")
    g, seed = cfg["gen"], cfg["seed"]
    gen = Generator(cfg)

    corrupt = [i["prompt"] + i["prefix_corrupt"] for i in injs]
    clean = [i["prompt"] + i["prefix_clean"] for i in injs]
    wait = cfg["intervene"]["wait_text"]
    print(f"[continue] {len(injs)} injections: greedy", flush=True)
    greedy = gen.generate(corrupt, 1, 0.0, g["cont_max_tokens"], seed)
    print("[continue] samples", flush=True)
    samples = gen.generate(corrupt, g["n_samples"], g["temperature"], g["cont_max_tokens"], seed)
    print("[continue] clean prefix", flush=True)
    clean_greedy = gen.generate(clean, 1, 0.0, g["cont_max_tokens"], seed)
    # intervention arm (Self-Correction Bench, Tsui 2025: appending "Wait" activates correction).
    # Run on every corrupted AND clean prefix, so any monitor policy can be simulated offline:
    # intervening on a corrupted step can rescue it; intervening on a clean step can break it.
    print("[continue] wait-intervention arms", flush=True)
    wait_corrupt = gen.generate([c + wait for c in corrupt], 1, 0.0, g["cont_max_tokens"], seed)
    wait_clean = gen.generate([c + wait for c in clean], 1, 0.0, g["cont_max_tokens"], seed)

    out, tally, tally_w = [], Counter(), Counter()
    for inj, gr, sm, cg, wc, wk in zip(injs, greedy, samples, clean_greedy, wait_corrupt, wait_clean):
        lab_g = label_continuation(gr[0], inj["gold"], inj)
        lab_s = [label_continuation(t, inj["gold"], inj) for t in sm]
        lab_w = label_continuation(wc[0], inj["gold"], inj)
        runs = [lab_g] + lab_s
        n_follow = sum(x["outcome"] == "FOLLOW" for x in runs)
        n_scored = sum(x["outcome"] in ("FOLLOW", "RECOVER", "OTHER") for x in runs)
        out.append({
            "inj_id": inj["inj_id"],
            "greedy_text": gr[0], "sample_texts": sm, "clean_text": cg[0],
            "wait_text": wc[0], "wait_clean_text": wk[0],
            "greedy": lab_g, "samples": lab_s, "wait": lab_w,
            "clean_cont_correct": close(extract_answer(cg[0]), inj["gold"], rel=1e-4),
            "wait_clean_correct": close(extract_answer(wk[0]), inj["gold"], rel=1e-4),
            "follow_rate": n_follow / n_scored if n_scored else None,
        })
        tally[lab_g["outcome"]] += 1
        tally_w[lab_w["outcome"]] += 1
    write_jsonl(rd / "continuations.jsonl", out)
    print(f"[continue] greedy outcomes: {dict(tally)}")
    print(f"[continue] with Wait:       {dict(tally_w)}")


# ----------------------------------------------------------------------------- extract

def stage_extract(cfg: dict) -> None:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rd = run_dir(cfg)
    injs = read_jsonl(rd / "injections.jsonl")
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], dtype=getattr(torch, cfg["extract"]["dtype"]), device_map="cuda").eval()
    n_layers = model.config.num_hidden_layers + 1
    d_model = model.config.hidden_size

    def enc(text):
        return tok(text, add_special_tokens=False)["input_ids"]

    @torch.no_grad()
    def run(text, base_text=None):
        """Hidden states at the last token (all layers) and, if `base_text` is given, the surprisal
        (-sum log p) of the tokens that come after it, i.e. of the injected/true number."""
        ids = enc(text)
        x = torch.tensor([ids], device="cuda")
        o = model(x, output_hidden_states=True)
        hs = torch.stack([h[0, -1] for h in o.hidden_states]).float().cpu().numpy().astype(np.float16)
        if base_text is None:
            return hs, None, 0, True, len(ids)
        base = enc(base_text)
        prefix_ok = ids[: len(base)] == base
        n_new = len(ids) - len(base) if prefix_ok and len(ids) > len(base) else 1
        logits = o.logits[0, len(ids) - n_new - 1: len(ids) - 1].float()
        lp = torch.log_softmax(logits, -1).gather(1, x[0, len(ids) - n_new:, None]).sum().item()
        return hs, -lp, n_new, prefix_ok, len(ids)

    N = len(injs)
    A_c = np.lib.format.open_memmap(rd / "acts_corrupt.npy", "w+", np.float16, (N, n_layers, d_model))
    A_k = np.lib.format.open_memmap(rd / "acts_clean.npy", "w+", np.float16, (N, n_layers, d_model))
    meta = []
    for i, inj in enumerate(injs):
        base = inj["prompt"] + inj["prefix_corrupt"][: inj["base_chars"]]
        hs_c, s_c, n_c, ok_c, L_c = run(inj["prompt"] + inj["prefix_corrupt"], base)
        hs_k, s_k, n_k, ok_k, _ = run(inj["prompt"] + inj["prefix_clean"], base)
        A_c[i], A_k[i] = hs_c, hs_k
        meta.append({"inj_id": inj["inj_id"], "surprise_corrupt": s_c, "surprise_clean": s_k,
                     "n_num_tokens": n_c, "prefix_ok": ok_c and ok_k, "n_tokens": L_c})
        if i % 500 == 0:
            print(f"[extract] {i}/{N}", flush=True)
    A_c.flush(); A_k.flush()

    # prompt-time activations (one per problem) for the susceptibility probe
    pids = sorted({inj["problem_id"] for inj in injs})
    prompt_of = {inj["problem_id"]: inj["prompt"] for inj in injs}
    A_p = np.lib.format.open_memmap(rd / "acts_prompt.npy", "w+", np.float16, (len(pids), n_layers, d_model))
    for i, pid in enumerate(pids):
        A_p[i], *_ = run(prompt_of[pid])
    A_p.flush()
    write_jsonl(rd / "extract_meta.jsonl", meta)
    write_jsonl(rd / "prompt_ids.jsonl", [{"problem_id": p} for p in pids])
    print(f"[extract] done: {N} injections, {len(pids)} prompts, {n_layers} layers x {d_model}")


# ----------------------------------------------------------------------------- steer

def stage_steer(cfg: dict) -> None:
    """Causal test: add alpha * (FOLLOW - RECOVER direction) to the residual stream at the probe
    layer and re-run held-out continuations. If the direction is used by the model, +alpha should
    raise FOLLOW and -alpha should raise RECOVER; random directions of equal size should do neither.
    (Jia & Mu 2026 report additive steering mostly fails; Phan et al. 2026 report a working
    critique vector. Either outcome is informative.)"""
    import json
    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rd = run_dir(cfg)
    sc = cfg["steer"]
    D = np.load(rd / "direction.npz")
    layer, unit, sd = int(D["layer"]), D["unit"], float(D["proj_sd"])
    tab = pd.read_csv(rd / "table.csv")
    pool = tab[tab.valid & tab.on_path & tab.catchable & (tab.split == "test")
               & tab.outcome.isin(["FOLLOW", "RECOVER"])]
    if len(pool) < 20:
        pool = tab[tab.valid & tab.on_path & (tab.split == "test") & tab.outcome.isin(["FOLLOW", "RECOVER"])]
    if len(pool) == 0:
        print("[steer] no held-out FOLLOW/RECOVER items; skipping", flush=True)
        return
    half = max(1, sc["n"] // 2)
    pick = pd.concat([g.sample(n=min(half, len(g)), random_state=cfg["seed"])
                      for _, g in pool.groupby("outcome")])
    injs = {i["inj_id"]: i for i in read_jsonl(rd / "injections.jsonl")}
    items = [injs[i] for i in pick.inj_id]
    orig = dict(zip(pick.inj_id, pick.outcome))
    print(f"[steer] layer {layer}, {len(items)} items ({pick.outcome.value_counts().to_dict()}), "
          f"proj sd {sd:.3f}", flush=True)

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.bfloat16, device_map="cuda").eval()
    block = model.model.layers[max(0, layer - 1)]      # hidden_states[l] is the output of block l-1
    state = {"delta": None}

    def hook(_m, _inp, out):
        if state["delta"] is None:
            return out
        if isinstance(out, tuple):
            return (out[0] + state["delta"],) + tuple(out[1:])
        return out + state["delta"]
    block.register_forward_hook(hook)

    rng = np.random.default_rng(cfg["seed"])
    conds = [("probe", a, unit) for a in sc["alphas"]]
    amax = max(abs(a) for a in sc["alphas"])
    for r in range(sc["n_random"]):
        v = rng.normal(size=unit.shape)
        v /= np.linalg.norm(v)
        conds += [(f"random{r}", -amax, v), (f"random{r}", amax, v)]

    prompts = [i["prompt"] + i["prefix_corrupt"] for i in items]
    rows, summary = [], []
    for name, a, v in conds:
        state["delta"] = None if a == 0 else torch.tensor(a * sd * v, dtype=torch.bfloat16, device="cuda")
        texts = []
        for b in range(0, len(prompts), sc["batch_size"]):
            enc = tok(prompts[b:b + sc["batch_size"]], return_tensors="pt", padding=True,
                      add_special_tokens=False).to("cuda")
            with torch.no_grad():
                ids = model.generate(**enc, max_new_tokens=sc["max_new_tokens"], do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            texts += tok.batch_decode(ids[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        tally = Counter()
        flips = Counter()
        for it, t in zip(items, texts):
            lab = label_continuation(t, it["gold"], it)
            tally[lab["outcome"]] += 1
            flips[(orig[it["inj_id"]], lab["outcome"])] += 1
            rows.append({"cond": name, "alpha": a, "inj_id": it["inj_id"], "orig": orig[it["inj_id"]],
                         "outcome": lab["outcome"], "text": t[:600]})
        n = len(items)
        n_f = sum(1 for it in items if orig[it["inj_id"]] == "FOLLOW")
        n_r = n - n_f
        rec = {"cond": name, "alpha": a, "n": n,
               "follow_rate": tally["FOLLOW"] / n, "recover_rate": tally["RECOVER"] / n,
               "other_rate": tally["OTHER"] / n,
               "FOLLOW_to_RECOVER": flips[("FOLLOW", "RECOVER")] / max(1, n_f),
               "RECOVER_to_FOLLOW": flips[("RECOVER", "FOLLOW")] / max(1, n_r)}
        summary.append(rec)
        print(f"[steer] {name:8s} alpha={a:+.1f}  follow={rec['follow_rate']:.3f} "
              f"recover={rec['recover_rate']:.3f} other={rec['other_rate']:.3f}", flush=True)
    write_jsonl(rd / "steer.jsonl", rows)
    with open(rd / "steer_summary.json", "w") as f:
        json.dump({"layer": layer, "proj_sd": sd, "conditions": summary}, f, indent=2)
