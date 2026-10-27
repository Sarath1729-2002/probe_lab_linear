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
    from .data import load_gsm8k, load_math500, render_prompt
    from .gen import Generator

    d, g, seed = cfg["data"], cfg["gen"], cfg["seed"]
    probs = load_gsm8k(d["gsm8k_n"], seed) + load_math500(d["math_n"], d["math_min_level"], seed)
    if limit:
        # keep the gate representative: half GSM8K, half MATH
        gs = [p for p in probs if p["dataset"] == "gsm8k"][: limit // 2]
        ms = [p for p in probs if p["dataset"] == "math"][: limit - len(gs)]
        probs = gs + ms

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
        cands = candidate_injections(eqs, r["gold"], numbers_in(r["question"]), kinds=tuple(ic["kinds"]))
        sel = select_injections(cands, ic["max_on"], ic["max_off"],
                                seed=cfg["seed"] + zlib.crc32(r["id"].encode()) % 10_000)
        if not sel:
            stats["skip_no_injection"] += 1
            continue
        stats["problems_used"] += 1
        for j, inj in enumerate(sel):
            e = eqs[inj.eq_idx]
            prefix_corrupt = trace[: e.rhs_start] + inj.new_text
            prefix_clean = trace[: e.rhs_end]
            rec = {
                "inj_id": f"{r['id']}#{j}", "problem_id": r["id"], "dataset": r["dataset"],
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


# ----------------------------------------------------------------------------- continue

def stage_continue(cfg: dict) -> None:
    from .gen import Generator
    rd = run_dir(cfg)
    injs = read_jsonl(rd / "injections.jsonl")
    g, seed = cfg["gen"], cfg["seed"]
    gen = Generator(cfg)

    corrupt = [i["prompt"] + i["prefix_corrupt"] for i in injs]
    clean = [i["prompt"] + i["prefix_clean"] for i in injs]
    greedy = gen.generate(corrupt, 1, 0.0, g["cont_max_tokens"], seed)
    samples = gen.generate(corrupt, g["n_samples"], g["temperature"], g["cont_max_tokens"], seed)
    clean_greedy = gen.generate(clean, 1, 0.0, g["cont_max_tokens"], seed)

    out, tally = [], Counter()
    for inj, gr, sm, cg in zip(injs, greedy, samples, clean_greedy):
        lab_g = label_continuation(gr[0], inj["gold"], inj)
        lab_s = [label_continuation(t, inj["gold"], inj) for t in sm]
        runs = [lab_g] + lab_s
        n_follow = sum(x["outcome"] == "FOLLOW" for x in runs)
        n_scored = sum(x["outcome"] in ("FOLLOW", "RECOVER", "OTHER") for x in runs)
        out.append({
            "inj_id": inj["inj_id"],
            "greedy_text": gr[0], "sample_texts": sm, "clean_text": cg[0],
            "greedy": lab_g, "samples": lab_s,
            "clean_cont_correct": close(extract_answer(cg[0]), inj["gold"], rel=1e-4),
            "follow_rate": n_follow / n_scored if n_scored else None,
        })
        tally[lab_g["outcome"]] += 1
    write_jsonl(rd / "continuations.jsonl", out)
    print(f"[continue] greedy outcomes: {dict(tally)}")


# ----------------------------------------------------------------------------- extract

def stage_extract(cfg: dict) -> None:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rd = run_dir(cfg)
    injs = read_jsonl(rd / "injections.jsonl")
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], torch_dtype=getattr(torch, cfg["extract"]["dtype"]), device_map="cuda").eval()
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
        if i % 100 == 0:
            print(f"[extract] {i}/{N}")
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
