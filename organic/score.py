"""Read-only scoring (GPU): for every prompt, how strongly does the model prefer the correct answer over the
planted wrong one, before it writes anything?

margin = log P("Answer: <correct>") - log P("Answer: <planted wrong>"), each scored by teacher forcing right
after the prompt (summed over the answer's tokens). One unmodified forward pass per candidate; the language-model
head is applied only at the answer positions. Nothing is generated and nothing in the model is changed.

Derived later (deference.py):
  prior  = margin on the bare question                       (how firmly it knows; ClashEval's prior probability)
  shift  = prior - margin with the wrong content in context  (how far the content pulls it; the regression target)
"""
from __future__ import annotations

import numpy as np

from .common import DEVICE, read_jsonl, render, write_jsonl
from .protocols import answer_string
from .run import bdir, is_probe_family, load


def stage_score(cfg: dict, buckets: list[str]) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    items, V = load(cfg)
    model = tok = None
    bs = cfg.get("score", {}).get("batch", 8)
    for b in buckets:
        out = bdir(cfg, b) / "score.jsonl"
        old = read_jsonl(out) if out.exists() else []
        have = {r["vid"] for r in old}
        vb = [v for v in V if v["bucket"] == b and v["vid"] not in have
              and (v["family"] == "base" or is_probe_family(v["family"]) or v["role"] == "doubt")]
        if not vb:
            print(f"[score] {b}: nothing new")
            continue
        if model is None:
            tok = AutoTokenizer.from_pretrained(cfg["model"])
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            tok.padding_side = "right"
            model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.bfloat16, device_map=DEVICE).eval()
        jobs = []                                         # (variant index, which, full text, char where the answer starts)
        for k, v in enumerate(vb):
            p, _ = render(tok, v)
            it = items[v["item"]]
            for which in ("gold", "wrong"):
                pre = p + "Answer: "
                jobs.append((k, which, pre + answer_string(it, which), len(pre)))
        order = np.argsort([len(j[2]) for j in jobs])
        lp = np.full((len(vb), 2), np.nan)
        for s in range(0, len(order), bs):
            chunk = [jobs[i] for i in order[s:s + bs]]
            enc = tok([c[2] for c in chunk], return_tensors="pt", padding=True, add_special_tokens=False,
                      return_offsets_mapping=True)
            offs = enc.pop("offset_mapping")
            enc = {k_: t.to(model.device) for k_, t in enc.items()}
            with torch.no_grad():
                h = model.model(**enc).last_hidden_state                  # final-norm hidden states
                for j, (k, which, text, start) in enumerate(chunk):
                    n = int(enc["attention_mask"][j].sum())
                    ans = [t for t in range(n) if offs[j, t, 1] > start]  # tokens covering the answer
                    if not ans or ans[0] == 0:
                        continue
                    pos = torch.tensor([t - 1 for t in ans], device=model.device)   # predicted from the previous token
                    logits = model.lm_head(h[j, pos]).float()
                    tgt = enc["input_ids"][j, ans]
                    lp[k, 0 if which == "gold" else 1] = float(torch.log_softmax(logits, -1).gather(1, tgt[:, None]).sum())
        rows = [{"vid": v["vid"], "lp_gold": float(lp[k, 0]), "lp_wrong": float(lp[k, 1]),
                 "margin": float(lp[k, 0] - lp[k, 1])} for k, v in enumerate(vb)]
        write_jsonl(out, old + rows)
        print(f"[score] {b}: {len(rows)} prompts scored", flush=True)
