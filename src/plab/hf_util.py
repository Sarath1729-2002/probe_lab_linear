import math
from typing import List

import torch


def enc(tok, text: str) -> List[int]:
    return list(tok.encode(text, add_special_tokens=False))


def load_model(model_id: str, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    except TypeError:                                   # older transformers
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    if torch.cuda.is_available():
        model = model.to("cuda")                        # no accelerate / device_map needed (one process per GPU)
    return model.eval(), tok


def pad_id(tok) -> int:
    p = getattr(tok, "pad_token_id", None)
    return p if p is not None else (getattr(tok, "eos_token_id", 0) or 0)


def left_pad(seqs: List[List[int]], pad: int, device):
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad, dtype=torch.long)
    att = torch.zeros((len(seqs), T), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, T - len(s):] = torch.tensor(s)
        att[i, T - len(s):] = 1
    return ids.to(device), att.to(device)


@torch.no_grad()
def generate(model, tok, prompts: List[str], n: int, max_new: int, temperature: float, top_p: float, max_seqs: int = 128) -> List[List[str]]:
    """n samples per prompt; sub-batches of max_seqs//n prompts with an OOM-halving guard.  -> [len(prompts)][n] strings."""
    dev = next(model.parameters()).device
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out: List[List[str]] = [None] * len(prompts)       # type: ignore
    bs = max(1, max_seqs // n)
    i = 0
    while i < len(order):
        idx = order[i:i + bs]
        try:
            seqs = [enc(tok, prompts[j]) for j in idx]
            ids, att = left_pad(seqs, pad_id(tok), dev)
            kw = dict(do_sample=temperature > 0, max_new_tokens=max_new, num_return_sequences=n, pad_token_id=pad_id(tok))
            if temperature > 0:
                kw.update(temperature=temperature, top_p=top_p, top_k=0)
            g = model.generate(input_ids=ids, attention_mask=att, **kw)
            new = g[:, ids.shape[1]:].cpu()
            for r, j in enumerate(idx):
                out[j] = [tok.decode(new[r * n + k].tolist(), skip_special_tokens=True) for k in range(n)]
            i += len(idx)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)
    return out


def hidden_and_logprobs(model, seqs: List[List[int]], state_pos: List[int], cont_start: List[int], pad: int):
    """One right-padded forward.  -> states [B, L+1, D] (float32, CPU) at state_pos[i] and
    the summed log-prob of tokens seq[cont_start[i]:] (teacher forced) -> [B]."""
    dev = next(model.parameters()).device
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad, dtype=torch.long)
    att = torch.zeros((len(seqs), T), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        att[i, :len(s)] = 1
    ids, att = ids.to(dev), att.to(dev)
    with torch.no_grad():
        o = model(input_ids=ids, attention_mask=att, output_hidden_states=True)
    rows = torch.arange(len(seqs), device=dev)
    pos = torch.tensor(state_pos, device=dev)
    states = torch.stack([h[rows, pos] for h in o.hidden_states], dim=1).float().cpu()      # [B, L+1, D]
    lp = torch.log_softmax(o.logits.float(), dim=-1)
    out = []
    for i, s in enumerate(seqs):
        tgt = torch.tensor(s[cont_start[i]:], device=dev)
        pred = lp[i, cont_start[i] - 1: len(s) - 1]
        out.append(float(pred.gather(1, tgt[:, None]).sum()))
    return states, out
