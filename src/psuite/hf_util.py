import math
from typing import List

import torch


def enc(tok, text: str) -> List[int]:
    return list(tok.encode(text, add_special_tokens=False))


def load_model(model_id: str, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if torch.cuda.is_available():
        torch.backends.cuda.enable_cudnn_sdp(False)     # cuDNN attention needs memory outside PyTorch's allocator and crashed a shard near 80 GB; flash/mem-efficient SDPA do not
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



def eos_set(model, tok) -> set:
    e = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    out = set(e) if isinstance(e, (list, tuple)) else ({e} if e is not None else set())
    if getattr(tok, "eos_token_id", None) is not None:
        out.add(tok.eos_token_id)
    return {int(x) for x in out}


@torch.no_grad()
def generate_ids(model, tok, prompts: List[str], max_new: int, n: int = 1, temperature: float = 0.0, top_p: float = 0.95, max_seqs: int = 96):
    """greedy (temperature 0) or sampled completions.  -> [len(prompts)][n] of (token_ids without EOS/padding, finished) where finished = an EOS was emitted
    before the cap.  Sub-batches of max_seqs//n prompts with an OOM-halving guard (cuDNN/CUDA errors included)."""
    dev = next(model.parameters()).device
    eos = eos_set(model, tok)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out = [None] * len(prompts)
    bs = max(1, max_seqs // n)
    i = 0
    while i < len(order):
        idx = order[i:i + bs]
        try:
            seqs = [enc(tok, prompts[j]) if isinstance(prompts[j], str) else list(prompts[j]) for j in idx]      # a prompt is a string or ready-made token ids (continuations)
            ids, att = left_pad(seqs, pad_id(tok), dev)
            kw = dict(do_sample=temperature > 0, max_new_tokens=max_new, num_return_sequences=n, pad_token_id=pad_id(tok), repetition_penalty=1.0)   # Qwen's generation_config ships a 1.05 penalty: off, so 'greedy' is greedy
            if temperature > 0:
                kw.update(temperature=temperature, top_p=top_p, top_k=0)
            else:
                kw.update(temperature=1.0, top_p=1.0, top_k=0)
            g = model.generate(input_ids=ids, attention_mask=att, **kw)
            new = g[:, ids.shape[1]:].cpu().tolist()
            for r, j in enumerate(idx):
                res = []
                for k in range(n):
                    row, cut, fin = new[r * n + k], None, False
                    for t, x in enumerate(row):
                        if x in eos:
                            cut, fin = t, True
                            break
                    res.append((row if cut is None else row[:cut], fin))
                out[j] = res
            i += len(idx)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            msg = str(exc).lower()
            if not isinstance(exc, torch.cuda.OutOfMemoryError) and not any(k in msg for k in ("out of memory", "cudnn", "cuda error")):
                raise
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)
    return out


@torch.no_grad()
def completion_entropy(model, seq: List[int], start: int, chunk: int = 256):
    """one teacher-forced pass over seq (prompt + completion): per-token entropy (nats) of the model's next-token distribution and NLL of the token actually
    taken, for the completion positions seq[start:].  -> (entropy [T], nll [T]) numpy float32."""
    import numpy as np
    dev = next(model.parameters()).device
    ids = torch.tensor([seq], device=dev)
    base = getattr(model, "model", model)
    h = base(input_ids=ids).last_hidden_state[0, start - 1: len(seq) - 1]
    head = model.lm_head if hasattr(model, "lm_head") else model.get_output_embeddings()
    tgt = torch.tensor(seq[start:], device=dev)
    ent, nll = [], []
    for i in range(0, h.shape[0], chunk):
        lp = torch.log_softmax(head(h[i:i + chunk]).float(), dim=-1)
        ent.append((-(lp.exp() * lp).sum(-1)).cpu())
        nll.append((-lp.gather(1, tgt[i:i + chunk, None])[:, 0]).cpu())
    if not ent:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    return torch.cat(ent).numpy().astype("float32"), torch.cat(nll).numpy().astype("float32")


def _right_pad(seqs, pad, dev):
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad, dtype=torch.long)
    att = torch.zeros((len(seqs), T), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        att[i, :len(s)] = 1
    return ids.to(dev), att.to(dev)


@torch.no_grad()
def prompt_states(model, seqs: List[List[int]], pad: int):
    """residual stream (every layer, embeddings included) at the LAST token of each sequence -> [B, L+1, D] float16 (CPU).  No lm_head."""
    dev = next(model.parameters()).device
    ids, att = _right_pad(seqs, pad, dev)
    base = getattr(model, "model", model)
    o = base(input_ids=ids, attention_mask=att, output_hidden_states=True)
    rows = torch.arange(len(seqs), device=dev)
    pos = torch.tensor([len(s) - 1 for s in seqs], device=dev)
    return torch.stack([h[rows, pos] for h in o.hidden_states], dim=1).half().cpu()


@torch.no_grad()
def checkpoint_forward(model, seqs: List[List[int]], state_pos: List[int], ans_start: List[int], layers: List[int], pad: int):
    """One right-padded forward over  prefix + cue + answer.  Causality means the state at state_pos is the state of the model WITHOUT the cue/answer.
    -> (states [B, len(layers), D] float16, summed log-prob of seq[ans_start:] given everything before it, [B])."""
    dev = next(model.parameters()).device
    ids, att = _right_pad(seqs, pad, dev)
    base = getattr(model, "model", model)
    o = base(input_ids=ids, attention_mask=att, output_hidden_states=True)
    rows = torch.arange(len(seqs), device=dev)
    pos = torch.tensor(state_pos, device=dev)
    states = torch.stack([o.hidden_states[l][rows, pos] for l in layers], dim=1).half().cpu()
    head = model.lm_head if hasattr(model, "lm_head") else model.get_output_embeddings()
    lps = []
    for i, s in enumerate(seqs):
        a = ans_start[i]
        logits = head(o.last_hidden_state[i, a - 1: len(s) - 1]).float()
        lp = torch.log_softmax(logits, dim=-1)
        tgt = torch.tensor(s[a:], device=dev)
        lps.append(float(lp.gather(1, tgt[:, None]).sum()))
    return states, lps
