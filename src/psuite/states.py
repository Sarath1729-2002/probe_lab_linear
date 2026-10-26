"""Prompt-end hidden states (every layer, last prompt token) for the PLAIN and the CONSTRAINED prompt of every problem.  No generation involved."""
from pathlib import Path

import numpy as np

from .hf_util import enc, pad_id, prompt_states
from .prompts import chat_prompt, con_text, plain_text
from .sched import Budget, schedule


def run(problems, model, tok, out_dir: Path, shard=0, nshards=1, fwd_bs=8, chunk=128, tag="", budget_s=0, sub="", log=print) -> int:
    d = Path(out_dir) / "states"
    d.mkdir(parents=True, exist_ok=True)
    pad = pad_id(tok)
    bud = Budget(budget_s)
    done = 0
    for ds, k, batch in schedule(problems, shard, nshards, chunk, sub):
        f = d / f"x{shard:02d}_{ds}_{k:04d}{tag}.npz"
        if f.exists():
            continue
        if bud.spent():
            log("[states] time budget reached - stopping cleanly")
            break
        pack = {"ids": np.array([p["id"] for p in batch])}
        for name, fn in (("plain", plain_text), ("con", con_text)):
            seqs = [enc(tok, chat_prompt(tok, fn(p))) for p in batch]
            order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
            H = None
            for s in range(0, len(order), fwd_bs):
                sel = order[s:s + fwd_bs]
                h = prompt_states(model, [seqs[i] for i in sel], pad).numpy()
                if H is None:
                    H = np.zeros((len(batch),) + h.shape[1:], dtype=np.float16)
                H[sel] = h
            pack[name], pack[f"{name}_ntok"] = H, np.array([len(s) for s in seqs], dtype=np.int32)
        tmp = f.with_name(f.stem + ".tmp.npz")
        np.savez(tmp, **pack)
        tmp.rename(f)
        done += len(batch)
        log(f"[states shard {shard}] {ds} chunk {k}: +{len(batch)} problems ({done} this run)")
    return done
