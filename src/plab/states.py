"""Prompt-end hidden states (every layer) + forced-answer log-probs, per condition.

For each (problem, cond) one forward over  prompt + CUE + "<answer>}"  gives
  * the residual stream at the LAST PROMPT TOKEN (before the cue: the model has seen the question, produced nothing)
  * log P(answer digits | prompt + cue)   - the model's direct (no-CoT) belief in that answer
clean: answer = gold.   auth/user: answers = planted wrong hint AND gold.
"""
import json
from pathlib import Path

import numpy as np

from .hf_util import enc, hidden_and_logprobs, pad_id
from .prompts import CONDS, CUE, chat_prompt, user_text


def run(problems, model, tok, out_dir: Path, shard=0, nshards=1, conds=CONDS, fwd_bs=8, chunk=256, tag="", log=print) -> int:
    d = Path(out_dir) / "states"
    d.mkdir(parents=True, exist_ok=True)
    mine = [p for p in problems if p["id"] % nshards == shard]
    pad = pad_id(tok)
    cue_ids = enc(tok, CUE)
    done = 0
    for k in range(0, len(mine), chunk):
        f = d / f"x{shard:02d}{tag}_{k // chunk:05d}.npz"
        if f.exists():
            continue
        batch = mine[k:k + chunk]
        pack = {"ids": np.array([p["id"] for p in batch])}
        for cond in conds:
            jobs = []        # (problem_index, which, seq, state_pos, cont_start)
            for bi, p in enumerate(batch):
                pr = enc(tok, chat_prompt(tok, user_text(p["question"], cond, p["hint"])))
                for which, ans in (("gold", p["gold"]), ("hint", p["hint"])):
                    if cond == "clean" and which == "hint":
                        continue
                    cont = enc(tok, f"{ans}}}")
                    seq = pr + cue_ids + cont
                    jobs.append((bi, which, seq, len(pr) - 1, len(pr) + len(cue_ids)))
            states = np.zeros((len(batch), 0, 0), dtype=np.float16)
            lps = {"gold": np.full(len(batch), np.nan), "hint": np.full(len(batch), np.nan)}
            ntok = np.zeros(len(batch), dtype=np.int32)
            order = sorted(range(len(jobs)), key=lambda j: len(jobs[j][2]))
            for s in range(0, len(order), fwd_bs):
                sel = [jobs[j] for j in order[s:s + fwd_bs]]
                st, lp = hidden_and_logprobs(model, [j[2] for j in sel], [j[3] for j in sel], [j[4] for j in sel], pad)
                for r, (bi, which, seq, sp, cs) in enumerate(sel):
                    lps[which][bi] = lp[r]
                    ntok[bi] = sp + 1
                    if which == "gold":
                        if states.shape[1] == 0:
                            states = np.zeros((len(batch),) + tuple(st.shape[1:]), dtype=np.float16)
                        states[bi] = st[r].numpy().astype(np.float16)
            pack[f"{cond}_states"] = states
            pack[f"{cond}_lp_gold"] = lps["gold"]
            pack[f"{cond}_lp_hint"] = lps["hint"]
            pack[f"{cond}_ntok"] = ntok
        tmp = f.with_name(f.stem + ".tmp.npz")
        np.savez(tmp, **pack)
        tmp.rename(f)
        done += len(batch)
        log(f"[states shard {shard}] {done}/{len(mine)} problems")
    return done
