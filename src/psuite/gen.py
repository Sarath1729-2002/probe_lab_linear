"""Behaviour of the frozen model on every problem (greedy => deterministic labels), resumable chunk files, one process per GPU.

Per problem:
  plain   greedy solution to the plain prompt (token ids kept: the trace stage re-reads the model along exactly this text)
  spread  `n_spread` sampled solutions to the plain prompt (T=0.7): token counts + finished flags -> length spread and a soft runaway rate
  con     greedy solution to the prompt with ONE deterministic constraint (verify.py), compliance checked in code
"""
import json
from pathlib import Path

from .hf_util import generate_ids
from .prompts import chat_prompt, con_text, plain_text
from .sched import Budget, schedule
from .textutil import parse_answer, strip_special
from .verify import complies, constraint_for, n_words, overshoot

MAX_NEW = {"gsm8k": 512, "math": 1024, "arc": 512, "arith": 512, "prog": 512}


def run(problems, model, tok, out_dir: Path, shard=0, nshards=1, chunk=64, max_new=0, n_spread=4, temperature=0.7, top_p=0.95, max_seqs=96,
        tag="", budget_s=0, sub="", log=print) -> int:
    d = Path(out_dir) / "gen"
    d.mkdir(parents=True, exist_ok=True)
    bud = Budget(budget_s)
    done = 0
    for ds, k, batch in schedule(problems, shard, nshards, chunk, sub):
        f = d / f"g{shard:02d}_{ds}_{k:04d}{tag}.jsonl"
        if f.exists():
            continue
        if bud.spent():
            log(f"[gen shard {shard}] time budget reached - stopping cleanly (resubmit to continue)")
            break
        mn = max_new or MAX_NEW.get(ds, 512)
        pl = generate_ids(model, tok, [chat_prompt(tok, plain_text(p)) for p in batch], mn, 1, 0.0, top_p, max_seqs)
        co = generate_ids(model, tok, [chat_prompt(tok, con_text(p)) for p in batch], mn, 1, 0.0, top_p, max_seqs)
        sp = generate_ids(model, tok, [chat_prompt(tok, plain_text(p)) for p in batch], mn, n_spread, temperature, top_p, max_seqs) if n_spread else [[] for _ in batch]
        rows = []
        for p, (pi, pf), (ci, cf), ss in zip(batch, (x[0] for x in pl), (x[0] for x in co), sp):
            pt, ct = tok.decode(pi, skip_special_tokens=True), tok.decode(ci, skip_special_tokens=True)
            ctype = constraint_for(p["id"])
            rows.append({"id": p["id"], "ds": ds, "ctype": ctype,
                         "plain_ids": pi, "plain_fin": pf, "plain_text": pt, "plain_ans": parse_answer(pt, p["kind"]), "plain_words": n_words(pt),
                         "spread_n": [len(x[0]) for x in ss], "spread_fin": [x[1] for x in ss],
                         "con_ntok": len(ci), "con_fin": cf, "con_text": ct, "con_ans": parse_answer(ct, p["kind"]), "con_words": n_words(ct),
                         "con_comply": complies(ctype, ct, cf), "con_over": overshoot(ctype, ct, cf)})
        tmp = f.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        tmp.rename(f)
        done += len(batch)
        log(f"[gen shard {shard}] {ds} chunk {k}: +{len(batch)} problems ({done} this run)")
    return done
