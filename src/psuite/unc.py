"""UNCERTAINTY POC - sampling stage.  Re-uses problems / greedy generations / hidden states / trace checkpoints of the main run; only adds ROLLOUTS.

For a deterministic subset of problems (gsm8k + math, `n_per_ds` each), per problem:
  roll   `n_roll` sampled solutions to the plain prompt                          -> parsed final answers: label = pass rate / answer entropy / majority share
  cont   for the trace checkpoints o48 and o96 (absolute token offsets into the model's own greedy solution), `n_cont` sampled CONTINUATIONS
         of prompt + greedy[:t]                                                  -> the same labels, now conditional on the first t tokens ("is the model still uncertain after 48 tokens?")
  tokent teacher-forced pass over the greedy solution: mean next-token entropy / NLL over all tokens and over the first 48 / 96   (the FREE uncertainty signal a probe has to beat)

All labels are exact functions of the sampled answers (no judge, no entailment model): answers are canonical integers (textutil), so 'semantic' clustering is string equality.
The labels are noisy estimates of the true answer distribution; analysis reports split-half reliability so a probe is judged against what the label can carry."""
import json
import zlib
from pathlib import Path

from .gen import MAX_NEW
from .hf_util import completion_entropy, enc, generate_ids
from .prompts import chat_prompt, plain_text
from .sched import Budget, schedule
from .textutil import parse_answer

CPS = (48, 96)                  # continuation checkpoints = trace offsets o48, o96
MIN_LEFT = 8                    # same validity rule as trace.checkpoint_ts
POC_DATASETS = ("gsm8k", "math")


def pick_subset(problems, G, n_per_ds: int, datasets=POC_DATASETS, skip: int = 0):
    """deterministic: the n problems per dataset with the smallest crc32(id) among those that have a greedy generation.  Nested: n=40 is a subset of n=400."""
    out = []
    for ds in datasets:
        ps = [p for p in problems if p["ds"] == ds and p["id"] in G]
        ps.sort(key=lambda p: zlib.crc32(f"unc{p['id']}".encode()))
        out += ps[skip:skip + n_per_ds]
    return sorted(out, key=lambda p: p["id"])


def _parse(tok, ids, kind):
    return parse_answer(tok.decode(ids, skip_special_tokens=True), kind)


def _mean(x, a, b):
    seg = x[a:b]
    return float(seg.mean()) if len(seg) >= (b - a) and len(seg) else None


def run(problems, G, model, tok, out_dir: Path, shard=0, nshards=1, n_roll=16, n_cont=8, temperature=0.7, top_p=0.95, max_seqs=96, chunk=16,
        cps=CPS, min_left=MIN_LEFT, max_new=0, budget_s=0, log=print) -> int:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    bud = Budget(budget_s)
    done = 0
    for ds, k, batch in schedule(problems, shard, nshards, chunk):
        f = d / f"u{shard:02d}_{ds}_{k:04d}.jsonl"
        if f.exists():
            continue
        if bud.spent():
            log(f"[unc shard {shard}] time budget reached - stopping cleanly (resubmit to continue)")
            break
        cap = max_new or MAX_NEW.get(ds, 512)
        prompts = [enc(tok, chat_prompt(tok, plain_text(p))) for p in batch]
        roll = generate_ids(model, tok, [chat_prompt(tok, plain_text(p)) for p in batch], cap, n_roll, temperature, top_p, max_seqs)
        rows = []
        for p, pr, rl in zip(batch, prompts, roll):
            comp = G[p["id"]]["plain_ids"]
            ent, nll = completion_entropy(model, pr + comp, len(pr)) if comp else (None, None)
            rows.append({"id": p["id"], "ds": ds, "n_greedy": len(comp),
                         "roll_ans": [_parse(tok, ids, p["kind"]) for ids, _ in rl], "roll_fin": [bool(fin) for _, fin in rl], "roll_ntok": [len(ids) for ids, _ in rl],
                         "ent_all": float(ent.mean()) if ent is not None and len(ent) else None, "nll_all": float(nll.mean()) if nll is not None and len(nll) else None,
                         **{f"ent{t}": _mean(ent, 0, t) if ent is not None else None for t in cps}, **{f"nll{t}": _mean(nll, 0, t) if nll is not None else None for t in cps},
                         "cont": {}})
        for t in cps:                                          # continuations from the model's OWN greedy prefix of t tokens
            sel = [i for i, p in enumerate(batch) if t <= len(G[p["id"]]["plain_ids"]) - min_left and cap - t > 0]
            if not sel:
                continue
            seqs = [prompts[i] + G[batch[i]["id"]]["plain_ids"][:t] for i in sel]
            cont = generate_ids(model, tok, seqs, cap - t, n_cont, temperature, top_p, max_seqs)
            for i, cs in zip(sel, cont):
                comp = G[batch[i]["id"]]["plain_ids"][:t]
                rows[i]["cont"][str(t)] = {"ans": [_parse(tok, comp + ids, batch[i]["kind"]) for ids, _ in cs], "fin": [bool(fin) for _, fin in cs]}
        tmp = f.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        tmp.rename(f)
        done += len(batch)
        log(f"[unc shard {shard}] {ds} chunk {k}: +{len(batch)} problems ({done} this run)")
    return done
