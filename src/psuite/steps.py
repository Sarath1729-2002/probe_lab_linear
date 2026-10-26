"""THREAD 1 - causal step importance.  Label from an INTERVENTION on the model's own chain of thought, not from a judge.

For a numeric step of the model's greedy solution we edit the step and let the model continue (greedy), then compare its final answer with the original:
   n1    the step's result number c is replaced by c + 1
   n2    ... by c + max(10, c // 10)
   del   the whole step line is deleted
   null  the prefix is left UNTOUCHED and regenerated (noise floor: batching / numerics can flip answers on their own; reported, never subtracted)
label  flip_x = 1 if the final answer after the edit differs from the original final answer (no parsable answer counts as a flip)
       load = mean(flip_n1, flip_n2, flip_del) in {0, 1/3, 2/3, 1}: how load-bearing the step is
The probe reads the natural hidden state at the LAST TOKEN OF THE STEP (the future of the chain does not exist yet, so text baselines cannot see whether the step's number is reused later).
One forward over the whole original solution gives the states at every step boundary (causality).  Resumable: <steps>/s<shard>_<chunk>.{npz,jsonl}; the jsonl is written last.
"""
import bisect
import json
import re
import zlib
from pathlib import Path
from typing import List, Optional

import numpy as np

from .hf_util import enc, generate_ids, pad_id
from .prompts import chat_prompt, plain_text
from .sched import Budget, schedule
from .textutil import parse_answer

LAYER_STEP = 2
INT = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?![\w]|\.\d)")
ENUM = re.compile(r"^\s*(?:\*\*)?(?:Step\s*)?\d+\s*[\.\):]", re.I)


def candidate_steps(text: str, max_steps: int = 8) -> List[dict]:
    """numeric steps of a solution: one candidate per line that holds an editable integer 'result' (last integer after the last '=', else the last integer of the line).
    Never the line with the final \\boxed answer, never a bare enumeration like 'Step 3:'.  Offsets are character offsets into `text`."""
    out, pos = [], 0
    for li, line in enumerate(text.split("\n")):
        start, end = pos, pos + len(line)
        pos = end + 1
        if not line.strip() or "\\boxed" in line:
            continue
        lo = line.rfind("=") + 1 if "=" in line else 0
        enum_end = ENUM.match(line).end() if ENUM.match(line) else 0
        ms = [m for m in INT.finditer(line) if m.start() >= lo and m.start() >= enum_end]
        if not ms:
            continue
        m = ms[-1]
        v = int(m.group(1).replace(",", ""))
        out.append({"li": li, "start": start, "end": end, "line": line, "span": (m.start(), m.end()), "value": v, "comma": "," in m.group(1)})
    if len(out) > max_steps:
        keep = np.unique(np.linspace(0, len(out) - 1, max_steps).round().astype(int))
        out = [out[i] for i in keep]
    for k, c in enumerate(out):
        c["k"] = k
    return out


def edit_value(v: int, variant: str) -> int:
    return v + 1 if variant == "n1" else v + max(10, v // 10)


def edited_prefix(text: str, step: dict, variant: str) -> Optional[str]:
    """text up to and including the (edited / unedited / deleted) step line plus its newline; None if the step is the last line."""
    if step["end"] >= len(text):
        return None
    head = text[:step["start"]]
    if variant == "del":
        return head
    line = step["line"]
    if variant in ("n1", "n2"):
        s, e = step["span"]
        nv = edit_value(step["value"], variant)
        line = line[:s] + (f"{nv:,}" if step["comma"] else str(nv)) + line[e:]
    return head + line + "\n"


def reuse_count(text: str, step: dict) -> int:
    """oracle-only feature: in how many LATER lines does the step's result string appear again (what the probe cannot see at the step boundary)."""
    s = str(step["value"])
    later = text[step["end"]:]
    return len(re.findall(rf"(?<![\d]){re.escape(s)}(?![\d])", later))


def end_token_index(tok, ids: List[int], end_char: int) -> int:
    """number of completion tokens needed so that the decoded prefix covers `end_char` characters (binary search)."""
    lo = bisect.bisect_left(range(len(ids) + 1), end_char, key=lambda t: len(tok.decode(ids[:t])))
    return min(max(lo, 1), len(ids))


def run(problems, G, model, tok, out_dir: Path, shard=0, nshards=1, chunk=8, max_steps=8, max_new=512, max_seqs=96, budget_s=0, log=print) -> int:
    import torch
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    bud = Budget(budget_s)
    dev = next(model.parameters()).device
    base = getattr(model, "model", model)
    layers = None
    done = 0
    problems = [p for p in problems if p["id"] in G and G[p["id"]]["plain_fin"] and G[p["id"]]["plain_ans"] is not None]
    problems.sort(key=lambda p: zlib.crc32(f"steps{p['id']}".encode()))        # a budget cut-off leaves a RANDOM subset of problems
    for ds, k, batch in schedule(problems, shard, nshards, chunk):
        f = d / f"s{shard:02d}_{k:04d}.jsonl"
        if f.exists():
            continue
        if bud.spent():
            log(f"[steps shard {shard}] time budget reached - stopping cleanly (resubmit to continue)")
            break
        rows, Hs, jobs = [], [], []
        for p in batch:
            g = G[p["id"]]
            comp = g["plain_ids"]
            text = tok.decode(comp, skip_special_tokens=True)
            steps = [s for s in candidate_steps(text, max_steps) if edited_prefix(text, s, "n1") is not None]
            if not steps:
                continue
            pr = enc(tok, chat_prompt(tok, plain_text(p)))
            with torch.no_grad():
                hs = base(input_ids=torch.tensor([pr + comp], device=dev), output_hidden_states=True).hidden_states
            if layers is None:
                layers = list(range(LAYER_STEP, len(hs), LAYER_STEP))
                if len(hs) - 1 not in layers:
                    layers.append(len(hs) - 1)
            for s in steps:
                te = end_token_index(tok, comp, s["end"])
                H = torch.stack([hs[l][0, len(pr) + te - 1] for l in layers]).half().cpu().numpy()
                row = {"id": p["id"], "k": s["k"], "n_steps": len(steps), "li": s["li"], "start": s["start"], "end": s["end"], "value": s["value"], "t_end": te, "n_tok": len(comp),
                       "reuse": reuse_count(text, s), "ans_orig": g["plain_ans"]}
                rows.append(row)
                Hs.append(H)
                variants = ["n1", "n2", "del"] + (["null"] if s["k"] == 0 else [])
                for v in variants:
                    pre = edited_prefix(text, s, "n1" if v == "null" else v)
                    if v == "null":
                        pre = text[:s["end"]] + "\n"
                    jobs.append((len(rows) - 1, v, p, pre, chat_prompt(tok, plain_text(p)) + pre))
        if not rows:
            f.with_suffix(".tmp").write_text("")
            f.with_suffix(".tmp").rename(f)
            continue
        out = generate_ids(model, tok, [j[4] for j in jobs], max_new, 1, 0.0, 0.95, max_seqs)
        for (ri, v, p, pre, _), res in zip(jobs, out):
            ids, fin = res[0]
            ans = parse_answer(pre + tok.decode(ids, skip_special_tokens=True), p["kind"])
            rows[ri][f"ans_{v}"] = ans
            rows[ri][f"flip_{v}"] = float(ans != rows[ri]["ans_orig"])
            rows[ri][f"fin_{v}"] = bool(fin)
        for r in rows:
            r["load"] = float(np.mean([r["flip_n1"], r["flip_n2"], r["flip_del"]]))
        np.savez(f.with_suffix(".npz"), H=np.stack(Hs), layers=np.array(layers))
        tmp = f.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        tmp.rename(f)
        done += len(batch)
        log(f"[steps shard {shard}] chunk {k}: {len(batch)} problems, {len(rows)} steps, {len(jobs)} regenerations ({done} problems this run)")
    return done
