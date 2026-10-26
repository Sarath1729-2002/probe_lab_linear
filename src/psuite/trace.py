"""Checkpoints along the model's own greedy PLAIN solution.  For checkpoint t (completion tokens included) one forward over
        prompt + completion[:t] + CUE + "<answer>}"
gives (causality!) both the natural residual stream at the last prefix token AND the forced-answer log-probs:
  lp_final  log P(the model's OWN final answer | prefix, cue)   -> how settled is the answer already?        (NaN if the solution has no parsable answer)
  lp_gold   log P(gold answer | prefix, cue)                    -> answer-readiness of the CORRECT answer
  rem       T - t completion tokens still to come (the remaining-length target of 'How Much is Left?'; NaN if the solution hit the cap = right-censored)
Checkpoints: ABSOLUTE token offsets 0, 24, 48, 96, 192 into the solution, and `pre` = just before the final \\boxed (all reasoning written, answer not yet).
Absolute offsets (not fractions of the solution) on purpose: a checkpoint at a fixed FRACTION of the solution has position t = f*T, so position alone would reveal the total length and
make 'remaining tokens' trivially solvable without reading the state.  An offset checkpoint is `valid` only if at least `min_left` tokens remain after it, so the model is
genuinely mid-solution; `pre` is not used for the remaining-length target (its position reveals the length).
`men_final` / `men_gold`: does the visible prefix already contain that answer string? (the text baseline's strongest feature)."""
import json
import re
from pathlib import Path

import numpy as np

from .hf_util import checkpoint_forward, enc, pad_id
from .prompts import CUE, chat_prompt, plain_text
from .sched import Budget, schedule

OFFS = (0, 24, 48, 96, 192)
CK_NAMES = ("o0", "o24", "o48", "o96", "o192", "pre")
MIN_LEFT = 8
LAYER_STEP = 2


def ck_layers(nstates: int):
    ls = list(range(LAYER_STEP, nstates, LAYER_STEP))
    if nstates - 1 not in ls:
        ls.append(nstates - 1)
    return ls


def checkpoint_ts(Tc: int, text: str, tok, offsets=OFFS, min_left=MIN_LEFT):
    """-> (token counts, valid flags)."""
    ts = list(offsets)
    valid = [(o == 0 and Tc >= 1) or (o > 0 and o <= Tc - min_left) for o in offsets]
    i = text.rfind("\\boxed")
    ts.append(min(max(len(enc(tok, text[:i])), 0), Tc) if i >= 0 else Tc)
    valid.append(i >= 0 and Tc >= 1)
    return ts, valid


def prefix_feats(text: str) -> list:
    """observable features of the visible solution-so-far (no access to the future or to any answer): the TEXT baseline for the checkpoint probes."""
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    last = float(nums[-1]) if nums else 0.0
    return [np.log1p(len(text)), np.log1p(len(text.split())), np.log1p(len(nums)), np.log1p(text.count("=")), np.log1p(text.count("\n")), np.log1p(last),
            float("\\boxed" in text), float(bool(re.search(r"\b(wait|however|but|let me|recheck|double-check)\b", text.lower())))]


def mentions(prefix_text: str, ans) -> bool:
    return bool(ans) and re.search(rf"(?<![\w.]){re.escape(ans)}(?![\w])", prefix_text) is not None


def run(problems, model, tok, out_dir: Path, shard=0, nshards=1, fwd_bs=4, chunk=64, tag="", budget_s=0, sub="", offsets=OFFS, min_left=MIN_LEFT, log=print) -> int:
    out_dir = Path(out_dir)
    d = out_dir / "trace"
    d.mkdir(parents=True, exist_ok=True)
    pad = pad_id(tok)
    cue = enc(tok, CUE)
    layers = None
    bud = Budget(budget_s)
    done = 0
    for ds, k, batch in schedule(problems, shard, nshards, chunk, sub):
        f = d / f"t{shard:02d}_{ds}_{k:04d}{tag}.npz"
        gfile = out_dir / "gen" / f"g{shard:02d}_{ds}_{k:04d}{tag}.jsonl"
        if f.exists() or not gfile.exists():
            continue
        if bud.spent():
            log("[trace] time budget reached - stopping cleanly")
            break
        G = {}
        for l in open(gfile):
            r = json.loads(l)
            G[r["id"]] = r
        if layers is None:
            layers = ck_layers(model.config.num_hidden_layers + 1)
        K, N = len(CK_NAMES), len(batch)
        ts = np.zeros((N, K), dtype=np.int32)
        valid = np.zeros((N, K), dtype=bool)
        jobs = []                                    # (n, k, which, seq, state_pos, ans_start)
        meta = []
        for n, p in enumerate(batch):
            g = G[p["id"]]
            pr = enc(tok, chat_prompt(tok, plain_text(p)))
            comp = g["plain_ids"]
            Tc = len(comp)
            ts[n], valid[n] = checkpoint_ts(Tc, g["plain_text"], tok, offsets, min_left)
            final, gold = g["plain_ans"], p["gold"]
            pre_txt = [tok.decode(comp[:t]) for t in ts[n]]
            meta.append((Tc, g["plain_fin"], final, gold, [mentions(x, final) for x in pre_txt], [mentions(x, gold) for x in pre_txt], [prefix_feats(x) for x in pre_txt]))
            for kk, t in enumerate(ts[n]):
                if not valid[n, kk]:
                    continue
                base = pr + comp[:t] + cue
                for which, ans in (("gold", gold), ("final", final)):
                    if ans is None or (which == "final" and final == gold):
                        continue
                    a = enc(tok, f"{ans}}}")
                    jobs.append((n, kk, which, base + a, len(pr) + t - 1, len(base)))
        H = np.zeros((N, K, len(layers), model.config.hidden_size), dtype=np.float16)
        lp = {"gold": np.full((N, K), np.nan), "final": np.full((N, K), np.nan)}
        order = sorted(range(len(jobs)), key=lambda j: len(jobs[j][3]))
        for s in range(0, len(order), fwd_bs):
            sel = [jobs[j] for j in order[s:s + fwd_bs]]
            st, lps = checkpoint_forward(model, [j[3] for j in sel], [j[4] for j in sel], [j[5] for j in sel], layers, pad)
            for r, (n, kk, which, *_rest) in enumerate(sel):
                lp[which][n, kk] = lps[r]
                if which == "gold":
                    H[n, kk] = st[r].numpy()
        for n, p in enumerate(batch):                        # same answer => same number; no parsable answer => stays NaN
            if meta[n][2] is not None and meta[n][2] == meta[n][3]:
                lp["final"][n] = lp["gold"][n]
        rem = np.array([[(m[0] - t) if (m[1] and valid[n, kk]) else np.nan for kk, t in enumerate(ts[n])] for n, m in enumerate(meta)], dtype=np.float64)
        pack = {"ids": np.array([p["id"] for p in batch]), "H": H, "layers": np.array(layers), "t": ts, "valid": valid, "lp_final": lp["final"], "lp_gold": lp["gold"], "rem": rem,
                "Tc": np.array([m[0] for m in meta], dtype=np.int32), "fin": np.array([m[1] for m in meta]),
                "men_final": np.array([m[4] for m in meta]), "men_gold": np.array([m[5] for m in meta]),
                "pf": np.array([m[6] for m in meta], dtype=np.float32)}
        tmp = f.with_name(f.stem + ".tmp.npz")
        np.savez(tmp, **pack)
        tmp.rename(f)
        done += N
        log(f"[trace shard {shard}] {ds} chunk {k}: +{N} ({done} this run)")
    return done
