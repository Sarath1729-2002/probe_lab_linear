"""Wide per-problem tables + baseline ladders.

LABEL GLOSSARY (plain prompt; greedy unless noted - all deterministic functions of the stored text)
  runaway        1 = the greedy solution hit the token cap without finishing         (resource / overthinking risk)
  runaway_rate   fraction of the `n_spread` sampled solutions that hit the cap       (soft version of runaway, 0..1)
  logT           log(tokens) of the greedy solution, finished solutions only         (output length; NaN when right-censored)
  len_sd         std of log(tokens) over the sampled solutions                       (how variable the model's own length is)
  noans          1 = no parsable final answer                                        (silent failure to deliver)
  correct        1 = greedy final answer equals gold
  steps          non-empty lines in the finished solution
  ans_logmag / ans_ndig   log10(|gold|+1) / digit count of the gold answer (integer tasks)   (answer size, known before any reasoning)
CONSTRAINED prompt (one deterministic constraint per problem, verify.py)
  comply  1 = the response satisfies the constraint     over  log(words/limit) for word limits (finished responses)
  c_logT, c_runaway, c_correct   length / cap / correctness under the constraint
CHECKPOINT labels (trace.py) at 6 points along the plain solution
  settled_p / settled_lp   P / log P of the model's OWN final answer when the answer is forced now
  ready_p / ready_lp       same for the GOLD answer
  log_rem                  log(1 + tokens still to come)
"""
import glob
import json
import re
import zlib
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .bench import read_problems
from .trace import CK_NAMES
from .verify import CONSTRAINTS, LIMIT

DSN = ["gsm8k", "math", "arith", "prog", "arc"]


# ----------------------------------------------------------------------------- loading
def load_gen(run_dir: Path, datasets=None) -> Dict[int, dict]:
    out = {}
    for f in sorted(glob.glob(str(Path(run_dir) / "gen" / "g*.jsonl"))):
        for l in open(f):
            r = json.loads(l)
            if datasets is None or r["ds"] in datasets:
                out[r["id"]] = r
    return out


def load_states(run_dir: Path, names=("plain", "con")):
    idl, st = [], {n: [] for n in names}
    for f in sorted(glob.glob(str(Path(run_dir) / "states" / "x*.npz"))):
        if ".tmp" in f:
            continue
        z = np.load(f)
        idl.append(z["ids"])
        for n in names:
            st[n].append(z[n])
    if not idl:
        return None
    ids = np.concatenate(idl)
    o = np.argsort(ids)
    return {"ids": ids[o], "states": {n: np.concatenate(v)[o] for n, v in st.items()}}


TKEYS = ("H", "t", "valid", "lp_final", "lp_gold", "rem", "Tc", "fin", "men_final", "men_gold", "pf")


def load_trace(run_dir: Path):
    idl, acc, layers = [], {k: [] for k in TKEYS}, None
    for f in sorted(glob.glob(str(Path(run_dir) / "trace" / "t*.npz"))):
        if ".tmp" in f:
            continue
        z = np.load(f)
        idl.append(z["ids"])
        layers = z["layers"]
        for k in TKEYS:
            acc[k].append(z[k])
    if not idl:
        return None
    ids = np.concatenate(idl)
    o = np.argsort(ids)
    return {"ids": ids[o], "layers": layers, **{k: np.concatenate(v)[o] for k, v in acc.items()}}


# ----------------------------------------------------------------------------- features
def question_features(q: str) -> list:
    nums = re.findall(r"\d+(?:\.\d+)?", q)
    return [np.log1p(len(q)), np.log1p(len(q.split())), np.log1p(len(nums)), np.log1p(len(re.findall(r"[.?!]", q))),
            np.log1p(max([float(x) for x in nums], default=0)), np.log1p(q.count("\n"))]


def bow(q: str, dim=512) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    for w in re.findall(r"[a-z]+|\d+", q.lower()):
        v[zlib.crc32(w.encode()) % dim] += 1
    return np.log1p(v)


def ds_onehot(df: pd.DataFrame) -> np.ndarray:
    return np.stack([(df.ds.values == n).astype(np.float32) for n in DSN], 1)


def ctype_onehot(df: pd.DataFrame) -> np.ndarray:
    return np.stack([(df.ctype.values == c).astype(np.float32) for c in CONSTRAINTS], 1)


def d1_rank(df: pd.DataFrame) -> np.ndarray:
    r = df.groupby("ds").d1.rank(pct=True).values.astype(np.float32)
    return np.stack([r, r ** 2], 1)


def _lg(x):
    return np.log(np.maximum(np.asarray(x, dtype=float), 1.0))


# ----------------------------------------------------------------------------- the problem table
def build(run_dir: Path, datasets=None, need_states=True, names=("plain", "con"), log=print):
    """-> (df, sub).  Rows = problems with a generation record (and prompt-end states when need_states).  sub[name] = [N, L+1, D] float16."""
    run_dir = Path(run_dir)
    probs = {p["id"]: p for p in read_problems(run_dir, datasets)}
    G = load_gen(run_dir, datasets)
    st = load_states(run_dir, names) if need_states else None
    if need_states and st is None:
        raise RuntimeError("no states yet")
    have = set(st["ids"].tolist()) if need_states else None
    ids = sorted(i for i in G if i in probs and (not need_states or i in have))
    if not ids:
        raise RuntimeError("no problem has both a generation record and states")
    rows = []
    for i in ids:
        p, g = probs[i], G[i]
        T = len(g["plain_ids"])
        sn, sf = np.array(g["spread_n"], dtype=float), np.array(g["spread_fin"], dtype=bool)
        r = {"id": i, "ds": p["ds"], "kind": p["kind"], "split": p["split"], "d1": p["d1"], "gold": p["gold"], "ctype": g["ctype"], "_q": p["question"],
             "T": T, "logT_all": float(np.log(max(T, 1))), "logT": float(np.log(max(T, 1))) if g["plain_fin"] else np.nan,
             "runaway": float(not g["plain_fin"]), "runaway_rate": float(1 - sf.mean()) if len(sf) else np.nan,
             "len_sd": float(np.log(np.maximum(sn, 1)).std()) if len(sn) > 1 else np.nan,
             "noans": float(g["plain_ans"] is None), "correct": float(g["plain_ans"] == p["gold"]),
             "steps": float(len([x for x in g["plain_text"].split("\n") if x.strip()])) if g["plain_fin"] else np.nan, "words": g["plain_words"],
             "ans_ndig": float(len(str(abs(int(p["gold"]))))) if p["kind"] == "int" else np.nan,
             "ans_logmag": float(np.log10(abs(int(p["gold"])) + 1)) if p["kind"] == "int" else np.nan,
             "comply": float(g["con_comply"]), "over": np.nan if g["con_over"] is None else float(g["con_over"]),
             "c_logT": float(np.log(max(g["con_ntok"], 1))) if g["con_fin"] else np.nan, "c_logT_all": float(np.log(max(g["con_ntok"], 1))),
             "c_runaway": float(not g["con_fin"]), "c_correct": float(g["con_ans"] == p["gold"]), "c_words": g["con_words"],
             "logN": float(np.log(LIMIT[g["ctype"]])) if g["ctype"] in LIMIT else 0.0}
        rows.append(r)
    df = pd.DataFrame(rows)
    sub = {}
    if need_states:
        sel = np.searchsorted(st["ids"], ids)
        sub = {n: a[sel] for n, a in st["states"].items()}
    log(f"table: {len(df)} problems {df.ds.value_counts().to_dict()}")
    return df, sub


# ----------------------------------------------------------------------------- baseline ladders (names containing 'oracle' use what the model cannot know)
def _early(sub, cond, early=(4, 8)):
    return {f"H{l}(early-layer)": sub[cond][:, l, :].astype(np.float32) for l in early if cond in sub and l < sub[cond].shape[1]}


def ladder_plain(df, sub) -> Dict[str, np.ndarray]:
    oh = ds_onehot(df)
    Bq = np.concatenate([np.array([question_features(q) for q in df._q], dtype=np.float32), oh], 1)
    Bbow = np.concatenate([np.stack([bow(q) for q in df._q]), oh], 1)
    Bd1 = np.concatenate([Bq, d1_rank(df)], 1)
    con = np.stack([df.c_logT_all, df.c_runaway, df.c_correct, df.comply], 1).astype(np.float32)
    out = {"Bds": oh, "Bq": Bq, "Bbow": Bbow, "Bd1(oracle)": Bd1, "Bother(oracle)": np.concatenate([Bd1, con, con ** 2, ctype_onehot(df)], 1)}
    out.update(_early(sub, "plain"))
    return out


def ladder_con(df, sub) -> Dict[str, np.ndarray]:
    oh, co = ds_onehot(df), ctype_onehot(df)
    qf = np.array([question_features(q) for q in df._q], dtype=np.float32)
    logN = df.logN.values.astype(np.float32)[:, None]
    Bcon = np.concatenate([qf, oh, co, logN, co * qf[:, :1], logN * qf[:, :1]], 1)          # constraint identity/limit x question length: all readable from the prompt text
    Bbow = np.concatenate([np.stack([bow(q) for q in df._q]), oh, co, logN], 1)
    plain = np.stack([df.logT_all, df.runaway, df.words, df.steps.fillna(df.steps.median()), df.correct, df.noans, df.runaway_rate, df.len_sd], 1).astype(np.float32)
    plain = np.nan_to_num(plain)
    out = {"Bds": oh, "Bcon": Bcon, "Bbow+con": Bbow, "Bplain(oracle)": np.concatenate([Bcon, plain, plain ** 2], 1),
           "Bd1(oracle)": np.concatenate([Bcon, d1_rank(df)], 1)}
    out.update(_early(sub, "con"))
    return out


# ----------------------------------------------------------------------------- checkpoint tables
def build_trace(run_dir: Path, datasets=None, log=print):
    """-> (base df, trace dict, frames, tks).  frames[k] = df rows whose checkpoint k is VALID + the checkpoint labels; tks[k] = the matching arrays
    (H [n, nl, D], pf, Tc, layers), so every probe/baseline row is one problem and bootstrap CIs are over problems."""
    run_dir = Path(run_dir)
    tr = load_trace(run_dir)
    if tr is None:
        raise RuntimeError("no trace yet")
    df, _ = build(run_dir, datasets, need_states=False, log=lambda *_: None)
    keep = np.isin(tr["ids"], df.id.values)
    df = df.set_index("id").loc[tr["ids"][keep]].reset_index()
    tr = {k: (v[keep] if k != "layers" else v) for k, v in tr.items()}
    frames, tks = [], []
    for k, name in enumerate(CK_NAMES):
        v = tr["valid"][:, k]
        f = df[v].copy().reset_index(drop=True)
        f["t_k"] = tr["t"][v, k].astype(float)
        f["frac_k"] = f.t_k / np.maximum(tr["Tc"][v], 1)
        f["settled_lp"] = np.clip(tr["lp_final"][v, k], -20, 0)
        f["settled_p"] = np.exp(f.settled_lp)
        f["ready_lp"] = np.clip(tr["lp_gold"][v, k], -20, 0)
        f["ready_p"] = np.exp(f.ready_lp)
        f["log_rem"] = np.nan if name == "pre" else np.log1p(tr["rem"][v, k])        # at `pre` the position reveals the length: not a fair remaining-length target
        f["men_final"] = tr["men_final"][v, k].astype(float)
        f["men_gold"] = tr["men_gold"][v, k].astype(float)
        f["settled_hi"] = np.where(np.isnan(f.settled_lp), np.nan, (f.settled_p > 0.5).astype(float))
        frames.append(f)
        tks.append({"H": tr["H"][v, k], "pf": tr["pf"][v, k], "Tc": tr["Tc"][v], "layers": tr["layers"]})
    log(f"trace table: {len(df)} problems; valid per checkpoint {dict(zip(CK_NAMES, [len(f) for f in frames]))}; {len(tr['layers'])} stored layers")
    return df, tr, frames, tks


def ladder_trace(f: pd.DataFrame, tk: dict) -> Dict[str, np.ndarray]:
    """observable: dataset identity, the question, the visible solution-so-far (Btext).  Position is constant within an absolute-offset checkpoint (it is added as a feature
    for completeness / for `pre`).  oracle: whether the answer string is already in the text, total length, correctness, difficulty."""
    oh = ds_onehot(f)
    t = f.t_k.values.astype(np.float32)[:, None]
    pos = np.concatenate([t / 100.0, np.log1p(t), oh], 1)
    qf = np.array([question_features(q) for q in f._q], dtype=np.float32)
    Btext = np.concatenate([pos, tk["pf"], qf], 1)
    oracle = np.nan_to_num(np.stack([f.men_final, f.men_gold, np.log1p(tk["Tc"]), f.frac_k, f.correct], 1).astype(np.float32))
    out = {"Bpos": pos, "Btext": Btext, "Btext+answer-mentions(oracle)": np.concatenate([Btext, oracle], 1), "Bd1(oracle)": np.concatenate([Btext, d1_rank(f)], 1)}
    layers = list(tk["layers"])
    for l in (4, 8):
        if l in layers:
            out[f"H{l}(early-layer)"] = tk["H"][:, layers.index(l), :].astype(np.float32)
    return out
