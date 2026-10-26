"""Problem sets with DETERMINISTIC ground truth, and the audit that decides whether a dataset is admitted.

A dataset enters the study only if every check in `audit()` passes on >= 99.5% of its problems (failing problems are dropped and counted):
  G1  gold is ONE canonical string (integer, or a letter that exists among the options)
  G2  gold is re-derived by a path independent of the stored label  (GSM8K: replay every <<expr=val>> calculator annotation exactly;
      MATH: the reference solution has exactly one distinct boxed value; synthetic: re-parse the PROMPT TEXT and recompute / re-execute it)
  G3  the planted wrong answer differs from gold, lies in the answer space, and is reproducible from (id, gold)
  G4  parse round-trip: parse(format(gold)) == gold and parse(format(hint)) == hint
  G5  no duplicate prompt with a different gold
Tier A = answer is computed or proof-checkable (GSM8K, MATH-integer, synthetic).  Tier B = expert-keyed multiple choice (ARC):
keyed by humans, so a small label-noise rate is possible; it is analysed separately and never carries a headline claim alone.
"""
import ast
import contextlib
import io
import json
import re
from fractions import Fraction
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from .textutil import all_boxed, canon_int, canon_letter, parse_answer

TIER = {"gsm8k": "A", "math": "A", "arith": "A", "prog": "A", "arc": "B"}
KIND = {"gsm8k": "int", "math": "int", "arith": "int", "prog": "int", "arc": "mcq"}
DS_ORDER = ["gsm8k", "arith", "prog", "math", "arc"]
LETTERS = "ABCD"
MATH_CANDIDATES = [("EleutherAI/hendrycks_math", "subjects"), ("DigitalLearningGmbH/MATH-lighteval", None), ("lighteval/MATH", "all"),
                   ("nlile/hendrycks-MATH-benchmark", None), ("qwedsacf/competition_math", None)]
MATH_SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]


# ----------------------------------------------------------------------------- hints
def hint_wrong_int(pid: int, gold: int) -> int:
    """A plausible WRONG integer, deterministic per problem.  The signed offset is randomly assigned (a manipulated variable)."""
    rng = np.random.RandomState(1000 + pid)
    d = int(rng.choice([-10, -3, -2, -1, 1, 2, 3, 10]))
    x = gold + d
    if (gold >= 0 and x < 0) or x == gold:
        x = gold + abs(d)
    return int(x)


def hint_wrong_letter(pid: int, gold: str) -> str:
    rng = np.random.RandomState(2000 + pid)
    wrong = [c for c in LETTERS if c != gold]
    return wrong[int(rng.randint(len(wrong)))]


def make_hint(kind: str, pid: int, gold: str) -> str:
    return str(hint_wrong_int(pid, int(gold))) if kind == "int" else hint_wrong_letter(pid, gold)


def hint_distance(kind: str, gold: str, hint: str) -> float:
    """signed offset: integers h-g; MCQ: option-position of hint minus gold."""
    if kind == "int":
        return float(int(hint) - int(gold))
    return float(LETTERS.index(hint) - LETTERS.index(gold))


# ----------------------------------------------------------------------------- safe exact arithmetic (GSM8K annotation replay)
_BIN = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b}


def safe_eval(expr: str) -> Fraction:
    """exact value of a +,-,*,/,**(small int) expression; raises ValueError on anything else (so it can never execute code)."""
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return Fraction(str(n.value))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.BinOp):
            if type(n.op) in _BIN:
                return _BIN[type(n.op)](ev(n.left), ev(n.right))
            if isinstance(n.op, ast.Pow):
                a, b = ev(n.left), ev(n.right)
                if b.denominator != 1 or abs(b) > 12:
                    raise ValueError("pow")
                return a ** int(b)
        raise ValueError(f"unsupported: {ast.dump(n)[:40]}")
    try:
        return ev(ast.parse(expr.strip(), mode="eval"))
    except (SyntaxError, ZeroDivisionError, RecursionError, OverflowError) as e:
        raise ValueError(str(e))


_ANN = re.compile(r"<<([^<>]*?)=([^<>=]*?)>>")


def gsm8k_check(answer_field: str):
    """-> (gold:int|None, status).  status 'ok' | 'no_gold' | 'non_integer' | 'annotation_mismatch' | 'final_not_derived'."""
    if "####" not in answer_field:
        return None, "no_gold"
    tail = answer_field.split("####")[-1].replace(",", "").replace("$", "").strip()
    m = re.fullmatch(r"-?\d+(\.0+)?", tail)
    if not m:
        return None, "non_integer"
    gold = int(float(tail))
    results = []
    for expr, val in _ANN.findall(answer_field):
        try:
            v = safe_eval(expr.replace(",", "").replace("$", ""))
            want = Fraction(val.replace(",", "").replace("$", "").strip())
        except (ValueError, ZeroDivisionError):
            continue                                            # not replayable (percent, units, ...): neither pass nor fail
        if abs(v - want) > Fraction(1, 100) * max(1, abs(want)) * Fraction(1, 10):
            return None, "annotation_mismatch"
        results.append(v)
    if results and not any(r == gold for r in results):
        return None, "final_not_derived"
    return gold, "ok"


def n_steps(answer_field: str) -> int:
    k = answer_field.count("<<")
    return k if k else max(1, len([l for l in answer_field.split("\n") if l.strip() and not l.startswith("####")]))


# ----------------------------------------------------------------------------- synthetic families (answers computed, never labelled)
def _gen_arith(rng):
    for _ in range(100):
        k = int(rng.randint(2, 8))
        v0 = int(rng.randint(2, 100))
        v, ops, ok = v0, [], True
        for _s in range(k):
            op = str(rng.choice(["+", "-", "*"], p=[0.35, 0.3, 0.35]))
            b = int(rng.randint(2, 13 if abs(v) > 1000 else 100)) if op == "*" else int(rng.randint(2, 1000))
            v = v + b if op == "+" else v - b if op == "-" else v * b
            ops.append([op, b])
            if abs(v) > 10 ** 9:
                ok = False
                break
        if ok:
            return v0, ops, v
    raise RuntimeError("could not generate")


_VERB = {"+": "Add {b}.", "-": "Subtract {b}.", "*": "Multiply by {b}."}


def arith_text(v0, ops) -> str:
    return " ".join([f"Start with {v0}."] + [_VERB[o].format(b=b) for o, b in ops]) + " What is the final result?"


def arith_recompute_from_text(text: str) -> Optional[int]:
    """independent re-derivation: parse the PROMPT TEXT, not the generator's spec."""
    m = re.match(r"Start with (-?\d+)\.", text)
    if not m:
        return None
    v = Fraction(int(m.group(1)))
    for sent in re.findall(r"(Add|Subtract|Multiply by) (\d+)\.", text):
        b = int(sent[1])
        v = v + b if sent[0] == "Add" else v - b if sent[0] == "Subtract" else v * b
    return int(v)


_PROG = {
    "mod": "s = {s0}\nfor i in range(1, {n} + 1):\n    s = (s * {a} + i) % {m}\nprint(s)",
    "branch": "s = {s0}\nfor i in range(1, {n} + 1):\n    if i % {c} == 0:\n        s += i * {a}\n    else:\n        s -= {b}\nprint(s)",
    "fib": "a, b = {s0}, {b}\nfor _ in range({n}):\n    a, b = b, (a + b) % {m}\nprint(a)",
}


def prog_value(t: str, p: dict) -> int:
    """direct re-implementation of each template (second code path; the audit also exec()s the prompt's code block)."""
    if t == "mod":
        s = p["s0"]
        for i in range(1, p["n"] + 1):
            s = (s * p["a"] + i) % p["m"]
        return s
    if t == "branch":
        s = p["s0"]
        for i in range(1, p["n"] + 1):
            s = s + i * p["a"] if i % p["c"] == 0 else s - p["b"]
        return s
    a, b = p["s0"], p["b"]
    for _ in range(p["n"]):
        a, b = b, (a + b) % p["m"]
    return a


def _gen_prog(rng):
    t = str(rng.choice(list(_PROG)))
    n = int(rng.randint(3, 15))
    p = {"s0": int(rng.randint(1, 10)), "n": n, "a": int(rng.randint(2, 9)), "m": int(rng.randint(7, 100)), "c": int(rng.randint(2, 5)),
         "b": int(rng.randint(1, 10))}
    return t, p


def prog_text(t: str, p: dict) -> str:
    code = _PROG[t].format(**p)
    return f"What does this Python program print?\n\n```python\n{code}\n```"


def run_code_block(text: str) -> Optional[int]:
    m = re.search(r"```python\n(.*?)\n```", text, re.S)
    if not m:
        return None
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(m.group(1), "<prompt>", "exec"), {"__builtins__": {"range": range, "print": print}}, {})
        return int(buf.getvalue().strip())
    except Exception:
        return None


# ----------------------------------------------------------------------------- loaders -> raw rows (no ids / splits yet)
def rows_gsm8k(raw: Dict[str, List[dict]], n_total: int, seed=0, log=print) -> List[dict]:
    rng = np.random.RandomState(seed)
    rows, drops = [], {}
    pool = [("test", i) for i in range(len(raw["test"]))]
    extra = [("train", int(i)) for i in rng.permutation(len(raw["train"]))]
    for split, i in pool + extra:
        if len(rows) >= n_total:
            break
        r = raw[split][i]
        gold, status = gsm8k_check(r["answer"])
        if status != "ok":
            drops[status] = drops.get(status, 0) + 1
            continue
        rows.append({"ds": "gsm8k", "src": f"{split}{i}", "question": r["question"].strip(), "gold": str(gold), "d1": float(n_steps(r["answer"])),
                     "diff": {"steps": n_steps(r["answer"])}})
    log(f"gsm8k: kept {len(rows)}; dropped at source {drops}")
    return rows


def _norm_math_columns(df):
    ren = {}
    for want, alts in (("problem", ["problem", "question"]), ("solution", ["solution", "answer_text"]), ("level", ["level"]), ("type", ["type", "subject"])):
        for a in alts:
            if a in df.columns:
                ren[a] = want
                break
    df = df.rename(columns=ren)
    for c in ("level", "type"):
        if c not in df.columns:
            df[c] = ""
    return df[["problem", "solution", "level", "type"]]


def load_math(loader: Optional[Callable] = None, log=print):
    import pandas as pd
    if loader is None:
        from datasets import load_dataset as loader
    last = None
    for name, cfg in MATH_CANDIDATES:
        try:
            if cfg == "subjects":
                out = {}
                for split in ("train", "test"):
                    out[split] = _norm_math_columns(pd.concat([pd.DataFrame(loader(name, s, split=split)).assign(type=s) for s in MATH_SUBJECTS], ignore_index=True))
            else:
                ds = loader(name, cfg) if cfg else loader(name)
                if hasattr(ds, "keys") and "train" in ds and "test" in ds:
                    out = {s: _norm_math_columns(pd.DataFrame(ds[s])) for s in ("train", "test")}
                else:
                    allp = _norm_math_columns(pd.DataFrame(ds["train"] if hasattr(ds, "keys") else ds)).sample(frac=1.0, random_state=0).reset_index(drop=True)
                    k = int(0.6 * len(allp))
                    out = {"train": allp.iloc[:k].reset_index(drop=True), "test": allp.iloc[k:].reset_index(drop=True)}
            log(f"MATH from {name}: train {len(out['train'])} test {len(out['test'])}")
            return out
        except Exception as exc:                                # noqa: BLE001
            last = exc
            log(f"  {name}: {type(exc).__name__}: {str(exc)[:100]}")
    raise RuntimeError(f"no MATH dataset could be loaded ({last})")


def rows_math(splits, n_total: int, seed=0, log=print) -> List[dict]:
    rng = np.random.RandomState(seed)
    cand, drops = [], {"no_boxed": 0, "non_integer": 0, "ambiguous_multi_boxed": 0, "no_level": 0}
    for split in ("test", "train"):
        df = splits[split]
        for r in df.itertuples(index=False):
            vals = all_boxed(r.solution)
            if not vals:
                drops["no_boxed"] += 1
                continue
            ints = {canon_int(v) for v in vals}
            if None in ints:
                drops["non_integer"] += 1
                continue
            if len(ints) != 1:
                drops["ambiguous_multi_boxed"] += 1
                continue
            m = re.search(r"(\d)", str(r.level))
            if not m:
                drops["no_level"] += 1
                continue
            cand.append((split, {"ds": "math", "src": f"{split}:{len(cand)}", "question": r.problem.strip(), "gold": ints.pop(),
                                 "d1": float(m.group(1)), "diff": {"level": int(m.group(1)), "type": str(r.type)}}))
    order = rng.permutation(len(cand))
    # test split first (as for GSM8K) so the benchmark's own test problems are always in
    ordered = sorted(order, key=lambda i: (cand[i][0] != "test", i))
    rows = [cand[i][1] for i in ordered[:n_total]]
    log(f"math: {len(cand)} integer-answer problems available, kept {len(rows)}; dropped at source {drops}")
    return rows


def rows_arc(loader: Optional[Callable] = None, n_total: int = 1500, seed=0, log=print) -> List[dict]:
    if loader is None:
        from datasets import load_dataset
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")
        loader = lambda split: [dict(r) for r in ds[split]]      # noqa: E731
    rows, drops = [], {"not4": 0, "bad_key": 0}
    for split in ("test", "validation", "train"):
        for r in loader(split):
            labels, texts = list(r["choices"]["label"]), list(r["choices"]["text"])
            if len(texts) != 4:
                drops["not4"] += 1
                continue
            key = str(r["answerKey"])
            mp = {l: LETTERS[i] for i, l in enumerate(labels)}
            if key not in mp:
                drops["bad_key"] += 1
                continue
            body = r["question"].strip() + "\n\n" + "\n".join(f"{LETTERS[i]}. {t.strip()}" for i, t in enumerate(texts))
            rows.append({"ds": "arc", "src": f"{split}:{r.get('id', len(rows))}", "question": body, "gold": mp[key], "d1": float(len(r["question"].split())),
                         "diff": {"words": len(r["question"].split())}, "choices": [t.strip() for t in texts]})
    rng = np.random.RandomState(seed)
    rows = [rows[i] for i in rng.permutation(len(rows))[:n_total]]
    log(f"arc: kept {len(rows)}; dropped {drops}")
    return rows


def rows_arith(n: int, seed=0) -> List[dict]:
    rng = np.random.RandomState(seed + 11)
    out = []
    while len(out) < n:
        v0, ops, g = _gen_arith(rng)
        out.append({"ds": "arith", "src": f"arith{len(out)}", "question": arith_text(v0, ops), "gold": str(g), "d1": float(len(ops)),
                    "diff": {"k": len(ops), "v0": v0, "ops": ops}})
    return out


def rows_prog(n: int, seed=0) -> List[dict]:
    rng = np.random.RandomState(seed + 23)
    out = []
    while len(out) < n:
        t, p = _gen_prog(rng)
        text = prog_text(t, p)
        g = run_code_block(text)
        if g is None:
            continue
        out.append({"ds": "prog", "src": f"prog{len(out)}", "question": text, "gold": str(g), "d1": float(p["n"]), "diff": {"k": p["n"], "t": t, "p": p}})
    return out


# ----------------------------------------------------------------------------- audit
def audit_problem(r: dict) -> List[str]:
    """-> list of failed check names for one finished problem row (empty = admitted)."""
    bad = []
    kind, g, h = KIND[r["ds"]], r["gold"], r["hint"]
    canon = canon_int if kind == "int" else canon_letter
    # G1
    if canon(g) != g:
        bad.append("G1_gold_not_canonical")
    if kind == "mcq" and (g not in LETTERS or not re.search(rf"\n{g}\. ", r["question"])):
        bad.append("G1_gold_not_an_option")
    # G2
    if r["ds"] == "arith":
        if arith_recompute_from_text(r["question"]) != int(g):
            bad.append("G2_arith_recompute")
    elif r["ds"] == "prog":
        if run_code_block(r["question"]) != int(g) or prog_value(r["diff"]["t"], r["diff"]["p"]) != int(g):
            bad.append("G2_prog_reexec")
    # G3
    if h == g or canon(h) != h:
        bad.append("G3_hint_invalid")
    if make_hint(kind, r["id"], g) != h:
        bad.append("G3_hint_not_reproducible")
    if kind == "mcq" and h not in LETTERS:
        bad.append("G3_hint_outside_answer_space")
    # G4
    for v in (g, h):
        if parse_answer(f"so \\boxed{{{v}}}<|im_end|>", kind) != v:
            bad.append("G4_roundtrip")
            break
    return bad


def audit(rows: List[dict], log=print) -> dict:
    """returns {'kept': rows, 'report': {...}}; drops failing rows and duplicate prompts (G5)."""
    kept, fails, seen = [], {}, {}
    for r in rows:
        key = re.sub(r"\s+", " ", r["question"].strip().lower())
        if key in seen:
            if seen[key] != r["gold"]:
                fails["G5_duplicate_conflicting_gold"] = fails.get("G5_duplicate_conflicting_gold", 0) + 1
            else:
                fails["G5_duplicate_same_gold"] = fails.get("G5_duplicate_same_gold", 0) + 1
            continue
        seen[key] = r["gold"]
        b = audit_problem(r)
        if b:
            for x in b:
                fails[x] = fails.get(x, 0) + 1
            continue
        kept.append(r)
    ds = rows[0]["ds"] if rows else "?"
    rep = {"dataset": ds, "tier": TIER.get(ds, "?"), "n_in": len(rows), "n_kept": len(kept), "pass_rate": len(kept) / max(len(rows), 1), "fails": fails,
           "admitted": bool(len(kept) / max(len(rows), 1) >= 0.995 and len(kept) >= 50)}
    log(f"audit {ds}: {len(kept)}/{len(rows)} pass ({rep['pass_rate']:.4f}) fails={fails} -> {'ADMITTED' if rep['admitted'] else 'REJECTED'}")
    return {"kept": kept, "report": rep}


# ----------------------------------------------------------------------------- build / read
def finalize(rows_by_ds: Dict[str, List[dict]], seed=0, log=print):
    """assign global ids, hints, 70/10/20 splits within each dataset, run the audit.  -> (problems, audit_reports)."""
    problems, reports = [], []
    rng = np.random.RandomState(seed)
    nid = 0
    for ds in DS_ORDER:
        rows = rows_by_ds.get(ds)
        if not rows:
            continue
        kind = KIND[ds]
        for r in rows:
            r["id"] = nid
            r["kind"] = kind
            r["hint"] = make_hint(kind, nid, r["gold"])
            r["hint_off"] = hint_distance(kind, r["gold"], r["hint"])
            nid += 1
        res = audit(rows, log)
        reports.append(res["report"])
        if not res["report"]["admitted"]:
            log(f"!! {ds} REJECTED by the ground-truth audit - not used")
            continue
        kept = res["kept"]
        perm = rng.permutation(len(kept))
        n = len(kept)
        role = np.empty(n, dtype=object)
        role[perm[: int(0.7 * n)]] = "fit"
        role[perm[int(0.7 * n): int(0.8 * n)]] = "val"
        role[perm[int(0.8 * n):]] = "test"
        for r, ro in zip(kept, role):
            r["split"] = str(ro)
            r["tier"] = TIER[ds]
        problems += kept
        log(f"{ds}: {n} problems (fit/val/test = {(role == 'fit').sum()}/{(role == 'val').sum()}/{(role == 'test').sum()})")
    return problems, reports


def write(out_dir: Path, problems: List[dict], reports: List[dict]):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "problems.jsonl", "w") as f:
        for p in problems:
            f.write(json.dumps(p) + "\n")
    (out_dir / "ground_truth_audit.json").write_text(json.dumps(reports, indent=1))


def read_problems(run_dir: Path, datasets=None) -> List[dict]:
    with open(Path(run_dir) / "problems.jsonl") as f:
        ps = [json.loads(l) for l in f if l.strip()]
    return [p for p in ps if datasets is None or p["ds"] in datasets]


def build(out_dir: Path, sizes: Dict[str, int], loaders: Optional[Dict[str, Callable]] = None, seed=0, log=print):
    """LOGIN node.  sizes e.g. {'gsm8k': 2400, 'math': 1800, 'arc': 1500, 'arith': 1000, 'prog': 1000}.  loaders override downloads (tests)."""
    loaders = loaders or {}
    rows = {}

    def attempt(name, fn):
        """a download problem must not kill the whole setup: skip that dataset loudly and carry on with the rest."""
        try:
            rows[name] = fn()
        except Exception as exc:                                   # noqa: BLE001
            log(f"!! {name} SKIPPED (could not be loaded/built): {type(exc).__name__}: {str(exc)[:200]}")

    def _gsm():
        if "gsm8k" in loaders:
            raw = loaders["gsm8k"]()
        else:
            from datasets import load_dataset
            d = load_dataset("openai/gsm8k", "main")
            raw = {s: [dict(r) for r in d[s]] for s in ("train", "test")}
        return rows_gsm8k(raw, sizes["gsm8k"], seed, log)
    if sizes.get("gsm8k"):
        attempt("gsm8k", _gsm)
    if sizes.get("math"):
        attempt("math", lambda: rows_math(loaders["math"]() if "math" in loaders else load_math(log=log), sizes["math"], seed, log))
    if sizes.get("arc"):
        attempt("arc", lambda: rows_arc(loaders.get("arc"), sizes["arc"], seed, log))
    if sizes.get("arith"):
        rows["arith"] = rows_arith(sizes["arith"], seed)
    if sizes.get("prog"):
        rows["prog"] = rows_prog(sizes["prog"], seed)
    problems, reports = finalize(rows, seed, log)
    write(out_dir, problems, reports)
    return problems, reports
