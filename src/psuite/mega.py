"""MEGA RUN analysis (GSM8K only).  Four threads, each with a decision rule written in docs/MEGA.md BEFORE the run; a failure in one thread never blocks the others.

  steps   causal step importance (interventions on the chain of thought) read from the hidden state at the step boundary
  unc     answer-distribution uncertainty (pass rate / answer entropy / majority share) on 2000 UNSEEN GSM8K problems, corrected (cross-validated) protocol
  steer   is the probe's length direction causal?  dose-response vs norm-matched random / mean-difference directions
  suite   GSM8K-only re-analysis of the signals that survived earlier (length, remaining length, overshoot, answer size, correctness) under the corrected protocol

Corrected protocol = cv_eval.py: grouped K-fold choice of (layer, lambda) over the training pool, bootstrap over problems.  Verdicts compare with the best OBSERVABLE baseline.
"""
import glob
import json
import re
import traceback
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from . import core, cv_eval, steer, steps, table, unc_study
from .bench import read_problems
from .probes import prompt_layers
from .trace import CK_NAMES, prefix_feats

MIN_STD = 0.1          # steps: a label that hardly varies cannot be probed
NOISE_MAX = 0.15       # steps: regenerating an UNEDITED prefix may not flip more than this share of answers


# ----------------------------------------------------------------------------- shared
def rows_df(rows: List[dict]) -> pd.DataFrame:
    return pd.DataFrame([{k: v for k, v in r.items() if k != "curve"} for r in rows])


def generic_status(rdf: pd.DataFrame, prefix: str, target: str, degenerate: str = "") -> dict:
    """SUPPORTED = beats the best observable baseline (CI lower bound > 0, R2 > 0.05) in iid AND shift; PARTIAL = iid only; NEGATIVE = the CI upper bound of the gain is < 0.05 in every regime;
    INCONCLUSIVE otherwise (or when the label is degenerate)."""
    s = rdf[(rdf.prefix == prefix) & (rdf.target == target)] if len(rdf) else rdf
    if degenerate:
        return {"status": "INCONCLUSIVE", "why": degenerate}
    iid, sh = (s[s.regime == r] for r in ("iid", "shift")) if len(s) else (s, s)
    if iid.empty:
        return {"status": "n/a", "why": "no iid result"}
    i = iid.iloc[0]
    if not all(np.isfinite(x) and abs(x) < 5 for x in (i.probe_test_r2, i.obs_r2)):
        return {"status": "INCONCLUSIVE", "why": f"degenerate label in the test split (R2 probe {i.probe_test_r2:.3g}, baseline {i.obs_r2:.3g}): essentially constant"}
    base = {"iid_probe_r2": i.probe_test_r2, "iid_obs_r2": i.obs_r2, "iid_delta": i.delta_vs_obs, "iid_lo": i.obs_lo, "iid_hi": i.obs_hi, "probe_layer": i.probe_layer}
    if not sh.empty:
        base.update(shift_probe_r2=sh.iloc[0].probe_test_r2, shift_obs_r2=sh.iloc[0].obs_r2, shift_delta=sh.iloc[0].delta_vs_obs, shift_lo=sh.iloc[0].obs_lo, shift_hi=sh.iloc[0].obs_hi)
    shp = (not sh.empty) and sh.iloc[0].verdict_obs == "PASS"
    if i.verdict_obs == "PASS" and shp:
        return {"status": "SUPPORTED", "why": "beats the best observable baseline in iid and under easy->hard shift", **base}
    if i.verdict_obs == "PASS":
        return {"status": "PARTIAL", "why": "beats the best observable baseline in iid only" + ("" if not sh.empty else " (no shift result)"), **base}
    his = [i.obs_hi] + ([sh.iloc[0].obs_hi] if not sh.empty else [])
    if max(his) < 0.05:
        return {"status": "NEGATIVE", "why": f"gain over the best observable baseline ruled out above 0.05 R2 (CI upper bounds {', '.join(f'{h:.3f}' for h in his)})", **base}
    if (not sh.empty) and shp:
        return {"status": "PARTIAL", "why": "beats the best observable baseline under shift only", **base}
    return {"status": "INCONCLUSIVE", "why": f"no clear gain; CI upper bound on the gain {max(his):.3f} (too wide to rule out 0.05)", **base}


def tab(rdf: pd.DataFrame, cols=None) -> str:
    if not len(rdf):
        return "(none)\n"
    cols = cols or ["prefix", "target", "regime", "n_test", "n_test_groups", "probe_layer", "probe_cv_r2", "probe_test_r2", "best_observable", "obs_r2", "delta_vs_obs", "obs_lo", "obs_hi", "verdict_obs", "best_baseline", "oracle_r2", "verdict_oracle"]
    return core.md(rdf[[c for c in cols if c in rdf.columns]].sort_values([c for c in ("prefix", "target", "regime") if c in rdf.columns]))


def numbers(q: str) -> List[float]:
    return [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*\.?\d*", q) if x.strip(",")]


def arith_feats(df: pd.DataFrame) -> np.ndarray:
    """OBSERVABLE answer-size heuristics from the question's numbers (sum, product, largest, two largest ...): the baseline the first suite lacked for 'answer size'."""
    out = []
    for q in df._q:
        v = sorted([abs(x) for x in numbers(q)] or [0.0], reverse=True)
        lg = lambda x: np.log10(max(x, 0.0) + 1.0)                    # noqa: E731
        out.append([lg(sum(v)), lg(float(np.prod(v[:2]))), lg(float(np.prod(v[:3]))), lg(float(np.prod(v))) if len(v) < 8 else lg(1e8), lg(v[0]), lg(v[-1]), lg(v[0] - v[-1]), len(v)])
    return np.array(out, dtype=np.float32)


# ----------------------------------------------------------------------------- thread 1: causal step importance
def load_steps(steps_dir: Path):
    rows, Hs, layers = [], [], None
    for f in sorted(glob.glob(str(Path(steps_dir) / "s*.jsonl"))):
        lines = [json.loads(l) for l in open(f) if l.strip()]
        if not lines:
            continue
        z = np.load(Path(f).with_suffix(".npz"))
        rows += lines
        Hs.append(z["H"])
        layers = z["layers"]
    return rows, (np.concatenate(Hs) if Hs else None), layers


def step_text_feats(line: str, value: int, q: str) -> list:
    nums = re.findall(r"\d+", line)
    return [np.log1p(len(line)), np.log1p(len(line.split())), len(nums), float("=" in line), line.count("+"), line.count("-"), line.count("\\times") + line.count("*"), line.count("\\div") + line.count("/"),
            np.log1p(value), float(re.search(rf"(?<!\d){value}(?!\d)", q) is not None), float("\\(" in line or "$" in line), sum(c.isdigit() for c in line) / max(len(line), 1)]


def analyze_steps(run_dir: Path, steps_dir: Path, log=print) -> dict:
    rows, H, layers = load_steps(steps_dir)
    if H is None or len(rows) < 200:
        return {"status": "n/a", "why": f"only {len(rows)} step rows - thread did not produce enough data", "md": "(not enough data)\n", "statuses": []}
    probs = {p["id"]: p for p in read_problems(run_dir, ["gsm8k"])}
    G = table.load_gen(run_dir, ["gsm8k"])
    df = pd.DataFrame(rows)
    df["flip_num"] = (df.flip_n1 + df.flip_n2) / 2
    df["split"] = [probs[i]["split"] for i in df.id]
    df["d1"] = [probs[i]["d1"] for i in df.id]
    q = [probs[i]["question"] for i in df.id]
    text = [G[i]["plain_text"] for i in df.id]
    line = [t[s:e] for t, s, e in zip(text, df.start, df.end)]
    pos = np.stack([df.k, np.log1p(df.k), df.t_end / 100.0, np.log1p(df.t_end)], 1).astype(np.float32)
    stp = np.array([step_text_feats(l, v, qq) for l, v, qq in zip(line, df.value, q)], dtype=np.float32)
    qf = np.array([table.question_features(qq) for qq in q], dtype=np.float32)
    pf = np.array([prefix_feats(t[:e]) for t, e in zip(text, df.end)], dtype=np.float32)
    txt = np.concatenate([pos, stp, qf, pf], 1)
    orc = np.nan_to_num(np.stack([df.reuse, np.log1p(df.reuse), df.n_steps, df.k / df.n_steps, np.log1p(df.n_tok - df.t_end)], 1).astype(np.float32))
    ly = list(layers)
    bl = {"Bpos": pos, "Bstep": np.concatenate([pos, stp], 1), "Btext": txt, "Bfuture(oracle)": np.concatenate([txt, orc], 1)}
    for l in (4, 8):
        if l in ly:
            bl[f"H{l}(early-layer)"] = H[:, ly.index(l), :].astype(np.float32)
    targets = ["load", "flip_num", "flip_del"]
    Y = df[targets].values.astype(np.float64)
    noise = float(df.flip_null.dropna().mean()) if "flip_null" in df else float("nan")
    log(f"steps: {len(df)} steps from {df.id.nunique()} problems; noise floor (null regeneration flips) {noise:.3f}; label means {dict(zip(targets, Y.mean(0).round(3)))}")
    res = cv_eval.run_block(df.split.values, df.d1.values, df.id.values, lambda li: H[:, li, :].astype(np.float32), list(range(len(ly))), bl, Y, targets, prefix="S", log=log)
    for r in res:
        r["probe_layer"] = ly[r["probe_layer"]]
    rdf = rows_df(res)
    sts = []
    for t in targets:
        deg = ""
        if Y[:, targets.index(t)].std() < MIN_STD:
            deg = f"label std {Y[:, targets.index(t)].std():.3f} < {MIN_STD}: nearly constant (base rate {Y[:, targets.index(t)].mean():.3f}), nothing to probe"
        elif noise > NOISE_MAX:
            deg = f"noise floor {noise:.3f} > {NOISE_MAX}: regenerating an unedited prefix already flips too many answers"
        sts.append({"thread": "steps", "label": t, **generic_status(rdf, "S", t, deg)})
    by_k = df.groupby("k").agg(n=("id", "size"), load=("load", "mean"), flip_num=("flip_num", "mean"), flip_del=("flip_del", "mean"), reuse=("reuse", "mean")).reset_index()
    mixed = float(((df.load > 0) & (df.load < 1)).mean())
    md = [f"{len(df)} numeric steps from {df.id.nunique()} problems (<= {df.groupby('id').size().max()} per problem). Folds and bootstraps are over problems.", "",
          f"- noise floor: regenerating an UNEDITED prefix changes the final answer in {noise:.3f} of {int(df.flip_null.notna().sum())} cases (limit {NOISE_MAX})",
          f"- label means: " + ", ".join(f"{t} {m:.3f}" for t, m in zip(targets, Y.mean(0))) + f"; share of steps that are neither always nor never load-bearing across the 3 edits: {mixed:.3f}",
          f"- share of steps whose result number is reused later in the chain (oracle feature): {(df.reuse > 0).mean():.3f}", "", "By step index:", "", core.md(by_k), "", tab(rdf)]
    return {"status": "ok", "md": "\n".join(md) + "\n", "statuses": sts, "results": rdf}


# ----------------------------------------------------------------------------- thread 2: uncertainty on unseen problems
def analyze_unc(run_dir: Path, unc_dir: Path, log=print) -> dict:
    rows = unc_study.load_unc(unc_dir)
    if len(rows) < 300:
        return {"status": "n/a", "why": f"only {len(rows)} problems with rollouts", "md": "(not enough data)\n", "statuses": []}
    df, sub = table.build(run_dir, ["gsm8k"], need_states=True, names=("plain",), log=log)
    d, s = core.take(df, sub, df.id.isin(list(rows)).values)
    rows = {i: rows[i] for i in d.id if i in rows}
    lab = unc_study.label_table(rows, dict(zip(d.id, d.gold)))
    d = d.merge(lab, on="id", how="left")
    rel = unc_study.reliability(lab, d.set_index("id").loc[lab.id, "ds"].reset_index(drop=True))
    bl = unc_study.ladder_unc(d, s)
    bl["Barith(observable)"] = np.concatenate([arith_feats(d), table.ds_onehot(d)], 1)
    targets = ["pass_rate", "ans_ent", "maj_frac", "correct", "ans_logmag"]
    Y = d[targets].values.astype(np.float64)
    layers = prompt_layers(s["plain"].shape[1])
    res = cv_eval.run_block(d.split.values, d.d1.values, d.id.values, lambda l: s["plain"][:, l, :].astype(np.float32), layers, bl, Y, targets, prefix="U[gsm8k]", log=log)
    rdf = rows_df(res)
    sp = rdf.assign(prefix="U[gsm8k]")
    ctrl_r2 = float(rdf[(rdf.target == "ans_logmag") & (rdf.regime == "iid")].probe_test_r2.iloc[0]) if len(rdf) else float("nan")
    lc = core.spearman(d.pass_rate.values, d.correct.values)
    ok = bool(ctrl_r2 >= unc_study.CTRL["ans_logmag"] and lc >= unc_study.CTRL["label_corr"])
    sts = [{"thread": "unc", "label": t, **unc_study.status(sp, rel, "U[gsm8k]", t, controls_ok=ok)} for t in ("ans_ent", "pass_rate", "maj_frac")]
    mixed = float((d.maj_frac < 1).mean())
    md = [f"{len(d)} UNSEEN GSM8K problems (the 400 used in the POC are excluded), {len(rows[next(iter(rows))]['roll_ans'])} rollouts each. Mixed problems (rollouts disagree): {mixed:.3f}; mean pass rate {d.pass_rate.mean():.3f}.", "",
          f"- controls: answer-size probe iid R2 {ctrl_r2:.3f} (needs >= {unc_study.CTRL['ans_logmag']}), Spearman(pass_rate, greedy correct) {lc:.3f} (needs >= {unc_study.CTRL['label_corr']}); controls OK: {ok}", "",
          "Label reliability (ceiling for probe R2):", "", core.md(rel[rel.scope.isin(["gsm8k"])]), "", tab(rdf)]
    return {"status": "ok", "md": "\n".join(md) + "\n", "statuses": sts, "results": rdf}


# ----------------------------------------------------------------------------- thread 3: steering
def analyze_steer(steer_dir: Path, log=print) -> dict:
    r = steer.analyze(steer_dir)
    prep = json.loads((Path(steer_dir) / "prep.json").read_text()) if (Path(steer_dir) / "prep.json").exists() else {}
    if r["status"] == "n/a":
        return {"status": "n/a", "why": r["why"], "md": f"(no result: {r['why']})\n", "statuses": []}
    sl = pd.DataFrame([{"family": f, "slope_log_tokens_per_sd": v[0], "lo95": v[1], "hi95": v[2]} for f, v in sorted(r["slopes"].items())])
    md = [f"{r['n']} test problems finished in every condition.", "", f"probe direction: layer {prep.get('layer')}, CV R2 {prep.get('cv_r2', float('nan')):.3f}, test R2 {prep.get('test_r2', float('nan')):.3f}; cos(probe, mean-diff) {prep.get('cos_probe_diff', float('nan')):.3f}; largest |cos| with a random direction {prep.get('cos_probe_rand_max', float('nan')):.3f}", "",
          f"**{r['status']}**: {r['why']}", "", "Slope of per-problem log tokens on alpha (alpha in standard deviations of the probe projection; bootstrap over problems):", "", core.md(sl), "", "Per condition:", "", core.md(r["table"])]
    return {"status": "ok", "md": "\n".join(md) + "\n", "statuses": [{"thread": "steer", "label": "length direction", "status": r["status"], "why": r["why"]}]}


# ----------------------------------------------------------------------------- thread 4: the surviving signals, GSM8K only, corrected protocol
def analyze_suite(run_dir: Path, log=print) -> dict:
    out_md, sts, allr = [], [], []
    df, sub = table.build(run_dir, ["gsm8k"], need_states=True, names=("plain", "con"), log=log)
    L = sub["plain"].shape[1]
    layers = prompt_layers(L)
    bl = table.ladder_plain(df, sub)
    bl["Barith(observable)"] = np.concatenate([arith_feats(df), table.ds_onehot(df)], 1)
    tp = ["logT", "len_sd", "correct", "ans_logmag", "runaway"]
    res = cv_eval.run_block(df.split.values, df.d1.values, df.id.values, lambda l: sub["plain"][:, l, :].astype(np.float32), layers, bl, df[tp].values.astype(np.float64), tp, prefix="P[gsm8k]", log=log)
    rdf = rows_df(res)
    allr.append(rdf)
    sts += [{"thread": "suite", "label": f"P:{t}", **generic_status(rdf, "P[gsm8k]", t, "base rate under 5%: too few positives" if t == "runaway" and df.runaway.mean() < 0.05 else "")} for t in tp]
    out_md += ["### Plain prompt-end state", "", tab(rdf), ""]
    tc = ["c_logT", "over", "comply"]
    resc = cv_eval.run_block(df.split.values, df.d1.values, df.id.values, lambda l: sub["con"][:, l, :].astype(np.float32), layers, table.ladder_con(df, sub), df[tc].values.astype(np.float64), tc, prefix="C[gsm8k]", log=log)
    rc = rows_df(resc)
    allr.append(rc)
    sts += [{"thread": "suite", "label": f"C:{t}", **generic_status(rc, "C[gsm8k]", t, "base rate over 90%: nothing to detect" if t == "comply" and df.comply.mean() > 0.9 else "")} for t in tc]
    out_md += ["### Constrained prompt-end state", "", tab(rc), ""]
    try:
        _, tr, frames, tks = table.build_trace(run_dir, ["gsm8k"], log=log)
        for name in ("o24", "o48", "o96", "o192"):
            k = CK_NAMES.index(name)
            f, tk = frames[k], tks[k]
            if len(f) < 200:
                continue
            lt = list(range(len(tk["layers"])))
            rr = cv_eval.run_block(f.split.values, f.d1.values, f.id.values, lambda li, tk=tk: tk["H"][:, li, :].astype(np.float32), lt, table.ladder_trace(f, tk), f[["log_rem"]].values.astype(np.float64), ["log_rem"], prefix=f"K[{name}]", log=log)
            for r in rr:
                r["probe_layer"] = int(tk["layers"][r["probe_layer"]])
            kd = rows_df(rr)
            allr.append(kd)
            sts.append({"thread": "suite", "label": f"K:log_rem@{name}", **generic_status(kd, f"K[{name}]", "log_rem")})
            out_md += [f"### Remaining length at {name} ({len(f)} problems)", "", tab(kd), ""]
    except RuntimeError as e:
        out_md.append(f"(no trace: {e})")
    return {"status": "ok", "md": "\n".join(out_md) + "\n", "statuses": sts, "results": pd.concat(allr) if allr else pd.DataFrame()}


# ----------------------------------------------------------------------------- driver + report
def where_to_pull(sts: pd.DataFrame) -> str:
    if not len(sts):
        return "No thread produced a result. See the sections below for the reason."
    sup = sts[sts.status == "SUPPORTED"]
    par = sts[sts.status == "PARTIAL"]
    lines = []
    if len(sup):
        lines.append("SUPPORTED (beats observable baselines under the corrected protocol and holds under shift / passes its own controls): " + "; ".join(f"{r.thread}:{r.label}" for r in sup.itertuples()) + ".")
    if len(par):
        lines.append("PARTIAL (one regime only; needs confirmation): " + "; ".join(f"{r.thread}:{r.label}" for r in par.itertuples()) + ".")
    neg = sts[sts.status == "NEGATIVE"]
    if len(neg):
        lines.append("NEGATIVE (gain ruled out): " + "; ".join(f"{r.thread}:{r.label}" for r in neg.itertuples()) + ".")
    inc = sts[sts.status.isin(["INCONCLUSIVE", "n/a"])]
    if len(inc):
        lines.append("INCONCLUSIVE / no data: " + "; ".join(f"{r.thread}:{r.label}" for r in inc.itertuples()) + ".")
    if not len(sup) and not len(par):
        lines.append("Nothing is supported beyond the already-known remaining-length / overshoot results. The honest reading is a negative sweep; the useful artefact is the ledger of what was ruled out and why.")
    return "\n\n".join(lines)


def run_all(run_dir, mega_dir=None, parts=("steps", "unc", "steer", "suite"), log=print) -> Path:
    run_dir = Path(run_dir)
    mega_dir = Path(mega_dir) if mega_dir else run_dir / "mega"
    out = mega_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    fns = {"steps": lambda: analyze_steps(run_dir, mega_dir / "steps", log), "unc": lambda: analyze_unc(run_dir, mega_dir / "unc", log),
           "steer": lambda: analyze_steer(mega_dir / "steer", log), "suite": lambda: analyze_suite(run_dir, log)}
    titles = {"steps": "Thread 1 - causal step importance", "unc": "Thread 2 - answer-distribution uncertainty (unseen problems)", "steer": "Thread 3 - is the length direction causal?", "suite": "Thread 4 - surviving signals, GSM8K only"}
    secs, sts = [], []
    for p in parts:
        log(f"=== {p} ===")
        try:
            r = fns[p]()
        except Exception:                                                  # noqa: BLE001 - a broken thread must not take the report down
            r = {"status": "error", "md": "```\n" + traceback.format_exc() + "```\n", "statuses": [{"thread": p, "label": "(thread crashed)", "status": "INCONCLUSIVE", "why": "analysis raised an exception, see the section"}]}
            log(traceback.format_exc())
        if r.get("results") is not None and len(r["results"]):
            r["results"].to_csv(out / f"{p}_results.csv", index=False)
        sts += r.get("statuses", [])
        secs.append(f"## {titles[p]}\n\n" + (f"_{r['why']}_\n\n" if r.get("why") and r["status"] == "n/a" else "") + r["md"])
    sdf = pd.DataFrame(sts)
    keep = [c for c in ("thread", "label", "status", "why", "iid_probe_r2", "iid_obs_r2", "iid_delta", "iid_lo", "iid_hi", "shift_probe_r2", "shift_obs_r2", "shift_delta", "shift_lo", "shift_hi") if c in sdf.columns]
    if len(sdf):
        sdf[keep].to_csv(out / "mega_status.csv", index=False)
    rep = ["# Mega run - GSM8K (Qwen2.5-7B-Instruct)", "",
           "Four independent threads, decision rules fixed in docs/MEGA.md before the run. Corrected protocol: cross-validated layer/lambda, bootstrap over problems, verdict against the best OBSERVABLE baseline.", "",
           "## Where to pull", "", where_to_pull(sdf), "", "## Status table", "", core.md(sdf[keep]) if len(sdf) else "(none)", ""] + secs
    (out / "MEGA_REPORT.md").write_text("\n".join(rep))
    log(f"wrote {out / 'MEGA_REPORT.md'}")
    return out / "MEGA_REPORT.md"
