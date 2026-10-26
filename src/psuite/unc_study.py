"""UNCERTAINTY POC - analysis + report.  Linear (ridge) probes, regression labels, the same baseline ladder / split regimes / paired bootstrap as the main study.

LABELS (exact functions of sampled answers; answers are canonical integers so 'semantic equivalence' is string equality)
  prompt-end (state = last prompt token, label from `n_roll` rollouts of the plain prompt)
     pass_rate  fraction of rollouts equal to gold            ans_ent  Shannon entropy (nats) of the empirical answer distribution ('no answer' = one category)
     maj_frac   share of the most common answer (gold-free)   controls: correct (greedy), ans_logmag (gold answer size)
  checkpoint (state = last token of the model's own greedy prefix of 48 / 96 tokens; label from `n_cont` continuations of that prefix)
     c_pass, c_ent, c_maj   as above, conditional on the prefix      control: log_rem (remaining tokens of the greedy solution)
RELIABILITY: rollout labels are noisy estimates of the true answer distribution.  Split-half (even vs odd rollouts) Pearson r -> Spearman-Brown reliability of the full label;
this is the CEILING for any probe's R^2 against the observed label.  Probes are read against it.
BASELINES: observable (dataset id, question features, bag of words, early layers H4/H8; at checkpoints also the visible prefix text AND the model's own prefix token entropy) and
oracle (greedy-run summaries incl. token entropy and correctness; difficulty; at checkpoints the prompt-end rollout statistics).  The verdict is against the best OBSERVABLE baseline.
"""
import glob
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from . import core, probes, table, trace
from .unc import CPS, POC_DATASETS

OFFSETS = trace.OFFS            # slot k of the trace checkpoints holds offset OFFSETS[k] (tests override both CPS and OFFSETS)

TU = ["pass_rate", "ans_ent", "maj_frac", "correct", "ans_logmag"]
TUC = ["c_pass", "c_ent", "c_maj", "log_rem"]
PRIMARY = ["ans_ent", "pass_rate"]
MIN_REL = 0.6                  # a label with lower reliability cannot support a negative claim
MIN_ABS = 0.05                 # same R^2 floor as the main study's PASS
CTRL = {"ans_logmag": 0.15, "log_rem": 0.30, "label_corr": 0.40}


# ----------------------------------------------------------------------------- labels
def load_unc(unc_dir: Path) -> dict:
    out = {}
    for f in sorted(glob.glob(str(Path(unc_dir) / "u*.jsonl"))):
        for l in open(f):
            r = json.loads(l)
            out[r["id"]] = r
    return out


def answer_stats(ans, gold) -> dict:
    n = len(ans)
    if n == 0:
        return {"pass": np.nan, "ent": np.nan, "maj": np.nan, "none": np.nan}
    _, cnt = np.unique([a if a is not None else "<none>" for a in ans], return_counts=True)
    p = cnt / n
    return {"pass": float(np.mean([a == gold for a in ans])), "ent": float(-(p * np.log(p)).sum()), "maj": float(p.max()), "none": float(np.mean([a is None for a in ans]))}


def label_table(rows: dict, gold: dict) -> pd.DataFrame:
    out = []
    for i, r in rows.items():
        g, a = gold[i], r["roll_ans"]
        s, sa, sb = answer_stats(a, g), answer_stats(a[0::2], g), answer_stats(a[1::2], g)
        row = {"id": i, "none_rate": s["none"]}
        for nm, k in (("pass_rate", "pass"), ("ans_ent", "ent"), ("maj_frac", "maj")):
            row[nm], row[nm + "_a"], row[nm + "_b"] = s[k], sa[k], sb[k]
        for cp in CPS:
            c = r["cont"].get(str(cp))
            if c is None:
                continue
            s, sa, sb = answer_stats(c["ans"], g), answer_stats(c["ans"][0::2], g), answer_stats(c["ans"][1::2], g)
            for nm, k in ((f"c{cp}_pass", "pass"), (f"c{cp}_ent", "ent"), (f"c{cp}_maj", "maj")):
                row[nm], row[nm + "_a"], row[nm + "_b"] = s[k], sa[k], sb[k]
        for k in ("ent_all", "nll_all") + tuple(f"{p}{t}" for p in ("ent", "nll") for t in CPS):
            row[k] = np.nan if r.get(k) is None else r[k]
        out.append(row)
    return pd.DataFrame(out)


def spearman_brown(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 10 or a[m].std() == 0 or b[m].std() == 0:
        return float("nan"), float("nan")
    r = float(np.corrcoef(a[m], b[m])[0, 1])
    return r, float(max(0.0, 2 * r / (1 + r))) if r > -1 else 0.0


def reliability(lab: pd.DataFrame, ds: pd.Series, cps=None) -> pd.DataFrame:
    cps = CPS if cps is None else cps
    rows = []
    names = [("pass_rate", "prompt"), ("ans_ent", "prompt"), ("maj_frac", "prompt")] + [(f"c{cp}_{k}", f"o{cp}") for cp in cps for k in ("pass", "ent", "maj")]
    for nm, kind in names:
        if nm + "_a" not in lab:
            continue
        for scope in [*sorted(ds.unique()), "pooled"]:
            m = (ds == scope).values if scope != "pooled" else np.ones(len(ds), bool)
            r, rel = spearman_brown(lab[nm + "_a"][m], lab[nm + "_b"][m])
            rows.append({"label": nm, "kind": kind, "scope": scope, "n": int(m.sum()), "split_half_r": r, "reliability": rel})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- baselines
def ladder_unc(df, sub):
    out = table.ladder_plain(df, sub)
    oh = table.ds_onehot(df)
    qf = np.array([table.question_features(q) for q in df._q], dtype=np.float32)
    g = np.nan_to_num(np.stack([df.ent_all, df.ent48.fillna(df.ent_all), df.ent96.fillna(df.ent_all), df.nll_all, df.logT_all, df.runaway, df.noans, df.correct], 1).astype(np.float32))
    out["Bgreedy(oracle)"] = np.concatenate([oh, qf, g, g ** 2], 1)          # needs a full greedy run: the 'free' uncertainty signals (token entropy / NLL / length / correctness)
    return out


def ladder_ck(f, tk):
    out = table.ladder_trace(f, tk)
    ent = np.nan_to_num(np.stack([f.ent_p, f.nll_p], 1).astype(np.float32))
    out["Btext+ent"] = np.concatenate([out["Btext"], ent, ent ** 2], 1)      # observable: the visible prefix AND how uncertain the model was while writing it
    p0 = np.nan_to_num(np.stack([f.pass_rate, f.ans_ent, f.maj_frac], 1).astype(np.float32))
    out["Bprompt-rollouts(oracle)"] = np.concatenate([out["Btext+ent"], p0, p0 ** 2], 1)
    return out


# ----------------------------------------------------------------------------- compute
def compute(run_dir, unc_dir=None, log=print):
    run_dir = Path(run_dir)
    unc_dir = Path(unc_dir) if unc_dir else run_dir / "unc"
    rows = load_unc(unc_dir)
    if not rows:
        raise RuntimeError(f"no rollout files in {unc_dir}")
    dss = [d for d in POC_DATASETS if any(r["ds"] == d for r in rows.values())]
    df, sub = table.build(run_dir, dss, need_states=True, names=("plain",), log=log)
    d, s = core.take(df, sub, df.id.isin(list(rows)).values)
    lab = label_table(rows, dict(zip(d.id, d.gold)))
    d = d.merge(lab, on="id", how="left")
    L = s["plain"].shape[1]
    layers = probes.prompt_layers(L)
    log(f"unc prompt-end: n={len(d)} datasets={dss} layers={layers}")
    res_p, _ = probes._block(d, s, "plain", TU, dss, layers, ladder_unc, "U", log)
    rel = reliability(lab, d.set_index("id").loc[lab.id, "ds"].reset_index(drop=True))
    res_c = []
    try:
        _, tr, frames, tks = table.build_trace(run_dir, dss, log=log)
    except RuntimeError:
        frames = None
        log("no trace found: checkpoint part skipped")
    if frames is not None:
        for cp in CPS:
            k = list(OFFSETS).index(cp)
            f, tk = frames[k], tks[k]
            if f"c{cp}_pass" not in lab:
                continue
            have = lab.set_index("id")[f"c{cp}_pass"].notna()
            m = f.id.map(lambda i: bool(have.get(i, False))).values
            if m.sum() < 3 * core.MIN_ROWS:
                log(f"  checkpoint o{cp}: only {int(m.sum())} rows - skipped")
                continue
            f2 = f[m].reset_index(drop=True)
            tk2 = {key: (v[m] if key != "layers" else v) for key, v in tk.items()}
            cols = {f"c{cp}_{a}": f"c_{a}" for a in ("pass", "ent", "maj")}
            cols.update({f"c{cp}_{a}_{h}": f"c_{a}_{h}" for a in ("pass", "ent", "maj") for h in ("a", "b")})
            keep = ["id", "pass_rate", "ans_ent", "maj_frac", f"ent{cp}", f"nll{cp}", *cols]
            f2 = f2.merge(lab[keep].rename(columns={**cols, f"ent{cp}": "ent_p", f"nll{cp}": "nll_p"}), on="id", how="left")
            lt = list(tk2["layers"])
            res, _ = core.run_targets(f2, {"k": tk2["H"]}, "k", "k", TUC, TUC, probes.pooled_regs(dss), ladder_ck(f2, tk2), list(range(len(lt))), f"UC[o{cp}]")
            for r in res:
                r["probe_layer"] = lt[r["probe_layer"]]
            res_c += res
            log(f"  checkpoint o{cp} done ({len(f2)} problems)")
    out = unc_dir / "analysis"
    out.mkdir(exist_ok=True)
    sp, sc = core.strip(res_p), core.strip(res_c)
    sp.to_csv(out / "unc_prompt_results.csv", index=False)
    sc.to_csv(out / "unc_checkpoint_results.csv", index=False)
    rel.to_csv(out / "unc_reliability.csv", index=False)
    lab.merge(d[["id", "ds", "correct"]], on="id").to_csv(out / "unc_labels.csv", index=False)
    with open(out / "unc.pkl", "wb") as fh:
        pickle.dump({"sp": sp, "sc": sc, "rel": rel, "lab": lab.merge(d[["id", "ds", "correct"]], on="id"), "dss": dss, "n_roll": len(next(iter(rows.values()))["roll_ans"])}, fh)
    return sp, sc, rel


# ----------------------------------------------------------------------------- decision rule (written BEFORE seeing results; see docs/UNCERTAINTY_POC.md)
def _rel(rel, label, scope):
    if label in ("correct", "ans_logmag", "log_rem"):
        return 1.0
    m = rel[(rel.label == label) & (rel.scope == scope)]
    return float(m.reliability.iloc[0]) if len(m) else float("nan")


def scope_of(prefix: str) -> str:
    return prefix.split("[", 1)[1].rstrip("]")


def ceil_for(rel, prefix: str, target: str) -> float:
    """reliability of the label behind (prefix, target): prompt-end labels per scope; checkpoint labels (c_pass of UC[o48]) from the pooled c48_pass row."""
    if target in ("correct", "ans_logmag", "log_rem"):
        return 1.0
    if prefix.startswith("UC"):
        return _rel(rel, f"c{scope_of(prefix)[1:]}_{target[2:]}", "pooled")
    return _rel(rel, target, scope_of(prefix))


def status(sdf: pd.DataFrame, rel: pd.DataFrame, prefix: str, target: str, controls_ok=True) -> dict:
    """SUPPORTED / PARTIAL / NEGATIVE / INCONCLUSIVE for one label in one scope."""
    s = sdf[(sdf.prefix == prefix) & (sdf.target == target)]
    iid = s[s.regime == "iid"]
    if iid.empty:
        return {"status": "n/a", "why": "no iid result"}
    iid = iid.iloc[0]
    oth = s[s.regime != "iid"]
    npass = int((oth.verdict_obs == "PASS").sum())
    need = int(np.ceil(2 * len(oth) / 3)) if len(oth) else 0
    rl = ceil_for(rel, prefix, target)
    base = {"iid_probe_r2": iid.probe_test_r2, "iid_obs_r2": iid.obs_r2, "iid_delta": iid.delta_vs_obs, "iid_lo": iid.obs_lo, "iid_hi": iid.obs_hi, "other_pass": f"{npass}/{len(oth)}", "reliability": rl}
    if iid.verdict_obs == "PASS":
        if len(oth) and npass >= need:
            return {"status": "SUPPORTED", "why": f"beats best observable in iid and {npass}/{len(oth)} shift/leave-dataset-out tests", **base}
        return {"status": "PARTIAL", "why": f"beats best observable in iid but only {npass}/{len(oth)} shift/leave-dataset-out tests", **base}
    if not controls_ok:
        return {"status": "INCONCLUSIVE", "why": "positive controls failed - pipeline not validated", **base}
    if np.isnan(rl) or rl < MIN_REL:
        return {"status": "INCONCLUSIVE", "why": f"label reliability {rl:.2f} < {MIN_REL}: a null cannot be told from label noise (raise rollouts)", **base}
    if iid.obs_hi < MIN_ABS:
        return {"status": "NEGATIVE", "why": f"reliable label (rel {rl:.2f}); gain over best observable ruled out above {MIN_ABS} R2 (CI hi {iid.obs_hi:.3f})", **base}
    return {"status": "INCONCLUSIVE", "why": f"reliable label but CI on the gain over observable is too wide (hi {iid.obs_hi:.3f}): underpowered", **base}


def controls(sp, sc, lab) -> dict:
    out = {}
    r = sp[(sp.prefix == "U[pooled]") & (sp.target == "ans_logmag") & (sp.regime == "iid")]
    out["ans_logmag_r2"] = float(r.probe_test_r2.iloc[0]) if len(r) else float("nan")
    out["label_corr"] = core.spearman(lab.pass_rate.values, lab.correct.values)
    r = sc[(sc.prefix == f"UC[o{CPS[0]}]") & (sc.target == "log_rem") & (sc.regime == "iid")] if len(sc) else sc
    out["log_rem_o48_r2"] = float(r.probe_test_r2.iloc[0]) if len(r) else float("nan")
    out["ok_prompt"] = bool(out["ans_logmag_r2"] >= CTRL["ans_logmag"] and out["label_corr"] >= CTRL["label_corr"])
    out["ok_ck"] = bool(out["ok_prompt"] and (np.isnan(out["log_rem_o48_r2"]) or out["log_rem_o48_r2"] >= CTRL["log_rem"]))
    return out


def decide(stat: pd.DataFrame, ctrl: dict) -> str:
    pr = stat[stat.kind == "prompt-end"]
    prim = pr[(pr.scope == "pooled") & pr.label.isin(PRIMARY)]
    ck = stat[stat.kind == "checkpoint"]
    if not ctrl["ok_prompt"]:
        return "**CONTROLS FAILED - no conclusion either way.** The positive controls (gold-answer size probe, label sanity) did not reproduce, so the pipeline or the subset is suspect. Debug before reading any entropy result."
    if (prim.status == "SUPPORTED").any():
        w = ", ".join(prim[prim.status == "SUPPORTED"].label)
        return f"**GO (prompt-end).** A linear probe predicts {w} beyond every observable baseline and the effect holds under shift / leave-dataset-out. Scale up (more problems, 32 rollouts, a second model) and add the group's semantic-entropy probe as a baseline."
    if (ck.status == "SUPPORTED").any():
        w = ", ".join(sorted(set(ck[ck.status == "SUPPORTED"].label + "@" + ck[ck.status == "SUPPORTED"].scope)))
        return f"**GO (trace-time only).** Prompt-end shows nothing supported, but the checkpoint label(s) {w} are linearly readable beyond observable baselines. The paper would be about uncertainty DURING reasoning; scale the checkpoint part."
    if (pd.concat([prim, ck]).status == "PARTIAL").any():
        return "**LEAD, not yet a result.** Something beats the observable baselines in-distribution but does not survive shift / leave-dataset-out. Worth one more round at larger scale (and a third dataset) before any claim."
    if len(prim) and (prim.status == "NEGATIVE").all():
        return "**STOP on prompt-end uncertainty.** Reliable labels, controls reproduce, and the gain over observable baselines is ruled out. That is a clean negative worth one line in the writeup (linear probes do not read answer-distribution entropy beyond cheap observables here). Look at the checkpoint table before dropping the direction."
    return "**INCONCLUSIVE.** Not enough label reliability or power to separate 'no signal' from 'noise'. The status table says which (reliability vs CI width); the fix is more rollouts and/or more problems, not a conclusion."


# ----------------------------------------------------------------------------- report
def _fmt(sdf, rel, cols):
    if not len(sdf):
        return "(none)\n"
    d = sdf.copy()
    d["ceiling"] = [ceil_for(rel, p, t) for p, t in zip(d.prefix, d.target)]
    d["r2/ceiling"] = d.probe_test_r2 / d.ceiling
    return core.md(d[cols])


def write(unc_dir, log=print):
    unc_dir = Path(unc_dir)
    with open(unc_dir / "analysis" / "unc.pkl", "rb") as fh:
        z = pickle.load(fh)
    sp, sc, rel, lab, dss = z["sp"], z["sc"], z["rel"], z["lab"], z["dss"]
    ctrl = controls(sp, sc, lab)
    srows = []
    for pref in [p for p in sp.prefix.unique() if p.startswith("U[")]:
        for t in PRIMARY + ["maj_frac"]:
            srows.append({"kind": "prompt-end", "scope": scope_of(pref), "label": t, **status(sp, rel, pref, t, controls_ok=ctrl["ok_prompt"])})
    for pref in sorted(sc.prefix.unique()) if len(sc) else []:
        for t in ("c_ent", "c_pass", "c_maj"):
            srows.append({"kind": "checkpoint", "scope": scope_of(pref), "label": t, **status(sc, rel, pref, t, controls_ok=ctrl["ok_ck"])})
    stat = pd.DataFrame(srows)
    stat.to_csv(unc_dir / "analysis" / "unc_status.csv", index=False)
    cols = ["prefix", "target", "regime", "n_test", "probe_layer", "probe_test_r2", "ceiling", "r2/ceiling", "best_observable", "obs_r2", "delta_vs_obs", "obs_lo", "obs_hi", "verdict_obs", "best_baseline", "verdict_oracle"]
    mixed = lab.assign(mixed=(lab.maj_frac < 1)).groupby("ds").agg(n=("id", "size"), pass_rate=("pass_rate", "mean"), ans_ent=("ans_ent", "mean"), mixed_frac=("mixed", "mean"), none_rate=("none_rate", "mean"))
    L = ["# Uncertainty probing - proof of concept", "",
         f"{len(lab)} problems ({', '.join(dss)}), {z['n_roll']} rollouts each at T=0.7; checkpoint continuations at o48 / o96. Linear (ridge) probes, regression labels, paired-bootstrap CIs over problems.", "",
         "## 0. DECISION", "", decide(stat, ctrl) if len(stat) else "no results", "",
         "Rule fixed before the run (docs/UNCERTAINTY_POC.md): SUPPORTED = probe beats the best OBSERVABLE baseline (CI lower bound > 0, R2 > 0.05) in the iid split AND in at least 2/3 of {shift, leave-dataset-out}; "
         "NEGATIVE only if the label is reliable (>= 0.6), the controls reproduce and the CI upper bound of the gain is < 0.05; otherwise INCONCLUSIVE. GO requires a SUPPORTED primary label (ans_ent or pass_rate) and passing controls.", "",
         "## 1. Positive controls", "",
         f"- gold-answer-size probe, pooled iid R2 = {ctrl['ans_logmag_r2']:.3f} (needs >= {CTRL['ans_logmag']})",
         f"- Spearman(pass_rate label, greedy correct) = {ctrl['label_corr']:.3f} (needs >= {CTRL['label_corr']})",
         f"- remaining-length probe at o48, pooled iid R2 = {ctrl['log_rem_o48_r2']:.3f} (needs >= {CTRL['log_rem']})",
         f"- controls OK: prompt-end {ctrl['ok_prompt']}, checkpoints {ctrl['ok_ck']}", "",
         "## 2. Label diagnostics", "", "`mixed_frac` = share of problems whose rollouts do not all agree - the only problems on which entropy / pass rate carry information. A low value means label saturation, not a probe failure.", "",
         core.md(mixed.reset_index()), "", "## 3. Label reliability (ceiling for probe R2)", "", "Split-half Pearson r between even / odd rollouts, Spearman-Brown corrected to the full label.", "", core.md(rel), "",
         "## 4. Status per label", "", core.md(stat[["kind", "scope", "label", "status", "iid_probe_r2", "iid_obs_r2", "iid_delta", "iid_lo", "iid_hi", "other_pass", "reliability", "why"]]) if len(stat) else "(none)", ""]
    for title, sdf in (("5. Prompt-end probes (state = last prompt token)", sp), ("6. Checkpoint probes (state after 48 / 96 tokens of the greedy solution)", sc)):
        L += [f"## {title}", ""]
        if not len(sdf):
            L += ["(not run)", ""]
            continue
        for pref in sdf.prefix.unique():
            L += [f"### {pref}", "", _fmt(sdf[sdf.prefix == pref].sort_values(["target", "regime"]), rel, cols), ""]
    L += ["## 7. Reading this honestly", "",
          "- Entropy / pass-rate labels come from a finite sample (see section 3); R2 close to the ceiling is as good as it can get, R2 near 0.05 against a ceiling of 0.9 is not.",
          "- `ans_ent` and `pass_rate` are tightly related when answers are mostly right or wrong; section 2 shows how many problems are actually mixed.",
          "- Observable baselines include the question text, dataset identity, early layers and (at checkpoints) the visible prefix and its token entropy. Oracle baselines (greedy-run summaries, prompt-end rollouts) are context, not the verdict.",
          "- Two datasets only: leave-dataset-out here is 'train on one, test on the other'. A SUPPORTED label is a lead, not a general claim.",
          "- Reasoning Theater (Boppana et al. 2026) already shows answer decodability along reasoning traces with attention probes on multiple choice; this POC asks a different question (linear probe, continuous answer-distribution labels, exact-answer tasks).", ""]
    (unc_dir / "analysis" / "UNC_REPORT.md").write_text("\n".join(L))
    log(f"wrote {unc_dir / 'analysis' / 'UNC_REPORT.md'}")
