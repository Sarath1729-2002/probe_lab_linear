"""The probing experiments (everything goes through core.run_targets: ridge probe per layer vs a ladder of baselines, paired bootstrap).

  P  plain prompt-end state       -> will it run away / how long / how variable / will it fail to answer / correct / answer size
  C  constrained prompt-end state -> will it comply / by how much will it overshoot / length under the constraint
  K  checkpoints along the solution -> is the answer already settled / ready / how many tokens remain   (per checkpoint, pooled over datasets)
Scopes: every dataset separately (iid + easy->hard shift) and pooled (iid + shift + leave-one-dataset-out)."""
from typing import Dict, List

import numpy as np
import pandas as pd

from . import core
from .table import ladder_con, ladder_plain, ladder_trace
from .trace import CK_NAMES

TP = ["runaway", "runaway_rate", "logT", "len_sd", "noans", "correct", "steps", "ans_logmag", "ans_ndig"]
TC = ["comply", "over", "c_logT", "c_runaway", "c_correct"]
TK = ["settled_p", "settled_lp", "ready_p", "ready_lp", "log_rem", "settled_hi"]
BINARY = {"runaway", "noans", "correct", "comply", "c_runaway", "c_correct", "settled_hi"}


def prompt_layers(L: int) -> List[int]:
    ls = list(range(2, L, 2))
    if L - 1 not in ls:
        ls.append(L - 1)
    return ls


def add_auroc(results, B=300):
    rng = np.random.RandomState(0)
    for r in results:
        if r["target"] not in BINARY:
            continue
        y = r["_y"]
        m = ~np.isnan(y)
        yb = y[m] > 0.5
        r["base_rate"] = float(yb.mean()) if len(yb) else float("nan")
        if len(yb) < 30 or yb.sum() < 5 or (~yb).sum() < 5:
            continue
        P = {k: np.asarray(r[v], dtype=float)[m] for k, v in (("probe", "_pred"), ("obs", "_pred_obs"), ("oracle", "_pred_oracle"))}
        for k, p in P.items():
            r[f"auroc_{k}"] = core.auroc(yb, p)
        d = []
        for _ in range(B):
            i = rng.randint(0, len(yb), len(yb))
            if yb[i].all() or (~yb[i]).all():
                continue
            d.append(core.auroc(yb[i], P["probe"][i]) - core.auroc(yb[i], P["obs"][i]))
        r["auroc_delta_obs"] = r["auroc_probe"] - r["auroc_obs"]
        r["auroc_delta_obs_lo"], r["auroc_delta_obs_hi"] = (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))) if d else (np.nan, np.nan)
    return results


def _scopes(df, datasets, min_rows=120):
    for ds in datasets:
        m = (df.ds == ds).values
        if m.sum() >= min_rows:
            yield ds, m


def pooled_regs(datasets):
    return ["iid", "shift"] + [f"lodo:{ds}" for ds in datasets]


def _block(df, sub, cond, targets, datasets, layers, ladder_fn, prefix, log, per_dataset=True):
    res_all, cur_all = [], []
    if per_dataset:
        for ds, m in _scopes(df, datasets):
            d, s = core.take(df, sub, m)
            res, cur = core.run_targets(d, s, cond, cond, targets, targets, ("iid", "shift"), ladder_fn(d, s), layers, f"{prefix}[{ds}]")
            res_all += res
            cur_all.append(cur.assign(scope=ds))
            log(f"  {prefix}[{ds}] done")
    res, cur = core.run_targets(df, sub, cond, cond, targets, targets, pooled_regs(datasets), ladder_fn(df, sub), layers, f"{prefix}[pooled]")
    res_all += res
    cur_all.append(cur.assign(scope="pooled"))
    log(f"  {prefix}[pooled] done")
    return add_auroc(res_all), pd.concat([c for c in cur_all if len(c)])


def block_P(df, sub, datasets, layers, log=print):
    return _block(df, sub, "plain", TP, datasets, layers, ladder_plain, "P", log)


def block_C(df, sub, datasets, layers, log=print):
    return _block(df, sub, "con", TC, datasets, layers, ladder_con, "C", log)


def block_K(frames, tks, layers_true, datasets, log=print):
    """one fit per checkpoint (rows = problems, so bootstrap CIs are over problems).  probe_layer is returned as the TRUE layer index."""
    layers_true = list(layers_true)
    idx = list(range(len(layers_true)))
    res_all, cur_all = [], []
    for k, name in enumerate(CK_NAMES):
        f = frames[k]
        if len(f) < 3 * core.MIN_ROWS:
            log(f"  K[{name}] skipped: only {len(f)} valid rows")
            continue
        sub = {"k": tks[k]["H"]}
        res, cur = core.run_targets(f, sub, "k", "k", TK, TK, pooled_regs(datasets), ladder_trace(f, tks[k]), idx, f"K[{name}]")
        for r in res:
            r["probe_layer"] = layers_true[r["probe_layer"]]
            r["ck"] = name
        cur["layer"] = [layers_true[i] for i in cur.layer]
        res_all += res
        cur_all.append(cur.assign(scope=name))
        log(f"  K[{name}] done")
    return add_auroc(res_all), pd.concat(cur_all)


def per_ctype(res, df) -> pd.DataFrame:
    """compliance AUROC of the probe vs the observable and oracle baselines WITHIN each constraint type (pooled iid test rows)."""
    r = next((x for x in res if x["prefix"] == "C[pooled]" and x["target"] == "comply" and x["regime"] == "iid"), None)
    if r is None:
        return pd.DataFrame()
    ct = df.set_index("id").loc[r["_test_ids"], "ctype"].values
    rows = []
    for c in np.unique(ct):
        m = ct == c
        y = r["_y"][m] > 0.5
        row = {"ctype": c, "n": int(m.sum()), "base_rate": float(y.mean())}
        if y.sum() >= 5 and (~y).sum() >= 5:
            row.update(auroc_probe=core.auroc(y, r["_pred"][m]), auroc_obs=core.auroc(y, r["_pred_obs"][m]), auroc_oracle=core.auroc(y, r["_pred_oracle"][m]))
        rows.append(row)
    return pd.DataFrame(rows)


def silent(res, frames) -> pd.DataFrame:
    """settled-answer monitoring on rows where the visible text does NOT yet contain the model's final answer (a text monitor has nothing to read)."""
    rows = []
    for r in res:
        if r["target"] != "settled_hi" or r["regime"] != "iid":
            continue
        f = frames[CK_NAMES.index(r["ck"])].set_index("id")
        mf = f.loc[r["_test_ids"], "men_final"].values
        y = r["_y"]
        m = (mf == 0) & ~np.isnan(y)
        yb = y[m] > 0.5
        row = {"ck": r["ck"], "n_silent": int(m.sum()), "n_pos": int(yb.sum())}
        if yb.sum() >= 5 and (~yb).sum() >= 5:
            row.update(auroc_probe=core.auroc(yb, r["_pred"][m]), auroc_obs=core.auroc(yb, r["_pred_obs"][m]), auroc_oracle=core.auroc(yb, r["_pred_oracle"][m]))
        rows.append(row)
    return pd.DataFrame(rows)
