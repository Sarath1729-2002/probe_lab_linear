"""Analysis driver.  Blocks are independent (the cluster runs them on separate GPUs):  P | C | K | tables.   Each writes analysis/<block>.pkl (+ CSVs)."""
import pickle
from pathlib import Path

from . import core, probes, table
from .bench import DS_ORDER, read_problems

BLOCKS = ("P", "C", "K", "tables")


def _datasets(run_dir, datasets):
    have = {p["ds"] for p in read_problems(run_dir)}
    return [d for d in DS_ORDER if d in have and (datasets is None or d in datasets)]


def _save(out: Path, name: str, obj):
    with open(out / f"{name}.pkl", "wb") as fh:
        pickle.dump(obj, fh)


def compute(run_dir, block: str, datasets=None, layers=None, log=print):
    run_dir = Path(run_dir)
    out = run_dir / "analysis"
    out.mkdir(exist_ok=True)
    dss = _datasets(run_dir, datasets)
    if block == "tables":
        df, _ = table.build(run_dir, dss, need_states=False, log=log)
        _save(out, "tables", {"df": df.drop(columns=["_q"])})
        df.drop(columns=["_q"]).to_csv(out / "problem_table.csv", index=False)
        return df
    if block == "K":
        df, tr, frames, tks = table.build_trace(run_dir, dss, log=log)
        res, cur = probes.block_K(frames, tks, tr["layers"], dss, log)
        sil = probes.silent(res, frames)
        _save(out, "K", {"res": res, "cur": cur, "silent": sil})
        core.strip(res).to_csv(out / "K_results.csv", index=False)
        sil.to_csv(out / "K_silent.csv", index=False)
        return res
    df, sub = table.build(run_dir, dss, need_states=True, names=("plain",) if block == "P" else ("con",), log=log)
    L = next(iter(sub.values())).shape[1]
    ls = layers or probes.prompt_layers(L)
    log(f"block {block}: n={len(df)} datasets={dss} layers={ls}")
    if block == "P":
        res, cur = probes.block_P(df, sub, dss, ls, log)
        _save(out, "P", {"res": res, "cur": cur})
        core.strip(res).to_csv(out / "P_results.csv", index=False)
    else:
        res, cur = probes.block_C(df, sub, dss, ls, log)
        ct = probes.per_ctype(res, df)
        _save(out, "C", {"res": res, "cur": cur, "ctype": ct})
        core.strip(res).to_csv(out / "C_results.csv", index=False)
        ct.to_csv(out / "C_per_ctype.csv", index=False)
    return df
