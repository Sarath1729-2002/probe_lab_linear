"""python -m organic.cli <stage> [<stage> ...] [--config config_organic.yaml] [--buckets b1 b2 ...] [--set k=v ...]

Login node (internet):  download   build
Login node (CPU):       variants   (new prompt families on the SAME items.jsonl)   regrade
GPU:                    gen   confidence   extract   traj        (--buckets limits them; default all)
CPU:                    fit   report
Helper:                 shard --n 4                  (prints one bucket list per GPU)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

from .common import load_config, read_jsonl

STAGES = ["download", "build", "variants", "regrade", "gen", "confidence", "extract", "fit", "traj", "report", "shard"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="+", choices=STAGES)
    ap.add_argument("--config", default="config_organic.yaml")
    ap.add_argument("--buckets", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in a.set)})
    dd = Path(os.path.expandvars(cfg["data_dir"]))
    all_b = [b for b, n in cfg["buckets"].items() if n]
    bks = a.buckets or all_b
    for s in a.stage:
        if s != "shard":
            print(f"===== {s} =====", flush=True)
        if s == "download":
            from .sources import download_all
            st = download_all(dd / "raw", bks if a.buckets else None)
            bad = [k for k, v in st.items() if v.startswith("FAILED")]
            print("download:", st, "\nFAILED: " + ", ".join(bad) if bad else "\nall sources OK")
        elif s == "build":
            from .build import build
            build(cfg, dd, Path(os.path.expandvars(cfg["report_dir"])) / cfg["run_prefix"] / "SAMPLES.md")
        elif s == "variants":
            from .build import rebuild_variants
            rebuild_variants(dd, Path(os.path.expandvars(cfg["report_dir"])) / cfg["run_prefix"] / "SAMPLES.md", cfg["seed"])
        elif s == "regrade":
            from .run import stage_regrade
            stage_regrade(cfg, bks)
        elif s == "confidence":
            from .run import stage_confidence
            stage_confidence(cfg, bks)
        elif s == "gen":
            from .run import stage_gen
            stage_gen(cfg, bks)
        elif s == "extract":
            from .run import stage_extract
            stage_extract(cfg, bks)
        elif s == "fit":
            from .fit import stage_fit
            stage_fit(cfg)
        elif s == "traj":
            from .traj import stage_traj
            stage_traj(cfg, bks)
        elif s == "report":
            from .report import write_report
            write_report(cfg)
        elif s == "shard":                       # balance buckets over GPUs by (prompts x max tokens)
            V = read_jsonl(dd / "variants.jsonl")
            mt = cfg["gen"]["max_tokens"]
            load = {b: sum(v["bucket"] == b for v in V) * mt.get(b, mt["default"]) for b in all_b}
            bins = [[] for _ in range(a.n)]
            tot = [0] * a.n
            for b in sorted(load, key=load.get, reverse=True):
                k = tot.index(min(tot))
                bins[k].append(b)
                tot[k] += load[b]
            for x in bins:
                print(" ".join(x))


if __name__ == "__main__":
    main()
