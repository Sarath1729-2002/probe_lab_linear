"""python -m organic.cli <stage> [<stage> ...] [--config config_organic.yaml] [--buckets b1 b2 ...] [--set k=v ...]

Login node (internet):  download   build
Login node (CPU):       variants   (new prompt families on the SAME items.jsonl)   regrade
GPU:                    gen   confidence   extract   traj        (--buckets limits them; default all)
CPU:                    fit   report
Helper:                 shard --n 4                  (prints one bucket list per GPU)

v3 (config_organic_v3.yaml, one model per --model tag; outputs under <run_prefix>/<tag>):
  login:  variants                       GPU:  gen confidence extract traj steer       CPU: fit report
  once, after all models:  compare
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

from .common import load_config, read_jsonl

STAGES = ["download", "build", "variants", "regrade", "gen", "confidence", "extract", "score", "fit", "deference",
          "traj", "steer", "report", "compare", "shard"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="+", choices=STAGES)
    ap.add_argument("--config", default="config_organic.yaml")
    ap.add_argument("--buckets", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--model", default=None, help="v3: model tag from the config's models list")
    a = ap.parse_args()
    cfg = load_config(a.config, {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in a.set)})
    prefix = cfg["run_prefix"]
    if a.model:
        m = next((m for m in cfg.get("models", []) if m["tag"] == a.model), None)
        if m is None:
            raise SystemExit(f"unknown --model {a.model}; config has {[m['tag'] for m in cfg.get('models', [])]}")
        cfg["model"], cfg["run_prefix"] = m["name"], f"{prefix}/{a.model}"
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
            rebuild_variants(dd, Path(os.path.expandvars(cfg["report_dir"])) / prefix / "SAMPLES.md", cfg["seed"],
                             cfg.get("families"), Path(os.path.expandvars(cfg["items_from"])) if cfg.get("items_from") else None,
                             cfg["buckets"] if cfg.get("version") in (4, 5) else all_b)
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
        elif s == "score":
            from .score import stage_score
            stage_score(cfg, bks)
        elif s == "deference":
            from .deference import stage_deference
            stage_deference(cfg)
        elif s == "steer":
            if cfg.get("version") == 5:
                raise SystemExit("v5 is observation-only: there is no steering stage")
            if cfg.get("version") == 4:
                from .steer import stage_steer_multi
                stage_steer_multi(cfg)
            else:
                from .steer import stage_steer
                stage_steer(cfg)
        elif s == "report":
            if cfg.get("version") == 5:
                from .report_v5 import write_report_v5
                write_report_v5(cfg)
            elif cfg.get("version") == 4:
                from .report_v4 import write_report_v4
                write_report_v4(cfg)
            elif cfg.get("version") == 3:
                from .report_v3 import write_report_v3
                write_report_v3(cfg)
            else:
                from .report import write_report
                write_report(cfg)
        elif s == "compare":
            if cfg.get("version") == 5:
                from .report_v5 import write_compare_v5
                write_compare_v5(dict(cfg, run_prefix=prefix), [m["tag"] for m in cfg.get("models", [])])
            elif cfg.get("version") == 4:
                from .report_v4 import write_compare_v4
                write_compare_v4(dict(cfg, run_prefix=prefix), [m["tag"] for m in cfg.get("models", [])])
            else:
                from .report_v3 import write_compare
                write_compare(dict(cfg, run_prefix=prefix), [m["tag"] for m in cfg.get("models", [])])
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
