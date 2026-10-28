"""python -m errprop.cli <stage> [--config config.yaml] [--limit N] [--set key=value ...]

stages (one or more, run in order): clean inject continue extract probes steer report
        shortcuts: gpu (clean..extract) | cpu (probes, report) | all (everything)
"""
from __future__ import annotations

import argparse

import yaml

from .io import load_config

GPU = ["clean", "inject", "continue", "extract"]
CPU = ["probes", "report"]
ALL = GPU + ["probes", "steer", "report"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="+", choices=GPU + CPU + ["steer", "gpu", "cpu", "all"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--limit", type=int, default=None, help="only N problems (feasibility gate)")
    ap.add_argument("--set", nargs="*", default=[], help="override config, e.g. run_name=gate model=...")
    a = ap.parse_args()
    cfg = load_config(a.config, {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in a.set)})

    stages = []
    for st in a.stage:
        stages += {"gpu": GPU, "cpu": CPU, "all": ALL}.get(st, [st])
    for s in stages:
        print(f"===== {s} =====", flush=True)
        if s == "clean":
            from .stages import stage_clean
            stage_clean(cfg, a.limit)
        elif s == "inject":
            from .stages import stage_inject
            stage_inject(cfg)
        elif s == "continue":
            from .stages import stage_continue
            stage_continue(cfg)
        elif s == "extract":
            from .stages import stage_extract
            stage_extract(cfg)
        elif s == "probes":
            from .analysis import run_probes
            run_probes(cfg)
        elif s == "steer":
            from .stages import stage_steer
            stage_steer(cfg)
        elif s == "report":
            from .report import write_report
            write_report(cfg)


if __name__ == "__main__":
    main()
