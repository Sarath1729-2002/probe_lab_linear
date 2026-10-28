"""python -m errprop.cli <stage> [<stage> ...] [--config config.yaml] [--model TAG] [--limit N] [--set key=value ...]

Stages (run in the order given):
  pipeline     clean inject continue extract
  paper run    fit trajectory paper          (config_paper.yaml; `paper` aggregates every model)
  look/full    probes steer report           (earlier single-model analysis)
Shortcuts: gpu = clean..extract | cpu = probes report | all = everything of the earlier analysis
"""
from __future__ import annotations

import argparse

import yaml

from .io import load_config

GPU = ["clean", "inject", "continue", "extract"]
CPU = ["probes", "report"]
ALL = GPU + ["probes", "steer", "report"]
STAGES = GPU + CPU + ["steer", "fit", "trajectory", "paper"]


def select_model(cfg: dict, tag: str | None) -> dict:
    """With a `models:` list in the config, --model TAG picks one and names the run after it."""
    models = cfg.get("models")
    if not models or tag is None:
        return cfg
    m = next((m for m in models if m["tag"] == tag), None)
    if m is None:
        raise SystemExit(f"unknown --model {tag}; known: {[m['tag'] for m in models]}")
    cfg["model"] = m["name"]
    cfg["model_tag"] = tag
    cfg["run_name"] = f"{cfg['run_prefix']}-{tag}"
    for k, v in (m.get("overrides") or {}).items():
        node = cfg
        *head, last = k.split(".")
        for h in head:
            node = node[h]
        node[last] = v
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="+", choices=STAGES + ["gpu", "cpu", "all"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", default=None, help="model tag from the config's `models:` list")
    ap.add_argument("--limit", type=int, default=None, help="only N problems (quick test)")
    ap.add_argument("--set", nargs="*", default=[], help="override config, e.g. run_name=gate seed=1")
    a = ap.parse_args()
    cfg = load_config(a.config, {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in a.set)})
    cfg = select_model(cfg, a.model)

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
        elif s == "fit":
            from .paper import stage_fit
            stage_fit(cfg)
        elif s == "trajectory":
            from .paper import stage_trajectory
            stage_trajectory(cfg)
        elif s == "paper":
            from .paper_report import write_paper
            write_paper(cfg)


if __name__ == "__main__":
    main()
