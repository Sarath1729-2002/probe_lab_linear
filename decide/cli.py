"""python -m decide.cli <stage> [<stage> ...] --config config_decide.yaml [--model TAG]

Login node:   data                         (needs prepare_decide.py first)
Per model:    a1 a2_gen a2_extract b_know b_extract c_gen c_extract fit
Once:         report
"""
from __future__ import annotations

import argparse

from errprop.io import load_config

from . import thread_a, thread_b, thread_c
from .report import write_report

MODEL_STAGES = ["a1", "a2_gen", "a2_extract", "b_know", "b_extract", "c_gen", "c_extract", "fit"]


def _fit_all(cfg, tag):
    """Each thread's fit runs even if another's inputs are missing."""
    import traceback
    bad = []
    for f in (thread_a.stage_a2_fit, thread_b.stage_b_fit, thread_c.stage_c_fit):
        try:
            f(cfg, tag)
        except Exception:
            traceback.print_exc()
            bad.append(f.__name__)
    if bad:
        raise SystemExit(f"fit failed: {bad}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="+", choices=["data", "report"] + MODEL_STAGES)
    ap.add_argument("--config", default="config_decide.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides (YAML values)")
    a = ap.parse_args()
    import yaml
    cfg = load_config(a.config, {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in a.set)})
    name = None
    if a.model:
        m = next((m for m in cfg["models"] if m["tag"] == a.model), None)
        if m is None:
            raise SystemExit(f"unknown --model {a.model}")
        name = m["name"]
    for s in a.stage:
        print(f"===== {s} =====", flush=True)
        if s == "data":
            thread_b.build_items(cfg)
            thread_c.build_items(cfg)
        elif s == "report":
            write_report(cfg)
        else:
            if not a.model:
                raise SystemExit(f"stage {s} needs --model")
            {"a1": lambda: thread_a.stage_a1(cfg, a.model),
             "a2_gen": lambda: thread_a.stage_a2_gen(cfg, a.model, name),
             "a2_extract": lambda: thread_a.stage_a2_extract(cfg, a.model, name),
             "b_know": lambda: thread_b.stage_b_know(cfg, a.model, name),
             "b_extract": lambda: thread_b.stage_b_extract(cfg, a.model, name),
             "c_gen": lambda: thread_c.stage_c_gen(cfg, a.model, name),
             "c_extract": lambda: thread_c.stage_c_extract(cfg, a.model, name),
             "fit": lambda: _fit_all(cfg, a.model),
             }[s]()


if __name__ == "__main__":
    main()
