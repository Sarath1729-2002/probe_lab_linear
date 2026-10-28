from __future__ import annotations

import json
from pathlib import Path

import yaml


def load_config(path: str, overrides: dict | None = None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k, v in (overrides or {}).items():
        node = cfg
        *head, last = k.split(".")
        for h in head:
            node = node[h]
        node[last] = v
    return cfg


def run_dir(cfg: dict) -> Path:
    """Run folder. `out_dir` may use env vars, e.g. ${SCRATCH}/errprop_runs (big activation files
    must not go to home: TamIA home has a 250K-file / 25 GB quota)."""
    import os
    d = Path(os.path.expandvars(cfg["out_dir"])) / cfg["run_name"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_jsonl(path: Path, rows) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
