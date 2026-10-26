#!/bin/bash
# ONE-TIME, on the LOGIN node (needs internet), from the repo root:   bash cluster/setup.sh
# Adds this package to the venv, makes sure the model is cached, downloads GSM8K / MATH / ARC, generates the synthetic set, runs the GROUND-TRUTH AUDIT and writes problems.jsonl.
# Safe to run once per RUN_NAME.  Do NOT re-run after generation has started: it would reassign problem ids.
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
source cluster/config.sh
export HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0
module purge 2>/dev/null || true
module load StdEnv/2023 gcc arrow python/3.12 2>/dev/null || echo "[warn] module load failed - try: module spider arrow python"
mkdir -p "$ROOT/outputs" "$HF_HOME" "$(dirname "$VENV")"
[[ -d "$VENV" ]] || python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -c "import torch" 2>/dev/null || { pip install --upgrade pip; pip install --no-index "torch>=2.9.0" 2>/dev/null || pip install "torch>=2.9.0"; }
pip install -e .
mkdir -p "$RUN_DIR"
[[ -f "$RUN_DIR/problems.jsonl" ]] && { echo "!! $RUN_DIR/problems.jsonl already exists - refusing to rebuild (ids would change). Use a new RUN_NAME for a new problem set."; exit 1; }
python - <<PY
import os
from huggingface_hub import snapshot_download
snapshot_download(os.environ["MODEL_ID"], allow_patterns=["*.json", "*.safetensors", "*.txt"])
print("model cached")
PY
python -m psuite.cli data --run-dir "$RUN_DIR" --sizes "$SIZES"
echo; echo "=== ground-truth audit ==="; python -c "import json;[print(a['dataset'], 'tier', a['tier'], 'kept', a['n_kept'], '/', a['n_in'], 'ADMITTED' if a['admitted'] else 'REJECTED', a['fails']) for a in json.load(open('$RUN_DIR/ground_truth_audit.json'))]"
wc -l "$RUN_DIR/problems.jsonl"
echo "next:  MODE=gate bash cluster/submit.sh   (optional ~1h dry run)   or   MODE=full bash cluster/submit.sh"
