#!/bin/bash
# Login node, once (needs internet): model weights + the sel_v2 item set. Run from the repo root.
#   bash slurm/paper_prep.sh
set -uo pipefail
source ~/errprop_env.sh
export HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0
python - <<'PY'
from huggingface_hub import snapshot_download
for m in ["Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3",
          "Qwen/Qwen3-8B", "Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-32B-Instruct",
          "meta-llama/Llama-3.2-3B-Instruct"]:
    try:
        snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "tokenizer*"])
        print("ok  ", m, flush=True)
    except Exception as e:
        print("FAIL", m, type(e).__name__, str(e)[:120], flush=True)
PY
python -m organic.selective build --config config_sel_v2.yaml
python -c "import organic.sel_more, organic.honesty, peft, scipy; print('imports ok')"
df -h $SCRATCH | tail -1     # 32B adds ~65 GB of weights
