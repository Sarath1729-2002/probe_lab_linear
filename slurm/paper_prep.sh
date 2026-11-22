#!/bin/bash
# Login node, once (needs internet): model weights + the sel_v2 item set. Run from the repo root.
#   bash slurm/paper_prep.sh
set -uo pipefail
source ~/errprop_env.sh
export HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0
python - <<'PY'
from huggingface_hub import snapshot_download
for m in ["Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3",
          "Qwen/Qwen3-8B", "Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-3B-Instruct", "allenai/OLMo-2-1124-7B-Instruct",
          "microsoft/Phi-3.5-mini-instruct", "google/gemma-2-9b-it"]:
    try:
        snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "tokenizer*"])
        print("ok  ", m, flush=True)
    except Exception as e:
        print("FAIL", m, type(e).__name__, str(e)[:120], flush=True)
PY
python -m organic.selective build --config config_sel_v2.yaml
python -c "import organic.sel_more, peft, scipy; print('imports ok')"
