"""Run once on a LOGIN node (needs internet) before slurm/paper.sbatch:

    HF_HUB_OFFLINE=0 python scripts/predownload_paper.py              # all models in config_paper.yaml
    HF_HUB_OFFLINE=0 python scripts/predownload_paper.py qwen7b r1qwen7b

Llama-3.1 and Mistral are gated: accept the licence on their HF pages, then `huggingface-cli login`
(or export HF_TOKEN=...). A model that fails is reported and skipped; paper.sbatch skips it too.
Then runs scripts/prepare_data.py (MATH, MMLU-Pro, OpenThoughts-1k, TriviaQA -> $SCRATCH/errprop_data)."""
import os
import sys

for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ[k] = "0"

import yaml  # noqa: E402
from datasets import load_dataset  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

cfg = yaml.safe_load(open("config_paper.yaml"))
want = set(sys.argv[1:])
ok, bad = [], []
for m in cfg["models"]:
    if want and m["tag"] not in want:
        continue
    try:
        snapshot_download(m["name"], allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "*.tiktoken"],
                          ignore_patterns=["consolidated*", "original/*"])   # Mistral/Llama duplicate weights
        ok.append(m["tag"])
    except Exception as e:
        bad.append(f"{m['tag']} ({type(e).__name__}: {str(e)[:120]})")

try:
    print("gsm8k:", len(load_dataset("openai/gsm8k", "main", split="test")), "rows")
except Exception as e:
    bad.append(f"dataset gsm8k ({type(e).__name__})")
import subprocess  # noqa: E402
r = subprocess.run([sys.executable, "scripts/prepare_data.py"] + ([os.environ["ERRPROP_DATA"]] if "ERRPROP_DATA" in os.environ else []))
if r.returncode:
    bad.append("scripts/prepare_data.py")
print("models cached:", ok)
if bad:
    print("FAILED:", *bad, sep="\n  ")
