"""Run once on a LOGIN node (needs internet):  HF_HUB_OFFLINE=0 python scripts/predownload.py
Caches the model and every dataset the full run uses; compute nodes are offline."""
import os
import sys

for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ[k] = "0"

from datasets import load_dataset  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-7B-Instruct"
snapshot_download(model)
print("gsm8k", len(load_dataset("openai/gsm8k", "main", split="test")))
print("MATH-500", len(load_dataset("HuggingFaceH4/MATH-500", split="test")))
for c in ("p1", "p2"):
    ds = load_dataset("apple/GSM-Symbolic", c, split="test")
    print(f"GSM-Symbolic {c}", len(ds), "fields:", ds.column_names)
print("cached:", model, "+ datasets. Synthetic chains and BBH-style arithmetic are generated locally.")
