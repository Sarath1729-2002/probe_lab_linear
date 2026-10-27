"""Run once on a login node (internet access) to cache the model and both datasets."""
import sys

from datasets import load_dataset
from huggingface_hub import snapshot_download

model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-7B-Instruct"
snapshot_download(model)
load_dataset("openai/gsm8k", "main", split="test")
load_dataset("HuggingFaceH4/MATH-500", split="test")
print("cached:", model, "gsm8k", "MATH-500")
