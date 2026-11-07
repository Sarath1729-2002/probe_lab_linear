"""Run once on a LOGIN node (needs internet) before slurm/decide.sbatch:

    HF_HUB_OFFLINE=0 python scripts/prepare_decide.py

Writes to $SCRATCH/decide_data:
  popqa.jsonl                       PopQA test split (akariasai/PopQA)
  truth_is_universal/{cities,neg_cities}.csv   Bürger et al. 2024 (MIT)
  poulis/arith_{1op,2ops,3ops}.csv  Poulis et al.; if unreachable, decide/ generates the same format
  b_items.jsonl, c_items.jsonl      the built item sets
Model weights are the ones paper_v1 already cached."""
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ[k] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from errprop.io import load_config  # noqa: E402

cfg = load_config(os.environ.get("DECIDE_CONFIG", "config_decide.yaml"))
dd = Path(os.path.expandvars(cfg["data_dir"]))
dd.mkdir(parents=True, exist_ok=True)
problems = []


def fetch(repo, files, dest):
    """Copy CSVs from a shallow git clone; fall back to raw.githubusercontent."""
    dest.mkdir(parents=True, exist_ok=True)
    if all((dest / Path(f).name).exists() for f in files):
        print(f"{dest.name}: already present")
        return
    tmp = dd / f"_clone_{repo.split('/')[1]}"
    if not tmp.exists():
        subprocess.run(["git", "clone", "--depth", "1", f"https://github.com/{repo}.git", str(tmp)],
                       capture_output=True)
    for f in files:
        out = dest / Path(f).name
        if (tmp / f).exists():
            out.write_bytes((tmp / f).read_bytes())
            continue
        for br in ("main", "master"):
            try:
                urllib.request.urlretrieve(f"https://raw.githubusercontent.com/{repo}/{br}/{f}", out)
                break
            except Exception:
                pass
        if not out.exists():
            problems.append(f"{repo}:{f}")
    print(f"{dest.name}: {sum((dest / Path(f).name).exists() for f in files)}/{len(files)} files")


# ---- PopQA
pq = dd / "popqa.jsonl"
if not pq.exists():
    try:
        from datasets import load_dataset
        ds = load_dataset("akariasai/PopQA", split="test")
        with open(pq, "w") as f:
            for r in ds:
                ans = r["possible_answers"]
                ans = json.loads(ans) if isinstance(ans, str) else list(ans)
                f.write(json.dumps({"id": r["id"], "prop": r["prop"], "subj": r["subj"], "obj": r["obj"],
                                    "answers": ans, "s_pop": int(r["s_pop"]), "question": r["question"]}) + "\n")
        print(f"popqa: {len(ds)} rows")
    except Exception as e:
        problems.append(f"PopQA ({type(e).__name__}: {str(e)[:120]})")
else:
    print("popqa: already present")

fetch("sciai-lab/Truth_is_Universal", ["datasets/cities.csv", "datasets/neg_cities.csv"], dd / "truth_is_universal")
fetch("angelosps/llm-truth-directions-limits",
      [f"datasets/arithmetic/arith_{x}.csv" for x in ("1op", "2ops", "3ops")], dd / "poulis")

from decide import thread_b, thread_c  # noqa: E402

thread_b.build_items(cfg)
thread_c.build_items(cfg)
print("\nOK" if not problems else "\nPROBLEMS (the run still works; missing families are skipped):\n  " + "\n  ".join(problems))
