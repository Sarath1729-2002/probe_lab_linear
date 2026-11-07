# Decision run on TamIA

Three experiments run in one job. Each has a kill criterion that is fixed before the run, and the report applies those criteria automatically. A thread is **LIVE** when its criterion holds in at least 2 of the 3 models (Qwen2.5-7B, Llama-3.1-8B, Mistral-7B).

| Thread | Question | GPU work | Kill criterion |
|---|---|---|---|
| **A1** | Is Goel et al.'s "detectable but no failure prediction" an artefact of reusing the detection score? | none (reuses paper_v1 activations) | R2 − R1 ≥ 0.20, CI above 0 |
| **A2** | Does it matter whether the corrupted working is the model's own or the user's? | ~1.5k × 2 generations | FOLLOW shifts ≥ 10 pts, CI excludes 0 |
| **B** | Is the truth direction a knowledge direction? (PopQA, cities, arithmetic; known vs unknown, surprisal-matched) | ~20k short generations + ~15k forward passes | PopQA matched gap ≥ 0.10, CI above 0 |
| **C** | Do models silently adopt false premises about numbers they demonstrably know? | ~45 facts × few variants | ≥ 15% adopt **and** ≥ 15% reject/override |

## 1. Login node (internet)

```bash
cd ~/error_prop_probe            # unpack the tarball over the existing repo
source ~/errprop_env.sh
HF_HUB_OFFLINE=0 python scripts/prepare_decide.py
```

This writes `$SCRATCH/decide_data` (PopQA, the Bürger cities CSVs, the Poulis arithmetic CSVs, `b_items.jsonl` and `c_items.jsonl`) and prints the counts. If a GitHub fetch fails, that family is skipped. For arithmetic, a generated set in the same format is used instead. The model weights are the ones paper_v1 already cached.

## 2. Submit

```bash
sbatch slurm/decide.sbatch                        # all three models, about 1–2 h on h100:4
MODELS="qwen7b" sbatch slurm/decide.sbatch          # one model
STAGES="fit" sbatch slurm/decide.sbatch            # refit only (finished files are reused)
SETS="b.popqa_n=2000" sbatch slurm/decide.sbatch   # overrides
```

The threads are independent, so a failed stage is logged (`logs/dec_<model>-<job>-<stage>.log`) and the chain moves on.

## 3. Read

`reports/decide_v1/DECISION.md` has one table per thread, the criterion verdicts and a line saying what to do next. The raw JSONs sit next to it. Send me the folder:

```bash
tar czf decide_v1_report.tgz reports/decide_v1
```

## Local check (already passed)

```bash
python tests/smoke_decide.py /path/to/tiny_model [ext_clones]   # CPU, every stage, planted signals
```
