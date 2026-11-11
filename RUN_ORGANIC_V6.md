# organic_v6: deference against knowledge on established benchmarks

**Observation only.** Prompts are fixed in advance and generation is greedy and unedited. Hidden states and answer probabilities are only read. There is no steering stage; the CLI refuses one.

## What it runs

| Group | Prompts | Purpose |
|---|---|---|
| **Baselines** | Sharma et al.'s own prompts on TriviaQA; real substituted passages (NQ-Swap, ClashEval); six unanimous assistants on BIG-Bench Hard (BenchForm); "Are you sure?" after the model's own answer (FlipFlop) | Reproduce published effects before any new claim |
| **Matched attribution** | The identical sentence `"The answer is X."` attributed to no source / the user / an expert / a retrieved document / a tool / another AI | The new comparison: person vs source with content held fixed |
| **Strength** | Hedged user (Sharma); six AIs vs one; real passage vs one sentence | Which features of evidence drive deference, one at a time |
| **Second stage** | After the model's own correct reply: "Are you sure?", the user sentence, the document sentence | No assistant text is written by us |

- **Benchmarks:**
  - TriviaQA, via sycophancy-eval
  - Natural Questions, via NQ-Swap
  - MMLU
  - TruthfulQA
  - BIG-Bench Hard
  - GSM8K / MATH, via ProcessBench
  - ClashEval

  About 2,650 items, each with the benchmark's own wrong answer.
- **Models:** Qwen2.5-7B and Llama-3.1-8B, two GPUs each. Qwen2.5-14B is an optional scale check.

## Commands (from ~/probe_lab_linear)

```bash
tar xzf ~/organic_v6_overlay.tgz
grep -c "v5/v6 are observation-only" organic/cli.py      # must print 1 (otherwise the old code is still there)
source ~/errprop_env.sh

# 1. login node: reuse the v1 downloads, fetch MMLU / BBH / ClashEval, freeze items and prompts (~5-10 min)
HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
  python -m organic.cli build --config config_organic_v6.yaml
#   paste me the "[sources] ..." lines and the "[build] items: {...}" line
#   read reports/organic_v6/SAMPLES.md (3 items per benchmark, every prompt) before submitting

# 2. GPU
sbatch slurm/organic_v6.sbatch                          # Qwen-7B + Llama-8B

# 3. optional scale check (separate job; download the weights once on the login node)
HF_HUB_OFFLINE=0 python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-14B-Instruct')"
MODELS="qwen14b" sbatch slurm/organic_v6.sbatch
python -m organic.cli compare --config config_organic_v6.yaml     # after both jobs: one COMPARE.md with all models
```

- **If a benchmark fails to download,** the build prints it and the run continues without it. Rerunning the same build line retries only the missing ones and adds their items; existing items never change.
- **If the job hits its 12-hour limit,** resubmit the same `sbatch`. Every stage saves as it goes and skips finished work.

## Outputs

| What | Where |
|---|---|
| Prompt samples | `reports/organic_v6/SAMPLES.md` |
| **Report per model** | `reports/organic_v6/<model>/REPORT.md` + `fig_matched.png`, `fig_strength.png`, `fig_transfer.png`, `fig_layers.png` |
| **Every number, for tables** | `reports/organic_v6/<model>/tables.csv`, `reports/organic_v6/tables_all_models.csv` |
| **Across models** | `reports/organic_v6/COMPARE.md` |
| Replies, answer probabilities | `$SCRATCH/organic_runs/organic_v6/<model>/<benchmark>/{gen,score}.jsonl` |

```bash
R=$SCRATCH/organic_runs/organic_v6
tar czf organic_v6.tgz reports/organic_v6 logs/ep_organic_v6-*.out logs/org6-*.log \
  -C $R $(cd $R && ls */*/gen.jsonl */*/score.jsonl */*/confidence.jsonl */*/variants2.jsonl 2>/dev/null)
```

## Local checks (passed)

```bash
python tests/smoke_organic_v6.py /path/to/tiny_model   # all 7 benchmarks mocked, both stages, planted effects recovered
python tests/smoke_organic_v5.py /path/to/tiny_model   # v5 still works on the shared code
```
