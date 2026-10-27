# error_prop_probe

Linear probes for **error propagation in chain of thought**: when a wrong value is written into the
model's reasoning, can we read off, at that moment, whether the model will follow it, recover
visibly, or recover silently? The research framing, hypotheses and decision rule are in
[`DIRECTION.md`](DIRECTION.md). This README is how to run it.

## What happens

```
clean     greedy CoT + 4 samples on GSM8K/MATH-500        -> clean.jsonl        (GPU, vLLM)
inject    parse verified equations, build dependency graph,
          inject +1 / x10 / digit-swap, compute the exact
          propagated answer                                -> injections.jsonl   (CPU)
continue  continue from each corrupted prefix (greedy + 4),
          plus the clean prefix; label every run            -> continuations.jsonl (GPU, vLLM)
extract   residual stream at the injected number, all
          layers; surprisal of the injected digits          -> acts_*.npy          (GPU, HF)
probes    P0-P5 + baselines + controls                      -> results.json        (CPU)
report    tables, figures, example continuations            -> REPORT.md           (CPU)
```

Every label is computed, never judged: FOLLOW means the final answer equals the answer you get by
re-executing the trace's own equations with the wrong value; RECOVER means it equals gold.

## Setup (TamIA)

```bash
# login node (has internet)
python -m venv ~/venvs/errprop && source ~/venvs/errprop/bin/activate
pip install -r requirements.txt
export HF_HOME=$SCRATCH/hf_cache
python scripts/predownload.py                       # model + GSM8K + MATH-500 into the cache
python -m pytest -q tests && python tests/synthetic_run.py   # CPU sanity checks, ~15 s
```

## Run

**1. Feasibility gate first (50 problems, ~20 min).** It answers the one question that decides
whether the full run is worth it: are there enough FOLLOW *and* RECOVER cases?

```bash
sbatch slurm/tamia.sbatch --limit 50 --set run_name=gate
cat runs/gate/REPORT.md      # look at "Labels" and the examples at the bottom
```

Go on if on-path FOLLOW and RECOVER are each ≥ ~15% and "valid" is ≥ ~0.8. If FOLLOW is rare, raise
`data.math_n` / `data.math_min_level` (harder problems propagate more) before the full run.

**2. Look-through run (~750 problems; my estimate is 1-2 hours on one H100 including the CPU
probe sweep, but this has not been timed on TamIA yet, so the job asks for 6).**

```bash
sbatch slurm/tamia.sbatch                     # run_name=look_v1 from config.yaml
```

Stages can be rerun on their own, e.g. iterate on probes without touching the GPU:

```bash
python -m errprop.cli probes && python -m errprop.cli report
python -m errprop.cli probes --set probes.layer_stride=2      # faster sweep
```

**Second model (once the first looks alive):**

```bash
sbatch slurm/tamia.sbatch --set run_name=r1_v1 model=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    reasoning_model=true gen.clean_max_tokens=4096 gen.cont_max_tokens=3072 gen.max_model_len=8192
```

## Reading REPORT.md

1. **Verdict**: the pre-registered rule from DIRECTION.md applied to P1.
2. **Labels**: outcome counts by dataset and operator. Check these first; everything else is
   meaningless if one class is tiny.
3. **P1 table**: the probe vs four covariate baselines (difficulty, structure, the model's own
   surprisal at the injected digits, all together) and the stacked model. Read the
   **within-problem AUROC** column: it only compares injections into the same problem, so
   difficulty cannot help there.
4. **Controls**: shuffled labels (≈ 0.5) and off-path injections (corrupted but cannot propagate;
   if the probe scores them like FOLLOW, it is reading "corrupted", not "will propagate").
5. **P2 figure**: which digit a clean-trained probe reads on corrupted runs, by layer. Layer 0 reads
   the injected digit by construction; the question is whether mid layers read the true digit more
   on RECOVER than on FOLLOW runs.
6. **Examples** at the bottom: read them. They are the fastest way to catch a parsing bug.

Per-injection probe scores are in `p1_scores.csv`; the full table is `table.csv`.

## Files

```
DIRECTION.md          research direction, hypotheses, decision rule
config.yaml           all settings (override any with --set key=value)
errprop/arith.py      equation parser, dependency graph, injections, exact propagation
errprop/labels.py     answer extraction, FOLLOW/RECOVER/OTHER, recovery type, horizon
errprop/stages.py     clean / inject / continue / extract
errprop/probes.py     layer sweeps, grouped CV, within-problem AUROC, cluster bootstrap
errprop/analysis.py   P0-P5, baselines, controls, verdict
errprop/report.py     REPORT.md + figures
tests/                unit tests + synthetic end-to-end run with a planted signal
slurm/tamia.sbatch    job script (account aip-oberman)
```

## Known limits of this run

- One model, one prompt format, greedy primary labels (samples feed the follow rate only).
- Injections are value-only (+1, ×10, digit swap). Operator and premise errors are a later run.
- The parser keeps only equations that check out arithmetically; traces written mostly in prose
  or heavy LaTeX contribute few injection sites (the inject stage prints how many were skipped).
- Steering (the causal test) is not in this run; it comes after P1 shows a direction worth steering.
