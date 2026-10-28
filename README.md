# error_prop_probe

Linear probes for **error propagation in chain of thought**: when a wrong value is written into the
model's reasoning, does the residual stream at that token predict whether the model will follow it
or catch it, and can a monitor use that to decide where to intervene? Design, borrowed protocols,
datasets and the pre-registered decision rule are in [`DIRECTION.md`](DIRECTION.md).

## Pipeline

```
clean     greedy CoT + 4 samples on 6 datasets                       (GPU, vLLM)
inject    parse verified equations, dependency graph, 6 error
          operators, exact propagated answer for every injection     (CPU, seconds)
continue  per injection: greedy, 4 samples, clean prefix,
          + "Wait" arm on the corrupted and on the clean prefix       (GPU, vLLM)
extract   residual stream at the injected number, all layers;
          surprisal of the injected digits; prompt-time states        (GPU, transformers)
probes    P0-P5, catchable headline, robustness, 50-shuffle null,
          Wait-policy simulation, steering direction                  (CPU, many cores)
steer     add ±alpha * direction at the probe layer, re-generate      (GPU, transformers)
report    REPORT.md + figures, copied to reports/<run_name>/          (CPU)
```

Big files (activations, ~10 GB) go to `${SCRATCH}/errprop_runs/<run_name>/`. The report, figures,
`results.json` and per-injection probe scores are copied to `reports/<run_name>/` in this folder.

## Paper recreation: "How Much is Left?" for error propagation (probing only)

A level-1 recreation of *How Much is Left? LLMs Linearly Encode Their Remaining Output Length*, with
"will the model propagate this injected error?" in place of "how many tokens are left?". There is no
steering: this is an initial probing result.

**Models:**
- `Qwen/Qwen2.5-7B-Instruct`: the pilot model.
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`: a reasoning model. It fills the paper's Qwen3-4B role in the retraction analysis.
- `meta-llama/Llama-3.1-8B-Instruct` and `mistralai/Mistral-7B-Instruct-v0.3`: the paper's own models. Both are gated.

**Datasets** follow the paper's §3.7 (2 synthetic + 5 standard), each restricted to answers that can be checked exactly:

| Dataset | Paper's counterpart | Constraint |
|---|---|---|
| Chains (synthetic) | Count | — |
| BBH-Arith (synthetic) | Countdown | — |
| GSM8K | GSM8K | — |
| MATH | MATH | full test set; `\boxed` answer must be a plain number |
| MMLU-Pro | MMLU-Pro | numeric-option questions; options withheld; 2% tolerance |
| OpenThoughts-1k | OpenThoughts-1k | 1k numeric math problems |
| TriviaQA | TriviaQA | numeric answers; the injected error is a wrong recalled fact at the first place the model states it |

**Metrics, mapped to the paper:**

| Paper | Here |
|---|---|
| Constant / exact countdown / Completion Length / Remaining Count | Constant / prompt-only + structure / prompt-only probe / error-token probe. The black-box observer is an extra row. |
| MAE and MAE ÷ floor | AUROC; Brier and log-loss ÷ constant |
| §4.3 retraction analysis | Pre-specified `Wait` trigger. Δ of the probe logit against matched controls, Cohen's d, cluster sign-flip and label-shuffle tests, plus a discovery scan on a disjoint half |
| §5 transfer matrix ÷ eval floor | Same, plus the natural-vs-synthetic asymmetry count and a Fisher test |
| App. A.7 | Exclusions |
| App. A.12 | TF-IDF surface baselines; global and within-problem shuffled labels |
| App. A.13 | Sampling-noise floor (6 samples at T 0.7 / top-p 0.8) |
| App. A.14 | Cohen's κ |
| App. A.2 | Layer sweep |
| Remaining-count MAE | Contamination horizon: MAE against the constant-median floor |

All results are the mean ± sd over 3 problem-level splits.

**To run:**
```bash
source ~/errprop_env.sh && cd ~/probe_lab_linear
tar xzf error_prop_probe_paper.tar.gz --strip-components=1
# gated models: accept the licences on the HF pages of Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3, then
huggingface-cli login                                   # or: export HF_TOKEN=hf_...
HF_HUB_OFFLINE=0 python scripts/predownload_paper.py    # login node: 4 models + GSM8K + prepare_data.py
python -m pytest -q tests && python tests/synthetic_paper.py
sbatch slurm/paper.sbatch                               # 1 node, 4 models in parallel, 1 GPU each
```

`prepare_data.py` prints each source's column names and row counts. If a row says FAILED, that
dataset is skipped and the rest run. Each model logs every stage to
`logs/ep_<tag>-<jobid>-<stage>.log`; the main log shows which stages ran and how long each took.

- **Resume:** `STAGES="fit trajectory" MODELS="r1qwen7b" sbatch slurm/paper.sbatch`.
- **Rebuild the report on the login node:** `python -m errprop.cli paper --config config_paper.yaml`.
- **Output:** `reports/paper_v1/PAPER.md` plus figures.
- **R1-Distill is the slow model:** traces of up to 4k tokens and continuations of up to 3k. If the job
  runs short, give it its own job: `MODELS=r1qwen7b sbatch slurm/paper.sbatch`.

## Running the full experiment on TamIA

Assumes the environment from before (`~/errprop_env.sh`, venv on scratch, `accelerate` installed).

**1. Unpack and cache the new datasets** (login node, needs internet, ~1 min):

```bash
source ~/errprop_env.sh && cd ~/probe_lab_linear
tar xzf error_prop_probe_full.tar.gz --strip-components=1
HF_HUB_OFFLINE=0 python scripts/predownload.py      # adds GSM-Symbolic P1/P2
python -m pytest -q tests && python tests/synthetic_run.py
```

**2. Smoke test (recommended, ~40 min):** 10 problems per dataset through every stage, so a typo in
a new dataset or the steering hook fails in minutes, not hours.

```bash
sbatch --time=02:00:00 slurm/full.sbatch --limit 60 --set run_name=smoke steer.n=20
tail -f logs/errprop_full-<JOBID>.out
```

Check `reports/smoke/REPORT.md` exists and every dataset appears in the "Outcome rates by dataset"
table. Numbers will be meaningless at this size.

**3. Full run (~10-14 h, 18 h limit):**

```bash
sbatch slurm/full.sbatch
```

Rough stage times on one H100: clean 30 min · continue 2-3 h · extract 1-1.5 h · probes 2-4 h
(CPU) · steer 1-1.5 h. Each stage prints a header with its duration in the log.

**Resume after a failure** from the stage that failed; finished files are reused:

```bash
STAGES="probes steer report" sbatch slurm/full.sbatch
```

## What to show (reports/full_v1/REPORT.md)

1. **Verdict**: the pre-registered rule on the catchable headline and on all injections.
2. **Outcome rates by error size**: small errors are followed, large ones sometimes caught.
3. **P1 catchable table**: probe vs difficulty / structure / surprisal / all covariates, gain, and
   the permutation null.
4. **Robustness**: by operator (the strict test), by dataset, by error size, across 5 splits.
5. **W, the Wait-policy table and figure**: final accuracy at 5-50% intervention budget for probe-,
   surprisal-, covariate- and random-ranked policies, with the oracle as ceiling. The mixed pool
   includes clean steps, so false alarms cost accuracy there.
6. **Steering table and figure**: FOLLOW rate vs alpha, with random-direction controls.

## Files

```
DIRECTION.md          question, protocol lineage, datasets, pre-registered rule
config.yaml           all settings (override with --set key=value)
errprop/arith.py      equation parser, dependency graph, 6 injection operators, exact propagation
errprop/synth.py      synthetic serial chains + BBH-format multistep arithmetic
errprop/data.py       GSM8K, GSM-Symbolic, MATH-500 loaders
errprop/labels.py     answer extraction, FOLLOW/RECOVER/OTHER, recovery type, horizon
errprop/stages.py     clean / inject / continue / extract / steer
errprop/probes.py     layer sweeps, grouped CV, within-problem AUROC, cluster bootstrap
errprop/analysis.py   all probes, robustness, permutation null, Wait-policy simulation
errprop/report.py     REPORT.md + figures
slurm/full.sbatch     whole-node job, one process per stage, resumable via STAGES
tests/                unit tests + synthetic end-to-end run with a planted signal
```
