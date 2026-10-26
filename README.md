# model_introspection_probe_suite

Linear probes for what a frozen LLM's residual stream knows about its **own continuation**, in the quantities a monitor would act on:
will it run away, how long, will it obey a constraint, is its answer already settled, how much is left. Every label is a deterministic function of greedy-decoded text
(plus deterministic verifiers for the constraints), the baselines are explicit ladders, and the scorecard in `REPORT.md` applies pre-declared strict rules (`docs/PROTOCOL.md`).

Model: Qwen2.5-7B-Instruct. Data: GSM8K, MATH (integer answers), ARC (multiple choice), synthetic arithmetic, each passed through a ground-truth audit (admitted only at >= 99.5% pass).
No model is trained: the only fitted objects are closed-form ridge probes.

## Stages (all resumable; one process per GPU; whole-node jobs)
`setup.sh` (login node, once) -> `states` (prompt-end states for the plain and the constrained prompt) -> `gen` (greedy plain, greedy constrained, 4 sampled plain solutions)
-> `trace` (hidden states + forced-answer log-probs at 6 checkpoints along the greedy plain solution) -> `analyze` (blocks P, C, K, tables, one per GPU) -> `report`.

## Run it
```
bash cluster/setup.sh                       # once. Reuses ~/venvs/susc and the cached model if they exist. Refuses to rebuild an existing problem set.
MODE=gate bash cluster/submit.sh            # optional dry run on 64 problems per dataset
MODE=full bash cluster/submit.sh            # states -> gen -> trace -> analyze -> report, chained with --dependency=afterok
```
Recovery: `MODE=resume` (gen hit its time budget), `MODE=rescue RESCUE_SHARD=<n>` (one shard crashed: its leftover chunks run on all 4 GPUs, then trace -> analyze -> report), `MODE=analyze`, `MODE=report`.
Never edit on the cluster and never pull while jobs run.

## Look at the results
```
cat <RUN_DIR>/analysis/REPORT.md
cat <RUN_DIR>/analysis/scorecard.csv
```
where `<RUN_DIR>` is `$SCRATCH/psuite/outputs/qwen7b-psuite`.

## Layout
`src/psuite`: `bench.py` (datasets + ground-truth audit), `verify.py` (constraints and their verifiers), `prompts.py`, `hf_util.py`, `gen.py`, `states.py`, `trace.py`, `table.py` (labels + baseline ladders),
`probes.py`, `study.py`, `report.py`, `core.py`/`ridge.py` (probe machinery), `cli.py`. `tests/`: verifier tests, a tiny-model end-to-end test that checks the forced log-probs and the
causality of the stored states against direct forwards, and planted-signal tests of the analysis (including a signal that does NOT generalise and must not be scored SUPPORTED).
