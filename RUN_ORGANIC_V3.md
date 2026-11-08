# organic_v3: framing, three models, causal test

This run uses the same 1,112 frozen ProcessBench items as v1/v2: real wrong solutions with a human-labelled first error, plus correct solutions as controls.

**What it does, for each of Qwen2.5-7B, Llama-3.1-8B and Mistral-7B:**

- **Framing grid.** 4 sources × 3 asks.
  - Sources: the user's working, a student's, another AI's, or the model's own earlier reply.
  - Asks: plain, "please check", or critic. The critic ask is ProcessBench's task: name the first wrong step, then answer.
- **Baselines:** the bare-claim baseline (`sugg`), plus knowledge strength from 5 re-answers at temperature 0.7.
- **Probes.** Run per frame:
  - detection (wrong vs right solution);
  - outcome (follow vs recover) before the reply;
  - comparison against confidence;
  - transfer across frames.
- **Per-position probes**, from the end of the solution into the reply.
- **Steering.** Push the residual stream along mean(recover) − mean(follow) at the last prompt token.
  - The direction is fitted on half the items and tested on the other half.
  - It is compared with a random direction of the same norm, and checked on correct solutions for side effects.

## Commands (from ~/probe_lab_linear)

```bash
# 0. code: unpack over the repo (or commit these files to the repo and pull)
tar xzf ~/organic_v3_overlay.tgz

# 1. login node, once: copy the frozen ProcessBench items and write the 14 prompt families (~seconds)
source ~/errprop_env.sh
python -m organic.cli variants --config config_organic_v3.yaml
#    -> "[variants] copied 1112 frozen items ..." and "[variants] ~16000 prompts" (15 per wrong solution, 14 per correct one)
#    -> look at reports/organic_v3/SAMPLES.md: check frame_self_* (multi-turn) and the critic wording

# 2. GPU: all three models in parallel, one H100 each (~3-4 h)
sbatch slurm/organic_v3.sbatch
tail -f logs/ep_organic_v3-<jobid>.out

# 3. after "ALL DONE": package
tar czf organic_v3.tgz reports/organic_v3 logs/ep_organic_v3-<jobid>.out logs/org3-<jobid>-*.log \
  -C $SCRATCH/organic_runs/organic_v3 \
  $(cd $SCRATCH/organic_runs/organic_v3 && ls */processbench/gen.jsonl */processbench/steer.jsonl */processbench/confidence.jsonl 2>/dev/null)
ls -lh organic_v3.tgz          # tens of MB at most; no .npy files
```

A model whose weights aren't cached is skipped. All three were used in paper_v1, so they should be cached.

**Partial reruns** reuse finished files. Examples:

```bash
MODELS="qwen7b" sbatch ...
STAGES="fit report" sbatch ...
```

## Outputs

| What | Where |
|---|---|
| Prompt samples | `reports/organic_v3/SAMPLES.md` |
| **Per-model report** | `reports/organic_v3/<model>/REPORT.md`, plus `fig_frames.png`, `fig_trajectory.png`, `results.json`, `steer.json` |
| **Cross-model summary** | `reports/organic_v3/COMPARE.md` |
| Replies, graded | `$SCRATCH/organic_runs/organic_v3/<model>/processbench/gen.jsonl` |
| Steering replies | `$SCRATCH/organic_runs/organic_v3/<model>/processbench/steer.jsonl` |

## Local checks (passed)

```bash
python tests/smoke_organic_v3.py /path/to/tiny_model    # every v3 stage, two model tags, planted signals
python tests/smoke_organic.py /path/to/tiny_model       # v1/v2 still works with the shared code
```
