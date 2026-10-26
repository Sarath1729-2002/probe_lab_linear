#!/bin/bash
# STAGE = states | gen | trace | analyze | report      (states/gen/trace: one process per GPU, resumable chunk files)
#SBATCH --job-name=psuite
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/cluster/config.sh"
[[ -n "${GATE:-}" ]] && { RUN_DIR="${RUN_DIR}-gate"; }
module purge 2>/dev/null || true
module load StdEnv/2023 gcc arrow python/3.12 2>/dev/null || true
source "$VENV/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p "$RUN_DIR"
for f in problems.jsonl ground_truth_audit.json; do [[ -f "$RUN_DIR/$f" ]] || ln -sf "$ROOT/outputs/$RUN_NAME/$f" "$RUN_DIR/$f"; done
T0=$SECONDS
trap 'notify "FAILED: stage=${STAGE} (line $LINENO)" "$(tail -n 25 "${SLURM_SUBMIT_DIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>/dev/null)"' ERR
elapsed() { printf '%dh%02dm' $(((SECONDS - T0) / 3600)) $((((SECONDS - T0) % 3600) / 60)); }
LIM=(); [[ -n "${GATE:-}" ]] && LIM=(--limit "$GATE_LIMIT")
DS=(); [[ -n "${DATASETS:-}" ]] && DS=(--datasets "$DATASETS")
BUDGET_S="$(python3 -c "print(int(float('${GEN_BUDGET_H}')*3600))")"
run_sharded() {
  local n i rc=0 pids=()
  n="$(nvidia-smi -L | wc -l)"
  for ((i = 0; i < n; i++)); do
    if [[ -n "${RESCUE_SHARD:-}" && ( "${STAGE}" == "gen" ) ]]; then       # all GPUs work on the leftovers of the one crashed shard
      CUDA_VISIBLE_DEVICES=$i python -m psuite.cli "$@" --shard "$RESCUE_SHARD" --nshards "$ORIG_NSHARDS" --sub "$i/$n" &
    else
      CUDA_VISIBLE_DEVICES=$i python -m psuite.cli "$@" --shard "$i" --nshards "$n" &
    fi
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  return $rc
}
case "${STAGE:?set STAGE}" in
states)
  notify "states STARTED"
  run_sharded states --run-dir "$RUN_DIR" --model-id "$MODEL_ID" --fwd-bs "$FWD_BS" ${LIM[@]+"${LIM[@]}"} ${DS[@]+"${DS[@]}"}
  notify "states DONE in $(elapsed)" ;;
gen)
  notify "gen STARTED" "greedy plain + constrained + ${N_SPREAD} sampled, budget ${GEN_BUDGET_H}h"
  [[ -n "${GATE:-}" ]] && BUDGET_S=0
  run_sharded gen --run-dir "$RUN_DIR" --model-id "$MODEL_ID" --n-spread "$N_SPREAD" --chunk "$CHUNK" --max-seqs "$MAX_SEQS" --budget-s "$BUDGET_S" ${LIM[@]+"${LIM[@]}"} ${DS[@]+"${DS[@]}"}
  notify "gen DONE in $(elapsed)" "chunks: $(ls "$RUN_DIR"/gen/*.jsonl 2>/dev/null | wc -l)" ;;
trace)
  notify "trace STARTED"
  run_sharded trace --run-dir "$RUN_DIR" --model-id "$MODEL_ID" --fwd-bs "$FWD_BS" --chunk "$CHUNK" ${LIM[@]+"${LIM[@]}"} ${DS[@]+"${DS[@]}"}
  notify "trace DONE in $(elapsed)" ;;
analyze)
  notify "analysis STARTED"
  pids=()
  ( export CUDA_VISIBLE_DEVICES=0; python -m psuite.cli analyze --run-dir "$RUN_DIR" --block P ${DS[@]+"${DS[@]}"} ) & pids+=($!)
  ( export CUDA_VISIBLE_DEVICES=1; python -m psuite.cli analyze --run-dir "$RUN_DIR" --block C ${DS[@]+"${DS[@]}"} ) & pids+=($!)
  ( export CUDA_VISIBLE_DEVICES=2; python -m psuite.cli analyze --run-dir "$RUN_DIR" --block K ${DS[@]+"${DS[@]}"} ) & pids+=($!)
  ( export CUDA_VISIBLE_DEVICES=3; python -m psuite.cli analyze --run-dir "$RUN_DIR" --block tables ${DS[@]+"${DS[@]}"} ) & pids+=($!)
  rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
  [[ $rc == 0 ]] || { notify "analysis: a block FAILED"; exit 1; }
  notify "analysis DONE in $(elapsed)" ;;
report)
  python -m psuite.cli report --run-dir "$RUN_DIR"
  notify "REPORT ready" "$(sed -n '/## 3. Scorecard/,/## 4/p' "$RUN_DIR/analysis/REPORT.md" | cut -c1-300 | head -n 40)" ;;
*) echo "unknown STAGE"; exit 2 ;;
esac
