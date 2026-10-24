#!/bin/bash
# STAGE = sample | states | introspect | analyze | analyze3 | steer | report3   (sample/states: 1 process per GPU, resumable chunks)
#SBATCH --job-name=plab
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
[[ -f "$RUN_DIR/problems.jsonl" ]] || ln -sf "$ROOT/outputs/$RUN_NAME/problems.jsonl" "$RUN_DIR/problems.jsonl"
T0=$SECONDS
trap 'notify "FAILED: stage=${STAGE} (line $LINENO)" "$(tail -n 25 "${SLURM_SUBMIT_DIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>/dev/null)"' ERR
elapsed() { printf '%dh%02dm' $(((SECONDS - T0) / 3600)) $((((SECONDS - T0) % 3600) / 60)); }
LIM=(); [[ -n "${GATE:-}" ]] && LIM=(--limit "$GATE_N")
run_sharded() {
  local n i rc=0 pids=()
  n="$(nvidia-smi -L | wc -l)"
  for ((i = 0; i < n; i++)); do
    CUDA_VISIBLE_DEVICES=$i python -m plab.cli "$@" --shard "$i" --nshards "$n" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  return $rc
}
case "${STAGE:?set STAGE}" in
states)
  notify "states STARTED"
  run_sharded states --run-dir "$RUN_DIR" --model-id "$MODEL_ID" --fwd-bs "$FWD_BS" --conds "$CONDS_RUN" --tag "$TAG" ${LIM[@]+"${LIM[@]}"}
  notify "states DONE in $(elapsed)" ;;
sample)
  notify "sample STARTED" "N=$N_SAMPLES per condition"
  run_sharded sample --run-dir "$RUN_DIR" --model-id "$MODEL_ID" --n "$N_SAMPLES" --max-new "$MAX_NEW" --temperature "$TEMP" --top-p "$TOP_P" --chunk "$CHUNK" --max-seqs "$MAX_SEQS" --conds "$CONDS_RUN" --tag "$TAG" ${LIM[@]+"${LIM[@]}"}
  notify "sample DONE in $(elapsed)" ;;
introspect)
  CUDA_VISIBLE_DEVICES=0 python -m plab.cli introspect --run-dir "$RUN_DIR" --model-id "$MODEL_ID"
  notify "introspect DONE in $(elapsed)" ;;
analyze)
  python -m plab.cli analyze --run-dir "$RUN_DIR"
  notify "ANALYSIS ready" "$(sed -n 1,60p "$RUN_DIR/analysis/report.md" | cut -c1-220)" ;;
analyze3)
  python -m plab.cli analyze3 --run-dir "$RUN_DIR" --part compute
  notify "analysis (compute) DONE in $(elapsed)" "$(cat "$RUN_DIR/analysis_v3/directions.json")" ;;
steer)
  notify "steer STARTED" "n=$STEER_N problems"
  run_sharded steer --run-dir "$RUN_DIR" --model-id "$MODEL_ID" --steer-n "$STEER_N" --max-seqs "$MAX_SEQS"
  notify "steer DONE in $(elapsed)" ;;
report3)
  python -m plab.cli analyze3 --run-dir "$RUN_DIR" --part report
  notify "REPORT ready" "$(sed -n 1,25p "$RUN_DIR/analysis_v3/REPORT.md" | cut -c1-260)" ;;
*) echo "unknown STAGE"; exit 2 ;;
esac
