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
UNC_BUDGET_S="$(python3 -c "print(int(float('${UNC_BUDGET_H}')*3600))")"
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
unc-roll)
  notify "unc-roll STARTED" "n=${UNC_N}/dataset, ${N_ROLL} rollouts, ${N_CONT} continuations at o48/o96, budget ${UNC_BUDGET_H}h"
  run_sharded unc-roll --run-dir "$RUN_DIR" --model-id "$MODEL_ID" --unc-n "$UNC_N" --n-roll "$N_ROLL" --n-cont "$N_CONT" --max-seqs "$MAX_SEQS" --budget-s "$UNC_BUDGET_S"
  notify "unc-roll DONE in $(elapsed)" "chunks: $(ls "$RUN_DIR"/unc_n${UNC_N}/u*.jsonl 2>/dev/null | wc -l)" ;;
unc-analyze)
  python -m psuite.cli unc-analyze --run-dir "$RUN_DIR" --unc-n "$UNC_N"
  notify "UNC REPORT ready" "$(sed -n '/## 0. DECISION/,/## 2. Label/p' "$RUN_DIR/unc_n${UNC_N}/analysis/UNC_REPORT.md" | cut -c1-400 | head -n 30)" ;;
mega-steps)
  notify "mega-steps STARTED" "edit + regenerate numeric steps, budget ${MEGA_STEPS_H}h"
  run_sharded steps --run-dir "$RUN_DIR" --mega-dir "$MEGA_DIR" --model-id "$MODEL_ID" --chunk "$MEGA_CHUNK" --max-steps "$MEGA_MAX_STEPS" --max-seqs "$MAX_SEQS" --budget-s "$(python3 -c "print(int(float('${MEGA_STEPS_H}')*3600))")"
  notify "mega-steps DONE in $(elapsed)" "chunks: $(ls "$MEGA_DIR"/steps/s*.jsonl 2>/dev/null | wc -l)" ;;
mega-unc)
  notify "mega-unc STARTED" "${MEGA_UNC_N} unseen GSM8K problems x ${MEGA_N_ROLL} rollouts, budget ${MEGA_UNC_H}h"
  run_sharded unc-roll --run-dir "$RUN_DIR" --mega-dir "$MEGA_DIR" --model-id "$MODEL_ID" --unc-datasets gsm8k --unc-n "$MEGA_UNC_N" --unc-skip "$MEGA_UNC_SKIP" --n-roll "$MEGA_N_ROLL" --n-cont 0 --max-seqs "$MAX_SEQS" --budget-s "$(python3 -c "print(int(float('${MEGA_UNC_H}')*3600))")"
  notify "mega-unc DONE in $(elapsed)" "chunks: $(ls "$MEGA_DIR"/unc/u*.jsonl 2>/dev/null | wc -l)" ;;
mega-steer-prep)
  CUDA_VISIBLE_DEVICES=0 python -m psuite.cli steer-prep --run-dir "$RUN_DIR" --mega-dir "$MEGA_DIR"
  notify "mega-steer-prep DONE in $(elapsed)" "$(cat "$MEGA_DIR"/steer/prep.json 2>/dev/null | tr -d '\n' | cut -c1-300)" ;;
mega-steer-run)
  notify "mega-steer-run STARTED" "45 conditions on the GSM8K test problems, budget ${MEGA_STEER_H}h"
  run_sharded steer-run --run-dir "$RUN_DIR" --mega-dir "$MEGA_DIR" --model-id "$MODEL_ID" --max-seqs "$MAX_SEQS" --steer-n "$MEGA_STEER_N" --budget-s "$(python3 -c "print(int(float('${MEGA_STEER_H}')*3600))")"
  notify "mega-steer-run DONE in $(elapsed)" "conditions: $(ls "$MEGA_DIR"/steer/r*.jsonl 2>/dev/null | wc -l)" ;;
mega-analyze)
  CUDA_VISIBLE_DEVICES=0 python -m psuite.cli mega-analyze --run-dir "$RUN_DIR" --mega-dir "$MEGA_DIR"
  notify "MEGA REPORT ready" "$(sed -n '/## Where to pull/,/## Thread 1/p' "$MEGA_DIR/analysis/MEGA_REPORT.md" | cut -c1-600 | head -n 40)" ;;
report)
  python -m psuite.cli report --run-dir "$RUN_DIR"
  notify "REPORT ready" "$(sed -n '/## 3. Scorecard/,/## 4/p' "$RUN_DIR/analysis/REPORT.md" | cut -c1-300 | head -n 40)" ;;
*) echo "unknown STAGE"; exit 2 ;;
esac
