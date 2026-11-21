#!/bin/bash
#SBATCH --job-name=ep_sel
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4          # TamIA allocates H100s by whole node; one GPU per model
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out

# Selective deference (organic/selective.py). Needs: python -m organic.selective build (login node) done first.
#   sbatch slurm/sel.sbatch
#   MODELS="qwen14b" sbatch slurm/sel.sbatch
#   STAGES="eval report" sbatch slurm/sel.sbatch            # resume (eval skips finished conditions)
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
set +eu
source ~/errprop_env.sh
set -u
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
python -c "import organic.common, organic.selective" || { echo "!! organic package incomplete"; exit 1; }
CFG=config_sel.yaml
MODELS="${MODELS:-qwen7b llama8b mistral7b qwen3_8b}"
STAGES="${STAGES:-base press fit alpha eval report}"

chain() {
  local t=$1 g=$2 log="logs/sel-${SLURM_JOB_ID}-$1.log"
  for s in $STAGES; do
    echo "== $(date '+%H:%M:%S') [$t gpu$g] $s"
    local n0; n0=$( (wc -l < "$log") 2>/dev/null || echo 0)
    if ! CUDA_VISIBLE_DEVICES=$g python -m organic.selective "$s" --config $CFG --model "$t" >> "$log" 2>&1; then
      echo "!! [$t] $s FAILED ($log)"; tail -8 "$log"; return 1
    fi
    tail -n +$((n0 + 1)) "$log" | grep "^\[sel\]" | head -n 40
  done
}
i=0
for t in $MODELS; do chain "$t" $i & i=$((i + 1)); done
wait
python -m organic.selective compare --config $CFG --models $MODELS
echo "ALL DONE $(date)   reports: reports/sel_v1/<model>/REPORT_sel.md   reports/sel_v1/COMPARE_sel.md"
