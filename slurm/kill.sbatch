#!/bin/bash
#SBATCH --job-name=ep_kill
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:2          # one GPU per model
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

# Two kill tests (organic/kill.py). Needs the v6 run and the steering v2 run (s2/dirs.json, calib.json) on disk.
#   sbatch slurm/kill.sbatch
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
set +eu
source ~/errprop_env.sh
set -u
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"
python -c "import organic.common, organic.kill, organic.steer2" || { echo "!! organic package incomplete"; exit 1; }
CFG=config_organic_v6.yaml
MODELS="${MODELS:-qwen7b llama8b}"

run() {
  local t=$1 g=$2 s=$3 log="logs/kill-${SLURM_JOB_ID}-$1-$3.log"
  echo "== $(date '+%H:%M:%S') [$t gpu$g] $s"
  if CUDA_VISIBLE_DEVICES=$g python -m organic.cli "$s" --config $CFG --model "$t" > "$log" 2>&1; then
    echo "== $(date '+%H:%M:%S') [$t] $s done"; grep -h "^\[k[12]\] \(best\|m_user\|m_document\|both\)" "$log"; return 0
  fi
  echo "!! [$t] $s FAILED ($log)"; tail -8 "$log"; return 1
}
chain() { run "$1" $2 k2; run "$1" $2 k1; run "$1" $2 kill_report; }
i=0
for t in $MODELS; do chain "$t" $i & i=$((i + 1)); done
wait
echo "ALL DONE $(date)   reports: reports/organic_v6/<model>/REPORT_kill.md"
