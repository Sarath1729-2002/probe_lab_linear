#!/bin/bash
#SBATCH --job-name=ep_hd
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4          # 2 models x 2 GPUs (steering conditions split over each model's 2 GPUs)
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

# Deference vs honesty on the v6 run (organic/honest_defer.py). Needs the v6 activations (acts.npy) on disk.
#   sbatch slurm/hd.sbatch
#   STAGES="hd_report" sbatch slurm/hd.sbatch           # finished files are reused
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
set +eu
source ~/errprop_env.sh
set -u
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"
python -c "import organic.common, organic.honest_defer, organic.deference" || { echo "!! organic package incomplete"; exit 1; }

CFG=config_organic_v6.yaml
MODELS="${MODELS:-qwen7b llama8b}"
STAGES="${STAGES:-hd_dirs hd_steer hd_report}"
has() { [[ " $STAGES " == *" $1 "* ]]; }

run() {   # $1 model, $2 gpu, $3 stage, $4 extra --set
  local t=$1 g=$2 s=$3 x=${4:-} log
  log="logs/hd-${SLURM_JOB_ID}-${t}-${s}-gpu${g}.log"
  echo "== $(date '+%H:%M:%S') [$t gpu$g] $s $x"
  if CUDA_VISIBLE_DEVICES=$g python -m organic.cli "$s" --config $CFG --model "$t" ${x:+--set $x} > "$log" 2>&1; then
    echo "== $(date '+%H:%M:%S') [$t] $s done"; return 0
  fi
  echo "!! [$t] $s FAILED ($log)"; tail -8 "$log"; return 1
}

chain() {   # $1 model, $2 first gpu
  local t=$1 g=$2
  has hd_dirs && { run "$t" $g hd_dirs || return; }
  if has hd_steer; then
    run "$t" $g hd_steer "hd.conditions=[none,def_gate,hon_gate]" &
    run "$t" $((g + 1)) hd_steer "hd.conditions=[def_uniform,hon_uniform,rand_uniform]" &
    wait
  fi
  has hd_report && run "$t" $g hd_report
}

i=0
for t in $MODELS; do chain "$t" $((2 * i)) & i=$((i + 1)); done
wait
python -m organic.cli hd_compare --config $CFG || echo "!! hd_compare failed"
echo "ALL DONE $(date)   reports: reports/organic_v6/COMPARE_hd.md, reports/organic_v6/<model>/REPORT_hd.md, fig_hd.png"
