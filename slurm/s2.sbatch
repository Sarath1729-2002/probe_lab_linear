#!/bin/bash
#SBATCH --job-name=ep_s2
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4          # 2 models x 2 GPUs (steering conditions split over each model's 2 GPUs)
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

# Steering v2 on the v6 run (organic/steer2.py): held-out items, three layers, a gate calibrated on generated tokens,
# directional ablation, a strength sweep and a random control. Needs the v6 activations (acts.npy) and replies on disk.
#   sbatch slurm/s2.sbatch
#   STAGES="s2_report" sbatch slurm/s2.sbatch           # finished files are reused
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
set +eu
source ~/errprop_env.sh
set -u
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"
python -c "import organic.common, organic.steer2, organic.honest_defer, organic.deference" || { echo "!! organic package incomplete"; exit 1; }

CFG=config_organic_v6.yaml
MODELS="${MODELS:-qwen7b llama8b}"
STAGES="${STAGES:-s2_dirs s2_calib s2_steer s2_report}"
has() { [[ " $STAGES " == *" $1 "* ]]; }

run() {   # $1 model, $2 gpu, $3 stage, $4 extra --set
  local t=$1 g=$2 s=$3 x=${4:-} log
  log="logs/s2-${SLURM_JOB_ID}-${t}-${s}-gpu${g}.log"
  echo "== $(date '+%H:%M:%S') [$t gpu$g] $s $x"
  if CUDA_VISIBLE_DEVICES=$g python -m organic.cli "$s" --config $CFG --model "$t" ${x:+--set $x} > "$log" 2>&1; then
    echo "== $(date '+%H:%M:%S') [$t] $s done"; return 0
  fi
  echo "!! [$t] $s FAILED ($log)"; tail -8 "$log"; return 1
}

chain() {   # $1 model, $2 first gpu
  local t=$1 g=$2
  has s2_dirs && { run "$t" $g s2_dirs || return; }
  has s2_calib && { run "$t" $g s2_calib || return; }
  if has s2_steer; then
    run "$t" $g s2_steer "s2.conditions=[none,ablate,gate,rand_2]" &
    run "$t" $((g + 1)) s2_steer "s2.conditions=[add_0.5,add_1,add_2]" &
    wait
  fi
  has s2_report && run "$t" $g s2_report
}

i=0
for t in $MODELS; do chain "$t" $((2 * i)) & i=$((i + 1)); done
wait
python -m organic.cli s2_compare --config $CFG || echo "!! s2_compare failed"
echo "ALL DONE $(date)   reports: reports/organic_v6/COMPARE_s2.md, fig_s2_tradeoff.png, <model>/REPORT_s2.md"
