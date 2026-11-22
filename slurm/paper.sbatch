#!/bin/bash
#SBATCH --job-name=ep_paper
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4          # whole node; each GPU runs a queue of models
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

# sel_v2 paper run (config_sel_v2.yaml). Prerequisite: bash slurm/paper_prep.sh on the login node.
# Per model: bare answers, training renditions, directions + probes, alpha sweep, eval (none / instruction / uniform
# deterrence / ablation / confidence-signed), causal layer sweep + steering there, 7 LoRA fine-tunes (mix frontier
# 0/natural/25/50/75% right claims, balanced x 3 seeds), report. Then the cross-model MEGA.md.
# Every stage is skipped when its output exists, so resubmitting resumes.
#   sbatch slurm/paper.sbatch
#   PLAN="qwen7b|llama8b|mistral7b|qwen3_8b" sbatch slurm/paper.sbatch     # one model per GPU, subset
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
set +eu
source ~/errprop_env.sh
set -u
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
python -c "import organic.common, organic.selective, organic.sel_more, peft, scipy" || { echo "!! imports failed"; exit 1; }
CFG=config_sel_v2.yaml
R="${SCRATCH}/organic_runs/sel_v2"
# GPU queues ("|" separates GPUs, spaces separate models), balanced by size
PLAN="${PLAN:-qwen14b qwen3b|gemma9b olmo7b|qwen3_8b mistral7b|llama8b qwen7b phi35}"

step() {   # step <tag> <gpu> <module> <stage> [skip-if-exists file]
  local t=$1 g=$2 mod=$3 s=$4 done_file=${5:-} log="logs/paper-${SLURM_JOB_ID}-$1.log"
  if [ -n "$done_file" ] && [ -f "$R/$t/$done_file" ]; then echo "-- [$t] $mod $s: done"; return 0; fi
  echo "== $(date '+%m-%d %H:%M') [$t gpu$g] $mod $s"
  local n0; n0=$( (wc -l < "$log") 2>/dev/null || echo 0)
  if CUDA_VISIBLE_DEVICES=$g python -m "organic.$mod" "$s" --config $CFG --model "$t" >> "$log" 2>&1; then
    tail -n +$((n0 + 1)) "$log" | grep "^\[sel\] \(base\|train\|layers\|chosen\)\|^\[more\] \(causal layers\|eval lora_bal \|eval cal_lp_c\)" | head -n 12
    return 0
  fi
  echo "!! [$t] $mod $s FAILED ($log)"; tail -5 "$log"; return 1
}
model_chain() {
  local t=$1 g=$2
  step "$t" "$g" selective base base_states.npz || return 1
  step "$t" "$g" selective press press_states.npz || return 1
  step "$t" "$g" selective fit dirs.npz || return 1
  step "$t" "$g" selective alpha alpha.json
  step "$t" "$g" selective eval
  step "$t" "$g" selective report
  step "$t" "$g" sel_more causal causal.json
  step "$t" "$g" sel_more ceval
  step "$t" "$g" sel_more lora
  step "$t" "$g" sel_more report
}
IFS='|' read -ra Q <<< "$PLAN"
ALL=""
for g in "${!Q[@]}"; do
  ( for t in ${Q[$g]}; do model_chain "$t" "$g"; done ) &
  ALL="$ALL ${Q[$g]}"
done
wait
python -m organic.sel_more compare --config $CFG --models $ALL
python -m organic.selective compare --config $CFG --models $ALL
echo "ALL DONE $(date)   reports/sel_v2/MEGA.md (+ MEGA.json)   reports/sel_v2/<model>/REPORT_more.md"
