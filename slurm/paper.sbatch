#!/bin/bash
#SBATCH --job-name=ep_paper
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4          # whole node; each GPU runs a queue of models
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

# sel_v2 paper run (config_sel_v2.yaml). Prerequisite on the login node: bash slurm/paper_prep.sh
# Phase 1 (every model first): bare answers, training renditions, directions, unsteered eval, then the honesty study
#   (instructed lying, post-conversation truth readout, geometry, cross-ablation) -> reports/sel_v2/HONESTY.md
# Phase 2 (support): alpha sweep, steering eval, causal layers, 7 LoRA fine-tunes -> reports/sel_v2/MEGA.md
# Finished stages are skipped, so resubmitting resumes.
#   sbatch slurm/paper.sbatch
#   PHASES=1 sbatch slurm/paper.sbatch                         # honesty study only
#   PLAN="qwen7b|llama8b" sbatch slurm/paper.sbatch            # subset; "|" separates GPUs
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
set +eu
source ~/errprop_env.sh
set -u
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
python -c "import organic.common, organic.selective, organic.sel_more, organic.honesty, peft, scipy" || { echo "!! imports failed"; exit 1; }
CFG=config_sel_v2.yaml
R="${SCRATCH}/organic_runs/sel_v2"
PLAN="${PLAN:-qwen32b qwen3b|qwen14b|llama8b qwen7b|mistral7b qwen3_8b llama3b}"
PHASES="${PHASES:-1 2}"
NO_FT="qwen32b"                          # honesty study only

run() {   # run <tag> <gpu> <done-file or -> <module> <stage> [extra args]
  local t=$1 g=$2 done_file=$3 mod=$4 s=$5; shift 5
  local log="logs/paper-${SLURM_JOB_ID}-$t.log"
  if [ "$done_file" != "-" ] && [ -f "$R/$t/$done_file" ]; then echo "-- [$t] $mod $s: done"; return 0; fi
  echo "== $(date '+%m-%d %H:%M') [$t gpu$g] $mod $s $*"
  local n0; n0=$( (wc -l < "$log") 2>/dev/null || echo 0)
  if CUDA_VISIBLE_DEVICES=$g python -m "organic.$mod" "$s" --config $CFG --model "$t" "$@" >> "$log" 2>&1; then
    tail -n +$((n0 + 1)) "$log" | grep "^\[sel\] \(base\|train\|layers\)\|^\[hon\]\|^\[more\] \(causal layers\|eval lora_bal \)" | head -n 12
    return 0
  fi
  echo "!! [$t] $mod $s FAILED ($log)"; tail -5 "$log"; return 1
}
phase1() {
  local t=$1 g=$2
  run "$t" "$g" base_states.npz selective base || return 1
  run "$t" "$g" press_states.npz selective press || return 1
  run "$t" "$g" dirs.npz selective fit || return 1
  run "$t" "$g" eval_none.jsonl selective eval --conds none || return 1
  run "$t" "$g" lie_states.npz honesty lie || return 1
  run "$t" "$g" post_states.npz honesty post || return 1
  run "$t" "$g" hfit.json honesty hfit || return 1
  run "$t" "$g" hsteer.json honesty hsteer
  run "$t" "$g" - honesty report
}
phase2() {
  local t=$1 g=$2
  case " $NO_FT " in *" $t "*) return 0;; esac
  run "$t" "$g" alpha.json selective alpha
  run "$t" "$g" - selective eval
  run "$t" "$g" - selective report
  run "$t" "$g" causal.json sel_more causal
  run "$t" "$g" - sel_more ceval
  run "$t" "$g" - sel_more lora
  run "$t" "$g" - sel_more report
}
IFS='|' read -ra Q <<< "$PLAN"
ALL=""
for g in "${!Q[@]}"; do
  ( for ph in $PHASES; do for t in ${Q[$g]}; do "phase$ph" "$t" "$g"; done; done ) &
  ALL="$ALL ${Q[$g]}"
done
wait
python -m organic.honesty compare --config $CFG --models $ALL
python -m organic.sel_more compare --config $CFG --models $ALL
python -m organic.selective compare --config $CFG --models $ALL
echo "ALL DONE $(date)   reports/sel_v2/HONESTY.md  reports/sel_v2/MEGA.md  reports/sel_v2/<model>/REPORT_honesty.md"
