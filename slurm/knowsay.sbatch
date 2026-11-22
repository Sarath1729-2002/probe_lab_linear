#!/bin/bash
#SBATCH --job-name=ep_knowsay
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x-%j.out

# Know vs say (organic/knowsay.py): 10 pressure families, belief readout (categorical + regression), 7 detectors on
# held-out families, recovery, route cross-ablation, gated correction. Reuses sel_v2 items + bare answers, so each
# model needs reports from the sel_v2 base stage (already done for the 6 below).
#   sbatch --exclude=tg10807 slurm/knowsay.sbatch
#   RUN=knowsay_peer PFX=kp sbatch --exclude=tg10807 slurm/knowsay.sbatch      # peer-pressure taxonomy (17 families)
# Every stage is skipped when its output exists; resubmitting resumes.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
set +eu
source ~/errprop_env.sh
set -u
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
python -c "import organic.knowsay, sklearn" || { echo "!! imports failed"; exit 1; }
CFG=config_sel_v2.yaml
R="${SCRATCH}/organic_runs/sel_v2"
PLAN="${PLAN:-qwen14b|llama8b qwen3b|qwen7b|mistral7b qwen3_8b}"
PFX="${PFX:-ks}"
RUNARG=(); [ -n "${RUN:-}" ] && RUNARG=(--run "$RUN")

run() {   # run <tag> <gpu> <done-file or -> <stage>
  local t=$1 g=$2 done_file=$3 s=$4 log="logs/knowsay-${PFX}-${SLURM_JOB_ID}-$1.log"
  if [ "$done_file" != "-" ] && [ -f "$R/$t/$done_file" ]; then echo "-- [$t] $s: done"; return 0; fi
  echo "== $(date '+%m-%d %H:%M') [$t gpu$g] $s"
  local n0; n0=$( (wc -l < "$log") 2>/dev/null || echo 0)
  if CUDA_VISIBLE_DEVICES=$g python -m organic.knowsay "$s" --config $CFG --model "$t" "${RUNARG[@]}" >> "$log" 2>&1; then
    tail -n +$((n0 + 1)) "$log" | grep "^\[ks\]" | head -n 14; return 0
  fi
  echo "!! [$t] $s FAILED ($log)"; tail -6 "$log"; return 1
}
chain() {
  local t=$1 g=$2
  [ -f "$R/$t/base.jsonl" ] || { echo "!! [$t] no sel_v2 base.jsonl, skipped"; return 1; }
  run "$t" "$g" ${PFX}_gen.jsonl gen || return 1
  run "$t" "$g" ${PFX}_follow_states.npz follow || return 1
  run "$t" "$g" ${PFX}_fit.json fit || return 1
  run "$t" "$g" ${PFX}_steer.json steer
  run "$t" "$g" - report
}
IFS='|' read -ra Q <<< "$PLAN"
ALL=""
for g in "${!Q[@]}"; do
  ( for t in ${Q[$g]}; do chain "$t" "$g"; done ) &
  ALL="$ALL ${Q[$g]}"
done
wait
python -m organic.knowsay compare --config $CFG --models $ALL "${RUNARG[@]}"
echo "ALL DONE $(date)   reports/sel_v2/KNOWSAY*.md   reports/sel_v2/<model>/REPORT_knowsay*.md"
