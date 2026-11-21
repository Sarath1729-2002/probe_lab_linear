#!/bin/bash
#SBATCH --job-name=ep_mega
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4          # whole node; one GPU per model
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

# Mega exploration on top of sel_v1 (organic/sel_more.py): checking gap (V), claim-signed gate + policies (G),
# LoRA for selectivity (L), causally chosen layers (C); plus the Qwen3 sel_v1 rerun and the no-activation policy check.
#   pip install peft   (login node, once, inside the venv) -- without it the LoRA stage is skipped
#   sbatch slurm/mega.sbatch
# A failed stage does not stop the chain; every stage is resumable.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
set +eu
source ~/errprop_env.sh
set -u
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
python -c "import organic.common, organic.selective, organic.sel_policy, organic.sel_more" || { echo "!! organic package incomplete"; exit 1; }
python -c "import peft" 2>/dev/null || echo "!! peft missing: LoRA stage will be skipped (pip install peft on the login node)"
CFG=config_sel.yaml
MODELS="${MODELS:-qwen7b llama8b mistral7b qwen3_8b}"
MORE="${MORE:-verify vfit veval lora causal ceval report}"

step() {   # step <tag> <gpu> <module> <stage>
  local t=$1 g=$2 mod=$3 s=$4 log="logs/mega-${SLURM_JOB_ID}-$1.log"
  echo "== $(date '+%H:%M:%S') [$t gpu$g] $mod $s"
  local n0; n0=$( (wc -l < "$log") 2>/dev/null || echo 0)
  if CUDA_VISIBLE_DEVICES=$g python -m "organic.$mod" "$s" --config $CFG --model "$t" >> "$log" 2>&1; then
    tail -n +$((n0 + 1)) "$log" | grep "^\[sel\]\|^\[more\]" | grep -v "alpha \|step [0-9]*0 loss\|causal L" | head -n 30
  else
    echo "!! [$t] $mod $s FAILED ($log)"; tail -6 "$log"
  fi
}
chain() {
  local t=$1 g=$2
  if [ ! -f "${SCRATCH}/organic_runs/sel_v1/$t/eval_rand.jsonl" ]; then
    for s in base press fit alpha eval report; do step "$t" "$g" selective "$s"; done
  fi
  for s in $MORE; do step "$t" "$g" sel_more "$s"; done
}
i=0
for t in $MODELS; do chain "$t" $i & i=$((i + 1)); done
wait
python -m organic.sel_policy --config $CFG --models $MODELS
python -m organic.selective compare --config $CFG --models $MODELS
python -m organic.sel_more compare --config $CFG --models $MODELS
echo "ALL DONE $(date)   reports: reports/sel_v1/MEGA.md  POLICY_sel.md  COMPARE_sel.md  <model>/REPORT_more.md"
