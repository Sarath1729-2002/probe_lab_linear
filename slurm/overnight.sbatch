#!/bin/bash
#SBATCH --job-name=ep_overnight
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4          # 2 models x 2 GPUs (datasets split across each model's 2 GPUs)
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

# Overnight: the v6 pipeline on two new model families (Qwen3-8B with thinking off, Mistral-7B-Instruct-v0.3), on the
# same frozen items and prompts, then the monitor audit (organic/audit.py: KT1 decision vs fragility, KT2 no-pressure
# false alarms) on all four models. Every stage resumes: if the time limit hits, submit again and it continues.
# Models must be downloaded once on the login node first (compute nodes are offline):
#   HF_HUB_OFFLINE=0 hf download Qwen/Qwen3-8B
#   HF_HUB_OFFLINE=0 hf download mistralai/Mistral-7B-Instruct-v0.3
#
#   sbatch slurm/overnight.sbatch
#   AUDIT_ONLY=1 sbatch slurm/overnight.sbatch            # skip generation, audit whatever is on disk
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
set +eu
source ~/errprop_env.sh
set -u
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

CFG=config_organic_v6.yaml
MODELS="${MODELS:-qwen3_8b mistral7b}"
AUDIT_MODELS="${AUDIT_MODELS:-qwen7b llama8b qwen3_8b mistral7b}"
STAGES="${STAGES:-gen gen2 confidence extract score}"
DD=$(python -c "import os,yaml; print(os.path.expandvars(yaml.safe_load(open('$CFG'))['data_dir']))")
[ -f "$DD/variants.jsonl" ] || { echo "no $DD/variants.jsonl: run 'python -m organic.cli build --config $CFG' on the login node"; exit 1; }
mapfile -t SHARDS < <(python -m organic.cli shard --n 2 --config $CFG)
echo "dataset shards per model: ${SHARDS[0]} | ${SHARDS[1]}"

cached() { local d="${HF_HOME}/hub/models--${1//\//--}/snapshots"; [ -d "$d" ] && ls "$d"/*/config.json >/dev/null 2>&1; }
repo_of() { python -c "import yaml,sys; print({m['tag']:m['name'] for m in yaml.safe_load(open('$CFG'))['models']}.get(sys.argv[1],''))" "$1"; }
has() { [[ " $STAGES " == *" $1 "* ]]; }

run() {   # $1 model, $2 gpu, $3 stage, $4 buckets (optional)
  local t=$1 g=$2 s=$3 b=${4:-} log t0
  log="logs/org6-${SLURM_JOB_ID}-${t}-${s}-gpu${g}.log"
  echo "== $(date '+%H:%M:%S') [$t gpu$g] $s ${b}"; t0=$(date +%s)
  if CUDA_VISIBLE_DEVICES=$g python -m organic.cli "$s" --config $CFG --model "$t" ${b:+--buckets $b} --set ${SETS:-} > "$log" 2>&1; then
    echo "== $(date '+%H:%M:%S') [$t] $s done in $(( ($(date +%s) - t0) / 60 )) min"; return 0
  fi
  echo "!! [$t] $s FAILED ($log)"; tail -8 "$log"; return 1
}

model_chain() {   # $1 model, $2 first gpu (uses $2 and $2+1)
  local t=$1 g0=$2 k
  for k in 0 1; do
    ( for s in gen gen2 confidence extract score; do has $s && run "$t" $((g0 + k)) $s "${SHARDS[$k]}"; done ) &
  done
  wait
  has deference && { run "$t" $g0 deference || return; }     # CPU: behaviour, probes, regression, anchors
  has report && run "$t" $g0 report
}

if [ -z "${AUDIT_ONLY:-}" ]; then
  i=0
  for t in $MODELS; do
    r=$(repo_of "$t")
    if [ -z "$r" ] || ! cached "$r"; then echo "SKIP $t ($r not cached in $HF_HOME)"; continue; fi
    model_chain "$t" $((2 * i)) &
    i=$((i + 1))
  done
  wait
fi

# monitor audit, one GPU per model (a1 is CPU; a2 reads the bare questions on the GPU)
audit_chain() { run "$1" $2 a1; run "$1" $2 a2; run "$1" $2 audit_report; grep -h "^\[a1\] .*\(@\|FAILED\)\|^\[a2\]" logs/org6-${SLURM_JOB_ID}-$1-a[12]-gpu$2.log 2>/dev/null; }
i=0
for t in $AUDIT_MODELS; do
  r=$(repo_of "$t")
  if [ -z "$r" ] || ! cached "$r"; then echo "SKIP audit $t ($r not cached)"; continue; fi
  audit_chain "$t" $((i % 4)) &
  i=$((i + 1))
done
wait
echo "ALL DONE $(date)   reports: reports/organic_v6/<model>/REPORT_audit.md"
