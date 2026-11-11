#!/bin/bash
#SBATCH --job-name=ep_organic_v61
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4          # 2 models x 2 GPUs (datasets split across each model's 2 GPUs)
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out

# organic_v6.1: three checks added to the finished v6 run (same items, same stored replies; observation only).
#   1. no-label controls (m_bare, m_key)  2. reason-first versions of every source (r_*)  3. regenerate cut-off replies
# Only the new prompts and the cut-off replies are generated; then the v6 analysis and report are redone (outcomes of
# extended replies may change) and the v6.1 report is written.
#
#   sbatch slurm/organic_v61.sbatch
#   STAGES="deference report report61" sbatch slurm/organic_v61.sbatch   # CPU part only, files are reused
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
set +eu
source ~/errprop_env.sh
set -u
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

CFG=config_organic_v6.yaml
MODELS="${MODELS:-qwen7b llama8b}"
STAGES="${STAGES:-gen extend deference report report61}"
DD=$(python -c "import os,yaml; print(os.path.expandvars(yaml.safe_load(open('$CFG'))['data_dir']))")
[ -f "$DD/variants.jsonl" ] || { echo "no $DD/variants.jsonl: the v6 run must exist first"; exit 1; }
python -c "import organic.common, organic.run, organic.design61, organic.deference, organic.report_v6" \
  || { echo "!! the organic package is incomplete in $(pwd)/organic: unpack organic_v61_full.tgz there"; exit 1; }
python -m organic.cli variants61 --config $CFG || { echo "!! variants61 failed"; exit 1; }   # appends new prompts once
mapfile -t SHARDS < <(python -m organic.cli shard --n 2 --config $CFG)
echo "dataset shards per model: ${SHARDS[0]} | ${SHARDS[1]}"

cached() { local d="${HF_HOME}/hub/models--${1//\//--}/snapshots"; [ -d "$d" ] && ls "$d"/*/config.json >/dev/null 2>&1; }
repo_of() { python -c "import yaml,sys; print({m['tag']:m['name'] for m in yaml.safe_load(open('$CFG'))['models']}.get(sys.argv[1],''))" "$1"; }
has() { [[ " $STAGES " == *" $1 "* ]]; }

run() {   # $1 model, $2 gpu, $3 stage, $4 buckets (optional)
  local t=$1 g=$2 s=$3 b=${4:-} log t0
  log="logs/org61-${SLURM_JOB_ID}-${t}-${s}-gpu${g}.log"
  echo "== $(date '+%H:%M:%S') [$t gpu$g] $s ${b}"; t0=$(date +%s)
  if CUDA_VISIBLE_DEVICES=$g python -m organic.cli "$s" --config $CFG --model "$t" ${b:+--buckets $b} --set ${SETS:-} > "$log" 2>&1; then
    echo "== $(date '+%H:%M:%S') [$t] $s done in $(( ($(date +%s) - t0) / 60 )) min"; return 0
  fi
  echo "!! [$t] $s FAILED ($log)"; tail -8 "$log"; return 1
}

model_chain() {   # $1 model, $2 first gpu (uses $2 and $2+1)
  local t=$1 g0=$2 k
  for k in 0 1; do
    ( for s in gen extend; do has $s && run "$t" $((g0 + k)) $s "${SHARDS[$k]}"; done ) &
  done
  wait
  has deference && { run "$t" $g0 deference || return; }     # CPU: behaviour, probes, regression, anchors
  has report && run "$t" $g0 report
  has report61 && run "$t" $g0 report61
}

i=0
for t in $MODELS; do
  r=$(repo_of "$t")
  if [ -z "$r" ] || ! cached "$r"; then echo "SKIP $t ($r not cached in $HF_HOME)"; continue; fi
  model_chain "$t" $((2 * i)) &
  i=$((i + 1))
done
wait
python -m organic.cli compare --config $CFG --set ${SETS:-} || echo "!! compare failed"
python -m organic.cli compare61 --config $CFG --set ${SETS:-} || echo "!! compare61 failed"
echo "ALL DONE $(date)   reports: reports/organic_v6/COMPARE_v61.md, reports/organic_v6/<model>/REPORT_v61.md (and the redone v6 REPORT.md)"
