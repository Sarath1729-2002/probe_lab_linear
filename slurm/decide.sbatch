#!/bin/bash
#SBATCH --job-name=ep_decide
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4          # one GPU per model, 3 models in parallel
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

# Decision run (threads A, B, C). Needs:
#   - scripts/prepare_decide.py run on the login node ($SCRATCH/decide_data)
#   - paper_v1 run dirs in $SCRATCH/errprop_runs (for A1 and A2's injections)
#   sbatch slurm/decide.sbatch
#   MODELS="qwen7b" sbatch slurm/decide.sbatch
#   STAGES="fit" sbatch slurm/decide.sbatch          # resume / refit only
#   SETS="a2.n_items=500 b.popqa_n=2000" sbatch ...   # overrides
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
set +eu
source ~/errprop_env.sh
set -u
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

CFG=config_decide.yaml
MODELS="${MODELS:-qwen7b llama8b mistral7b}"
STAGES="${STAGES:-a1 a2_gen a2_extract b_know b_extract c_gen c_extract fit}"
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NGPU" -lt 1 ] && NGPU=1
[ -f "${SCRATCH}/decide_data/b_items.jsonl" ] || { echo "run scripts/prepare_decide.py on the login node first"; exit 1; }

cached() { local d="${HF_HOME}/hub/models--${1//\//--}/snapshots"; [ -d "$d" ] && ls "$d"/*/config.json >/dev/null 2>&1; }
repo_of() { python -c "import yaml,sys; print({m['tag']:m['name'] for m in yaml.safe_load(open('$CFG'))['models']}.get(sys.argv[1],''))" "$1"; }

RUN=()
for t in $MODELS; do
  r=$(repo_of "$t")
  if [ -n "$r" ] && cached "$r"; then RUN+=("$t"); else echo "SKIP $t (not in config or not cached)"; fi
done
[ ${#RUN[@]} -eq 0 ] && { echo "no cached models"; exit 1; }
echo "models: ${RUN[*]}   gpus: $NGPU"

chain() {
  local t=$1 g=$2 s t0 log
  for s in $STAGES; do
    log="logs/dec_${t}-${SLURM_JOB_ID}-${s}.log"
    echo "== $(date '+%H:%M:%S') [$t] $s"; t0=$(date +%s)
    if ! CUDA_VISIBLE_DEVICES=$g python -m decide.cli "$s" --config $CFG --model "$t" --set ${SETS:-} > "$log" 2>&1; then
      echo "!! [$t] $s FAILED ($log)"; tail -5 "$log"
      continue                # threads are independent: a failed stage does not stop the others
    fi
    echo "== $(date '+%H:%M:%S') [$t] $s done in $(( ($(date +%s) - t0) / 60 )) min"
  done
}

i=0
for t in "${RUN[@]}"; do
  chain "$t" $(( i % NGPU )) &
  i=$((i + 1)); [ $(( i % NGPU )) -eq 0 ] && wait
done
wait
python -m decide.cli report --config $CFG --set ${SETS:-}
echo "ALL DONE $(date)   report: reports/decide_v1/DECISION.md"
