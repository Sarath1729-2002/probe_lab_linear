#!/bin/bash
#SBATCH --job-name=ep_paper
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4          # one node, one GPU per model, all models in parallel
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=12:00:00          # R1-Distill (long traces) sets the pace; the 3 others finish well before
#SBATCH --output=logs/%x-%j.out

# Level-1 "How Much is Left?" recreation for error propagation. Probing only, no steering.
# Each model runs its own stage chain on its own GPU; CPU cores are split between them for `fit`.
# Models whose weights are not cached are skipped (gated Llama/Mistral need HF_TOKEN at predownload).
# Needs the prepared datasets in $SCRATCH/errprop_data (scripts/prepare_data.py on the login node).
#
#   sbatch slurm/paper.sbatch                              # all cached models in config_paper.yaml
#   MODELS="qwen7b r1qwen7b" sbatch slurm/paper.sbatch       # a subset
#   STAGES="fit trajectory" sbatch slurm/paper.sbatch      # resume (finished files are reused)
#   (H200 node: sbatch --gpus-per-node=h200:8 --cpus-per-task=64 slurm/paper.sbatch)
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
set +eu
source ~/errprop_env.sh
set -u
nvidia-smi -i 0 --query-gpu=name,memory.total --format=csv,noheader || true

CFG=config_paper.yaml
MODELS="${MODELS:-qwen7b r1qwen7b llama8b mistral7b}"
STAGES="${STAGES:-clean inject continue extract fit trajectory}"
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NGPU" -lt 1 ] && NGPU=1

cached() {   # model repo id -> 0 if its weights are in the HF cache
  local d="${HF_HOME}/hub/models--${1//\//--}/snapshots"
  [ -d "$d" ] && ls "$d"/*/config.json >/dev/null 2>&1
}
repo_of() { python - "$1" <<'PY'
import sys, yaml
m = {x["tag"]: x["name"] for x in yaml.safe_load(open("config_paper.yaml"))["models"]}
print(m.get(sys.argv[1], ""))
PY
}

RUN=()
for t in $MODELS; do
  r=$(repo_of "$t")
  if [ -z "$r" ]; then echo "SKIP $t: not in $CFG"; continue; fi
  if cached "$r"; then RUN+=("$t"); else echo "SKIP $t: $r not cached (run scripts/predownload_paper.py)"; fi
done
[ ${#RUN[@]} -eq 0 ] && { echo "no cached models"; exit 1; }
ls "${SCRATCH}/errprop_data/"*.jsonl >/dev/null 2>&1 || echo "WARNING: no prepared datasets in \$SCRATCH/errprop_data (MATH/MMLU-Pro/OpenThoughts/TriviaQA will be skipped)"
NJ=$(( ${SLURM_CPUS_PER_TASK:-8} / ${#RUN[@]} )); [ "$NJ" -lt 1 ] && NJ=1
echo "models: ${RUN[*]}   gpus: $NGPU   cpu workers per model: $NJ"

chain() {    # $1 tag, $2 gpu index
  local t=$1 g=$2 s t0
  for s in $STAGES; do
    echo "== $(date '+%H:%M:%S') [$t] stage $s"
    t0=$(date +%s)
    if ! CUDA_VISIBLE_DEVICES=$g python -m errprop.cli "$s" --config $CFG --model "$t" --set probes.n_jobs=$NJ \
         > "logs/ep_${t}-${SLURM_JOB_ID}-${s}.log" 2>&1; then
      echo "!! [$t] stage $s FAILED (logs/ep_${t}-${SLURM_JOB_ID}-${s}.log)"; tail -5 "logs/ep_${t}-${SLURM_JOB_ID}-${s}.log"
      [ "$s" = trajectory ] && continue     # trajectory is a figure; the tables don't need it
      return 1
    fi
    echo "== $(date '+%H:%M:%S') [$t] stage $s done in $(( ($(date +%s) - t0) / 60 )) min"
  done
}

i=0
for t in "${RUN[@]}"; do
  chain "$t" $(( i % NGPU )) &
  i=$((i + 1))
  # more models than GPUs: wait for a batch before starting the next
  [ $(( i % NGPU )) -eq 0 ] && wait
done
wait
echo "== $(date '+%H:%M:%S') writing report"
python -m errprop.cli paper --config $CFG || echo "!! paper failed; rerun on login node: python -m errprop.cli paper --config $CFG"
echo "ALL DONE $(date)   report: reports/paper_v1/PAPER.md"
