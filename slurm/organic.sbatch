#!/bin/bash
#SBATCH --job-name=ep_organic
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4          # buckets split over the 4 GPUs; one Qwen copy per GPU
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

# Wrong information in the prompt, observed only. Needs the login-node step first:
#   python -m organic.cli download build --config config_organic.yaml
#
#   sbatch slurm/organic.sbatch
#   STAGES="fit report" sbatch slurm/organic.sbatch     # refit only (finished per-bucket files are reused)
#   SETS="run_prefix=organic_v1b" sbatch ...             # overrides
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
set +eu
source ~/errprop_env.sh
set -u
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

CFG=config_organic.yaml
STAGES="${STAGES:-gen extract fit traj report}"
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NGPU" -lt 1 ] && NGPU=1
DD=$(python -c "import os,yaml; print(os.path.expandvars(yaml.safe_load(open('$CFG'))['data_dir']))")
[ -f "$DD/variants.jsonl" ] || { echo "no $DD/variants.jsonl: run 'python -m organic.cli download build' on the login node"; exit 1; }
mapfile -t SHARDS < <(python -m organic.cli shard --n "$NGPU" --config $CFG)
echo "gpus: $NGPU"; printf '  shard: %s\n' "${SHARDS[@]}"

has() { [[ " $STAGES " == *" $1 "* ]]; }
gpu_stages() {   # $1 gpu, $2 "bucket list", rest: stages
  local g=$1 b=$2; shift 2
  for s in "$@"; do
    has "$s" || continue
    local log="logs/org-${SLURM_JOB_ID}-${s}-gpu${g}.log"
    echo "== $(date '+%H:%M:%S') gpu$g $s: $b"
    CUDA_VISIBLE_DEVICES=$g python -m organic.cli "$s" --config $CFG --buckets $b --set ${SETS:-} > "$log" 2>&1 \
      || { echo "!! gpu$g $s FAILED ($log)"; tail -5 "$log"; }
  done
}

i=0; for b in "${SHARDS[@]}"; do [ -n "$b" ] && gpu_stages $i "$b" gen extract & i=$((i+1)); done; wait
if has fit; then
  echo "== $(date '+%H:%M:%S') fit"
  python -m organic.cli fit --config $CFG --set ${SETS:-} > "logs/org-${SLURM_JOB_ID}-fit.log" 2>&1 \
    || { echo "!! fit FAILED"; tail -20 "logs/org-${SLURM_JOB_ID}-fit.log"; }
fi
i=0; for b in "${SHARDS[@]}"; do [ -n "$b" ] && gpu_stages $i "$b" traj & i=$((i+1)); done; wait
if has report; then
  python -m organic.cli report --config $CFG --set ${SETS:-} || echo "!! report failed: rerun 'python -m organic.cli report' on the login node"
fi
echo "ALL DONE $(date)   report: reports/organic_v1/REPORT.md"
