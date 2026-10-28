#!/bin/bash
#SBATCH --job-name=errprop_full
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4          # TamIA allocates H100s by whole node
#SBATCH --mem=0                         # all node memory (probe sweeps run many workers in parallel)
#SBATCH --time=18:00:00
#SBATCH --output=logs/%x-%j.out

# Full run: every stage in its own Python process (vLLM must release the GPU before transformers loads).
#
#   sbatch slurm/full.sbatch                                   # everything, run_name from config.yaml
#   STAGES="probes steer report" sbatch slurm/full.sbatch      # resume from a stage
#   sbatch slurm/full.sbatch --limit 60 --set run_name=smoke   # 10-problems-per-dataset smoke test
#
# Stage order: clean inject continue extract probes steer report
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

set +eu
source ~/errprop_env.sh
set -eu
echo "python: $(which python)   nvcc: $(which nvcc)   HF_HOME: ${HF_HOME}   offline: ${HF_HUB_OFFLINE:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

STAGES="${STAGES:-clean inject continue extract probes steer report}"
for s in $STAGES; do
  echo "=================== $(date '+%H:%M:%S')  stage: $s ==================="
  t0=$(date +%s)
  python -m errprop.cli "$s" "$@"
  echo "=================== stage $s done in $(( ($(date +%s) - t0) / 60 )) min ==================="
done
echo "ALL DONE $(date)"
