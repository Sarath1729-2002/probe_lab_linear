#!/bin/bash
#SBATCH --job-name=errprop
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:1      # if TamIA only schedules whole nodes for your account, use h100:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

# Usage:
#   sbatch slurm/tamia.sbatch                         # full look-through run
#   sbatch slurm/tamia.sbatch --limit 50 --set run_name=gate   # feasibility gate first
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

module load StdEnv/2023 python/3.11 cuda/12.2 arrow 2>/dev/null || true
source "${VENV:-$HOME/venvs/errprop}/bin/activate"

export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"   # compute nodes have no internet: pre-download (see README)
export TOKENIZERS_PARALLELISM=false

python -m errprop.cli gpu "$@"   # clean -> inject -> continue -> extract
python -m errprop.cli cpu "$@"   # probes -> report
