#!/bin/bash
#SBATCH --job-name=ep_organic_v3
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4          # one model per GPU (3 models)
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=07:00:00
#SBATCH --output=logs/%x-%j.out

# organic_v3. Needs, once on the login node:
#   python -m organic.cli variants --config config_organic_v3.yaml
#
#   sbatch slurm/organic_v3.sbatch
#   MODELS="qwen7b" sbatch slurm/organic_v3.sbatch
#   STAGES="fit report" sbatch slurm/organic_v3.sbatch     # finished per-model files are reused
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
set +eu
source ~/errprop_env.sh
set -u
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

CFG=config_organic_v3.yaml
MODELS="${MODELS:-qwen7b llama8b mistral7b}"
STAGES="${STAGES:-gen confidence extract fit traj steer report}"
DD=$(python -c "import os,yaml; print(os.path.expandvars(yaml.safe_load(open('$CFG'))['data_dir']))")
[ -f "$DD/variants.jsonl" ] || { echo "no $DD/variants.jsonl: run 'python -m organic.cli variants --config $CFG' on the login node"; exit 1; }

cached() { local d="${HF_HOME}/hub/models--${1//\//--}/snapshots"; [ -d "$d" ] && ls "$d"/*/config.json >/dev/null 2>&1; }
repo_of() { python -c "import yaml,sys; print({m['tag']:m['name'] for m in yaml.safe_load(open('$CFG'))['models']}.get(sys.argv[1],''))" "$1"; }

chain() {
  local t=$1 g=$2 s log t0
  for s in $STAGES; do
    log="logs/org3-${SLURM_JOB_ID}-${t}-${s}.log"
    echo "== $(date '+%H:%M:%S') [$t gpu$g] $s"; t0=$(date +%s)
    if ! CUDA_VISIBLE_DEVICES=$g python -m organic.cli "$s" --config $CFG --model "$t" --set ${SETS:-} > "$log" 2>&1; then
      echo "!! [$t] $s FAILED ($log)"; tail -8 "$log"
      [ "$s" = "fit" ] && { echo "!! [$t] stopping: later stages need fit"; return; }
      continue
    fi
    echo "== $(date '+%H:%M:%S') [$t] $s done in $(( ($(date +%s) - t0) / 60 )) min"
  done
}

i=0
for t in $MODELS; do
  r=$(repo_of "$t")
  if [ -z "$r" ] || ! cached "$r"; then echo "SKIP $t ($r not cached in $HF_HOME)"; continue; fi
  chain "$t" $i &
  i=$((i + 1))
done
wait
python -m organic.cli compare --config $CFG --set ${SETS:-} || echo "!! compare failed"
echo "ALL DONE $(date)   reports: reports/organic_v3/<model>/REPORT.md and reports/organic_v3/COMPARE.md"
