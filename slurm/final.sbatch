#!/bin/bash
#SBATCH --job-name=ep_final
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=23:59:00
#SBATCH --output=logs/%x-%j.out

# Everything for the Monday report, in one job. Every stage skips when its output exists: resubmit to resume.
#   Phase 1  know-vs-say (ks): refit the 3 discovery models (post-hoc scores for bare replies) and run the
#            frozen detector on the 3 confirmation models (Llama-3.1-8B, Qwen3-8B, Qwen2.5-14B).
#   Review 1 ks_review + ks_controls + compare on all 6 (CPU)  -> reports/sel_v2/REVIEW_ks.md, CONTROLS_ks.md
#   Phase 2  peer-pressure (kp) and natural-form (kn) runs on Qwen2.5-7B, Mistral-7B, then Llama-3.1-8B
#   Review 2 same CPU analyses for kp and kn, plus the self-knowledge wall
#     sbatch --exclude=tg10807 slurm/final.sbatch
#   Stale fits (no post-hoc probes) are detected and refit automatically; resubmitting is always safe.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
set +eu
source ~/errprop_env.sh
set -u
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "import organic.knowsay, organic.ks_review, sklearn" || { echo "!! imports failed"; exit 1; }
CFG=config_sel_v2.yaml
R="${SCRATCH}/organic_runs/sel_v2"
DISC="qwen3b qwen7b mistral7b"
CONF="llama8b qwen3_8b qwen14b"
echo "commit $(git rev-parse --short HEAD)   $(date)"

# stale fits (written before the post-hoc probes, or before post-hoc scores for bare replies) are deleted and refit
for p in ks kp kn; do
  for t in $DISC $CONF; do
    f="$R/$t/${p}_fit.json"; [ -f "$f" ] || continue
    python - "$f" "$R/$t/${p}_rows.jsonl" <<'PY' || { rm -f "$f"; echo "-- [$p $t] stale fit removed: refit"; }
import json, sys
fj = json.load(open(sys.argv[1]))
ok = "posthoc_layer" in fj
if ok:
    ok = False
    for line in open(sys.argv[2]):
        r = json.loads(line)
        if r.get("fam") == "base" and isinstance(r.get("barereg_lofo"), float):
            ok = True
            break
sys.exit(0 if ok else 1)
PY
  done
done

run() {   # run <pfx> <runarg|-> <tag> <gpu> <done-file|-> <stage>
  local p=$1 ra=$2 t=$3 g=$4 done_file=$5 s=$6 log="logs/final-${SLURM_JOB_ID}-$1-$3.log"
  local RA=(); [ "$ra" != "-" ] && RA=(--run "$ra")
  if [ "$done_file" != "-" ] && [ -f "$R/$t/$done_file" ]; then echo "-- [$p $t] $s: done"; return 0; fi
  echo "== $(date '+%m-%d %H:%M') [$p $t gpu$g] $s"
  local n0; n0=$( (wc -l < "$log") 2>/dev/null || echo 0)
  if CUDA_VISIBLE_DEVICES=$g python -m organic.knowsay "$s" --config $CFG --model "$t" "${RA[@]}" >> "$log" 2>&1; then
    tail -n +$((n0 + 1)) "$log" | grep "^\[ks\]" | head -n 14; return 0
  fi
  echo "!! [$p $t] $s FAILED ($log)"; tail -6 "$log"; return 1
}
chain() {  # chain <pfx> <runarg|-> <tag> <gpu>
  local p=$1 ra=$2 t=$3 g=$4
  [ -f "$R/$t/base.jsonl" ] || { echo "!! [$t] no sel_v2 base.jsonl, skipped"; return 1; }
  run "$p" "$ra" "$t" "$g" ${p}_gen.jsonl gen || return 1
  run "$p" "$ra" "$t" "$g" ${p}_follow_states.npz follow || return 1
  run "$p" "$ra" "$t" "$g" ${p}_fit.json fit || return 1
  run "$p" "$ra" "$t" "$g" ${p}_steer.json steer
  run "$p" "$ra" "$t" "$g" - report
}
review() {  # review <pfx> <models...>
  local p=$1; shift
  python -m organic.ks_review   --config $CFG --prefix "$p" --models "$@" > "logs/final-${SLURM_JOB_ID}-review-$p.log" 2>&1 \
    && echo "-- review $p ok" || { echo "!! review $p FAILED"; tail -8 "logs/final-${SLURM_JOB_ID}-review-$p.log"; }
  python -m organic.ks_controls --config $CFG --prefix "$p" --models "$@" > "logs/final-${SLURM_JOB_ID}-controls-$p.log" 2>&1 \
    && echo "-- controls $p ok" || { echo "!! controls $p FAILED"; tail -8 "logs/final-${SLURM_JOB_ID}-controls-$p.log"; }
}

# ---- phase 1: ks. 14B alone on gpu0; the refits are CPU-heavy and fast, then the 8Bs
( chain ks - qwen14b 0 ) &
( chain ks - llama8b 1 ) &
( chain ks - qwen3_8b 2 ) &
( for t in $DISC; do chain ks - "$t" 3; done ) &
wait
python -m organic.knowsay compare --config $CFG --models $DISC $CONF > "logs/final-${SLURM_JOB_ID}-compare-ks.log" 2>&1 || echo "!! compare ks failed"
review ks $DISC $CONF
echo "== PHASE 1 DONE $(date)   reports/sel_v2/REVIEW_ks.md  CONTROLS_ks.md  KNOWSAY*.md"

# ---- phase 2: peer (kp) and natural forms (kn)
( chain kp knowsay_peer qwen7b 0;    chain kp knowsay_peer llama8b 0 ) &
( chain kp knowsay_peer mistral7b 1 ) &
( chain kn knowsay_nat qwen7b 2;     chain kn knowsay_nat llama8b 2 ) &
( chain kn knowsay_nat mistral7b 3 ) &
wait
python -m organic.knowsay compare --config $CFG --run knowsay_peer --models qwen7b mistral7b llama8b > "logs/final-${SLURM_JOB_ID}-compare-kp.log" 2>&1 || echo "!! compare kp failed"
python -m organic.knowsay compare --config $CFG --run knowsay_nat --models qwen7b mistral7b llama8b > "logs/final-${SLURM_JOB_ID}-compare-kn.log" 2>&1 || echo "!! compare kn failed"
review kp qwen7b mistral7b llama8b
review kn qwen7b mistral7b llama8b
python -m organic.wall --config $CFG > "logs/final-${SLURM_JOB_ID}-wall.log" 2>&1 && echo "-- wall ok" || echo "!! wall failed"
echo "ALL DONE $(date)"
