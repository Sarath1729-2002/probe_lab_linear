#!/bin/bash
# From the repo root on the login node AFTER  bash cluster/setup.sh
#   MODE=gate    bash cluster/submit.sh   # optional dry run: GATE_LIMIT problems per dataset, every stage, ~1.5h incl. queue -> ${RUN_DIR}-gate/analysis/REPORT.md
#   MODE=full    bash cluster/submit.sh   # the study: states -> gen -> trace -> analyze -> report, chained (afterok).  Resubmitting is safe: finished chunks are skipped.
#   MODE=resume  bash cluster/submit.sh   # continue after the gen time budget cut the run short (gen -> trace -> analyze -> report)
#   MODE=rescue  RESCUE_SHARD=2 bash cluster/submit.sh   # a shard crashed: finish ITS leftover gen chunks on all 4 GPUs, then trace -> analyze -> report
#   MODE=analyze bash cluster/submit.sh   # re-run analysis + report on what exists
#   MODE=report  bash cluster/submit.sh   # just rebuild REPORT.md from the analysis pickles
#   MODE=unc     bash cluster/submit.sh   # UNCERTAINTY POC (needs the finished main run): rollouts on 4 GPUs -> analysis on 1 GPU -> $RUN_DIR/unc_n$UNC_N/analysis/UNC_REPORT.md
#   MODE=unc-analyze bash cluster/submit.sh   # redo only the POC analysis + report
#   MODE=mega-smoke bash cluster/submit.sh    # ~40 min dry run of the whole mega chain on tiny budgets -> $RUN_DIR/mega_smoke   (run this first if you are awake)
#   MODE=mega    bash cluster/submit.sh       # the overnight GSM8K mega run: 4 independent threads (steps / unc / steer / suite) -> $MEGA_DIR/analysis/MEGA_REPORT.md
#   MODE=mega-analyze bash cluster/submit.sh  # redo only the mega analysis + report
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
source cluster/config.sh
[[ -f "$RUN_DIR/problems.jsonl" ]] || { echo "run bash cluster/setup.sh first"; exit 1; }
MODE="${MODE:-gate}"
COMMON=(--parsable --account="$SBATCH_ACCOUNT" --export=ALL --mem=0)
[[ -n "$SLACK_EMAIL" ]] && COMMON+=(--mail-user="$SLACK_EMAIL" --mail-type=FAIL,TIME_LIMIT_80)
sub() { local stage="$1" t="$2" dep="$3"; shift 3
  local d=(); [[ -n "$dep" ]] && d=(--dependency="${DEPTYPE:-afterok}":"$dep")
  STAGE="$stage" sbatch "${COMMON[@]}" --job-name="psuite-$stage" --time="$t" ${d[@]+"${d[@]}"} "$@" cluster/run.sbatch; }
NODE=(--gpus-per-node=h100:4 --cpus-per-task=24)   # TamIA allocates whole nodes only
GEN_T="$(python3 -c "import math;h=float('${GEN_BUDGET_H}')+0.75;print('%02d:%02d:00'%(int(h),int((h-int(h))*60)))")"
st="-"; ge="-"; tr="-"; an="-"; rp="-"
case "$MODE" in
gate)
  export GATE=1
  st=$(sub states 01:00:00 "" "${NODE[@]}"); ge=$(sub gen 02:00:00 "$st" "${NODE[@]}"); tr=$(sub trace 01:00:00 "$ge" "${NODE[@]}")
  an=$(sub analyze 01:30:00 "$tr" "${NODE[@]}"); rp=$(sub report 00:20:00 "$an" "${NODE[@]}")
  echo "gate: states=$st gen=$ge trace=$tr analyze=$an report=$rp -> ${RUN_DIR}-gate/analysis/REPORT.md"; exit 0 ;;
full)
  st=$(sub states 01:30:00 "" "${NODE[@]}"); ge=$(sub gen "$GEN_T" "$st" "${NODE[@]}"); tr=$(sub trace 03:00:00 "$ge" "${NODE[@]}")
  an=$(sub analyze 04:00:00 "$tr" "${NODE[@]}"); rp=$(sub report 00:30:00 "$an" "${NODE[@]}") ;;
resume)
  ge=$(sub gen "$GEN_T" "" "${NODE[@]}"); tr=$(sub trace 03:00:00 "$ge" "${NODE[@]}")
  an=$(sub analyze 04:00:00 "$tr" "${NODE[@]}"); rp=$(sub report 00:30:00 "$an" "${NODE[@]}") ;;
rescue)
  [[ -n "${RESCUE_SHARD:-}" ]] || { echo "set RESCUE_SHARD=<crashed shard number>"; exit 1; }
  ge=$(sub gen "$GEN_T" "" "${NODE[@]}"); tr=$(sub trace 03:00:00 "$ge" "${NODE[@]}")
  an=$(sub analyze 04:00:00 "$tr" "${NODE[@]}"); rp=$(sub report 00:30:00 "$an" "${NODE[@]}") ;;
analyze)
  an=$(sub analyze 04:00:00 "" "${NODE[@]}"); rp=$(sub report 00:30:00 "$an" "${NODE[@]}") ;;
report)
  rp=$(sub report 00:30:00 "" "${NODE[@]}") ;;
unc)
  ur=$(sub unc-roll "$(python3 -c "h=float('${UNC_BUDGET_H}')+0.5;print('%02d:%02d:00'%(int(h),int((h-int(h))*60)))")" "" "${NODE[@]}")
  ua=$(sub unc-analyze 01:00:00 "$ur" "${NODE[@]}"); st="$ur"; ge="$ua"
  echo "unc: roll=$ur analyze=$ua -> $RUN_DIR/unc_n${UNC_N}/analysis/UNC_REPORT.md"; notify "unc SUBMITTED" "roll=$ur analyze=$ua"; exit 0 ;;
mega|mega-smoke)
  if [[ "$MODE" == "mega-smoke" ]]; then
    export MEGA_DIR="$RUN_DIR/mega_smoke" MEGA_STEPS_H=0.15 MEGA_UNC_H=0.15 MEGA_STEER_H=0.2 MEGA_UNC_N=96 MEGA_STEER_N=48 MEGA_CHUNK=4
    T_STEPS=01:00:00; T_UNC=01:00:00; T_STEER=01:00:00; T_AN=01:00:00
  else
    hh() { python3 -c "h=float('$1')+0.75;print('%02d:%02d:00'%(int(h),int((h-int(h))*60)))"; }
    T_STEPS=$(hh "$MEGA_STEPS_H"); T_UNC=$(hh "$MEGA_UNC_H"); T_STEER=$(hh "$MEGA_STEER_H"); T_AN=03:00:00
  fi
  j1=$(sub mega-steps "$T_STEPS" "" "${NODE[@]}")
  j2=$(sub mega-unc "$T_UNC" "" "${NODE[@]}")
  j3=$(sub mega-steer-prep 00:45:00 "" "${NODE[@]}")
  j4=$(DEPTYPE=afterany sub mega-steer-run "$T_STEER" "$j3" "${NODE[@]}")      # afterany: a failed prep must not leave steer-run (and so the analysis) pending forever; steer-run then fails fast without dirs.npz
  ja=$(STAGE=mega-analyze sbatch "${COMMON[@]}" --job-name="psuite-mega-analyze" --time="$T_AN" --dependency="afterany:$j1:$j2:$j4" "${NODE[@]}" cluster/run.sbatch)
  echo "$MODE: steps=$j1 unc=$j2 steer-prep=$j3 steer-run=$j4 analyze=$ja  -> $MEGA_DIR/analysis/MEGA_REPORT.md   (squeue -u \$USER)"
  notify "$MODE SUBMITTED" "steps=$j1 unc=$j2 prep=$j3 steer=$j4 analyze=$ja (the analysis runs when ALL threads have ended, successfully or not)"; exit 0 ;;
mega-analyze)
  ja=$(sub mega-analyze 03:00:00 "" "${NODE[@]}"); echo "mega-analyze: $ja -> $MEGA_DIR/analysis/MEGA_REPORT.md"; exit 0 ;;
unc-analyze)
  ua=$(sub unc-analyze 01:00:00 "" "${NODE[@]}")
  echo "unc-analyze: $ua -> $RUN_DIR/unc_n${UNC_N}/analysis/UNC_REPORT.md"; exit 0 ;;
*) echo "unknown MODE"; exit 2 ;;
esac
echo "$MODE: states=$st gen=$ge trace=$tr analyze=$an report=$rp   -> $RUN_DIR/analysis/REPORT.md   (squeue -u \$USER)"
notify "$MODE SUBMITTED" "states=$st gen=$ge trace=$tr analyze=$an report=$rp"
