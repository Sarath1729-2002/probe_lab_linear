#!/bin/bash
# From the repo root on the login node AFTER  bash cluster/setup.sh
#   MODE=gate bash cluster/submit.sh    # ~240 problems, all stages + report (<1h): is anything there? READ IT FIRST
#   MODE=full bash cluster/submit.sh    # states -> sample -> introspect -> analyze  (chained; resubmit is safe: finished chunks are skipped)
#   MODE=analyze bash cluster/submit.sh # just re-run the first-round analysis
#   MODE=v3gate bash cluster/submit.sh  # main study on 240 problems, all 8 conditions, whole chain (~3-4 h incl. queue): dry run of everything
#   MODE=v3 bash cluster/submit.sh      # main study: 5 new framings -> layer sweep + LOFO -> steering -> analysis_v3/REPORT.md
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
source cluster/config.sh
[[ -f "$RUN_DIR/problems.jsonl" ]] || { echo "run bash cluster/setup.sh first"; exit 1; }
MODE="${MODE:-gate}"
COMMON=(--parsable --account="$SBATCH_ACCOUNT" --export=ALL --mem=0)
[[ -n "$SLACK_EMAIL" ]] && COMMON+=(--mail-user="$SLACK_EMAIL" --mail-type=FAIL,TIME_LIMIT_80)
sub() { local stage="$1" t="$2" dep="$3"; shift 3
  local d=(); [[ -n "$dep" ]] && d=(--dependency=afterok:"$dep")
  STAGE="$stage" sbatch "${COMMON[@]}" --job-name="plab-$stage" --time="$t" ${d[@]+"${d[@]}"} "$@" cluster/run.sbatch; }
ONE=(--gpus-per-node=h100:4 --cpus-per-task=24)   # TamIA allocates whole nodes only
FOUR=(--gpus-per-node=h100:4 --cpus-per-task=24)
if [[ "$MODE" == "gate" ]]; then
  export GATE=1
  st=$(sub states 01:00:00 "" "${FOUR[@]}"); sa=$(sub sample 01:30:00 "$st" "${FOUR[@]}"); it=$(sub introspect 01:00:00 "$sa" "${ONE[@]}")
  an=$(sub analyze 00:30:00 "$it" "${ONE[@]}")
  echo "gate: states=$st sample=$sa introspect=$it analyze=$an   -> ${RUN_DIR}-gate/analysis/report.md"; exit 0
fi
if [[ "$MODE" == "v3" || "$MODE" == "v3gate" ]]; then
  # main study: 5 NEW framings (the clean/auth/user data from the first run are reused) -> analysis -> steering -> REPORT.md
  if [[ "$MODE" == "v3gate" ]]; then
    export GATE=1 CONDS_RUN="clean,auth,user,weak,peer,teacher,grader,pressure" TAG="_g" STEER_N=60
    T_ST=01:00:00; T_SA=02:30:00; T_AN=01:00:00; T_SV=01:30:00
  else
    export CONDS_RUN="weak,peer,teacher,grader,pressure" TAG="_v3"
    T_ST=03:00:00; T_SA=14:00:00; T_AN=03:00:00; T_SV=05:00:00
  fi
  st=$(sub states "$T_ST" "" "${FOUR[@]}")
  sa=$(sub sample "$T_SA" "$st" "${FOUR[@]}")
  an=$(sub analyze3 "$T_AN" "$sa" "${ONE[@]}")
  sv=$(sub steer "$T_SV" "$an" "${FOUR[@]}")
  rp=$(sub report3 00:30:00 "$sv" "${ONE[@]}")
  echo "$MODE: states=$st sample=$sa analyze3=$an steer=$sv report3=$rp   -> ${RUN_DIR}$([[ $MODE == v3gate ]] && echo -gate)/analysis_v3/REPORT.md"
  notify "$MODE SUBMITTED" "states=$st sample=$sa analyze3=$an steer=$sv report3=$rp"; exit 0
fi
if [[ "$MODE" == "analyze" ]]; then an=$(sub analyze 00:45:00 "" "${ONE[@]}"); echo "analyze=$an"; exit 0; fi
st=$(sub states 03:00:00 "" "${FOUR[@]}")
sa=$(sub sample 14:00:00 "$st" "${FOUR[@]}")
it=$(sub introspect 02:00:00 "$sa" "${ONE[@]}")
an=$(sub analyze 00:45:00 "$it" "${ONE[@]}")
echo "full: states=$st sample=$sa introspect=$it analyze=$an   (squeue -u \$USER)"; notify "full pipeline SUBMITTED" "states=$st sample=$sa introspect=$it analyze=$an"
