#!/bin/bash
# All settings in one place; sourced by every script.  Override on the command line, e.g.  GEN_BUDGET_H=3 MODE=full bash cluster/submit.sh
export SLACK_EMAIL="${SLACK_EMAIL:-}"
export SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-aip-oberman}"
export MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
export RUN_NAME="${RUN_NAME:-qwen7b-psuite}"
export ROOT="${ROOT:-${SCRATCH:-$HOME/scratch}/psuite}"
export VENV="${VENV:-$HOME/venvs/susc}"                       # reuse the existing venv (torch etc. already installed); setup.sh adds this package to it
if [[ -z "${HF_HOME:-}" ]]; then
  if [[ -d "${SCRATCH:-/nonexistent}/susc/hf_home" ]]; then export HF_HOME="$SCRATCH/susc/hf_home"; else export HF_HOME="$ROOT/hf_home"; fi    # reuse the cached model
fi
export RUN_DIR="${RUN_DIR:-$ROOT/outputs/$RUN_NAME}"

# ---- problem sets (all deterministic ground truth; src/psuite/bench.py audits them) ----
if [[ -z "${SIZES:-}" ]]; then SIZES='{"gsm8k":2400,"math":1800,"arc":1200,"arith":1000}'; fi
export SIZES
export DATASETS="${DATASETS:-}"                # comma list to restrict a stage to some datasets (empty = all)
export N_SPREAD="${N_SPREAD:-4}"               # sampled plain solutions per problem (length spread / soft runaway rate); the labels themselves come from greedy decoding
export CHUNK="${CHUNK:-64}"                    # problems per resumable chunk file (gen and trace must use the same value)
export MAX_SEQS="${MAX_SEQS:-96}"              # sequences per generate() call per GPU (H100 80GB)
export FWD_BS="${FWD_BS:-4}"
export GEN_BUDGET_H="${GEN_BUDGET_H:-5}"       # the gen job stops starting new chunks after this many hours and exits cleanly (datasets interleaved); MODE=resume continues
export RESCUE_SHARD="${RESCUE_SHARD:-}"        # MODE=rescue: finish ONE crashed shard's leftover chunks using all GPUs
export ORIG_NSHARDS="${ORIG_NSHARDS:-4}"
export UNC_N="${UNC_N:-400}"                   # uncertainty POC: problems per dataset (gsm8k, math); output goes to $RUN_DIR/unc_n$UNC_N
export N_ROLL="${N_ROLL:-16}"                  # prompt-end rollouts per problem
export N_CONT="${N_CONT:-8}"                   # continuations per checkpoint (o48, o96)
export UNC_BUDGET_H="${UNC_BUDGET_H:-3}"       # rollout job stops starting new chunks after this many hours (resumable)
# ---- mega run (GSM8K only; four independent threads, see docs/MEGA.md) ----
export MEGA_DIR="${MEGA_DIR:-$RUN_DIR/mega}"
export MEGA_STEPS_H="${MEGA_STEPS_H:-4}"       # budget (hours) of the step-edit generation job; resumable, a cut-off leaves a random subset of problems
export MEGA_UNC_H="${MEGA_UNC_H:-3}"           # budget of the uncertainty rollout job
export MEGA_STEER_H="${MEGA_STEER_H:-2}"       # budget of the steering job
export MEGA_UNC_N="${MEGA_UNC_N:-2000}"        # unseen GSM8K problems for the uncertainty thread (the 400 from the POC are skipped)
export MEGA_UNC_SKIP="${MEGA_UNC_SKIP:-400}"
export MEGA_N_ROLL="${MEGA_N_ROLL:-16}"
export MEGA_MAX_STEPS="${MEGA_MAX_STEPS:-8}"   # numeric steps edited per solution
export MEGA_CHUNK="${MEGA_CHUNK:-8}"           # problems per resumable chunk in the steps thread
export MEGA_STEER_N="${MEGA_STEER_N:-0}"       # 0 = all GSM8K test problems
export GATE_LIMIT="${GATE_LIMIT:-64}"          # problems per dataset in MODE=gate

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

# Stage reports. Never fails the job; always prints the subject to the log.
# Channels used if configured (put the secrets in ~/.bashrc as `export ...`, never in the repo):
#   SLACK_WEBHOOK_URL  Slack incoming webhook posting to your DM   (works from Fir/Nibi compute nodes: they have internet)
#   NTFY_TOPIC         private topic name for push notifications via ntfy.sh (phone app)
# If neither is set it falls back to e-mail to SLACK_EMAIL (sendmail, mail, local SMTP) - not available on Fir nodes.
# usage: notify "subject" ["body"]
notify() {
  local subject="[psuite ${RUN_NAME}] $1" body="${2:-}" text sent=0 sm msg
  echo "[notify] $subject"
  text="*${subject}*"$'\n'"${body}"$'\n'"host=$(hostname) job=${SLURM_JOB_ID:-login}"
  if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
    if printf '%s' "${text:0:30000}" | python3 -c 'import json,sys; print(json.dumps({"text": sys.stdin.read()}))' \
         | curl -sS --fail -m 20 -X POST -H 'Content-type: application/json' --data @- "$SLACK_WEBHOOK_URL" >/dev/null 2>&1; then
      sent=1
    else echo "[notify] Slack webhook post failed"; fi
  fi
  if [[ -n "${NTFY_TOPIC:-}" ]]; then
    if curl -sS --fail -m 20 -H "Title: ${subject}" -d "${text:0:3500}" "${NTFY_SERVER:-https://ntfy.sh}/${NTFY_TOPIC}" >/dev/null 2>&1; then
      sent=1
    else echo "[notify] ntfy post failed"; fi
  fi
  [[ $sent == 1 || -z "${SLACK_EMAIL:-}" ]] && return 0
  msg="$(printf 'To: %s\nFrom: %s@%s\nSubject: %s\n\n%s\n' "$SLACK_EMAIL" "${USER:-psuite}" "$(hostname)" "$subject" "$text")"
  sm="$(command -v sendmail 2>/dev/null || ls /usr/sbin/sendmail /usr/lib/sendmail 2>/dev/null | head -n1 || true)"
  if [[ -n "$sm" ]] && printf '%s\n' "$msg" | "$sm" -t 2>/dev/null; then return 0; fi
  if command -v mail >/dev/null 2>&1 && printf '%s\n' "$text" | mail -s "$subject" "$SLACK_EMAIL" 2>/dev/null; then return 0; fi
  if printf '%s\n' "$msg" | python3 -c 'import smtplib,sys
from email import message_from_string
m = message_from_string(sys.stdin.read())
with smtplib.SMTP("localhost", 25, timeout=20) as s:
    s.send_message(m)' 2>/dev/null; then return 0; fi
  echo "[notify] could not deliver the message anywhere; it is only in this log"
  return 0
}

