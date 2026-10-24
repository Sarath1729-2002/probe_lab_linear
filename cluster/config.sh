#!/bin/bash
# All settings in one place; sourced by every script.  Override on the command line, e.g.  N_SAMPLES=8 bash cluster/submit.sh
export SLACK_EMAIL="${SLACK_EMAIL:-}"
export SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-aip-oberman}"
export MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
export RUN_NAME="${RUN_NAME:-qwen7b-gsm8k}"
export ROOT="${ROOT:-${SCRATCH:-$HOME/scratch}/probe_lab}"
export VENV="${VENV:-$HOME/venvs/plab}"
export HF_HOME="${HF_HOME:-$ROOT/hf_home}"
export RUN_DIR="${RUN_DIR:-$ROOT/outputs/$RUN_NAME}"

export EXTRA_TRAIN="${EXTRA_TRAIN:-2700}"      # + all 1319 GSM8K test problems  => ~4000 problems
export N_SAMPLES="${N_SAMPLES:-16}"            # sampled answers per (problem, condition): the label resolution is 1/N
export MAX_NEW="${MAX_NEW:-512}"
export TEMP="${TEMP:-0.7}" TOP_P="${TOP_P:-0.95}"
export CHUNK="${CHUNK:-64}"
export MAX_SEQS="${MAX_SEQS:-192}"             # sequences per generate() call per GPU (H100 80GB)
export FWD_BS="${FWD_BS:-8}"
export GATE_N="${GATE_N:-240}"
export CONDS_RUN="${CONDS_RUN:-}"             # comma list of conditions for states/sample (empty = all 8)
export TAG="${TAG:-}"                          # file-name tag so new conditions do not collide with finished chunks
export STEER_N="${STEER_N:-400}"               # held-out problems used by the steering experiment

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

# Stage reports. Never fails the job; always prints the subject to the log.
# Channels used if configured (put the secrets in ~/.bashrc as `export ...`, never in the repo):
#   SLACK_WEBHOOK_URL  Slack incoming webhook posting to your DM   (works from Fir/Nibi compute nodes: they have internet)
#   NTFY_TOPIC         private topic name for push notifications via ntfy.sh (phone app)
# If neither is set it falls back to e-mail to SLACK_EMAIL (sendmail, mail, local SMTP) - not available on Fir nodes.
# usage: notify "subject" ["body"]
notify() {
  local subject="[plab ${RUN_NAME}] $1" body="${2:-}" text sent=0 sm msg
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
  msg="$(printf 'To: %s\nFrom: %s@%s\nSubject: %s\n\n%s\n' "$SLACK_EMAIL" "${USER:-plab}" "$(hostname)" "$subject" "$text")"
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

