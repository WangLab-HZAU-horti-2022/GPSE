#!/usr/bin/env bash
set -euo pipefail

GPSE_ROOT="${GPSE_ROOT:-/data/jzhang/project/Temp/gpse_26.6.4_test/GPSE}"
OUT_ROOT="${OUT_ROOT:-${GPSE_ROOT}/tests/manuscript_runs/six_species_maize_rice_8core_20260806}"
LAUNCHER="${GPSE_ROOT}/tests/launch_six_species_maize_rice_8.sh"
INITIAL_HT_PID="${INITIAL_HT_PID:?Set INITIAL_HT_PID to the active maize HT GPSE PID}"
INITIAL_QUEUE_PID="${INITIAL_QUEUE_PID:-}"
STATUS_FILE="${OUT_ROOT}/scheduler_status.tsv"
WATCHDOG_LOG="${OUT_ROOT}/watchdog.log"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${WATCHDOG_LOG}"
}

has_complete_summary() {
  local result_dir="$1"
  [[ -f "${result_dir}/reports/run_summary.json" ]] && \
    rg -q '"completed_models"[[:space:]]*:[[:space:]]*11' "${result_dir}/reports/run_summary.json"
}

has_active_gpse_for_queue() {
  pgrep -af 'gpse\.cli train' | rg -F "${OUT_ROOT}" >/dev/null
}

record_ht_terminal_status() {
  local result_dir="${OUT_ROOT}/results/maize/HT"
  local cv_file="${OUT_ROOT}/cv/maize/HT_cv_50x5.csv"
  local log_file="${OUT_ROOT}/logs/maize_HT.log"
  if rg -q $'^maize\tHT\t(COMPLETED|FAILED)\t' "${STATUS_FILE}"; then
    return 0
  fi
  if has_complete_summary "${result_dir}"; then
    printf 'maize\tHT\tCOMPLETED\t1\t0\t-\t%s\t8\t1\t1\t%s\t%s\t%s\tcompleted under provisional S1b protocol\n' \
      "$(date -Is)" "${cv_file}" "${log_file}" "${result_dir}" >> "${STATUS_FILE}"
    log 'maize/HT summary verified complete.'
  else
    printf 'maize\tHT\tFAILED\t1\tunknown\t-\t%s\t8\t1\t1\t%s\t%s\t%s\tactive process exited without complete summary\n' \
      "$(date -Is)" "${cv_file}" "${log_file}" "${result_dir}" >> "${STATUS_FILE}"
    log 'maize/HT exited without a complete summary; fallback launcher will retry it.'
  fi
}

log "Supervisor started; preserving active maize/HT PID ${INITIAL_HT_PID}."
while kill -0 "${INITIAL_HT_PID}" 2>/dev/null; do
  sleep 60
done

log "Active maize/HT PID ${INITIAL_HT_PID} has exited; allowing reports and the original queue to advance."
for _ in {1..12}; do
  if has_active_gpse_for_queue; then
    log 'The original queue has advanced to an active GPSE task; no recovery launch is needed.'
    exit 0
  fi
  sleep 10
done

record_ht_terminal_status
if has_active_gpse_for_queue; then
  log 'A GPSE process for this queue became active during finalization; no recovery launch is needed.'
  exit 0
fi

if [[ -n "${INITIAL_QUEUE_PID}" ]] && kill -0 "${INITIAL_QUEUE_PID}" 2>/dev/null; then
  log "Original queue owner PID ${INITIAL_QUEUE_PID} is still alive but idle; waiting 60 seconds before fallback."
  sleep 60
  if has_active_gpse_for_queue; then
    log 'The original queue resumed; no recovery launch is needed.'
    exit 0
  fi
fi

log 'No active original queue remains; starting recovery launcher. It skips verified-complete traits and continues sequentially.'
cd "${GPSE_ROOT}/tests"
exec "${LAUNCHER}"
