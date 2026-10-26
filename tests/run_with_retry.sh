#!/usr/bin/env bash
# run_with_retry.sh <logfile> <attempts> <cmd...>
# Run a queue script, retrying up to <attempts> times. Relies on GPSE
# repeat-level resume so retries fast-forward through completed work.
set -uo pipefail

log="$1"
attempts="$2"
shift 2

for ((i = 1; i <= attempts; i++)); do
  echo "[$(date -Is)] attempt ${i}/${attempts}: $*" | tee -a "${log}"
  if "$@" >>"${log}" 2>&1; then
    echo "[$(date -Is)] SUCCESS: $*" | tee -a "${log}"
    exit 0
  fi
  rc=$?
  echo "[$(date -Is)] attempt ${i} failed (exit=${rc}); sleeping 60s before retry" | tee -a "${log}"
  sleep 60
done

echo "[$(date -Is)] FAILED after ${attempts} attempts: $*" | tee -a "${log}"
exit 1
