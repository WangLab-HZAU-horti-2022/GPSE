#!/usr/bin/env bash
# Resume all incomplete manuscript queues after the 2026-08-19 server crash,
# using 50 cores in total (6 tmux sessions: 10+10+10+10+5+5 threads).
#
# What this covers (audited 2026-08-29):
#   - Partial traits (repeat-level resume fills only missing repeats/models):
#       tomato/FLNT, lettuce/3D_Angle_LAA0, maize/HT, sorghum/HT, switchgrass/ST
#   - Unstarted traits:
#       tomato:  FRNS FRNT FSD FSIN FSL IL OLD
#       lettuce: 3D_Angle_LAA2 3D_Angle_p1 3D_Angle_p3 3D_Angle_p5
#       maize FT/YLD, rice FT/HT, soy R8/YLD, switchgrass HT/AN
#   - Lettuce leaf-color svm_clf rerun (bug fixed; other 5 models resume from disk)
#   - S1b strict watcher restarted to auto-assemble six-species traits at 550/550
#
# Completed traits are skipped by the original scripts' own run_summary checks.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT}/manuscript_runs/resume_50core_20260829_logs"
mkdir -p "${LOG_DIR}"

launch() {
  local session="$1"; shift
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "[skip] tmux session ${session} already exists"
    return 0
  fi
  local log="${LOG_DIR}/${session}.log"
  tmux new-session -d -s "${session}" \
    "${ROOT}/run_with_retry.sh '${log}' 3 $*"
  echo "[launch] ${session}: $* (log: ${log})"
}

cd "${ROOT}"

# --- Six-species S1b queues (regression, 11 models) -------------------------
launch gpse_r_maize_rice  env THREADS=10 bash launch_six_species_maize_rice_8.sh
launch gpse_r_sorghum_soy env N_JOBS=10  bash launch_six_species_sorghum_soy_8.sh
launch gpse_r_spruce_sg   env THREADS=10 bash launch_six_species_spruce_switchgrass_8.sh

# --- High-dimensional provisional GWAS-input queues -------------------------
launch gpse_r_tomato  env THREADS_PER_MODEL=10 bash launch_highdim_tomato_gwas_provisional_15.sh
launch gpse_r_lettuce env THREADS_PER_MODEL=5  bash launch_highdim_lettuce_gwas_provisional_15.sh

# --- Lettuce leaf-color classification (svm_clf rerun after bug fix) --------
launch gpse_r_leafcolor bash resume_leafcolor_svm_5.sh

# --- S1b strict ensemble watcher (lightweight, 5-min polling) ---------------
if tmux has-session -t gpse_r_s1b_watcher 2>/dev/null; then
  echo "[skip] tmux session gpse_r_s1b_watcher already exists"
else
  tmux new-session -d -s gpse_r_s1b_watcher \
    "bash watch_s1b_strict_ensembles.sh >> '${LOG_DIR}/gpse_r_s1b_watcher.log' 2>&1"
  echo "[launch] gpse_r_s1b_watcher (log: ${LOG_DIR}/gpse_r_s1b_watcher.log)"
fi

echo
echo "All resume queues launched at $(date -Is). Monitor with:"
echo "  tmux ls | grep gpse_r_"
echo "  tail -f ${LOG_DIR}/<session>.log"
