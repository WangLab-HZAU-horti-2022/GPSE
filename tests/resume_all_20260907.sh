#!/usr/bin/env bash
# Resume all incomplete manuscript queues after the 2026-09-07 server reboot.
# Machine is idle: 104 cores / 503 GB RAM -> ~88 threads total.
#
# Audited 2026-09-07:
#   - Complete (skipped): all lowdim 3-species (11 traits), sorghum+soy (6 traits),
#     spruce DBH/HT/DE, rice/YLD, tomato FIN/FLNS/FLNT, lettuce LAA0/LAA1/LAA2,
#     lettuce leaf-color classification (6 models x 50 repeats, incl. svm_clf rerun).
#   - Partial (repeat-level resume fills missing repeats/models):
#       tomato/FRNS (384/550), lettuce/3D_Angle_p1 (344/550),
#       maize/HT (470/550), switchgrass/ST (353/550)
#   - Unstarted:
#       tomato: FRNT FSD FSIN FSL IL OLD
#       lettuce: 3D_Angle_p3 3D_Angle_p5
#       maize FT/YLD, rice FT/HT, switchgrass HT/AN
#   - S1b strict watcher restarted to auto-assemble six-species traits at 550/550.
#
# Completed traits are skipped by the original scripts' own run_summary checks.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT}/manuscript_runs/resume_20260907_logs"
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
# maize HT(partial)/FT/YLD + rice FT/HT (rice YLD done, auto-skipped)
launch gpse_r_maize_rice  env THREADS=24 bash launch_six_species_maize_rice_8.sh
# switchgrass ST(partial)/HT/AN (spruce all done, auto-skipped)
launch gpse_r_spruce_sg   env THREADS=20 bash launch_six_species_spruce_switchgrass_8.sh

# --- High-dimensional provisional GWAS-input queues -------------------------
# tomato FRNS(partial) + FRNT FSD FSIN FSL IL OLD
launch gpse_r_tomato  env THREADS_PER_MODEL=24 bash launch_highdim_tomato_gwas_provisional_15.sh
# lettuce 3D_Angle_p1(partial) + p3 p5
launch gpse_r_lettuce env THREADS_PER_MODEL=20 bash launch_highdim_lettuce_gwas_provisional_15.sh

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
