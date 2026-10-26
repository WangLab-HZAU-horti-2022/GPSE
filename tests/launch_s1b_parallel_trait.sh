#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "Usage: $0 <species> <trait> <results-dir> <cv-file>" >&2
  exit 64
fi

SPECIES="$1"
TRAIT="$2"
RESULTS_DIR="$3"
CV_FILE="$4"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="/data/xiaoliu/gpse_6pecies_model/6specise"
GENO_FILE="${DATA_ROOT}/geno/${SPECIES}_geno.csv"
PHENO_FILE="${DATA_ROOT}/pheno/${SPECIES}_pheno.csv"
LOG_FILE="${RESULTS_DIR}/parallel_train.log"
MODELS=(
  svr_reg kernelridge_reg knn_reg rf_reg gbdt_reg histgradientboost_reg
  adaboost_reg xgboost_reg lightgbm_reg catboost_reg gblup_reg
)

for input in "${GENO_FILE}" "${PHENO_FILE}"; do
  [[ -s "${input}" ]] || { echo "Missing input: ${input}" >&2; exit 2; }
done

if [[ -f "${RESULTS_DIR}/reports/run_summary.json" ]] && \
    rg -q '"completed_models"[[:space:]]*:[[:space:]]*11' "${RESULTS_DIR}/reports/run_summary.json"; then
  echo "[SKIP] ${SPECIES}/${TRAIT} already complete"
  exit 0
fi

available_gib() {
  awk '/MemAvailable:/ {printf "%d", $2 / 1024 / 1024}' /proc/meminfo
}

select_threads() {
  local available
  available="$(available_gib)"
  if (( available >= 192 )); then
    echo 8
  elif (( available >= 96 )); then
    echo 4
  elif (( available >= 48 )); then
    echo 2
  else
    echo 0
  fi
}

THREADS="$(select_threads)"
while (( THREADS == 0 )); do
  echo "[$(date -Is)] ${SPECIES}/${TRAIT}: MemAvailable=$(available_gib) GiB; waiting for 48 GiB before launch" >&2
  sleep 120
  THREADS="$(select_threads)"
done

mkdir -p "${RESULTS_DIR}" "$(dirname "${CV_FILE}")" "$(dirname "${LOG_FILE}")"
cat > "${RESULTS_DIR}/parallel_launch.md" <<EOF
# Parallel S1b base-model rerun

- Species / trait: ${SPECIES} / ${TRAIT}
- Started: $(date -Is)
- Input genotype: ${GENO_FILE}
- Input phenotype: ${PHENO_FILE}
- Fixed CV: ${CV_FILE}
- Requested models: ${MODELS[*]}
- Protocol: 50 repeats, 5 folds, 50 Optuna trials, patience 20, top-3 stacking
- Threads selected at launch: ${THREADS}; start-time MemAvailable: $(available_gib) GiB
- This companion run is safe for the sequential queue to skip after `run_summary.json` verifies all 11 models.
EOF

export OMP_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export PYTHONPATH="${ROOT}/..:${PYTHONPATH:-}"

echo "[START] ${SPECIES}/${TRAIT} at $(date -Is); threads=${THREADS}; MemAvailable=$(available_gib) GiB" | tee "${LOG_FILE}"
python -m gpse.cli train \
  --geno_file "${GENO_FILE}" \
  --pheno_file "${PHENO_FILE}" \
  --target_trait "${TRAIT}" \
  --task_type regression \
  --models "${MODELS[@]}" \
  --train_folds 5 \
  --n_repeats 50 \
  --trials 50 \
  --patience 20 \
  --random_seed 42 \
  --cv_file "${CV_FILE}" \
  --use_stacking \
  --top_n_models 3 \
  --cv_folds 5 \
  --n_jobs "${THREADS}" \
  --max_workers 1 \
  --repeat_workers 1 \
  --save_representative \
  --results_dir "${RESULTS_DIR}" \
  2>&1 | tee -a "${LOG_FILE}"
