#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 <tomato|lettuce> <trait> <run-root>" >&2
  exit 64
fi

SPECIES="$1"
TRAIT="$2"
RUN_ROOT="$(cd "$3" && pwd)"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/home/jzhang/.pyenv/versions/gpse/bin/python"
PLINK_PATH="/home/jzhang/software/plink1.9/plink"
MODELS=(
  svr_reg kernelridge_reg knn_reg rf_reg gbdt_reg histgradientboost_reg
  adaboost_reg xgboost_reg lightgbm_reg catboost_reg gblup_reg
)

case "${SPECIES}" in
  tomato)
    SOURCE_DIR="/data/xiaoliu/GBLUP_RRBLUP_tomato/tomato_gwas/${TRAIT}"
    PREPARED_DIR="${RUN_ROOT}/prepared/${TRAIT}"
    RESULT_DIR="${RUN_ROOT}/highdim/tomato/${TRAIT}"
    CV_FILE="${RUN_ROOT}/cv/tomato_${TRAIT}_cv_50x5.csv"
    ;;
  lettuce)
    SOURCE_DIR="/data/xiaoliu/GBLUP_RRBLUP_lettuce/lettuce_gwas/${TRAIT}"
    PREPARED_DIR="${RUN_ROOT}/prepared/lettuce_gwas/${TRAIT}"
    RESULT_DIR="${RUN_ROOT}/highdim/lettuce_gwas/${TRAIT}"
    CV_FILE="${RUN_ROOT}/cv/lettuce_gwas/${TRAIT}_cv_50x5.csv"
    ;;
  *)
    echo "Unsupported species: ${SPECIES}" >&2
    exit 64
    ;;
esac

for input in "${SOURCE_DIR}/genotype.bed" "${SOURCE_DIR}/genotype.bim" "${SOURCE_DIR}/genotype.fam" "${SOURCE_DIR}/phenotype.txt"; do
  [[ -s "${input}" ]] || { echo "Missing input: ${input}" >&2; exit 2; }
done

if [[ -f "${RESULT_DIR}/reports/run_summary.json" ]] && rg -q '"completed_models"[[:space:]]*:[[:space:]]*11' "${RESULT_DIR}/reports/run_summary.json"; then
  echo "[SKIP] ${SPECIES}/${TRAIT} already complete"
  exit 0
fi

available_gib() {
  awk '/MemAvailable:/ {printf "%d", $2 / 1024 / 1024}' /proc/meminfo
}

select_threads() {
  local available
  available="$(available_gib)"
  if (( available >= 192 )); then echo 8
  elif (( available >= 96 )); then echo 4
  elif (( available >= 48 )); then echo 2
  else echo 0
  fi
}

THREADS="$(select_threads)"
while (( THREADS == 0 )); do
  echo "[$(date -Is)] ${SPECIES}/${TRAIT}: MemAvailable=$(available_gib) GiB; waiting" >&2
  sleep 120
  THREADS="$(select_threads)"
done

mkdir -p "${PREPARED_DIR}" "${RESULT_DIR}" "${RUN_ROOT}/logs" "$(dirname "${CV_FILE}")"
LOG_FILE="${RUN_ROOT}/logs/${SPECIES}_${TRAIT}.parallel.log"
PREFIX="${PREPARED_DIR}/${TRAIT}"
GENOTYPE_FILE="$(find "${PREPARED_DIR}" -maxdepth 1 -type f -name '*_genotype.csv' -print -quit 2>/dev/null || true)"
PHENOTYPE_FILE="$(find "${PREPARED_DIR}" -maxdepth 1 -type f -name '*_phenotype.csv' ! -name '*_raw.csv' -print -quit 2>/dev/null || true)"

if [[ -z "${GENOTYPE_FILE}" || -z "${PHENOTYPE_FILE}" ]]; then
  echo "[CONVERT] ${SPECIES}/${TRAIT} at $(date -Is)" | tee "${LOG_FILE}"
  "${PYTHON_BIN}" -m gpse.cli convert \
    --bfile "${SOURCE_DIR}/genotype" \
    --direct \
    --pheno "${SOURCE_DIR}/phenotype.txt" \
    --out-prefix "${PREFIX}" \
    --out-format csv \
    --trait-name "${TRAIT}" \
    --plink-path "${PLINK_PATH}" \
    --threads "${THREADS}" >>"${LOG_FILE}" 2>&1
  GENOTYPE_FILE="$(find "${PREPARED_DIR}" -maxdepth 1 -type f -name '*_genotype.csv' -print -quit)"
  PHENOTYPE_FILE="$(find "${PREPARED_DIR}" -maxdepth 1 -type f -name '*_phenotype.csv' ! -name '*_raw.csv' -print -quit)"
fi

[[ -s "${GENOTYPE_FILE}" && -s "${PHENOTYPE_FILE}" ]] || { echo "Prepared CSV discovery failed" >&2; exit 1; }

cat > "${RESULT_DIR}/parallel_launch.md" <<EOF
# Parallel provisional high-dimensional rerun

- Species / trait: ${SPECIES} / ${TRAIT}
- Started: $(date -Is)
- Source PLINK prefix: ${SOURCE_DIR}/genotype
- Prepared genotype: ${GENOTYPE_FILE}
- Prepared phenotype: ${PHENOTYPE_FILE}
- Fixed CV: ${CV_FILE}
- Models: ${MODELS[*]}
- Protocol: 50 repeats, 5 folds, 50 Optuna trials, patience 20, top-3 stacking
- Threads selected at launch: ${THREADS}; MemAvailable: $(available_gib) GiB
- Status: provisional pending formal trait/SNP manifest confirmation.
EOF

export OMP_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
cd "${ROOT}"
echo "[TRAIN] ${SPECIES}/${TRAIT} at $(date -Is); threads=${THREADS}" | tee -a "${LOG_FILE}"
exec "${PYTHON_BIN}" -m gpse.cli train \
  --geno_file "${GENOTYPE_FILE}" \
  --pheno_file "${PHENOTYPE_FILE}" \
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
  --results_dir "${RESULT_DIR}" \
  >>"${LOG_FILE}" 2>&1
