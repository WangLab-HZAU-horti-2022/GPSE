#!/usr/bin/env bash
set -euo pipefail

GPSE_ROOT="${GPSE_ROOT:-/data/jzhang/project/Temp/gpse_26.6.4_test/GPSE}"
XIAOLIU_ROOT="${XIAOLIU_ROOT:-/data/xiaoliu}"
RUN_LABEL="${RUN_LABEL:-lowdim_watermelon_parallel_30core_20260805}"
OUT_ROOT="${OUT_ROOT:-${GPSE_ROOT}/tests/manuscript_runs/${RUN_LABEL}}"
MODEL_WORKERS="${MODEL_WORKERS:-1}"
THREADS_PER_MODEL="${THREADS_PER_MODEL:-30}"
TRIALS="${TRIALS:-50}"

REG_MODELS=(
  svr_reg kernelridge_reg knn_reg rf_reg gbdt_reg
  histgradientboost_reg adaboost_reg xgboost_reg
  lightgbm_reg catboost_reg gblup_reg
)

mkdir -p "${OUT_ROOT}/cv/watermelon" "${OUT_ROOT}/logs" "${OUT_ROOT}/state"
cd "${GPSE_ROOT}"
export PYTHONPATH="${GPSE_ROOT}:${PYTHONPATH:-}"

run_trait() {
  local trait="$1"
  local input_dir="${XIAOLIU_ROOT}/gpse_3species/watermelon/watermelon_processed_data"
  local result_dir="${OUT_ROOT}/lowdim/watermelon/${trait}"
  local cv_file="${OUT_ROOT}/cv/watermelon/${trait}_cv_50x5.csv"
  local log_file="${OUT_ROOT}/logs/watermelon_${trait}.log"
  local state_file="${OUT_ROOT}/state/watermelon_${trait}.tsv"

  if [[ -f "${result_dir}/reports/run_summary.json" ]] && \
    rg -q '"completed_models": 11' "${result_dir}/reports/run_summary.json"; then
    printf '[SKIP] watermelon/%s already completed at %s\n' "${trait}" "$(date -Is)" | tee -a "${OUT_ROOT}/scheduler.log"
    return 0
  fi

  for suffix in genotype phenotype; do
    if [[ ! -s "${input_dir}/${trait}_${suffix}.csv" ]]; then
      printf '[FAILED] watermelon/%s missing %s input at %s\n' "${trait}" "${suffix}" "$(date -Is)" | tee -a "${OUT_ROOT}/scheduler.log"
      return 2
    fi
  done

  mkdir -p "${result_dir}"
  printf 'trait\tpid\tstart_time\tmodel_workers\tthreads_per_model\tlog_file\tresult_dir\n' > "${state_file}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${trait}" "$$" "$(date -Is)" "${MODEL_WORKERS}" "${THREADS_PER_MODEL}" "${log_file}" "${result_dir}" >> "${state_file}"
  printf '[START] watermelon/%s at %s; MODEL_WORKERS=%s; THREADS_PER_MODEL=%s\n' \
    "${trait}" "$(date -Is)" "${MODEL_WORKERS}" "${THREADS_PER_MODEL}" | tee -a "${OUT_ROOT}/scheduler.log"

  set +e
  python -m gpse.cli train \
    --geno_file "${input_dir}/${trait}_genotype.csv" \
    --pheno_file "${input_dir}/${trait}_phenotype.csv" \
    --target_trait "${trait}" \
    --task_type regression \
    --models "${REG_MODELS[@]}" \
    --train_folds 5 \
    --n_repeats 50 \
    --trials "${TRIALS}" \
    --patience 20 \
    --random_seed 42 \
    --cv_file "${cv_file}" \
    --use_stacking \
    --top_n_models 3 \
    --cv_folds 5 \
    --n_jobs "${THREADS_PER_MODEL}" \
    --max_workers "${MODEL_WORKERS}" \
    --repeat_workers 1 \
    --save_representative \
    --results_dir "${result_dir}" > "${log_file}" 2>&1
  local exit_code=$?
  set -e

  printf '[END] watermelon/%s at %s; exit_code=%s\n' \
    "${trait}" "$(date -Is)" "${exit_code}" | tee -a "${OUT_ROOT}/scheduler.log"
  return "${exit_code}"
}

for trait in PM_race_2_leaf PM_race_2_stem BFB_fruit; do
  if ! run_trait "${trait}"; then
    printf '[FAILED] watermelon/%s; stopping this isolated watermelon queue\n' "${trait}" | tee -a "${OUT_ROOT}/scheduler.log"
    exit 1
  fi
done

printf '[COMPLETE] all watermelon low-dimensional traits finished at %s\n' "$(date -Is)" | tee -a "${OUT_ROOT}/scheduler.log"
