#!/usr/bin/env bash
set -euo pipefail

GPSE_ROOT="${GPSE_ROOT:-/data/jzhang/project/Temp/gpse_26.6.4_test/GPSE}"
XIAOLIU_ROOT="${XIAOLIU_ROOT:-/data/xiaoliu}"
RUN_LABEL="${RUN_LABEL:-lowdim_cucumber_parallel_30core_20260805}"
OUT_ROOT="${OUT_ROOT:-${GPSE_ROOT}/tests/manuscript_runs/${RUN_LABEL}}"
MODEL_WORKERS="${MODEL_WORKERS:-1}"
THREADS_PER_MODEL="${THREADS_PER_MODEL:-30}"
TRIALS="${TRIALS:-50}"

REG_MODELS=(
  svr_reg kernelridge_reg knn_reg rf_reg gbdt_reg
  histgradientboost_reg adaboost_reg xgboost_reg
  lightgbm_reg catboost_reg gblup_reg
)

mkdir -p "${OUT_ROOT}/cv/cucumber" "${OUT_ROOT}/logs" "${OUT_ROOT}/lowdim/cucumber"
cd "${GPSE_ROOT}"
export PYTHONPATH="${GPSE_ROOT}:${PYTHONPATH:-}"

run_trait() {
  local trait="$1"
  local input_dir="${XIAOLIU_ROOT}/gpse_3species/cucumber_n/cucumber_processed_data"
  local result_dir="${OUT_ROOT}/lowdim/cucumber/${trait}"
  local cv_file="${OUT_ROOT}/cv/cucumber/${trait}_cv_50x5.csv"
  local log_file="${OUT_ROOT}/logs/cucumber_${trait}.log"

  if [[ -f "${result_dir}/reports/run_summary.json" ]] && \
    rg -q '"completed_models": 11' "${result_dir}/reports/run_summary.json"; then
    printf '[SKIP] cucumber/%s already completed at %s\n' "${trait}" "$(date -Is)" | tee -a "${OUT_ROOT}/scheduler.log"
    return 0
  fi

  mkdir -p "${result_dir}"
  printf '[START] cucumber/%s at %s; model_workers=%s; threads_per_model=%s; repeats=50; folds=5; trials=%s; stacking=top3\n' \
    "${trait}" "$(date -Is)" "${MODEL_WORKERS}" "${THREADS_PER_MODEL}" "${TRIALS}" | tee -a "${OUT_ROOT}/scheduler.log"

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
    --results_dir "${result_dir}" >"${log_file}" 2>&1
  local exit_code=$?

  printf '[END] cucumber/%s at %s; exit_code=%s\n' \
    "${trait}" "$(date -Is)" "${exit_code}" | tee -a "${OUT_ROOT}/scheduler.log"
  return "${exit_code}"
}

traits=(Anthracnose Days_to_flower Downy_mildew Gummy_stem_blight)
for trait in "${traits[@]}"; do
  if ! run_trait "${trait}"; then
    printf '[FAILED] cucumber/%s; stopping cucumber-only queue at %s\n' \
      "${trait}" "$(date -Is)" | tee -a "${OUT_ROOT}/scheduler.log"
    exit 1
  fi
done

printf '[COMPLETE] All cucumber low-dimensional core traits finished at %s\n' "$(date -Is)" | tee -a "${OUT_ROOT}/scheduler.log"
