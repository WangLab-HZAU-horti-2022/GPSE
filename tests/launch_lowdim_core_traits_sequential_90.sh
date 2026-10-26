#!/usr/bin/env bash
set -euo pipefail

GPSE_ROOT="${GPSE_ROOT:-/data/jzhang/project/Temp/gpse_26.6.4_test/GPSE}"
XIAOLIU_ROOT="${XIAOLIU_ROOT:-/data/xiaoliu}"
RUN_LABEL="${RUN_LABEL:-lowdim_core_traits_sequential_88core_20260805}"
OUT_ROOT="${OUT_ROOT:-${GPSE_ROOT}/tests/manuscript_runs/${RUN_LABEL}}"
MODEL_WORKERS="${MODEL_WORKERS:-1}"
THREADS_PER_MODEL="${THREADS_PER_MODEL:-88}"
TRIALS="${TRIALS:-50}"

REG_MODELS=(
  svr_reg kernelridge_reg knn_reg rf_reg gbdt_reg
  histgradientboost_reg adaboost_reg xgboost_reg
  lightgbm_reg catboost_reg gblup_reg
)

mkdir -p "${OUT_ROOT}/cv" "${OUT_ROOT}/logs"
cd "${GPSE_ROOT}"
export PYTHONPATH="${GPSE_ROOT}:${PYTHONPATH:-}"

run_trait() {
  local species="$1"
  local input_dir="$2"
  local trait="$3"
  local result_dir="${OUT_ROOT}/lowdim/${species}/${trait}"
  local cv_file="${OUT_ROOT}/cv/${species}/${trait}_cv_50x5.csv"
  local log_file="${OUT_ROOT}/logs/${species}_${trait}.log"

  if [[ -f "${result_dir}/reports/run_summary.json" ]] && \
    rg -q '"completed_models": 11' "${result_dir}/reports/run_summary.json"; then
    printf '[SKIP] %s/%s already completed\n' "${species}" "${trait}" | tee -a "${OUT_ROOT}/scheduler.log"
    return
  fi

  mkdir -p "${result_dir}" "$(dirname "${cv_file}")"
  printf '[START] %s/%s at %s; model_workers=%s; threads_per_model=%s\n' \
    "${species}" "${trait}" "$(date -Is)" "${MODEL_WORKERS}" "${THREADS_PER_MODEL}" | tee -a "${OUT_ROOT}/scheduler.log"

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
  printf '[END] %s/%s at %s; exit_code=%s\n' \
    "${species}" "${trait}" "$(date -Is)" "${exit_code}" | tee -a "${OUT_ROOT}/scheduler.log"
  return "${exit_code}"
}

traits=(
  "cucumber|${XIAOLIU_ROOT}/gpse_3species/cucumber_n/cucumber_processed_data|Anthracnose"
  "cucumber|${XIAOLIU_ROOT}/gpse_3species/cucumber_n/cucumber_processed_data|Days_to_flower"
  "cucumber|${XIAOLIU_ROOT}/gpse_3species/cucumber_n/cucumber_processed_data|Downy_mildew"
  "cucumber|${XIAOLIU_ROOT}/gpse_3species/cucumber_n/cucumber_processed_data|Gummy_stem_blight"
  "melon|${XIAOLIU_ROOT}/gpse_3species/melon/melon_processed_data|100-seed_weight_g"
  "melon|${XIAOLIU_ROOT}/gpse_3species/melon/melon_processed_data|fruit_width_cm"
  "melon|${XIAOLIU_ROOT}/gpse_3species/melon/melon_processed_data|Fruit_weight_kg"
  "melon|${XIAOLIU_ROOT}/gpse_3species/melon/melon_processed_data|Soluble_solids_content_oBrix"
  "watermelon|${XIAOLIU_ROOT}/gpse_3species/watermelon/watermelon_processed_data|PM_race_2_leaf"
  "watermelon|${XIAOLIU_ROOT}/gpse_3species/watermelon/watermelon_processed_data|PM_race_2_stem"
  "watermelon|${XIAOLIU_ROOT}/gpse_3species/watermelon/watermelon_processed_data|BFB_fruit"
)

for item in "${traits[@]}"; do
  IFS='|' read -r species input_dir trait <<< "${item}"
  if ! run_trait "${species}" "${input_dir}" "${trait}"; then
    printf '[FAILED] %s/%s; stopping sequential queue to preserve evidence\n' \
      "${species}" "${trait}" | tee -a "${OUT_ROOT}/scheduler.log"
    exit 1
  fi
done

printf '[COMPLETE] All sequential low-dimensional core traits finished at %s\n' "$(date -Is)" | tee -a "${OUT_ROOT}/scheduler.log"
