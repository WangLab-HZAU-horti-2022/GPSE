#!/usr/bin/env bash
set -euo pipefail

GPSE_ROOT="${GPSE_ROOT:-/data/jzhang/project/Temp/gpse_26.6.4_test/GPSE}"
XIAOLIU_ROOT="${XIAOLIU_ROOT:-/data/xiaoliu}"
RUN_LABEL="${RUN_LABEL:-lowdim_core_traits_90core_20260805}"
OUT_ROOT="${OUT_ROOT:-${GPSE_ROOT}/tests/manuscript_runs/${RUN_LABEL}}"
THREADS_PER_TRAIT="${THREADS_PER_TRAIT:-8}"
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

  mkdir -p "${result_dir}" "$(dirname "${cv_file}")"
  nohup python -m gpse.cli train \
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
    --n_jobs 1 \
    --threads "${THREADS_PER_TRAIT}" \
    --save_representative \
    --results_dir "${result_dir}" >"${log_file}" 2>&1 < /dev/null &
  echo "$! ${species} ${trait} ${log_file}"
}

pid_file="${OUT_ROOT}/run_pids.tsv"
{
  echo -e "pid\tspecies\ttrait\tlog"
  run_trait cucumber "${XIAOLIU_ROOT}/gpse_3species/cucumber_n/cucumber_processed_data" Anthracnose
  run_trait cucumber "${XIAOLIU_ROOT}/gpse_3species/cucumber_n/cucumber_processed_data" Days_to_flower
  run_trait cucumber "${XIAOLIU_ROOT}/gpse_3species/cucumber_n/cucumber_processed_data" Downy_mildew
  run_trait cucumber "${XIAOLIU_ROOT}/gpse_3species/cucumber_n/cucumber_processed_data" Gummy_stem_blight
  run_trait melon "${XIAOLIU_ROOT}/gpse_3species/melon/melon_processed_data" 100-seed_weight_g
  run_trait melon "${XIAOLIU_ROOT}/gpse_3species/melon/melon_processed_data" fruit_width_cm
  run_trait melon "${XIAOLIU_ROOT}/gpse_3species/melon/melon_processed_data" Fruit_weight_kg
  run_trait melon "${XIAOLIU_ROOT}/gpse_3species/melon/melon_processed_data" Soluble_solids_content_oBrix
  run_trait watermelon "${XIAOLIU_ROOT}/gpse_3species/watermelon/watermelon_processed_data" PM_race_2_leaf
  run_trait watermelon "${XIAOLIU_ROOT}/gpse_3species/watermelon/watermelon_processed_data" PM_race_2_stem
  run_trait watermelon "${XIAOLIU_ROOT}/gpse_3species/watermelon/watermelon_processed_data" BFB_fruit
} >"${pid_file}"

printf 'Started 11 traits × %s threads = %s thread budget\n' "${THREADS_PER_TRAIT}" "$((11 * THREADS_PER_TRAIT))"
printf 'PID manifest: %s\n' "${pid_file}"
