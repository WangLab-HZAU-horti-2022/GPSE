#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${ROOT}/manuscript_runs/classification_outer_leaf_color_top3_8core_20260806"
GENO_FILE="/data/xiaoliu/gpse_lettuce_class/final_matrix_clean.csv"
PHENO_FILE="/data/xiaoliu/gpse_lettuce_class/phenotypes_renamed/Outer_leaf_color_OLT_2021.csv"
CV_FILE="${RUN_ROOT}/cv/Outer_leaf_color_OLT_2021_cv_50x5.csv"
RESULTS_DIR="${RUN_ROOT}/classification/Outer_leaf_color_OLT_2021"
LOG_FILE="${RUN_ROOT}/logs/outer_leaf_color_top3.train.log"

mkdir -p "$(dirname "${CV_FILE}")" "$(dirname "${LOG_FILE}")" "${RESULTS_DIR}"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

exec /home/jzhang/.pyenv/versions/gpse/bin/python -m gpse.cli train \
  --geno_file "${GENO_FILE}" \
  --pheno_file "${PHENO_FILE}" \
  --target_trait Outer \
  --task_type classification \
  --n_classes 4 \
  --models rf_clf xgboost_clf lightgbm_clf catboost_clf svm_clf mlp_clf \
  --train_folds 5 \
  --n_repeats 50 \
  --trials 50 \
  --patience 20 \
  --random_seed 42 \
  --cv_file "${CV_FILE}" \
  --use_stacking \
  --top_n_models 3 \
  --cv_folds 5 \
  --n_jobs 8 \
  --max_workers 1 \
  --repeat_workers 1 \
  --save_representative \
  --results_dir "${RESULTS_DIR}" \
  2>&1 | tee "${LOG_FILE}"
