#!/usr/bin/env bash
# Resume the lettuce outer-leaf-color classification run after the svm_clf fix.
#
# Background: the original run (classification_outer_leaf_color_top3_8core_20260806)
# lost all 50 svm_clf repeats because the Optuna search space passed LinearSVC
# parameters (loss/dual/penalty) to sklearn.svm.SVC. The search space in
# gpse/config/models.yaml was fixed on 2026-08-29. The broken svm_clf outputs
# were archived to <RUN_ROOT>/archived_failed/svm_clf_buggy_loss_param_20260810.
#
# Thanks to repeat-level resume in gpse.train._repeat_training, the five
# completed models (rf/xgboost/lightgbm/catboost/mlp, 50/50 each) are loaded
# from disk and only svm_clf is actually retrained; comparison tables and
# top-3 stacking are then rebuilt from all six models.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPSE_ROOT="${GPSE_ROOT:-/data/jzhang/project/Temp/gpse_26.6.4_test/GPSE}"
RUN_ROOT="${ROOT}/manuscript_runs/classification_outer_leaf_color_top3_8core_20260806"
GENO_FILE="/data/xiaoliu/gpse_lettuce_class/final_matrix_clean.csv"
PHENO_FILE="/data/xiaoliu/gpse_lettuce_class/phenotypes_renamed/Outer_leaf_color_OLT_2021.csv"
CV_FILE="${RUN_ROOT}/cv/Outer_leaf_color_OLT_2021_cv_50x5.csv"
RESULTS_DIR="${RUN_ROOT}/classification/Outer_leaf_color_OLT_2021"
LOG_FILE="${RUN_ROOT}/logs/outer_leaf_color_svm_resume_20260829.log"
THREADS="${THREADS:-5}"

mkdir -p "$(dirname "${LOG_FILE}")" "${RESULTS_DIR}"
export OMP_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
cd "${GPSE_ROOT}"
export PYTHONPATH="${GPSE_ROOT}:${PYTHONPATH:-}"

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
  --n_jobs "${THREADS}" \
  --max_workers 1 \
  --repeat_workers 1 \
  --save_representative \
  --results_dir "${RESULTS_DIR}" \
  2>&1 | tee "${LOG_FILE}"
