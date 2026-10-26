#!/usr/bin/env bash
set -euo pipefail

# GPSE manuscript missing-analysis command book.
# Default behavior is safe: commands are printed but not executed.
# Usage examples:
#   bash GPSE/tests/run_manuscript_missing_analyses.sh audit
#   bash GPSE/tests/run_manuscript_missing_analyses.sh lowdim
#   DRY_RUN=0 THREADS=60 bash GPSE/tests/run_manuscript_missing_analyses.sh lowdim

GPSE_ROOT="${GPSE_ROOT:-/data/jzhang/project/Temp/gpse_26.6.4_test/GPSE}"
XIAOLIU_ROOT="${XIAOLIU_ROOT:-/data/xiaoliu}"
OUT_ROOT="${OUT_ROOT:-${GPSE_ROOT}/tests/manuscript_runs}"
THREADS="${THREADS:-20}"
TRIALS="${TRIALS:-50}"
DRY_RUN="${DRY_RUN:-1}"

REG_MODELS=(
  svr_reg kernelridge_reg knn_reg rf_reg gbdt_reg
  histgradientboost_reg adaboost_reg xgboost_reg
  lightgbm_reg catboost_reg gblup_reg
)
CLF_MODELS=(rf_clf xgboost_clf lightgbm_clf catboost_clf svm_clf mlp_clf)

run_cmd() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
  if [[ "${DRY_RUN}" == "0" ]]; then
    "$@"
  fi
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || die "Missing file: $1"
}

activate_gpse() {
  cd "${GPSE_ROOT}"
  export PYTHONPATH="${GPSE_ROOT}:${PYTHONPATH:-}"
}

audit_inputs() {
  activate_gpse
  python -c 'import gpse; print(f"gpse package version: {gpse.__version__}")'
  python -m gpse.cli convert --check-deps

  require_file "${XIAOLIU_ROOT}/gpse_3species/cucumber_n/cucumber_processed_data/Anthracnose_genotype.csv"
  require_file "${XIAOLIU_ROOT}/gpse_3species/melon/melon_processed_data/100-seed_weight_g_genotype.csv"
  require_file "${XIAOLIU_ROOT}/gpse_3species/watermelon/watermelon_processed_data/BFB_fruit_genotype.csv"
  require_file "${XIAOLIU_ROOT}/gpse_lettuce_class/final_matrix_clean.csv"
  require_file "${XIAOLIU_ROOT}/gpse_lettuce_class/roc_plot_data/roc_data_all_traits.csv"
  require_file "${GPSE_ROOT}/../xiaoliu_result/6_species_result/genomic_prediction_results7/all_traits_detailed_cv_data.csv"

  echo "Input audit passed. Review trait and SNP manifests before DRY_RUN=0."
}

run_regression() {
  local geno_file="$1"
  local pheno_file="$2"
  local trait="$3"
  local results_dir="$4"
  local cv_file="$5"

  require_file "${geno_file}"
  require_file "${pheno_file}"
  if [[ "${DRY_RUN}" == "0" ]]; then
    mkdir -p "$(dirname "${cv_file}")" "${results_dir}"
  fi

  run_cmd python -m gpse.cli train \
    --geno_file "${geno_file}" \
    --pheno_file "${pheno_file}" \
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
    --threads "${THREADS}" \
    --save_representative \
    --results_dir "${results_dir}"
}

run_classification() {
  local geno_file="$1"
  local pheno_file="$2"
  local trait="$3"
  local n_classes="$4"
  local results_dir="$5"
  local cv_file="$6"

  require_file "${geno_file}"
  require_file "${pheno_file}"
  if [[ "${DRY_RUN}" == "0" ]]; then
    mkdir -p "$(dirname "${cv_file}")" "${results_dir}"
  fi

  run_cmd python -m gpse.cli train \
    --geno_file "${geno_file}" \
    --pheno_file "${pheno_file}" \
    --target_trait "${trait}" \
    --task_type classification \
    --n_classes "${n_classes}" \
    --models "${CLF_MODELS[@]}" \
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
    --threads "${THREADS}" \
    --save_representative \
    --results_dir "${results_dir}"
}

run_lowdim() {
  activate_gpse

  local cucumber_dir="${XIAOLIU_ROOT}/gpse_3species/cucumber_n/cucumber_processed_data"
  local melon_dir="${XIAOLIU_ROOT}/gpse_3species/melon/melon_processed_data"
  local watermelon_dir="${XIAOLIU_ROOT}/gpse_3species/watermelon/watermelon_processed_data"

  local cucumber_traits=(Anthracnose Days_to_flower Downy_mildew Gummy_stem_blight)
  local melon_traits=(100-seed_weight_g fruit_width_cm Fruit_weight_kg Soluble_solids_content_oBrix)
  local watermelon_traits=(PM_race_2_leaf PM_race_2_stem BFB_fruit)

  local trait
  for trait in "${cucumber_traits[@]}"; do
    run_regression \
      "${cucumber_dir}/${trait}_genotype.csv" \
      "${cucumber_dir}/${trait}_phenotype.csv" \
      "${trait}" \
      "${OUT_ROOT}/lowdim/cucumber/${trait}" \
      "${OUT_ROOT}/cv/cucumber/${trait}_cv_50x5.csv"
  done

  for trait in "${melon_traits[@]}"; do
    run_regression \
      "${melon_dir}/${trait}_genotype.csv" \
      "${melon_dir}/${trait}_phenotype.csv" \
      "${trait}" \
      "${OUT_ROOT}/lowdim/melon/${trait}" \
      "${OUT_ROOT}/cv/melon/${trait}_cv_50x5.csv"
  done

  for trait in "${watermelon_traits[@]}"; do
    run_regression \
      "${watermelon_dir}/${trait}_genotype.csv" \
      "${watermelon_dir}/${trait}_phenotype.csv" \
      "${trait}" \
      "${OUT_ROOT}/lowdim/watermelon/${trait}" \
      "${OUT_ROOT}/cv/watermelon/${trait}_cv_50x5.csv"
  done

  cat <<'EOF'

[NEXT] Run rrBLUP with exactly the same sample and fold manifests.
Current GPSE includes gblup_reg but does not include rrBLUP.
Aggregate the GPSE, GBLUP and rrBLUP outputs into Figure 2a and Table 1.
EOF
}

convert_plink_subset() {
  local bfile_prefix="$1"
  local phenotype_txt="$2"
  local trait="$3"
  local out_prefix="$4"

  require_file "${bfile_prefix}.bed"
  require_file "${bfile_prefix}.bim"
  require_file "${bfile_prefix}.fam"
  require_file "${phenotype_txt}"
  if [[ "${DRY_RUN}" == "0" ]]; then
    mkdir -p "$(dirname "${out_prefix}")"
  fi

  run_cmd python -m gpse.cli convert \
    --bfile "${bfile_prefix}" \
    --pheno "${phenotype_txt}" \
    --trait-name "${trait}" \
    --out-prefix "${out_prefix}" \
    --direct \
    --out-format csv
}

run_highdim_example() {
  activate_gpse

  cat <<'EOF'
[IMPORTANT]
This phase is an executable example for one tomato and one lettuce trait.
Before expanding it, freeze the manuscript's exact 10 tomato and 6 lettuce traits.
The *_gwas directories must be verified as EMMAX top-5000 inputs from logs/SNP manifests.
EOF

  local tomato_trait="IL"
  local tomato_base="${XIAOLIU_ROOT}/GBLUP_RRBLUP_tomato/tomato_gwas/${tomato_trait}"
  local tomato_prefix="${OUT_ROOT}/prepared/tomato_gwas/${tomato_trait}/${tomato_trait}"
  convert_plink_subset "${tomato_base}/genotype" "${tomato_base}/phenotype.txt" "${tomato_trait}" "${tomato_prefix}"

  echo "After conversion, inspect the exact generated genotype/phenotype filenames, then run:"
  echo "run_regression <generated_genotype.csv> <generated_phenotype.csv> ${tomato_trait} ${OUT_ROOT}/highdim/tomato_gwas/${tomato_trait} ${OUT_ROOT}/cv/tomato/${tomato_trait}_cv_50x5.csv"

  local lettuce_trait="3D_Angle_LAA0"
  local lettuce_base="${XIAOLIU_ROOT}/GBLUP_RRBLUP_lettuce/lettuce_gwas/${lettuce_trait}"
  local lettuce_prefix="${OUT_ROOT}/prepared/lettuce_gwas/${lettuce_trait}/${lettuce_trait}"
  convert_plink_subset "${lettuce_base}/genotype" "${lettuce_base}/phenotype.txt" "${lettuce_trait}" "${lettuce_prefix}"

  echo "After conversion, inspect the exact generated genotype/phenotype filenames, then run:"
  echo "run_regression <generated_genotype.csv> <generated_phenotype.csv> ${lettuce_trait} ${OUT_ROOT}/highdim/lettuce_gwas/${lettuce_trait} ${OUT_ROOT}/cv/lettuce/${lettuce_trait}_cv_50x5.csv"
}

run_classification_examples() {
  activate_gpse

  local geno="${XIAOLIU_ROOT}/gpse_lettuce_class/final_matrix_clean.csv"
  local pheno_dir="${XIAOLIU_ROOT}/gpse_lettuce_class/phenotypes_renamed"

  cat <<'EOF'
[IMPORTANT]
The task document requests lateral branching, leaf shape and leaf color.
No explicit lateral-branching file was found in the audited directory.
Do not silently replace it. Confirm the source field or revise the manuscript first.
EOF

  run_classification \
    "${geno}" \
    "${pheno_dir}/Leaf_shape_LS_2020.csv" \
    "Leaf" \
    3 \
    "${OUT_ROOT}/classification/Leaf_shape_LS_2020" \
    "${OUT_ROOT}/cv/lettuce_class/Leaf_shape_LS_2020_cv_50x5.csv"

  run_classification \
    "${geno}" \
    "${pheno_dir}/Outer_leaf_color_OLT_2021.csv" \
    "Outer" \
    4 \
    "${OUT_ROOT}/classification/Outer_leaf_color_OLT_2021" \
    "${OUT_ROOT}/cv/lettuce_class/Outer_leaf_color_OLT_2021_cv_50x5.csv"
}

print_sensitivity_plan() {
  cat <<'EOF'
Figure 3a sample-size loop:
  fractions=(0.05 0.10 0.20 0.40 0.60 0.80 1.00)
  traits=(cucumber/Downy_mildew cucumber/Anthracnose tomato/OLD tomato/IL)
  For each repeat and fraction:
    1. Keep the validation/test definition fixed.
    2. Deterministically subsample only the training IDs.
    3. Run GPSE and the best single model with the same IDs.
    4. Save species, trait, fraction, repeat, model, r and seed.

Figure 3b marker-density loop:
  densities=(0.05 0.10 0.20 0.40 0.60 0.80 1.00)
  strategies=(gwas_p ld)
  For each trait, strategy, density and repeat:
    1. Start from the verified strategy-specific SNP universe.
    2. Generate a deterministic SNP list.
    3. Use `gpse convert --bfile PREFIX --extract SNP_LIST --pheno PHENO --out-prefix OUT`.
    4. Run GPSE and the selected single-model comparator with the same CV file.

These loops require a dedicated manifest-generating helper. They are not run here because
the exact tomato/lettuce trait manifest and verified EMMAX SNP universes are not yet frozen.
EOF
}

print_s1b_plan() {
  cat <<EOF
Reuse base fold data:
  ${GPSE_ROOT}/../xiaoliu_result/6_species_result/genomic_prediction_results7/all_traits_detailed_cv_data.csv

Required ensemble definitions per species/trait/repeat:
  refined: Entropy-TOPSIS top-3 + Ridge stacking
  manual: manuscript-predeclared three-model set; freeze before seeing test folds
  full: all eligible base learners in one stacking model

Do not use the existing ranking table's Ensemble SD=0 as the Figure S1b error bar.
Generate repeat-level OOF ensemble predictions and calculate SD across 50 repeats.
EOF
}

case "${1:-help}" in
  audit) audit_inputs ;;
  lowdim) run_lowdim ;;
  highdim-example) run_highdim_example ;;
  classification) run_classification_examples ;;
  sensitivity-plan) print_sensitivity_plan ;;
  s1b-plan) print_s1b_plan ;;
  help|*)
    cat <<EOF
Usage: bash GPSE/tests/run_manuscript_missing_analyses.sh COMMAND

Commands:
  audit              Check software and key audited inputs
  lowdim             Run/print Figure 2a + Table 1 GPSE commands
  highdim-example    Convert and print high-dimensional example commands
  classification     Run/print leaf-shape and leaf-color classification
  sensitivity-plan   Print Figure 3a/3b execution specification
  s1b-plan           Print Figure S1b reconstruction specification

Environment:
  DRY_RUN=1          Print only; set DRY_RUN=0 to execute
  THREADS=20         Total GPSE parallelism
  TRIALS=50          Optuna trials per model
  OUT_ROOT=...       Output root
EOF
    ;;
esac
