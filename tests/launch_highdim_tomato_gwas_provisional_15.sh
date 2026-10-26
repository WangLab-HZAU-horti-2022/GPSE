#!/usr/bin/env bash
set -euo pipefail

GPSE_ROOT="${GPSE_ROOT:-/data/jzhang/project/Temp/gpse_26.6.4_test/GPSE}"
TOMATO_GWAS_ROOT="${TOMATO_GWAS_ROOT:-/data/xiaoliu/GBLUP_RRBLUP_tomato/tomato_gwas}"
RUN_LABEL="${RUN_LABEL:-highdim_tomato_gwas_provisional_15core_20260805}"
OUT_ROOT="${OUT_ROOT:-${GPSE_ROOT}/tests/manuscript_runs/${RUN_LABEL}}"
PLINK_PATH="${PLINK_PATH:-/home/jzhang/software/plink1.9/plink}"
MODEL_WORKERS="${MODEL_WORKERS:-1}"
THREADS_PER_MODEL="${THREADS_PER_MODEL:-15}"
TRIALS="${TRIALS:-50}"

TRAITS=(FIN FLNS FLNT FRNS FRNT FSD FSIN FSL IL OLD)
REG_MODELS=(
  svr_reg kernelridge_reg knn_reg rf_reg gbdt_reg
  histgradientboost_reg adaboost_reg xgboost_reg
  lightgbm_reg catboost_reg gblup_reg
)

mkdir -p "${OUT_ROOT}" "${OUT_ROOT}/prepared" "${OUT_ROOT}/cv" \
  "${OUT_ROOT}/logs" "${OUT_ROOT}/highdim/tomato"
cd "${GPSE_ROOT}"
export PYTHONPATH="${GPSE_ROOT}:${PYTHONPATH:-}"

manifest="${OUT_ROOT}/manifest.tsv"
status_file="${OUT_ROOT}/scheduler_status.tsv"

if [[ ! -f "${manifest}" ]]; then
  cat >"${manifest}" <<EOF
species	trait	provisional	input_status	plink_prefix	phenotype_source	snp_source	emmax_top5000_status	analysis_note
tomato	FIN	true	pending	${TOMATO_GWAS_ROOT}/FIN/genotype	${TOMATO_GWAS_ROOT}/FIN/phenotype.txt	existing tomato_gwas directory	PENDING_MANUAL_CONFIRMATION	GPSE univariate regression; not EMMAX
tomato	FLNS	true	pending	${TOMATO_GWAS_ROOT}/FLNS/genotype	${TOMATO_GWAS_ROOT}/FLNS/phenotype.txt	existing tomato_gwas directory	PENDING_MANUAL_CONFIRMATION	GPSE univariate regression; not EMMAX
tomato	FLNT	true	pending	${TOMATO_GWAS_ROOT}/FLNT/genotype	${TOMATO_GWAS_ROOT}/FLNT/phenotype.txt	existing tomato_gwas directory	PENDING_MANUAL_CONFIRMATION	GPSE univariate regression; not EMMAX
tomato	FRNS	true	pending	${TOMATO_GWAS_ROOT}/FRNS/genotype	${TOMATO_GWAS_ROOT}/FRNS/phenotype.txt	existing tomato_gwas directory	PENDING_MANUAL_CONFIRMATION	GPSE univariate regression; not EMMAX
tomato	FRNT	true	pending	${TOMATO_GWAS_ROOT}/FRNT/genotype	${TOMATO_GWAS_ROOT}/FRNT/phenotype.txt	existing tomato_gwas directory	PENDING_MANUAL_CONFIRMATION	GPSE univariate regression; not EMMAX
tomato	FSD	true	pending	${TOMATO_GWAS_ROOT}/FSD/genotype	${TOMATO_GWAS_ROOT}/FSD/phenotype.txt	existing tomato_gwas directory	PENDING_MANUAL_CONFIRMATION	GPSE univariate regression; not EMMAX
tomato	FSIN	true	pending	${TOMATO_GWAS_ROOT}/FSIN/genotype	${TOMATO_GWAS_ROOT}/FSIN/phenotype.txt	existing tomato_gwas directory	PENDING_MANUAL_CONFIRMATION	GPSE univariate regression; not EMMAX
tomato	FSL	true	pending	${TOMATO_GWAS_ROOT}/FSL/genotype	${TOMATO_GWAS_ROOT}/FSL/phenotype.txt	existing tomato_gwas directory	PENDING_MANUAL_CONFIRMATION	GPSE univariate regression; not EMMAX
tomato	IL	true	pending	${TOMATO_GWAS_ROOT}/IL/genotype	${TOMATO_GWAS_ROOT}/IL/phenotype.txt	existing tomato_gwas directory	PENDING_MANUAL_CONFIRMATION	GPSE univariate regression; not EMMAX
tomato	OLD	true	pending	${TOMATO_GWAS_ROOT}/OLD/genotype	${TOMATO_GWAS_ROOT}/OLD/phenotype.txt	existing tomato_gwas directory	PENDING_MANUAL_CONFIRMATION	GPSE univariate regression; not EMMAX
EOF
fi

if [[ ! -f "${status_file}" ]]; then
  printf 'species\ttrait\tstatus\tattempt\texit_code\tstart_time\tend_time\tthreads\tmax_workers\tprepared_genotype\tprepared_phenotype\tlog_file\tresult_dir\n' >"${status_file}"
fi

update_manifest_status() {
  local trait="$1"
  local new_status="$2"
  local temporary="${manifest}.tmp"
  awk -F '\t' -v OFS='\t' -v trait="${trait}" -v status="${new_status}" \
    'NR == 1 || $2 != trait { print; next } { $4 = status; print }' \
    "${manifest}" >"${temporary}"
  mv "${temporary}" "${manifest}"
}

find_prepared_file() {
  local prepared_dir="$1"
  local suffix="$2"
  local -a matches=()
  while IFS= read -r path; do
    matches+=("${path}")
  done < <(find "${prepared_dir}" -maxdepth 1 -type f -name "*_${suffix}.csv" -print | sort)

  if (( ${#matches[@]} != 1 )); then
    printf 'Expected exactly one *_%s.csv in %s; found %s\n' \
      "${suffix}" "${prepared_dir}" "${#matches[@]}" >&2
    printf '%s\n' "${matches[@]}" >&2
    return 1
  fi
  printf '%s\n' "${matches[0]}"
}

convert_trait() {
  local trait="$1"
  local source_dir="${TOMATO_GWAS_ROOT}/${trait}"
  local bfile="${source_dir}/genotype"
  local pheno="${source_dir}/phenotype.txt"
  local prepared_dir="${OUT_ROOT}/prepared/${trait}"
  local out_prefix="${prepared_dir}/${trait}"
  local convert_log="${OUT_ROOT}/logs/tomato_${trait}.convert.log"
  local paths_file="${prepared_dir}/prepared_paths.tsv"

  for input in "${bfile}.bed" "${bfile}.bim" "${bfile}.fam" "${pheno}"; do
    if [[ ! -s "${input}" ]]; then
      printf '[SKIP] tomato/%s incomplete input: %s\n' "${trait}" "${input}" | tee -a "${OUT_ROOT}/scheduler.log"
      update_manifest_status "${trait}" "INCOMPLETE_INPUT"
      return 2
    fi
  done

  mkdir -p "${prepared_dir}"
  local genotype_file phenotype_file
  if genotype_file="$(find_prepared_file "${prepared_dir}" genotype 2>/dev/null)" && \
     phenotype_file="$(find_prepared_file "${prepared_dir}" phenotype 2>/dev/null)"; then
    printf '[CONVERT-SKIP] tomato/%s prepared files found\n' "${trait}" | tee -a "${OUT_ROOT}/scheduler.log"
    printf '%s\t%s\n' "${genotype_file}" "${phenotype_file}" >"${paths_file}"
    return 0
  fi

  printf '[CONVERT-START] tomato/%s at %s\n' "${trait}" "$(date -Is)" | tee -a "${OUT_ROOT}/scheduler.log"
  rm -f "${prepared_dir}"/*_genotype.csv "${prepared_dir}"/*_phenotype.csv
  if ! python -m gpse.cli convert \
    --bfile "${bfile}" \
    --direct \
    --pheno "${pheno}" \
    --out-prefix "${out_prefix}" \
    --out-format csv \
    --trait-name "${trait}" \
    --plink-path "${PLINK_PATH}" \
    --threads "${THREADS_PER_MODEL}" >"${convert_log}" 2>&1; then
    printf '[CONVERT-FAILED] tomato/%s at %s\n' "${trait}" "$(date -Is)" | tee -a "${OUT_ROOT}/scheduler.log"
    update_manifest_status "${trait}" "CONVERT_FAILED"
    return 1
  fi

  genotype_file="$(find_prepared_file "${prepared_dir}" genotype)"
  phenotype_file="$(find_prepared_file "${prepared_dir}" phenotype)"
  printf '[CONVERT-END] tomato/%s genotype=%s phenotype=%s\n' \
    "${trait}" "${genotype_file}" "${phenotype_file}" | tee -a "${OUT_ROOT}/scheduler.log"
  printf '%s\t%s\n' "${genotype_file}" "${phenotype_file}" >"${paths_file}"
}

run_trait() {
  local trait="$1"
  local result_dir="${OUT_ROOT}/highdim/tomato/${trait}"
  local cv_file="${OUT_ROOT}/cv/tomato_${trait}_cv_50x5.csv"
  local train_log="${OUT_ROOT}/logs/tomato_${trait}.train.log"
  local start_time exit_code

  if [[ -f "${result_dir}/reports/run_summary.json" ]] && \
    rg -q '"completed_models": 1[12]' "${result_dir}/reports/run_summary.json" && \
    ! rg -q '"failed_models": [1-9]' "${result_dir}/reports/run_summary.json"; then
    printf '[SKIP] tomato/%s already completed at %s\n' "${trait}" "$(date -Is)" | tee -a "${OUT_ROOT}/scheduler.log"
    update_manifest_status "${trait}" "COMPLETED"
    return 0
  fi

  set +e
  convert_trait "${trait}"
  exit_code=$?
  set -e
  if [[ ${exit_code} -ne 0 ]]; then
    printf 'tomato\t%s\t%s\t1\t%s\t%s\t%s\t%s\t%s\t\t\t%s\t%s\n' \
      "${trait}" "$([[ ${exit_code} -eq 2 ]] && echo SKIPPED_INCOMPLETE || echo CONVERT_FAILED)" \
      "${exit_code}" "$(date -Is)" "$(date -Is)" "${THREADS_PER_MODEL}" "${MODEL_WORKERS}" \
      "${OUT_ROOT}/logs/tomato_${trait}.convert.log" "${result_dir}" >>"${status_file}"
    return "${exit_code}"
  fi

  local genotype_file phenotype_file
  IFS=$'\t' read -r genotype_file phenotype_file <"${OUT_ROOT}/prepared/${trait}/prepared_paths.tsv"
  if [[ -z "${genotype_file}" || -z "${phenotype_file}" || ! -s "${genotype_file}" || ! -s "${phenotype_file}" ]]; then
    printf '[TRAIN-BLOCKED] tomato/%s prepared paths are invalid\n' "${trait}" | tee -a "${OUT_ROOT}/scheduler.log"
    printf 'tomato\t%s\tPREPARED_PATHS_INVALID\t2\t1\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${trait}" "$(date -Is)" "$(date -Is)" "${THREADS_PER_MODEL}" "${MODEL_WORKERS}" \
      "${genotype_file}" "${phenotype_file}" "${train_log}" "${result_dir}" >>"${status_file}"
    update_manifest_status "${trait}" "PREPARED_PATHS_INVALID"
    return 1
  fi
  mkdir -p "${result_dir}"
  start_time="$(date -Is)"
  printf '[TRAIN-START] tomato/%s at %s; MODEL_WORKERS=%s; THREADS_PER_MODEL=%s; repeats=50; folds=5; trials=%s; stacking=top3; provisional=true\n' \
    "${trait}" "${start_time}" "${MODEL_WORKERS}" "${THREADS_PER_MODEL}" "${TRIALS}" | tee -a "${OUT_ROOT}/scheduler.log"

  set +e
  python -m gpse.cli train \
    --geno_file "${genotype_file}" \
    --pheno_file "${phenotype_file}" \
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
    --results_dir "${result_dir}" >"${train_log}" 2>&1
  exit_code=$?
  set -e

  printf '[TRAIN-END] tomato/%s at %s; exit_code=%s\n' \
    "${trait}" "$(date -Is)" "${exit_code}" | tee -a "${OUT_ROOT}/scheduler.log"
  printf 'tomato\t%s\t%s\t1\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${trait}" "$([[ ${exit_code} -eq 0 ]] && echo COMPLETED || echo TRAIN_FAILED)" \
    "${exit_code}" "${start_time}" "$(date -Is)" "${THREADS_PER_MODEL}" "${MODEL_WORKERS}" \
    "${genotype_file}" "${phenotype_file}" "${train_log}" "${result_dir}" >>"${status_file}"

  if [[ ${exit_code} -eq 0 ]]; then
    update_manifest_status "${trait}" "COMPLETED"
  else
    update_manifest_status "${trait}" "TRAIN_FAILED"
  fi
  return "${exit_code}"
}

printf '[QUEUE-START] tomato provisional high-dimensional GPSE analysis at %s\n' "$(date -Is)" | tee -a "${OUT_ROOT}/scheduler.log"
for trait in "${TRAITS[@]}"; do
  if ! run_trait "${trait}"; then
    printf '[QUEUE-STOP] tomato/%s failed; queue stopped at %s\n' "${trait}" "$(date -Is)" | tee -a "${OUT_ROOT}/scheduler.log"
    exit 1
  fi
done
printf '[QUEUE-COMPLETE] tomato provisional high-dimensional GPSE analysis finished at %s\n' "$(date -Is)" | tee -a "${OUT_ROOT}/scheduler.log"
