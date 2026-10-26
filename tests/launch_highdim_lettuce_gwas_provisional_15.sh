#!/usr/bin/env bash
set -euo pipefail

GPSE_ROOT="${GPSE_ROOT:-/data/jzhang/project/Temp/gpse_26.6.4_test/GPSE}"
XIAOLIU_ROOT="${XIAOLIU_ROOT:-/data/xiaoliu}"
RUN_LABEL="${RUN_LABEL:-highdim_lettuce_gwas_provisional_15core_20260805}"
OUT_ROOT="${OUT_ROOT:-${GPSE_ROOT}/tests/manuscript_runs/${RUN_LABEL}}"
MODEL_WORKERS="${MODEL_WORKERS:-1}"
THREADS_PER_MODEL="${THREADS_PER_MODEL:-15}"
TRIALS="${TRIALS:-50}"

REG_MODELS=(
  svr_reg kernelridge_reg knn_reg rf_reg gbdt_reg
  histgradientboost_reg adaboost_reg xgboost_reg
  lightgbm_reg catboost_reg gblup_reg
)
TRAITS=(
  3D_Angle_LAA0 3D_Angle_LAA1 3D_Angle_LAA2
  3D_Angle_p1 3D_Angle_p3 3D_Angle_p5
)

mkdir -p "${OUT_ROOT}"/{prepared/lettuce_gwas,cv/lettuce_gwas,highdim/lettuce_gwas,logs}
cd "${GPSE_ROOT}"
export PYTHONPATH="${GPSE_ROOT}:${PYTHONPATH:-}"

manifest_file="${OUT_ROOT}/manifest.md"
input_audit_file="${OUT_ROOT}/input_audit.tsv"
prepared_files_file="${OUT_ROOT}/prepared_files.tsv"
status_file="${OUT_ROOT}/scheduler_status.tsv"
scheduler_log="${OUT_ROOT}/scheduler.log"

write_manifest() {
  cat >"${manifest_file}" <<EOF
# Provisional lettuce high-dimensional GWAS-P analysis manifest

- **Status:** PROVISIONAL — the formal manuscript trait manifest has not been frozen.
- **Run label:** ${RUN_LABEL}
- **Date created:** $(date -Is)
- **Traits:** ${TRAITS[*]}
- **Input SNP provenance:** pre-existing per-trait PLINK data in
  ${XIAOLIU_ROOT}/GBLUP_RRBLUP_lettuce/lettuce_gwas/<trait>/genotype.{bed,bim,fam}.
- **EMMAX status:** the relationship of these SNPs to EMMAX top-5000 selection has **not**
  been confirmed. Manually audit EMMAX logs/SNP manifests before treating this run as a
  manuscript-final GWAS-P analysis.
- **Method scope:** GPSE univariate feature selection or GPSE regression must **not** be
  described as EMMAX. This run performs GPSE regression on existing PLINK SNP inputs.
- **Protocol:** 50 repeats, 5 outer folds, 50 Optuna trials, fixed seed 42, top-3 stacking.
- **Resources:** MODEL_WORKERS=${MODEL_WORKERS}; THREADS_PER_MODEL=${THREADS_PER_MODEL};
  repeat_workers=1; traits run sequentially.
EOF
}

append_status() {
  local trait="$1" status="$2" attempt="$3" exit_code="$4" start_time="$5" end_time="$6" prepared_geno="$7" prepared_pheno="$8" details="$9"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "lettuce" "${trait}" "${status}" "${attempt}" "${exit_code}" "${start_time}" "${end_time}" \
    "${MODEL_WORKERS}" "${THREADS_PER_MODEL}" "${prepared_geno}" "${prepared_pheno}" \
    >>"${status_file}"
  printf '[%s] lettuce/%s %s: %s\n' "$(date -Is)" "${trait}" "${status}" "${details}" | tee -a "${scheduler_log}"
}

find_prepared_file() {
  local prepared_dir="$1" suffix="$2"
  local -a candidates=()
  [[ -d "${prepared_dir}" ]] || return 1
  while IFS= read -r candidate; do
    candidates+=("${candidate}")
  done < <(find "${prepared_dir}" -maxdepth 1 -type f -name "*${suffix}.csv" -print | sort)

  if (( ${#candidates[@]} == 1 )); then
    printf '%s\n' "${candidates[0]}"
    return 0
  fi
  return 1
}

verify_inputs() {
  : >"${input_audit_file}"
  printf 'species\ttrait\tinput\tstatus\tbytes\tpath\n' >>"${input_audit_file}"
  local trait input input_dir complete
  for trait in "${TRAITS[@]}"; do
    input_dir="${XIAOLIU_ROOT}/GBLUP_RRBLUP_lettuce/lettuce_gwas/${trait}"
    complete=1
    for input in genotype.bed genotype.bim genotype.fam phenotype.txt; do
      if [[ -s "${input_dir}/${input}" ]]; then
        printf 'lettuce\t%s\t%s\tOK\t%s\t%s\n' "${trait}" "${input}" "$(stat -c '%s' "${input_dir}/${input}")" "${input_dir}/${input}" >>"${input_audit_file}"
      else
        printf 'lettuce\t%s\t%s\tMISSING\t0\t%s\n' "${trait}" "${input}" "${input_dir}/${input}" >>"${input_audit_file}"
        complete=0
      fi
    done
    if (( ! complete )); then
      printf '[SKIP] lettuce/%s has incomplete PLINK or phenotype input\n' "${trait}" | tee -a "${scheduler_log}"
    fi
  done
}

convert_trait() {
  local trait="$1" input_dir="$2" prepared_dir="$3" prefix="$4" log_file="$5"
  local prepared_geno prepared_pheno
  prepared_geno="$(find_prepared_file "${prepared_dir}" '_genotype' || true)"
  prepared_pheno="$(find_prepared_file "${prepared_dir}" '_phenotype' || true)"
  if [[ -n "${prepared_geno}" && -n "${prepared_pheno}" ]]; then
    printf '%s\t%s\t%s\t%s\t%s\n' "${trait}" "${prepared_geno}" "${prepared_pheno}" "$(sha256sum "${prepared_geno}" | awk '{print $1}')" "$(sha256sum "${prepared_pheno}" | awk '{print $1}')" >>"${prepared_files_file}"
    return 0
  fi

  mkdir -p "${prepared_dir}"
  printf '[CONVERT] lettuce/%s at %s\n' "${trait}" "$(date -Is)" | tee -a "${scheduler_log}"
  python -m gpse.cli convert \
    --bfile "${input_dir}/genotype" \
    --pheno "${input_dir}/phenotype.txt" \
    --trait-name "${trait}" \
    --out-prefix "${prefix}" \
    --direct \
    --out-format csv >>"${log_file}" 2>&1

  prepared_geno="$(find_prepared_file "${prepared_dir}" '_genotype' || true)"
  prepared_pheno="$(find_prepared_file "${prepared_dir}" '_phenotype' || true)"
  if [[ -z "${prepared_geno}" || -z "${prepared_pheno}" ]]; then
    printf '[CONVERT_FAILED] lettuce/%s; unable to discover one prepared genotype and phenotype CSV in %s\n' "${trait}" "${prepared_dir}" | tee -a "${scheduler_log}"
    return 1
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' "${trait}" "${prepared_geno}" "${prepared_pheno}" "$(sha256sum "${prepared_geno}" | awk '{print $1}')" "$(sha256sum "${prepared_pheno}" | awk '{print $1}')" >>"${prepared_files_file}"
}

run_trait() {
  local trait="$1" input_dir prepared_dir prefix result_dir cv_file log_file
  local start_time end_time exit_code prepared_geno prepared_pheno
  input_dir="${XIAOLIU_ROOT}/GBLUP_RRBLUP_lettuce/lettuce_gwas/${trait}"
  prepared_dir="${OUT_ROOT}/prepared/lettuce_gwas/${trait}"
  prefix="${prepared_dir}/${trait}"
  result_dir="${OUT_ROOT}/highdim/lettuce_gwas/${trait}"
  cv_file="${OUT_ROOT}/cv/lettuce_gwas/${trait}_cv_50x5.csv"
  log_file="${OUT_ROOT}/logs/lettuce_${trait}.log"

  if [[ ! -s "${input_dir}/genotype.bed" || ! -s "${input_dir}/genotype.bim" || ! -s "${input_dir}/genotype.fam" || ! -s "${input_dir}/phenotype.txt" ]]; then
    append_status "${trait}" "SKIPPED_INCOMPLETE_INPUT" 0 NA NA "$(date -Is)" NA NA "required input missing"
    return 0
  fi
  if [[ -f "${result_dir}/reports/run_summary.json" ]] && rg -q '"completed_models": 1[12]' "${result_dir}/reports/run_summary.json" && ! rg -q '"failed_models": [1-9]' "${result_dir}/reports/run_summary.json"; then
    append_status "${trait}" "SKIPPED_COMPLETED" 0 0 NA "$(date -Is)" NA NA "existing complete run summary"
    return 0
  fi

  start_time="$(date -Is)"
  : >"${log_file}"
  if ! convert_trait "${trait}" "${input_dir}" "${prepared_dir}" "${prefix}" "${log_file}"; then
    append_status "${trait}" "FAILED_CONVERSION" 1 1 "${start_time}" "$(date -Is)" NA NA "see ${log_file}"
    return 1
  fi
  prepared_geno="$(find_prepared_file "${prepared_dir}" '_genotype')"
  prepared_pheno="$(find_prepared_file "${prepared_dir}" '_phenotype')"

  mkdir -p "${result_dir}"
  printf '[START] lettuce/%s at %s; PROVISIONAL; model_workers=%s; threads_per_model=%s; repeats=50; folds=5; trials=%s; stacking=top3\n' \
    "${trait}" "${start_time}" "${MODEL_WORKERS}" "${THREADS_PER_MODEL}" "${TRIALS}" | tee -a "${scheduler_log}"
  if python -m gpse.cli train \
    --geno_file "${prepared_geno}" \
    --pheno_file "${prepared_pheno}" \
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
    --results_dir "${result_dir}" >>"${log_file}" 2>&1; then
    exit_code=0
  else
    exit_code=$?
  fi
  end_time="$(date -Is)"
  if (( exit_code == 0 )) && [[ -f "${result_dir}/reports/run_summary.json" ]] && rg -q '"completed_models": 1[12]' "${result_dir}/reports/run_summary.json" && ! rg -q '"failed_models": [1-9]' "${result_dir}/reports/run_summary.json"; then
    append_status "${trait}" "COMPLETED" 1 0 "${start_time}" "${end_time}" "${prepared_geno}" "${prepared_pheno}" "all expected models completed"
    return 0
  fi
  append_status "${trait}" "FAILED_TRAINING" 1 "${exit_code}" "${start_time}" "${end_time}" "${prepared_geno}" "${prepared_pheno}" "see ${log_file}"
  return 1
}

write_manifest
verify_inputs
if [[ ! -s "${prepared_files_file}" ]]; then
  printf 'trait\tgenotype_csv\tphenotype_csv\tgenotype_sha256\tphenotype_sha256\n' >"${prepared_files_file}"
fi
if [[ ! -s "${status_file}" ]]; then
  printf 'species\ttrait\tstatus\tattempt\texit_code\tstart_time\tend_time\tmodel_workers\tthreads_per_model\tgenotype_csv\tphenotype_csv\n' >"${status_file}"
fi
printf '[QUEUE_START] provisional lettuce high-dimensional GPSE regression at %s\n' "$(date -Is)" | tee -a "${scheduler_log}"

for trait in "${TRAITS[@]}"; do
  if ! run_trait "${trait}"; then
    printf '[QUEUE_STOP] lettuce/%s failed; preserving outputs and stopping at %s\n' "${trait}" "$(date -Is)" | tee -a "${scheduler_log}"
    exit 1
  fi
done

printf '[QUEUE_COMPLETE] all provisional lettuce traits completed at %s\n' "$(date -Is)" | tee -a "${scheduler_log}"
