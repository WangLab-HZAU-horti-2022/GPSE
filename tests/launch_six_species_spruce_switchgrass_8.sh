#!/usr/bin/env bash
set -uo pipefail

GPSE_ROOT="${GPSE_ROOT:-/data/jzhang/project/Temp/gpse_26.6.4_test/GPSE}"
DATA_ROOT="${DATA_ROOT:-/data/xiaoliu/gpse_6pecies_model/6specise}"
RUN_LABEL="${RUN_LABEL:-six_species_spruce_switchgrass_8core_20260806}"
OUT_ROOT="${OUT_ROOT:-${GPSE_ROOT}/tests/manuscript_runs/${RUN_LABEL}}"
THREADS="${THREADS:-8}"
TRIALS="${TRIALS:-50}"

REG_MODELS=(
  svr_reg kernelridge_reg knn_reg rf_reg gbdt_reg
  histgradientboost_reg adaboost_reg xgboost_reg
  lightgbm_reg catboost_reg gblup_reg
)

mkdir -p "${OUT_ROOT}"/{cv,logs,traits}
STATUS_FILE="${OUT_ROOT}/scheduler_status.tsv"
SCHEDULER_LOG="${OUT_ROOT}/scheduler.log"

if [[ ! -s "${STATUS_FILE}" ]]; then
  printf 'species\ttrait\tstatus\tattempt\texit_code\tstart_time\tend_time\tn_jobs\tmax_workers\trepeat_workers\tcv_file\tlog_file\tresult_dir\tnote\n' > "${STATUS_FILE}"
fi

log_scheduler() {
  printf '%s\n' "$*" | tee -a "${SCHEDULER_LOG}"
}

is_complete() {
  local result_dir="$1"
  local summary="${result_dir}/reports/run_summary.json"
  [[ -s "${summary}" ]] && \
    rg -q '"completed_models"[[:space:]]*:[[:space:]]*1[12]' "${summary}" && \
    ! rg -q '"failed_models"[[:space:]]*:[[:space:]]*[1-9]' "${summary}"
}

last_attempt() {
  local species="$1" trait="$2"
  awk -F '\t' -v species="${species}" -v trait="${trait}" \
    'NR > 1 && $1 == species && $2 == trait { value = $4 } END { print value + 0 }' "${STATUS_FILE}"
}

write_manifest() {
  local species="$1" trait="$2" manifest="$3" genotype="$4" phenotype="$5" cv_file="$6" result_dir="$7"
  cat > "${manifest}" <<EOF
# ${species}/${trait}: S1b base-model rerun (provisional)

- **Purpose:** Rebuild the 11 regression base models and top-3 stacking inputs required for Figure S1b.
- **Status:** Provisional S1b base-model rerun; use only after completion checks pass.
- **Genotype:** ${genotype}
- **Phenotype:** ${phenotype}; target column ${trait}
- **Fixed CV file:** ${cv_file} (5 folds × 50 repeats; seed 42)
- **Results directory:** ${result_dir}
- **Protocol:** regression; 11 specified base models; 50 Optuna trials; patience 20; n_jobs=${THREADS}, max_workers=1, repeat_workers=1; top-3 stacking.
- **Manual ensemble:** Not generated. It requires a user-defined model group and must not be inferred from this run.
EOF
}

run_trait() {
  local species="$1" trait="$2"
  local genotype="${DATA_ROOT}/geno/${species}_geno.csv"
  local phenotype="${DATA_ROOT}/pheno/${species}_pheno.csv"
  local trait_dir="${OUT_ROOT}/traits/${species}/${trait}"
  local result_dir="${trait_dir}/results"
  local cv_file="${OUT_ROOT}/cv/${species}/${trait}_cv_50x5.csv"
  local log_file="${OUT_ROOT}/logs/${species}_${trait}.log"
  local manifest="${trait_dir}/manifest.md"
  local start_time end_time exit_code attempt

  mkdir -p "${trait_dir}" "$(dirname "${cv_file}")" "$(dirname "${log_file}")" "${result_dir}"
  write_manifest "${species}" "${trait}" "${manifest}" "${genotype}" "${phenotype}" "${cv_file}" "${result_dir}"

  if is_complete "${result_dir}"; then
    log_scheduler "[SKIP] ${species}/${trait}: verified completed at $(date -Is)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${species}" "${trait}" "SKIPPED_COMPLETED" "$(last_attempt "${species}" "${trait}")" 0 "" "$(date -Is)" \
      "${THREADS}" 1 1 "${cv_file}" "${log_file}" "${result_dir}" "run_summary.json completed_models=11" >> "${STATUS_FILE}"
    return 0
  fi

  if [[ ! -s "${genotype}" || ! -s "${phenotype}" ]]; then
    log_scheduler "[FAILED_INPUT] ${species}/${trait}: missing input file"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${species}" "${trait}" "FAILED_INPUT" "$(($(last_attempt "${species}" "${trait}") + 1))" 2 "" "$(date -Is)" \
      "${THREADS}" 1 1 "${cv_file}" "${log_file}" "${result_dir}" "missing genotype or phenotype input" >> "${STATUS_FILE}"
    return 2
  fi

  attempt="$(($(last_attempt "${species}" "${trait}") + 1))"
  start_time="$(date -Is)"
  log_scheduler "[START] ${species}/${trait} at ${start_time}; S1b base-model rerun provisional; n_jobs=${THREADS}; max_workers=1; repeat_workers=1; repeats=50; folds=5; trials=${TRIALS}; stacking=top3"

  cd "${GPSE_ROOT}"
  export PYTHONPATH="${GPSE_ROOT}:${PYTHONPATH:-}"
  python -m gpse.cli train \
    --geno_file "${genotype}" \
    --pheno_file "${phenotype}" \
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
    --n_jobs "${THREADS}" \
    --max_workers 1 \
    --repeat_workers 1 \
    --save_representative \
    --results_dir "${result_dir}" > "${log_file}" 2>&1
  exit_code=$?
  end_time="$(date -Is)"

  if [[ "${exit_code}" -eq 0 ]] && is_complete "${result_dir}"; then
    log_scheduler "[COMPLETE] ${species}/${trait} at ${end_time}; attempt=${attempt}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${species}" "${trait}" "COMPLETED" "${attempt}" "${exit_code}" "${start_time}" "${end_time}" \
      "${THREADS}" 1 1 "${cv_file}" "${log_file}" "${result_dir}" "verified run_summary.json completed_models=11" >> "${STATUS_FILE}"
    return 0
  fi

  log_scheduler "[FAILED] ${species}/${trait} at ${end_time}; attempt=${attempt}; exit_code=${exit_code}; queue stops for inspection"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${species}" "${trait}" "FAILED" "${attempt}" "${exit_code}" "${start_time}" "${end_time}" \
    "${THREADS}" 1 1 "${cv_file}" "${log_file}" "${result_dir}" "inspect log before retry; no manual ensemble is defined" >> "${STATUS_FILE}"
  return "${exit_code}"
}

for item in spruce:DBH spruce:HT spruce:DE switchgrass:ST switchgrass:HT switchgrass:AN; do
  species="${item%%:*}"
  trait="${item##*:}"
  run_trait "${species}" "${trait}" || exit $?
done

log_scheduler "[COMPLETE] spruce + switchgrass S1b provisional base-model queue finished at $(date -Is)"
