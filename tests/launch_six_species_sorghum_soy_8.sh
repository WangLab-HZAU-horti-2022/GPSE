#!/usr/bin/env bash
set -uo pipefail

GPSE_ROOT="${GPSE_ROOT:-/data/jzhang/project/Temp/gpse_26.6.4_test/GPSE}"
DATA_ROOT="${DATA_ROOT:-/data/xiaoliu/gpse_6pecies_model/6specise}"
RUN_LABEL="${RUN_LABEL:-six_species_sorghum_soy_8core_20260806}"
OUT_ROOT="${OUT_ROOT:-${GPSE_ROOT}/tests/manuscript_runs/${RUN_LABEL}}"
N_JOBS="${N_JOBS:-8}"
MAX_WORKERS="${MAX_WORKERS:-1}"
REPEAT_WORKERS="${REPEAT_WORKERS:-1}"
TRIALS="${TRIALS:-50}"

REG_MODELS=(
  svr_reg kernelridge_reg knn_reg rf_reg gbdt_reg
  histgradientboost_reg adaboost_reg xgboost_reg
  lightgbm_reg catboost_reg gblup_reg
)

mkdir -p "${OUT_ROOT}/cv" "${OUT_ROOT}/logs" "${OUT_ROOT}/lowdim" "${OUT_ROOT}/manifests"
cd "${GPSE_ROOT}"
export PYTHONPATH="${GPSE_ROOT}:${PYTHONPATH:-}"

STATUS_FILE="${OUT_ROOT}/scheduler_status.tsv"
SCHEDULER_LOG="${OUT_ROOT}/scheduler.log"

if [[ ! -s "${STATUS_FILE}" ]]; then
  printf 'timestamp\tspecies\ttrait\tstatus\tattempt\texit_code\tn_jobs\tmax_workers\trepeat_workers\tcv_file\tlog_file\tresult_dir\tnote\n' > "${STATUS_FILE}"
fi

write_manifest() {
  local species="$1"
  local traits="$2"
  local manifest_file="${OUT_ROOT}/manifests/${species}_manifest.md"

  cat > "${manifest_file}" <<EOF
# ${species} S1b base-model rerun manifest

- Status: provisional
- Scope: Figure S1b base-model rerun only
- Traits: ${traits}
- Task: regression
- Models: ${REG_MODELS[*]}
- Protocol: 5 training folds, 50 repeats, 50 Optuna trials, patience 20, seed 42
- CV: fixed per-trait 50x5 folds derived from ${DATA_ROOT}/cv/${species}_CVFs.csv; source cv_1--cv_50 are renamed to cv0--cv49 and fold labels 1--5 are converted to 0--4 for GPSE.
- Parallelism: n_jobs=${N_JOBS}, max_workers=${MAX_WORKERS}, repeat_workers=${REPEAT_WORKERS}
- Stacking: enabled with top_n_models=3 and cv_folds=5
- Manual ensemble: pending user-defined model groups; no manual ensemble is inferred or fabricated by this run.
EOF
}

write_status() {
  local species="$1"
  local trait="$2"
  local status="$3"
  local attempt="$4"
  local exit_code="$5"
  local cv_file="$6"
  local log_file="$7"
  local result_dir="$8"
  local note="$9"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -Is)" "${species}" "${trait}" "${status}" "${attempt}" "${exit_code}" \
    "${N_JOBS}" "${MAX_WORKERS}" "${REPEAT_WORKERS}" "${cv_file}" "${log_file}" \
    "${result_dir}" "${note}" >> "${STATUS_FILE}"
}

prepare_fixed_cv() {
  local species="$1"
  local trait="$2"
  local source_cv="${DATA_ROOT}/cv/${species}_CVFs.csv"
  local cv_file="${OUT_ROOT}/cv/${species}/${trait}_fixed_cv_50x5.csv"
  local temporary_cv="${cv_file}.tmp.$$"

  mkdir -p "$(dirname "${cv_file}")"
  if [[ ! -s "${cv_file}" ]]; then
    awk -F, 'BEGIN { OFS="," }
      NR == 1 {
        printf "ID"
        for (column = 0; column < 50; column++) printf "%s%s", OFS, "cv" column
        printf "\n"
        next
      }
      {
        printf "%s", $1
        for (column = 2; column <= 51; column++) printf "%s%d", OFS, $column - 1
        printf "\n"
      }' "${source_cv}" > "${temporary_cv}" && mv "${temporary_cv}" "${cv_file}"
  fi

  printf '%s\n' "${cv_file}"
}

trait_is_complete() {
  local result_dir="$1"
  local summary_file="${result_dir}/reports/run_summary.json"
  [[ -f "${summary_file}" ]] && \
    rg -q '"completed_models": 1[12]' "${summary_file}" && \
    rg -q '"failed_models": 0' "${summary_file}"
}

run_trait() {
  local species="$1"
  local trait="$2"
  local result_dir="${OUT_ROOT}/lowdim/${species}/${trait}"
  local log_file="${OUT_ROOT}/logs/${species}_${trait}.log"
  local cv_file
  local attempt
  local exit_code

  cv_file="$(prepare_fixed_cv "${species}" "${trait}")"
  attempt="$(( $(awk -F'\t' -v species="${species}" -v trait="${trait}" 'NR > 1 && $2 == species && $3 == trait { count++ } END { print count + 1 }' "${STATUS_FILE}") ))"

  if trait_is_complete "${result_dir}"; then
    printf '[SKIP] %s/%s already completed at %s\n' "${species}" "${trait}" "$(date -Is)" | tee -a "${SCHEDULER_LOG}"
    write_status "${species}" "${trait}" "SKIPPED_COMPLETE" "${attempt}" "0" "${cv_file}" "${log_file}" "${result_dir}" "validated_completed_models_11"
    return 0
  fi

  mkdir -p "${result_dir}"
  printf '[START] %s/%s at %s; S1b_base_model_rerun=provisional; n_jobs=%s; max_workers=%s; repeat_workers=%s; repeats=50; folds=5; trials=%s; stacking=top3\n' \
    "${species}" "${trait}" "$(date -Is)" "${N_JOBS}" "${MAX_WORKERS}" "${REPEAT_WORKERS}" "${TRIALS}" | tee -a "${SCHEDULER_LOG}"
  write_status "${species}" "${trait}" "RUNNING" "${attempt}" "" "${cv_file}" "${log_file}" "${result_dir}" "S1b_base_model_rerun_provisional"

  if python -m gpse.cli train \
    --geno_file "${DATA_ROOT}/geno/${species}_geno.csv" \
    --pheno_file "${DATA_ROOT}/pheno/${species}_pheno.csv" \
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
    --n_jobs "${N_JOBS}" \
    --max_workers "${MAX_WORKERS}" \
    --repeat_workers "${REPEAT_WORKERS}" \
    --save_representative \
    --results_dir "${result_dir}" > "${log_file}" 2>&1; then
    exit_code=0
  else
    exit_code=$?
  fi

  if [[ "${exit_code}" -eq 0 ]] && trait_is_complete "${result_dir}"; then
    printf '[END] %s/%s at %s; exit_code=0; status=COMPLETED\n' "${species}" "${trait}" "$(date -Is)" | tee -a "${SCHEDULER_LOG}"
    write_status "${species}" "${trait}" "COMPLETED" "${attempt}" "0" "${cv_file}" "${log_file}" "${result_dir}" "validated_completed_models_11"
    return 0
  fi

  printf '[END] %s/%s at %s; exit_code=%s; status=FAILED_OR_INCOMPLETE\n' \
    "${species}" "${trait}" "$(date -Is)" "${exit_code}" | tee -a "${SCHEDULER_LOG}"
  write_status "${species}" "${trait}" "FAILED_OR_INCOMPLETE" "${attempt}" "${exit_code}" "${cv_file}" "${log_file}" "${result_dir}" "inspect_log_before_retry"
  return 1
}

write_manifest "sorghum" "YLD, MO, HT"
write_manifest "soy" "HT, R8, YLD"

failures=0
for entry in sorghum:YLD sorghum:MO sorghum:HT soy:HT soy:R8 soy:YLD; do
  species="${entry%%:*}"
  trait="${entry##*:}"
  if ! run_trait "${species}" "${trait}"; then
    failures=$((failures + 1))
    printf '[QUEUE] Continuing after %s/%s failure at %s\n' "${species}" "${trait}" "$(date -Is)" | tee -a "${SCHEDULER_LOG}"
  fi
done

if [[ "${failures}" -eq 0 ]]; then
  printf '[COMPLETE] sorghum+soy S1b base-model queue finished at %s\n' "$(date -Is)" | tee -a "${SCHEDULER_LOG}"
else
  printf '[COMPLETE_WITH_FAILURES] sorghum+soy queue finished at %s; failed_traits=%s\n' "$(date -Is)" "${failures}" | tee -a "${SCHEDULER_LOG}"
fi
