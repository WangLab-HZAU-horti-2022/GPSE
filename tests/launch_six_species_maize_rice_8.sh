#!/usr/bin/env bash
set -euo pipefail

GPSE_ROOT="${GPSE_ROOT:-/data/jzhang/project/Temp/gpse_26.6.4_test/GPSE}"
DATA_ROOT="${DATA_ROOT:-/data/xiaoliu/gpse_6pecies_model/6specise}"
RUN_LABEL="${RUN_LABEL:-six_species_maize_rice_8core_20260806}"
OUT_ROOT="${OUT_ROOT:-${GPSE_ROOT}/tests/manuscript_runs/${RUN_LABEL}}"
THREADS="${THREADS:-8}"
TRIALS="${TRIALS:-50}"

REG_MODELS=(
  svr_reg kernelridge_reg knn_reg rf_reg gbdt_reg
  histgradientboost_reg adaboost_reg xgboost_reg
  lightgbm_reg catboost_reg gblup_reg
)

mkdir -p "${OUT_ROOT}/cv" "${OUT_ROOT}/logs" "${OUT_ROOT}/results"
cd "${GPSE_ROOT}"
export PYTHONPATH="${GPSE_ROOT}:${PYTHONPATH:-}"

status_file="${OUT_ROOT}/scheduler_status.tsv"
if [[ ! -f "${status_file}" ]]; then
  printf 'species\ttrait\tstatus\tattempt\texit_code\tstart_time\tend_time\tn_jobs\tmax_workers\trepeat_workers\tcv_file\tlog_file\tresult_dir\tnote\n' > "${status_file}"
fi

write_manifest() {
  local species="$1"
  local trait="$2"
  local manifest_dir="${OUT_ROOT}/manifests/${species}"
  mkdir -p "${manifest_dir}"
  cat > "${manifest_dir}/${trait}.md" <<EOF
# ${species}/${trait} S1b base-model rerun

- Status: provisional
- Scope: Figure S1b reproducible base-model rerun; ensemble reconstruction is not complete.
- Task: regression
- Models: ${REG_MODELS[*]}
- Protocol: 5 training folds, 50 repeats, 50 Optuna trials, patience 20, seed 42.
- Fixed CV: ${OUT_ROOT}/cv/${species}/${trait}_cv_50x5.csv
- Parallelism: n_jobs=${THREADS}, max_workers=1, repeat_workers=1.
- Stacking: enabled with top_n_models=3.
- Manual ensemble: pending user-defined model groups; no manual ensemble is inferred or fabricated by this queue.
EOF
}

is_complete() {
  local result_dir="$1"
  local summary="${result_dir}/reports/run_summary.json"
  # completed_models counts the stacking entry too: 11 base models -> 11 or 12
  [[ -f "${summary}" ]] && \
    rg -q '"completed_models"[[:space:]]*:[[:space:]]*1[12]' "${summary}" && \
    ! rg -q '"failed_models"[[:space:]]*:[[:space:]]*[1-9]' "${summary}"
}

run_trait() {
  local species="$1"
  local trait="$2"
  local geno_file="${DATA_ROOT}/geno/${species}_geno.csv"
  local pheno_file="${DATA_ROOT}/pheno/${species}_pheno.csv"
  local result_dir="${OUT_ROOT}/results/${species}/${trait}"
  local cv_file="${OUT_ROOT}/cv/${species}/${trait}_cv_50x5.csv"
  local log_file="${OUT_ROOT}/logs/${species}_${trait}.log"
  local started ended exit_code attempt

  write_manifest "${species}" "${trait}"
  mkdir -p "${result_dir}" "$(dirname "${cv_file}")"

  if is_complete "${result_dir}"; then
    printf '%s\t%s\tSKIPPED_COMPLETE\t0\t0\t-\t%s\t%s\t1\t1\t%s\t%s\t%s\talready complete\n' \
      "${species}" "${trait}" "$(date -Is)" "${THREADS}" "${cv_file}" "${log_file}" "${result_dir}" >> "${status_file}"
    printf '[SKIP] %s/%s already complete at %s\n' "${species}" "${trait}" "$(date -Is)" | tee -a "${OUT_ROOT}/scheduler.log"
    return 0
  fi

  attempt=$(awk -F '\t' -v species="${species}" -v trait="${trait}" '$1 == species && $2 == trait {count++} END {print count + 1}' "${status_file}")
  started="$(date -Is)"
  printf '[START] %s/%s at %s; S1b base-model rerun provisional; n_jobs=%s; max_workers=1; repeat_workers=1; repeats=50; folds=5; trials=%s; stacking=top3\n' \
    "${species}" "${trait}" "${started}" "${THREADS}" "${TRIALS}" | tee -a "${OUT_ROOT}/scheduler.log"
  printf '%s\t%s\tRUNNING\t%s\t-\t%s\t-\t%s\t1\t1\t%s\t%s\t%s\tS1b provisional base-model rerun in progress\n' \
    "${species}" "${trait}" "${attempt}" "${started}" "${THREADS}" "${cv_file}" "${log_file}" "${result_dir}" >> "${status_file}"

  set +e
  python -m gpse.cli train \
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
    --n_jobs "${THREADS}" \
    --max_workers 1 \
    --repeat_workers 1 \
    --save_representative \
    --results_dir "${result_dir}" > "${log_file}" 2>&1
  exit_code=$?
  set -e

  ended="$(date -Is)"
  if [[ "${exit_code}" -eq 0 ]] && is_complete "${result_dir}"; then
    printf '%s\t%s\tCOMPLETED\t%s\t%s\t%s\t%s\t%s\t1\t1\t%s\t%s\t%s\tS1b provisional; manual ensemble pending user definition\n' \
      "${species}" "${trait}" "${attempt}" "${exit_code}" "${started}" "${ended}" "${THREADS}" "${cv_file}" "${log_file}" "${result_dir}" >> "${status_file}"
    printf '[END] %s/%s completed at %s\n' "${species}" "${trait}" "${ended}" | tee -a "${OUT_ROOT}/scheduler.log"
    return 0
  fi

  printf '%s\t%s\tFAILED\t%s\t%s\t%s\t%s\t%s\t1\t1\t%s\t%s\t%s\tinspect log; queue halted\n' \
    "${species}" "${trait}" "${attempt}" "${exit_code}" "${started}" "${ended}" "${THREADS}" "${cv_file}" "${log_file}" "${result_dir}" >> "${status_file}"
  printf '[FAILED] %s/%s at %s; exit_code=%s; queue halted\n' "${species}" "${trait}" "${ended}" "${exit_code}" | tee -a "${OUT_ROOT}/scheduler.log"
  return "${exit_code}"
}

queue=(
  'maize HT' 'maize FT' 'maize YLD'
  'rice YLD' 'rice FT' 'rice HT'
)

cat > "${OUT_ROOT}/README.md" <<EOF
# Maize and rice Figure S1b reruns

This queue reruns the 11 specified regression base models for maize HT/FT/YLD
and rice YLD/FT/HT under the manuscript protocol. All outputs are provisional
until the Figure S1b ensemble definitions are finalized. The manual ensemble
is explicitly pending user-defined model groups and is not inferred here.

Queue order: ${queue[*]}
EOF

for item in "${queue[@]}"; do
  read -r species trait <<< "${item}"
  write_manifest "${species}" "${trait}"
done

for item in "${queue[@]}"; do
  read -r species trait <<< "${item}"
  run_trait "${species}" "${trait}"
done

printf '[COMPLETE] maize/rice S1b provisional base-model queue finished at %s\n' "$(date -Is)" | tee -a "${OUT_ROOT}/scheduler.log"
