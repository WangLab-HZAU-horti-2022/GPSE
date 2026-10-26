#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUMMARY_LOG="${ROOT}/manuscript_runs/s1b_strict_watcher_20260806.log"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-300}"
MODELS=(
  svr_reg kernelridge_reg knn_reg rf_reg gbdt_reg histgradientboost_reg
  adaboost_reg xgboost_reg lightgbm_reg catboost_reg gblup_reg
)

TRAITS=(
  "maize|HT|${ROOT}/manuscript_runs/six_species_maize_rice_8core_20260806/results/maize/HT"
  "maize|FT|${ROOT}/manuscript_runs/six_species_maize_rice_8core_20260806/results/maize/FT"
  "maize|YLD|${ROOT}/manuscript_runs/six_species_maize_rice_8core_20260806/results/maize/YLD"
  "rice|YLD|${ROOT}/manuscript_runs/six_species_maize_rice_8core_20260806/results/rice/YLD"
  "rice|FT|${ROOT}/manuscript_runs/six_species_maize_rice_8core_20260806/results/rice/FT"
  "rice|HT|${ROOT}/manuscript_runs/six_species_maize_rice_8core_20260806/results/rice/HT"
  "sorghum|YLD|${ROOT}/manuscript_runs/six_species_sorghum_soy_8core_20260806/lowdim/sorghum/YLD"
  "sorghum|MO|${ROOT}/manuscript_runs/six_species_sorghum_soy_8core_20260806/lowdim/sorghum/MO"
  "sorghum|HT|${ROOT}/manuscript_runs/six_species_sorghum_soy_8core_20260806/lowdim/sorghum/HT"
  "soy|HT|${ROOT}/manuscript_runs/six_species_sorghum_soy_8core_20260806/lowdim/soy/HT"
  "soy|R8|${ROOT}/manuscript_runs/six_species_sorghum_soy_8core_20260806/lowdim/soy/R8"
  "soy|YLD|${ROOT}/manuscript_runs/six_species_sorghum_soy_8core_20260806/lowdim/soy/YLD"
  "spruce|DBH|${ROOT}/manuscript_runs/six_species_spruce_switchgrass_8core_20260806/traits/spruce/DBH/results"
  "spruce|HT|${ROOT}/manuscript_runs/six_species_spruce_switchgrass_8core_20260806/traits/spruce/HT/results"
  "spruce|DE|${ROOT}/manuscript_runs/six_species_spruce_switchgrass_8core_20260806/traits/spruce/DE/results"
  "switchgrass|ST|${ROOT}/manuscript_runs/six_species_spruce_switchgrass_8core_20260806/traits/switchgrass/ST/results"
  "switchgrass|HT|${ROOT}/manuscript_runs/six_species_spruce_switchgrass_8core_20260806/traits/switchgrass/HT/results"
  "switchgrass|AN|${ROOT}/manuscript_runs/six_species_spruce_switchgrass_8core_20260806/traits/switchgrass/AN/results"
)

check_trait() {
  local species="$1" trait="$2" results_dir="$3" model repeat completed=0
  local output_dir="${results_dir}/s1b_ensembles"
  local marker="${output_dir}/.strict_complete"

  if [[ -f "${marker}" ]]; then
    printf '%s\t%s\tcomplete\n' "${species}" "${trait}"
    return
  fi
  if [[ ! -f "${results_dir}/test_ids.txt" ]]; then
    printf '%s\t%s\twaiting_for_trait\n' "${species}" "${trait}"
    return
  fi
  for model in "${MODELS[@]}"; do
    for repeat in $(seq 1 50); do
      [[ -f "${results_dir}/${model}/repeat_${repeat}/repeat_results.json" ]] && ((completed += 1)) || true
    done
  done
  if [[ "${completed}" -ne 550 ]]; then
    printf '%s\t%s\twaiting\t%d/550\n' "${species}" "${trait}" "${completed}"
    return
  fi
  mkdir -p "${output_dir}"
  if "${ROOT}/assemble_s1b_ensembles.py" \
      --results-dir "${results_dir}" \
      --species "${species}" \
      --trait "${trait}" \
      --output-dir "${output_dir}"; then
    date -Is > "${marker}"
    printf '%s\t%s\taggregated\n' "${species}" "${trait}"
  else
    printf '%s\t%s\taggregation_failed\n' "${species}" "${trait}" >&2
  fi
}

run_once() {
  local entry species trait results_dir
  while IFS= read -r entry; do
    species=${entry%%|*}
    entry=${entry#*|}
    trait=${entry%%|*}
    results_dir=${entry#*|}
    check_trait "${species}" "${trait}" "${results_dir}"
  done < <(printf '%s\n' "${TRAITS[@]}")
}

mkdir -p "$(dirname "${SUMMARY_LOG}")"
if [[ "${1:-}" == "--once" ]]; then
  run_once | tee -a "${SUMMARY_LOG}"
  exit 0
fi

while true; do
  printf '[%s]\n' "$(date -Is)" | tee -a "${SUMMARY_LOG}"
  run_once | tee -a "${SUMMARY_LOG}"
  sleep "${INTERVAL_SECONDS}"
done
