#!/usr/bin/env python3
"""Assemble auditable Figure S1b regression ensembles from GPSE repeat outputs.

The script deliberately keeps model selection and meta-model fitting on the
cross-validation validation predictions saved by GPSE.  The held-out ``test``
predictions are used only once, for reporting.  A manual ensemble is emitted
only when it is declared in a user-supplied manifest; this prevents accidental
post-hoc selection of a favourable model combination.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_MODELS = (
    "svr_reg,kernelridge_reg,knn_reg,rf_reg,gbdt_reg,histgradientboost_reg,"
    "adaboost_reg,xgboost_reg,lightgbm_reg,catboost_reg,gblup_reg"
)
RIDGE_ALPHAS = (1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0)


@dataclass(frozen=True)
class PredictionSet:
    ids: np.ndarray
    true_values: np.ndarray
    predictions: np.ndarray


@dataclass(frozen=True)
class ModelRepeat:
    model: str
    repeat: int
    validation: PredictionSet
    test: PredictionSet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        help="One trait's GPSE results directory.")
    parser.add_argument("--output-dir", type=Path,
                        help="Defaults to <results-dir>/s1b_ensembles.")
    parser.add_argument("--species", default="", help="Optional manifest filter.")
    parser.add_argument("--trait", default="", help="Optional manifest filter.")
    parser.add_argument("--models", default=DEFAULT_MODELS,
                        help="Comma-separated required base-model names.")
    parser.add_argument("--expected-repeats", type=int, default=50)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--manual-manifest", type=Path,
                        help="CSV with ensemble_name,models and optional species,trait.")
    parser.add_argument("--allow-partial", action="store_true",
                        help="For live monitoring only: use finished models/repeats that exist.")
    parser.add_argument("--write-manifest-template", type=Path,
                        help="Write a manual ensemble manifest template and exit.")
    return parser.parse_args()


def write_manifest_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("species", "trait", "ensemble_name", "models"))
        writer.writeheader()
        writer.writerow({
            "species": "example_species",
            "trait": "example_trait",
            "ensemble_name": "manual_predeclared",
            "models": "svr_reg;rf_reg;gblup_reg",
        })


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return math.nan
    y_true_centered = y_true - np.mean(y_true)
    y_pred_centered = y_pred - np.mean(y_pred)
    denominator = math.sqrt(float(np.sum(y_true_centered ** 2) * np.sum(y_pred_centered ** 2)))
    return float(np.sum(y_true_centered * y_pred_centered) / denominator) if denominator else math.nan


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(np.mean((y_true - y_pred) ** 2))
    total_sum_squares = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "pearson": pearson(y_true, y_pred),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "r2": 1.0 - float(np.sum((y_true - y_pred) ** 2)) / total_sum_squares if total_sum_squares else math.nan,
    }


def load_partition(predictions: dict[str, Any], partition: str, aggregate_duplicates: bool) -> PredictionSet:
    folds = predictions.get(partition)
    if not isinstance(folds, list) or not folds:
        raise ValueError(f"Missing or empty {partition!r} predictions")

    values: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for fold_index, fold in enumerate(folds, start=1):
        if not isinstance(fold, dict):
            raise ValueError(f"{partition} fold {fold_index} is not an object")
        indices = fold.get("indices")
        true_values = fold.get("true_values")
        predicted_values = fold.get("predicted_values")
        if not (isinstance(indices, list) and isinstance(true_values, list) and isinstance(predicted_values, list)):
            raise ValueError(f"{partition} fold {fold_index} lacks indices/true_values/predicted_values")
        if not (len(indices) == len(true_values) == len(predicted_values)):
            raise ValueError(f"{partition} fold {fold_index} has unequal array lengths")
        for sample_id, true_value, predicted_value in zip(indices, true_values, predicted_values):
            values[int(sample_id)].append((float(true_value), float(predicted_value)))

    if not aggregate_duplicates and any(len(items) != 1 for items in values.values()):
        duplicate_ids = [sample_id for sample_id, items in values.items() if len(items) != 1]
        raise ValueError(f"{partition} contains duplicate sample IDs, e.g. {duplicate_ids[:5]}")

    ordered_ids = np.array(sorted(values), dtype=int)
    y_true: list[float] = []
    y_pred: list[float] = []
    for sample_id in ordered_ids:
        items = values[int(sample_id)]
        labels = np.array([item[0] for item in items], dtype=float)
        if not np.allclose(labels, labels[0], rtol=1e-8, atol=1e-10):
            raise ValueError(f"{partition} has inconsistent labels for sample {sample_id}")
        y_true.append(float(labels[0]))
        y_pred.append(float(np.mean([item[1] for item in items])))
    return PredictionSet(ordered_ids, np.array(y_true), np.array(y_pred))


def load_model_repeats(results_dir: Path, model: str) -> tuple[dict[int, ModelRepeat], list[str]]:
    repeats: dict[int, ModelRepeat] = {}
    errors: list[str] = []
    model_dir = results_dir / model
    if not model_dir.is_dir():
        return repeats, [f"Missing model directory: {model_dir}"]
    for repeat_dir in sorted(model_dir.glob("repeat_*"), key=lambda path: int(path.name.split("_")[-1])):
        try:
            repeat = int(repeat_dir.name.removeprefix("repeat_"))
            prediction_path = repeat_dir / "all_predictions.json"
            result_path = repeat_dir / "repeat_results.json"
            if not prediction_path.exists() or not result_path.exists():
                continue
            with prediction_path.open() as handle:
                predictions = json.load(handle)
            with result_path.open() as handle:
                result = json.load(handle)
            result_repeat = int(result.get("repeat_idx", repeat - 1)) + 1
            if result_repeat != repeat:
                raise ValueError(f"repeat index mismatch: directory={repeat}, result={result_repeat}")
            repeats[repeat] = ModelRepeat(
                model=model,
                repeat=repeat,
                validation=load_partition(predictions, "val", aggregate_duplicates=False),
                test=load_partition(predictions, "test", aggregate_duplicates=True),
            )
        except Exception as error:
            errors.append(f"{repeat_dir}: {error}")
    return repeats, errors


def align_prediction_sets(sets: Iterable[PredictionSet]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prediction_sets = list(sets)
    if not prediction_sets:
        raise ValueError("No prediction sets supplied")
    shared_ids = set(prediction_sets[0].ids.tolist())
    for prediction_set in prediction_sets[1:]:
        shared_ids.intersection_update(prediction_set.ids.tolist())
    if len(shared_ids) < 3:
        raise ValueError("Fewer than three shared samples across selected models")
    ids = np.array(sorted(shared_ids), dtype=int)
    y_columns: list[np.ndarray] = []
    prediction_columns: list[np.ndarray] = []
    for prediction_set in prediction_sets:
        positions = {int(sample_id): position for position, sample_id in enumerate(prediction_set.ids)}
        row_positions = np.array([positions[int(sample_id)] for sample_id in ids])
        y_columns.append(prediction_set.true_values[row_positions])
        prediction_columns.append(prediction_set.predictions[row_positions])
    first_labels = y_columns[0]
    if not all(np.allclose(first_labels, labels, rtol=1e-8, atol=1e-10) for labels in y_columns[1:]):
        raise ValueError("True labels disagree across selected model predictions")
    return ids, first_labels, np.column_stack(prediction_columns)


def fit_ridge(x_values: np.ndarray, y_values: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_means = np.mean(x_values, axis=0)
    feature_scales = np.std(x_values, axis=0)
    feature_scales[feature_scales == 0] = 1.0
    x_standardized = (x_values - feature_means) / feature_scales
    y_mean = float(np.mean(y_values))
    coefficients = np.linalg.solve(
        x_standardized.T @ x_standardized + alpha * np.eye(x_standardized.shape[1]),
        x_standardized.T @ (y_values - y_mean),
    )
    return coefficients, feature_means, np.array([*feature_scales, y_mean])


def ridge_predict(x_values: np.ndarray, fitted: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    coefficients, feature_means, scales_and_y_mean = fitted
    feature_scales = scales_and_y_mean[:-1]
    y_mean = scales_and_y_mean[-1]
    return ((x_values - feature_means) / feature_scales) @ coefficients + y_mean


def choose_ridge_alpha(x_values: np.ndarray, y_values: np.ndarray) -> float:
    folds = np.array_split(np.arange(len(y_values)), min(5, len(y_values)))
    best_alpha = RIDGE_ALPHAS[0]
    best_score = -math.inf
    for alpha in RIDGE_ALPHAS:
        scores: list[float] = []
        for validation_indices in folds:
            train_mask = np.ones(len(y_values), dtype=bool)
            train_mask[validation_indices] = False
            if int(np.sum(train_mask)) <= x_values.shape[1]:
                continue
            fitted = fit_ridge(x_values[train_mask], y_values[train_mask], alpha)
            score = pearson(y_values[validation_indices], ridge_predict(x_values[validation_indices], fitted))
            if math.isfinite(score):
                scores.append(score)
        score = float(np.mean(scores)) if scores else -math.inf
        if score > best_score:
            best_alpha, best_score = alpha, score
    return best_alpha


def make_ensemble_row(
    repeat: int,
    kind: str,
    name: str,
    selected: list[ModelRepeat],
) -> dict[str, Any]:
    _, validation_y, validation_x = align_prediction_sets(item.validation for item in selected)
    _, test_y, test_x = align_prediction_sets(item.test for item in selected)
    alpha = choose_ridge_alpha(validation_x, validation_y)
    fitted = fit_ridge(validation_x, validation_y, alpha)
    validation_metrics = metric_values(validation_y, ridge_predict(validation_x, fitted))
    test_metrics = metric_values(test_y, ridge_predict(test_x, fitted))
    return {
        "repeat": repeat,
        "row_type": "ensemble",
        "ensemble_kind": kind,
        "ensemble_name": name,
        "selected_models": ";".join(item.model for item in selected),
        "n_models": len(selected),
        "ridge_alpha": alpha,
        "n_validation_samples": len(validation_y),
        "n_test_samples": len(test_y),
        **{f"validation_{key}": value for key, value in validation_metrics.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }


def read_manual_manifest(path: Path | None, species: str, trait: str) -> list[tuple[str, list[str]]]:
    if path is None:
        return []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"ensemble_name", "models"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must have columns: ensemble_name,models")
        definitions: list[tuple[str, list[str]]] = []
        for row in reader:
            if row.get("species", "") not in ("", species) or row.get("trait", "") not in ("", trait):
                continue
            models = [model.strip() for model in row["models"].replace(",", ";").split(";") if model.strip()]
            if len(models) < 2:
                raise ValueError(f"Manual ensemble {row['ensemble_name']!r} contains fewer than two models")
            definitions.append((row["ensemble_name"], models))
    return definitions


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numeric_fields = ("validation_pearson", "test_pearson", "test_mse", "test_rmse", "test_r2")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["ensemble_kind"]), str(row["ensemble_name"]))].append(row)
    summaries: list[dict[str, Any]] = []
    for (kind, name), group in sorted(grouped.items()):
        summary: dict[str, Any] = {"ensemble_kind": kind, "ensemble_name": name, "n_repeats": len(group)}
        for field in numeric_fields:
            values = np.array([float(row[field]) for row in group if math.isfinite(float(row[field]))])
            summary[f"{field}_mean"] = float(np.mean(values)) if len(values) else math.nan
            summary[f"{field}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
        summaries.append(summary)
    return summaries


def main() -> int:
    args = parse_args()
    if args.write_manifest_template:
        write_manifest_template(args.write_manifest_template)
        print(f"Wrote {args.write_manifest_template}")
        return 0
    if args.results_dir is None:
        raise SystemExit("--results-dir is required unless --write-manifest-template is used")
    if args.top_n < 2:
        raise SystemExit("--top-n must be at least 2")
    results_dir = args.results_dir.resolve()
    output_dir = (args.output_dir or results_dir / "s1b_ensembles").resolve()
    expected_models = [model.strip() for model in args.models.split(",") if model.strip()]
    manual_definitions = read_manual_manifest(args.manual_manifest, args.species, args.trait)
    loaded: dict[str, dict[int, ModelRepeat]] = {}
    load_errors: dict[str, list[str]] = {}
    for model in expected_models:
        loaded[model], load_errors[model] = load_model_repeats(results_dir, model)

    discovered_repeats = sorted(set().union(*(set(repeats) for repeats in loaded.values())))
    expected_repeat_ids = set(range(1, args.expected_repeats + 1))
    audit: dict[str, Any] = {
        "results_dir": str(results_dir),
        "expected_models": expected_models,
        "expected_repeats": args.expected_repeats,
        "allow_partial": args.allow_partial,
        "completed_repeats_by_model": {model: sorted(repeats) for model, repeats in loaded.items()},
        "load_errors": load_errors,
        "skipped_repeats": {},
        "manual_ensembles": [{"ensemble_name": name, "models": models} for name, models in manual_definitions],
    }
    if not args.allow_partial:
        missing_by_model = {
            model: sorted(expected_repeat_ids.difference(repeats))
            for model, repeats in loaded.items()
            if expected_repeat_ids.difference(repeats)
        }
        if missing_by_model:
            output_dir.mkdir(parents=True, exist_ok=True)
            audit["strict_completion_error"] = {
                "message": "All required models must contain every expected repeat before strict aggregation.",
                "missing_repeats_by_model": missing_by_model,
            }
            with (output_dir / "ensemble_audit.json").open("w") as handle:
                json.dump(audit, handle, indent=2)
            print(f"Strict aggregation deferred; wrote audit to {output_dir / 'ensemble_audit.json'}")
            return 2
        all_repeats = sorted(expected_repeat_ids)
    else:
        all_repeats = discovered_repeats
    rows: list[dict[str, Any]] = []
    for repeat in all_repeats:
        available = [loaded[model][repeat] for model in expected_models if repeat in loaded[model]]
        missing = [model for model in expected_models if repeat not in loaded[model]]
        if missing and not args.allow_partial:
            audit["skipped_repeats"][str(repeat)] = {"reason": "missing required base models", "models": missing}
            continue
        if len(available) < args.top_n:
            audit["skipped_repeats"][str(repeat)] = {"reason": "fewer than top-n usable models", "models": missing}
            continue
        validation_scores = [(pearson(item.validation.true_values, item.validation.predictions), item.model, item) for item in available]
        selected_refined = [item for _, _, item in sorted(validation_scores, key=lambda value: (-np.nan_to_num(value[0], nan=-np.inf), value[1]))[:args.top_n]]
        try:
            rows.append(make_ensemble_row(repeat, "refined_top_n", f"top{args.top_n}_ridge", selected_refined))
            rows.append(make_ensemble_row(repeat, "full_base_models", "all_eligible_ridge", available))
            for name, models in manual_definitions:
                selected_manual = [loaded[model][repeat] for model in models if model in loaded and repeat in loaded[model]]
                if len(selected_manual) != len(models):
                    audit.setdefault("missing_manual_models", {}).setdefault(str(repeat), {})[name] = [
                        model for model in models if model not in loaded or repeat not in loaded[model]
                    ]
                    continue
                rows.append(make_ensemble_row(repeat, "manual_predeclared", name, selected_manual))
        except ValueError as error:
            audit["skipped_repeats"][str(repeat)] = {"reason": str(error), "models": [item.model for item in available]}

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "ensemble_repeat_metrics.csv", rows)
    write_csv(output_dir / "ensemble_summary.csv", summary_rows(rows))
    with (output_dir / "ensemble_audit.json").open("w") as handle:
        json.dump(audit, handle, indent=2)
    readme = output_dir / "README.md"
    readme.write_text(
        "# S1b ensemble outputs\n\n"
        "`ensemble_repeat_metrics.csv` contains one held-out-test result per completed repeat. "
        "Top-N model selection and ridge fitting use only the saved validation-fold predictions. "
        "`full_base_models` means all required models (or, only with `--allow-partial`, all currently available models). "
        "No manual ensemble is generated without an explicit manifest row. These are repeated hold-out results, "
        "not a claim of strict outer-fold OOF performance.\n"
    )
    print(f"Wrote {len(rows)} ensemble rows to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
