"""Recompute CV summaries without fitting models or modifying historical inputs.

Usage: python -m gpse.tools.reaggregate_cv RESULTS --output NEW_DIRECTORY
       --expected-repeats 50 --expected-folds 5
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from gpse.train._cv_statistics import (
    STABILITY_METRICS,
    aggregate_cv,
    digest,
    export_cv,
    validate_shared_splits,
)
from gpse.train.topsis import TOPSISEvaluator


def _prepare_historical_repeat(path, repeat, task_type, hashes, source, read_json):
    """Validate retained prediction evidence and recover missing split identifiers."""
    folds = repeat.get("fold_results", [])
    if not isinstance(folds, list) or any(not isinstance(fold, dict) for fold in folds):
        raise ValueError("fold_results must be a list of fold objects")
    # Original CV assignment includes sample IDs in the precise training row order.
    assignment = path.parent / "cv_train_only.csv"
    if assignment.exists():
        hashes[str(assignment.relative_to(source))] = hashlib.sha256(
            assignment.read_bytes()
        ).hexdigest()
        ids = pd.read_csv(assignment, dtype={"sample_id": str})["sample_id"].tolist()
        for fold in repeat.get("fold_results", []):
            if not fold.get("split_identifier"):
                fold["split_identifier"] = digest(
                    {"samples": ids, "train": fold["train_indices"], "val": fold["val_indices"]}
                )
    # Old code replaced undefined Pearson by zero. Detect that case when raw
    # validation predictions survived; do not legitimise it as valid CV evidence.
    predictions_path = path.parent / "all_predictions.json"
    if task_type == "regression" and predictions_path.exists():
        import numpy as np

        prediction_data = read_json(predictions_path)
        if not isinstance(prediction_data, dict):
            raise ValueError("all_predictions.json must contain an object")
        predictions = prediction_data.get("val", [])
        if not isinstance(predictions, list) or len(predictions) != len(folds):
            raise ValueError("Retained validation predictions do not match the fold count")
        for fold, prediction in zip(folds, predictions):
            if not isinstance(prediction, dict):
                raise ValueError("Validation prediction entries must be objects")
            observed = np.asarray(prediction["true_values"], dtype=float)
            predicted = np.asarray(prediction["predicted_values"], dtype=float)
            if observed.ndim != 1 or observed.shape != predicted.shape:
                raise ValueError("Observed and predicted validation arrays must match")
            if (
                len(observed) < 2
                or not np.isfinite(observed).all()
                or not np.isfinite(predicted).all()
                or np.ptp(observed) == 0
                or np.ptp(predicted) == 0
            ):
                fold["val_metrics"]["pearson"] = float("nan")


def reaggregate(
    source,
    output,
    expected_repeats=50,
    expected_folds=5,
    task_type="regression",
    stability_metric="cv_std_repeat",
):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents:
        raise ValueError("Offline output must be outside the historical results directory")
    if not source.is_dir():
        raise FileNotFoundError(f"Historical results directory does not exist: {source}")
    output.mkdir(parents=True, exist_ok=False)
    files = sorted(source.glob("*/repeat_*/repeat_results.json"))
    groups, aggregates, failures, hashes = {}, {}, {}, {}
    input_errors = {}

    def read_json(path):
        payload = path.read_bytes()
        hashes[str(path.relative_to(source))] = hashlib.sha256(payload).hexdigest()
        return json.loads(payload)

    for path in files:
        model = path.parent.parent.name
        groups.setdefault(model, [])
        try:
            repeat = read_json(path)
            if not isinstance(repeat, dict):
                raise ValueError("repeat_results.json must contain an object")
            _prepare_historical_repeat(path, repeat, task_type, hashes, source, read_json)
            groups[model].append(repeat)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            relative = str(path.relative_to(source))
            input_errors[relative] = str(exc)
            failures[model] = f"Unreadable/invalid historical inputs for {relative}: {exc}"
    # Include models for which only a historical summary survived.
    for pattern in ("*/summary_results.json", "*/cv_failure.json"):
        for path in source.glob(pattern):
            groups.setdefault(path.parent.name, [])
    for model, repeats in groups.items():
        if model in failures:
            continue
        try:
            aggregates[model] = aggregate_cv(
                repeats, expected_repeats, expected_folds, task_type, stability_metric
            )
        except (ValueError, KeyError, TypeError) as exc:
            failures[model] = str(exc)
    valid = {m: groups[m] for m in aggregates}
    original_manifest = source / "cv_run_manifest.json"
    original = {}
    if original_manifest.exists():
        try:
            original = read_json(original_manifest)
            if not isinstance(original, dict):
                raise ValueError("cv_run_manifest.json must contain an object")
        except (ValueError, OSError) as exc:
            input_errors["cv_run_manifest.json"] = str(exc)
            original = {}
    refit_required, refit_unknown, saved_artifacts = set(), set(), []
    for model in sorted(groups):
        for role in ("deployment_ensemble", "representative_model"):
            artifact_dir = source / model / role
            if not artifact_dir.is_dir():
                continue
            info_path = artifact_dir / "info.json"
            info = {}
            if info_path.exists():
                try:
                    info = read_json(info_path)
                    if not isinstance(info, dict):
                        raise ValueError("Saved artifact info must contain an object")
                except (ValueError, OSError) as exc:
                    input_errors[str(info_path.relative_to(source))] = str(exc)
                    info = {}
            trained_on = info.get("trained_on", "unknown_historical")
            if trained_on == "all_samples":
                status = "requires_train_partition_refit"
                refit_required.add(model)
            elif trained_on == "holdout_training_partition":
                status = "already_trained_without_holdout"
            else:
                status = "cannot_determine_training_partition"
                refit_unknown.add(model)
            saved_artifacts.append(
                {"model": model, "artifact_role": role, "trained_on": trained_on, "status": status}
            )
    provenance = {
        "code_version": original.get("code_version", "unknown_historical"),
        "config_version": original.get("config_version", "unknown_historical"),
        "operation": "offline_reaggregation_only_no_retraining",
        "historical_artifacts_migrated": False,
        "historical_score_validation": "finite/range checks; prediction checks where retained; undocumented legacy zero substitution cannot be reversed without predictions",
        "source": str(source),
        "source_sha256": hashes,
        "reaggregation_code_version": digest(
            {
                str(p.name): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (
                    Path(__file__),
                    Path(__file__).parents[1] / "train" / "_cv_statistics.py",
                    Path(__file__).parents[1] / "train" / "topsis.py",
                )
            }
        ),
    }
    export_cv(output, valid, aggregates, provenance)
    audit = {
        "recovered_models": sorted(valid),
        "requires_cv_retraining": failures,
        "input_errors": input_errors,
        "deployment_refit_required": sorted(refit_required),
        "deployment_refit_unknown": sorted(refit_unknown),
        "saved_artifact_audit": saved_artifacts,
        "note": (
            "Historical saved models are unchanged; refit requirements use saved training metadata. "
            "Missing metadata stays unknown. Restore damaged inputs before deciding to retrain CV."
        ),
        "ranking_status": "unavailable",
    }
    if valid and not failures:
        try:
            validate_shared_splits(valid)
            new = TOPSISEvaluator().evaluate(
                output / "model_comparison_cv.csv",
                output / "model_comparison_cv_topsis.csv",
                criteria=["cv_mean", stability_metric],
                criteria_types=["max", "min"],
                min_transform="neglog",
                simple_output=output / "model_comparison_cv_topsis_simple.csv",
            )
            frame = pd.read_csv(output / "model_comparison_cv.csv")
            legacy_scores, _, _ = TOPSISEvaluator().topsis(
                frame[["legacy_cv_mean", "legacy_cv_std"]], [0.8, 0.2], ["max", "min"], "neglog"
            )
            frame["legacy_TOPSIS_Score"] = legacy_scores
            frame["legacy_TOPSIS_Rank"] = legacy_scores.rank(method="min", ascending=False).astype(
                int
            )
            difference = frame[
                [
                    "Model",
                    "legacy_cv_mean",
                    "legacy_cv_std",
                    "legacy_TOPSIS_Score",
                    "legacy_TOPSIS_Rank",
                ]
            ].merge(
                new[["Model", "cv_mean", stability_metric, "TOPSIS_Score", "TOPSIS_Rank"]],
                on="Model",
            )
            difference.to_csv(output / "cv_ranking_comparison.csv", index=False)
            audit["ranking_status"] = "verified_shared_splits"
        except ValueError as exc:
            audit["ranking_error"] = str(exc)
    with (output / "reaggregation_audit.json").open("x") as handle:
        json.dump(audit, handle, indent=2)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-repeats", type=int, default=50)
    parser.add_argument("--expected-folds", type=int, default=5)
    parser.add_argument(
        "--task-type", choices=["regression", "classification"], default="regression"
    )
    parser.add_argument("--stability-metric", choices=STABILITY_METRICS, default="cv_std_repeat")
    args = parser.parse_args()
    audit = reaggregate(
        args.source,
        args.output,
        args.expected_repeats,
        args.expected_folds,
        args.task_type,
        args.stability_metric,
    )
    print(json.dumps(audit, indent=2))
    if (
        not audit["recovered_models"]
        or audit["requires_cv_retraining"]
        or audit["input_errors"]
        or audit.get("ranking_error")
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
