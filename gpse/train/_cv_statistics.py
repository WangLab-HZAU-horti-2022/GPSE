"""Strict training-only repeated-CV aggregation and auditable exports.

All folds/repeats are required. Undefined validation scores invalidate selection;
they are never imputed or silently dropped. Repeat SD is descriptive, not an SE.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

AGGREGATION_METHOD = "repeated_cv_v2"
STABILITY_METRICS = ("cv_std_repeat", "cv_std_within", "cv_std_all_folds")
METRIC_DEFINITIONS = {
    "cv_mean": "mean of all repeat validation means (equal repeat weights)",
    "cv_std_repeat": "sample SD of repeat validation means; ddof=1; not SE or CI",
    "cv_std_within": "RMS of within-repeat fold sample SDs; ddof=1",
    "cv_std_all_folds": "sample SD of all validation fold scores; ddof=1",
    "cv_best_repeat_mean": "maximum repeat validation mean; diagnostic only",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def runtime_provenance(predictor):
    """Hash actual source and effective config, including external config contents."""
    root = Path(__file__).resolve().parents[1]
    sources = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*.py"))
    }
    config = {
        k: v
        for k, v in vars(predictor).items()
        if isinstance(v, (str, int, float, bool, type(None)))
        and k not in {"results_dir", "main_log_filename", "resume_from", "automatic_fit_threads"}
        and not k.startswith("_checkpoint_")
    }
    config["run"] = getattr(predictor, "_cv_run_config", {})
    execution = getattr(predictor, "execution_config", None)
    if execution is not None:
        from dataclasses import asdict

        config["execution"] = asdict(execution)
    for name in ("feature_selection_config", "genotype_imputation_config"):
        config[name] = getattr(predictor, name).as_dict()
    config["config_files"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((root / "config").glob("*.yaml"))
    }
    for name in ("cv_file", "model_config", "topsis_config"):
        value = getattr(predictor, name, None)
        if value and Path(value).is_file():
            config[name + "_sha256"] = hashlib.sha256(Path(value).read_bytes()).hexdigest()
    return {
        "code_version": digest(sources),
        "config_version": digest(config),
        "effective_config": config,
    }


def aggregate_cv(
    repeats,
    expected_repeats,
    expected_folds,
    task_type="regression",
    stability_metric="cv_std_repeat",
):
    if stability_metric not in STABILITY_METRICS:
        raise ValueError(f"Unknown stability metric: {stability_metric}")
    if expected_repeats < 1 or expected_folds < 2:
        raise ValueError("CV requires >=1 repeat and >=2 folds")
    ids = [r["repeat_idx"] for r in repeats]
    if len(ids) != expected_repeats or set(ids) != set(range(expected_repeats)):
        raise ValueError(f"Incomplete/duplicate repeats: got {ids}; expected {expected_repeats}")
    metric = "accuracy" if task_type == "classification" else "pearson"
    values = []
    for repeat in sorted(repeats, key=lambda r: r["repeat_idx"]):
        folds = repeat.get("fold_results", [])
        fold_ids = [f["fold_idx"] for f in folds]
        if len(fold_ids) != expected_folds or set(fold_ids) != set(range(expected_folds)):
            raise ValueError(f"Repeat {repeat['repeat_idx']}: missing/duplicate folds")
        row = [float(f["val_metrics"][metric]) for f in sorted(folds, key=lambda f: f["fold_idx"])]
        if not np.isfinite(row).all():
            raise ValueError(f"Repeat {repeat['repeat_idx']}: non-finite validation {metric}")
        low = 0 if metric == "accuracy" else -1
        if np.any(np.asarray(row) < low) or np.any(np.asarray(row) > 1):
            raise ValueError(f"Repeat {repeat['repeat_idx']}: invalid validation {metric} range")
        values.append(row)
    values = np.asarray(values)
    means = values.mean(axis=1)
    within = values.std(axis=1, ddof=1)
    result = {
        "cv_mean": float(means.mean()),
        "cv_std_repeat": float(means.std(ddof=1)) if expected_repeats > 1 else None,
        "cv_std_within": float(np.sqrt(np.mean(within**2))),
        "cv_std_all_folds": float(values.std(ddof=1)),
        "cv_best_repeat_mean": float(means.max()),
        "aggregation_method": AGGREGATION_METHOD,
        "completed_repeats": len(repeats),
        "expected_repeats": expected_repeats,
        "completed_folds": int(values.size),
        "expected_folds_per_repeat": expected_folds,
        "selection_source": "train_only_cv",
        "accuracy_metric": "cv_mean",
        "stability_metric": stability_metric,
        "weight_method": "manual",
        "weights": [0.8, 0.2],
        "metric_definition": METRIC_DEFINITIONS,
        "validation_metric": metric,
        "invalid_score_policy": "reject_model_no_imputation_no_partial_selection",
    }
    if result[stability_metric] is None:
        raise ValueError(
            "cv_std_repeat needs at least two repeats; explicitly select another stability metric"
        )
    result["cv_std"] = result[stability_metric]  # compatibility alias with recorded definition
    # Exact historical policy: best repeat mean, then population SD, then repeat ID.
    best = min(range(len(means)), key=lambda i: (-means[i], values[i].std(ddof=0), i))
    result.update(
        legacy_cv_mean=float(means[best]),
        legacy_cv_std=float(values[best].std(ddof=0)),
        diagnostic_best_repeat_idx=best,
    )
    return result


def export_cv(directory, model_repeats, aggregates, provenance):
    """Write each artifact exclusively; callers must reserve a fresh directory/run."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fold_rows, repeat_rows, comparison = [], [], []
    for model, repeats in sorted(model_repeats.items()):
        aggregate = aggregates[model]
        meta = {
            k: aggregate[k]
            for k in (
                "aggregation_method",
                "completed_repeats",
                "expected_repeats",
                "completed_folds",
                "selection_source",
                "accuracy_metric",
                "stability_metric",
                "weight_method",
            )
        }
        meta.update({k: provenance.get(k, "unknown") for k in ("code_version", "config_version")})
        meta["metric_definition"] = json.dumps(METRIC_DEFINITIONS, sort_keys=True)
        meta["weights"] = "[0.8,0.2]"
        comparison.append({"Model": model, **aggregate, **meta})
        for repeat in sorted(repeats, key=lambda r: r["repeat_idx"]):
            scores = []
            for fold in sorted(repeat["fold_results"], key=lambda f: f["fold_idx"]):
                score = fold["val_metrics"][aggregate["validation_metric"]]
                scores.append(score)
                fold_rows.append(
                    {
                        "model": model,
                        "repeat_id": repeat["repeat_idx"],
                        "fold_id": fold["fold_idx"],
                        "validation_" + aggregate["validation_metric"]: score,
                        "split_seed": repeat.get("split_seed"),
                        "split_identifier": fold.get("split_identifier", "unknown"),
                        **meta,
                    }
                )
            repeat_rows.append(
                {
                    "model": model,
                    "repeat_id": repeat["repeat_idx"],
                    "validation_mean": float(np.mean(scores)),
                    "validation_std": float(np.std(scores, ddof=1)),
                    "split_seed": repeat.get("split_seed"),
                    **meta,
                }
            )
    for name, rows in (
        ("cv_fold_metrics.csv", fold_rows),
        ("cv_repeat_metrics.csv", repeat_rows),
        ("model_comparison_cv.csv", comparison),
    ):
        with (directory / name).open("x", encoding="utf-8") as handle:
            pd.DataFrame(rows).to_csv(handle, index=False)
    with (directory / "cv_aggregation_summary.json").open("x", encoding="utf-8") as handle:
        json.dump({"models": aggregates, **provenance}, handle, indent=2, allow_nan=False)


def validate_shared_splits(model_repeats):
    """Reject cross-model comparisons when the actual sample assignments differ."""
    reference = None
    for model, repeats in model_repeats.items():
        signature = {
            (r["repeat_idx"], f["fold_idx"]): f.get("split_identifier")
            for r in repeats
            for f in r["fold_results"]
        }
        if not signature or any(not v or v == "unknown" for v in signature.values()):
            raise ValueError(f"Cannot verify shared CV sample splits for {model}")
        if reference is not None and signature != reference:
            raise ValueError(f"Candidate {model} uses different CV splits")
        reference = signature
