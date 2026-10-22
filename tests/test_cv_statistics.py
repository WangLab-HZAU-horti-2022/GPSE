"""Acceptance tests for repeated CV, provenance, ranking and offline recovery."""

import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from gpse.train._cv_statistics import aggregate_cv, export_cv, validate_shared_splits
from gpse.train._cv_manager import generate_cv_folds_from_file
from gpse.train.predictor import GenomicPredictorV2
from gpse.train.topsis import TOPSISEvaluator
from gpse.tools.reaggregate_cv import reaggregate
from gpse.utils.genomic_utils import calculate_metrics


def repeats(values, model="ridge"):
    return [
        {
            "model_name": model,
            "repeat_idx": r,
            "split_seed": 42 + r,
            "fold_results": [
                {
                    "fold_idx": f,
                    "val_metrics": {"pearson": float(v)},
                    "split_identifier": f"shared-{r}-{f}",
                }
                for f, v in enumerate(row)
            ],
        }
        for r, row in enumerate(values)
    ]


def test_all_formulas_50_by_5_and_outlier():
    rng = np.random.default_rng(17)
    values = rng.uniform(0.2, 0.5, (50, 5))
    values[-1] = [0.90, 0.91, 0.92, 0.93, 0.94]
    result = aggregate_cv(repeats(values), 50, 5)
    assert result["cv_mean"] == pytest.approx(values.mean(axis=1).mean())
    assert result["cv_std_repeat"] == pytest.approx(values.mean(axis=1).std(ddof=1))
    assert result["cv_std_within"] == pytest.approx(
        np.sqrt(np.mean(values.std(axis=1, ddof=1) ** 2))
    )
    assert result["cv_std_all_folds"] == pytest.approx(values.std(ddof=1))
    assert result["cv_best_repeat_mean"] == pytest.approx(0.92)
    assert result["cv_mean"] != result["cv_best_repeat_mean"]
    assert result["legacy_cv_std"] == pytest.approx(values[-1].std(ddof=0))
    assert result["completed_folds"] == 250
    assert result["cv_std_repeat"] != pytest.approx(values.mean(axis=1).std(ddof=0))
    shuffled = repeats(values)[::-1]
    for repeat in shuffled:
        repeat["fold_results"].reverse()
    assert aggregate_cv(shuffled, 50, 5) == result


@pytest.mark.parametrize(
    "problem",
    ["missing_repeat", "duplicate_repeat", "missing_fold", "duplicate_fold", "nan", "inf", "range"],
)
def test_incomplete_and_invalid_scores_rejected(problem):
    data = repeats(np.full((50, 5), 0.5))
    if problem == "missing_repeat":
        data.pop()
    elif problem == "duplicate_repeat":
        data[-1]["repeat_idx"] = 0
    elif problem == "missing_fold":
        data[0]["fold_results"].pop()
    elif problem == "duplicate_fold":
        data[0]["fold_results"][-1]["fold_idx"] = 0
    else:
        data[0]["fold_results"][0]["val_metrics"]["pearson"] = {
            "nan": np.nan,
            "inf": np.inf,
            "range": 1.1,
        }[problem]
    with pytest.raises(ValueError):
        aggregate_cv(data, 50, 5)


def test_single_repeat_and_zero_variance_policy():
    data = repeats([[0.5] * 5])
    with pytest.raises(ValueError, match="at least two"):
        aggregate_cv(data, 1, 5)
    within = aggregate_cv(data, 1, 5, stability_metric="cv_std_within")
    assert within["cv_std_repeat"] is None
    assert within["cv_std"] == 0
    assert aggregate_cv(repeats([[0.5] * 5] * 50), 50, 5)["cv_std_repeat"] == 0


def test_constant_and_short_pearson_are_not_zero():
    for observed, predicted in (([1, 1, 1], [1, 2, 3]), ([1, 2, 3], [1, 1, 1]), ([1], [2])):
        with pytest.warns((RuntimeWarning, UserWarning)):
            metrics = calculate_metrics(observed, predicted)
        assert np.isnan(metrics["pearson"])
        assert metrics["pearson_status"] != "valid"


def test_split_reproducibility_across_candidate_models():
    from types import SimpleNamespace

    X = pd.DataFrame(np.random.default_rng(5).normal(size=(60, 3)))
    y = pd.Series(np.arange(60))
    config = SimpleNamespace(
        cv_file=None, n_splits=5, n_repeats=50, task_type="regression", random_seed=42
    )
    for r in range(50):
        left = generate_cv_folds_from_file(config, X, y, None, r)
        right = generate_cv_folds_from_file(copy.copy(config), X, y, None, r)
        assert all(
            np.array_equal(a, b) and np.array_equal(c, d) for (a, c), (b, d) in zip(left, right)
        )
    data = {"a": repeats(np.full((50, 5), 0.2)), "b": repeats(np.full((50, 5), 0.3))}
    validate_shared_splits(data)
    data["b"][0]["fold_results"][0]["split_identifier"] = "different"
    with pytest.raises(ValueError, match="different CV splits"):
        validate_shared_splits(data)


def test_topsis_edges_and_intermediate_alignment(tmp_path):
    evaluator = TOPSISEvaluator()
    data = pd.DataFrame({"accuracy": [0.2, 0.8], "sd": [0, 0]})
    scores, _, normalized = evaluator.topsis(data, [0.8, 0.2], ["max", "min"], "neglog")
    assert scores[1] > scores[0]
    assert np.isfinite(normalized.to_numpy()).all()
    for values in ([[0, 0], [0, 0]], [[0.5, 0], [0.5, 0]], [[0.5, 0]]):
        score, _, _ = evaluator.topsis(pd.DataFrame(values), [0.8, 0.2], ["max", "min"], "neglog")
        assert (score == 0.5).all()
    for values in ([[np.nan, 0], [0.5, 0]], [[0.5, -1], [0.4, 0]], [[np.inf, 0], [0.4, 0]]):
        with pytest.raises(ValueError):
            evaluator.topsis(pd.DataFrame(values), [0.8, 0.2], ["max", "min"], "neglog")
    data = {"a": repeats([[0.1, 0.3]] * 3), "b": repeats([[0.7, 0.9]] * 3)}
    export_cv(tmp_path, data, {m: aggregate_cv(r, 3, 2) for m, r in data.items()}, {})
    ranked = evaluator.evaluate(tmp_path / "model_comparison_cv.csv", tmp_path / "ranking.csv")
    assert ranked.iloc[0]["Model"] == "b"
    assert ranked.iloc[0]["cv_mean_positive"] == pytest.approx(0.8)
    assert (ranked["selection_source"] == "train_only_cv").all()
    # Adding adversarial holdout results cannot change selection.
    frame = pd.read_csv(tmp_path / "model_comparison_cv.csv")
    frame["Test Pearson"] = [1, -1]
    frame.to_csv(tmp_path / "input.csv", index=False)
    again = evaluator.evaluate(tmp_path / "input.csv", tmp_path / "ranking2.csv")
    assert again["Model"].tolist() == ranked["Model"].tolist()
    with pytest.raises(ValueError, match="training-side"):
        evaluator.evaluate(
            tmp_path / "input.csv", tmp_path / "bad.csv", criteria=["Test Pearson", "cv_std_repeat"]
        )
    frame["cv_mean"] = np.nan
    frame.to_csv(tmp_path / "invalid.csv", index=False)
    with pytest.raises(ValueError, match="finite"):
        evaluator.evaluate(tmp_path / "invalid.csv", tmp_path / "bad.csv")


def write_history(directory, data):
    for name, entries in data.items():
        for entry in entries:
            path = directory / name / f"repeat_{entry['repeat_idx']+1}" / "repeat_results.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(entry))


def test_offline_no_overwrite_and_ranking_difference(tmp_path):
    source = tmp_path / "history"
    # a has a lucky repeat; b has a consistently higher mean over all repeats.
    data = {
        "a": repeats([[0.19, 0.21]] * 49 + [[0.89, 0.91]], "a"),
        "b": repeats([[0.49, 0.51]] * 50, "b"),
    }
    write_history(source, data)
    before = {p: p.read_bytes() for p in source.rglob("*.json")}
    audit = reaggregate(source, tmp_path / "new", 50, 2)
    assert audit["ranking_status"] == "verified_shared_splits"
    difference = pd.read_csv(tmp_path / "new" / "cv_ranking_comparison.csv").set_index("Model")
    assert difference.loc["a", "legacy_TOPSIS_Rank"] == 1
    assert difference.loc["b", "TOPSIS_Rank"] == 1
    assert all(p.read_bytes() == value for p, value in before.items())
    summary = json.loads((tmp_path / "new" / "cv_aggregation_summary.json").read_text())
    assert summary["code_version"] == "unknown_historical"
    assert summary["historical_artifacts_migrated"] is False
    with pytest.raises(FileExistsError):
        reaggregate(source, tmp_path / "new", 50, 2)
    with pytest.raises(ValueError):
        reaggregate(source, source / "new", 50, 2)
    next(iter((source / "a").rglob("repeat_results.json"))).unlink()
    incomplete = reaggregate(source, tmp_path / "incomplete", 50, 2)
    assert "a" in incomplete["requires_cv_retraining"]
    assert incomplete["ranking_status"] == "unavailable"


def test_export_does_not_overwrite(tmp_path):
    data = {"a": repeats([[0.1, 0.2]] * 2)}
    args = (tmp_path, data, {"a": aggregate_cv(data["a"], 2, 2)}, {})
    export_cv(*args)
    original = (tmp_path / "cv_fold_metrics.csv").read_bytes()
    with pytest.raises(FileExistsError):
        export_cv(*args)
    assert (tmp_path / "cv_fold_metrics.csv").read_bytes() == original


def test_real_pipeline_selection_stacking_and_training_only_refit(tmp_path, monkeypatch):
    """Run actual folds, aggregation, TOPSIS, OOF stacking and both saved refits."""
    rng = np.random.default_rng(123)
    X = pd.DataFrame(rng.normal(size=(80, 5)), index=[f"s{i}" for i in range(80)])
    y = pd.Series(X.iloc[:, 0].to_numpy() * 2 + rng.normal(size=80) * 0.1, index=X.index)
    predictor = GenomicPredictorV2(
        results_dir=str(tmp_path / "run"),
        n_repeats=50,
        n_splits=5,
        use_default_params=True,
        save_models=False,
    )
    monkeypatch.setattr(predictor, "load_data", lambda *a: (X, y, pd.DataFrame({"y": y})))
    monkeypatch.setattr(
        predictor, "create_model", lambda name, params: Ridge(alpha=params["alpha"])
    )
    monkeypatch.setattr(
        predictor, "get_default_params", lambda name: {"alpha": 1 if name == "a" else 10}
    )
    fit_rows = []
    fit_targets = []
    original_fit = Ridge.fit

    def record_fit(model, features, target, *args, **kwargs):
        fit_rows.append(len(target))
        fit_targets.extend(np.asarray(target).tolist())
        return original_fit(model, features, target, *args, **kwargs)

    monkeypatch.setattr(Ridge, "fit", record_fit)
    results = predictor.run_all_models(
        "geno", "pheno", "y", models=["a", "b"], use_stacking=True, top_n_models=2, cv_folds=5
    )
    assert "stacking_ensemble" in results
    assert 80 not in fit_rows
    assert 64 in fit_rows  # final base models refit on complete holdout training set
    from gpse.train._repeat_training import create_holdout_indices

    training, holdout = create_holdout_indices(y, 0.2, 42, "regression")
    assert set(fit_targets).issubset(set(y.iloc[training]))
    assert not set(fit_targets).intersection(set(y.iloc[holdout]))
    ranking = pd.read_csv(
        tmp_path / "run" / "selection" / "model_comparison_cv_topsis_simple.csv"
    )
    chosen = json.loads(
        (tmp_path / "run" / "selection" / "stacking_selected_models.json").read_text()
    )
    assert chosen["selected_models"] == ranking["Model"].tolist()
    assert results["stacking_ensemble"]["selected_models"] == chosen["selected_models"]
    folds = pd.read_csv(tmp_path / "run" / "selection" / "cv_fold_metrics.csv")
    assert len(folds) == 500
    assert (folds.groupby(["repeat_id", "fold_id"])["split_identifier"].nunique() == 1).all()
    for name in ("a", "b"):
        selection = results[name]["training_selection"]
        assert selection["cv_mean"] == pytest.approx(
            folds[folds.model == name].validation_pearson.mean()
        )
        assert results[name]["deployment_ensemble"]["trained_on"] == "holdout_training_partition"
    fingerprint = hashlib.sha256(
        (tmp_path / "run" / "selection" / "model_comparison_cv.csv").read_bytes()
    ).hexdigest()
    with pytest.raises(FileExistsError):
        predictor.run_all_models("geno", "pheno", "y", models=["a"])
    assert (
        hashlib.sha256(
            (tmp_path / "run" / "selection" / "model_comparison_cv.csv").read_bytes()
        ).hexdigest()
        == fingerprint
    )
    # Same seed + training data reproduces CV even with adversarial holdout labels.
    changed_y = y.copy()
    changed_y.iloc[holdout] = changed_y.iloc[holdout] * -100 + 999
    again = GenomicPredictorV2(
        results_dir=str(tmp_path / "run_again"),
        n_repeats=50,
        n_splits=5,
        use_default_params=True,
        save_models=False,
    )
    monkeypatch.setattr(
        again, "load_data", lambda *a: (X, changed_y, pd.DataFrame({"y": changed_y}))
    )
    monkeypatch.setattr(again, "create_model", lambda name, params: Ridge(alpha=params["alpha"]))
    monkeypatch.setattr(
        again, "get_default_params", lambda name: {"alpha": 1 if name == "a" else 10}
    )
    repeated = again.run_all_models("geno", "pheno", "y", models=["a", "b"])
    for name in ("a", "b"):
        for metric in ("cv_mean", "cv_std_repeat", "cv_std_within", "cv_std_all_folds"):
            assert (
                repeated[name]["training_selection"][metric]
                == results[name]["training_selection"][metric]
            )
    assert not set(fit_targets).intersection(set(changed_y.iloc[holdout]))


def test_offline_legacy_zero_from_constant_predictions_is_rejected(tmp_path):
    source = tmp_path / "old"
    data = {"a": repeats([[0, 0.5]] * 2)}
    write_history(source, data)
    for path in source.rglob("repeat_results.json"):
        (path.parent / "all_predictions.json").write_text(
            json.dumps(
                {
                    "val": [
                        {"true_values": [1, 2, 3], "predicted_values": [1, 1, 1]},
                        {"true_values": [1, 2, 3], "predicted_values": [1, 2, 3]},
                    ]
                }
            )
        )
    with pytest.raises(FileNotFoundError):
        reaggregate(tmp_path / "missing", tmp_path / "output", 2, 2)
    audit = reaggregate(source, tmp_path / "output", 2, 2)
    assert "non-finite validation" in audit["requires_cv_retraining"]["a"]


def test_aggregation_uses_fold_values_not_cached_repeat_means_or_holdout():
    data = repeats([[0.1, 0.3], [0.4, 0.6]])
    for r in data:
        r.update(avg_val_pearson=1, std_val_pearson=0, avg_test_pearson=-1)
    result = aggregate_cv(data, 2, 2)
    assert result["cv_mean"] == pytest.approx(0.35)
    assert result["cv_std_repeat"] == pytest.approx(np.std([0.2, 0.5], ddof=1))


def test_topsis_normalization_overflow_rejected():
    with pytest.raises(ValueError, match="normalization denominator"):
        TOPSISEvaluator().topsis(
            pd.DataFrame([[1e308, 0], [1e308, 0]]), [0.8, 0.2], ["max", "min"], "neglog"
        )


def test_optuna_constant_pearson_prunes_trials_instead_of_zero(tmp_path, monkeypatch):
    from sklearn.dummy import DummyRegressor

    predictor = GenomicPredictorV2(
        results_dir=str(tmp_path), n_repeats=2, n_trials=3, n_splits=5, patience=0
    )
    monkeypatch.setattr(predictor, "get_param_func", lambda name: lambda trial: {})
    monkeypatch.setattr(predictor, "create_model", lambda name, params: DummyRegressor())
    X = pd.DataFrame(np.arange(40).reshape(20, 2))
    y = pd.Series(np.arange(20, dtype=float))
    with pytest.raises(ValueError, match="No valid complete CV trials"):
        predictor.optimize_model_parameters(X, y, "constant", 0)


@pytest.mark.parametrize("corruption", ["json", "object", "folds", "predictions", "length"])
def test_offline_corrupt_history_produces_audit_without_partial_ranking(tmp_path, corruption):
    source = tmp_path / "old"
    write_history(source, {"bad": repeats([[0.2, 0.4]] * 2), "good": repeats([[0.4, 0.6]] * 2)})
    path = source / "bad" / "repeat_1" / "repeat_results.json"
    if corruption == "json":
        path.write_text('{"incomplete":')
    elif corruption == "object":
        path.write_text("[]")
    elif corruption == "folds":
        entry = json.loads(path.read_text())
        entry["fold_results"] = [None]
        path.write_text(json.dumps(entry))
    else:
        (path.parent / "all_predictions.json").write_text(
            "[]" if corruption == "predictions" else '{"val": []}'
        )
    original = {p: p.read_bytes() for p in source.rglob("*.json")}
    audit = reaggregate(source, tmp_path / "analysis", 2, 2)
    assert audit["recovered_models"] == ["good"]
    assert "bad" in audit["requires_cv_retraining"]
    assert "bad/repeat_1/repeat_results.json" in audit["input_errors"]
    assert audit["ranking_status"] == "unavailable"
    assert (tmp_path / "analysis" / "reaggregation_audit.json").exists()
    assert not (tmp_path / "analysis" / "model_comparison_cv_topsis.csv").exists()
    assert all(p.read_bytes() == contents for p, contents in original.items())


def test_offline_existing_split_identifier_needs_no_index_reconstruction(tmp_path):
    source = tmp_path / "old"
    write_history(source, {"a": repeats([[0.2, 0.4]] * 2)})
    for path in source.rglob("repeat_results.json"):
        (path.parent / "cv_train_only.csv").write_text("sample_id,fold\ns1,0\ns2,1\n")
    manifest = source / "cv_run_manifest.json"
    manifest.write_text('{"code_version":"original", "config_version":"v1"}')
    audit = reaggregate(source, tmp_path / "analysis", 2, 2)
    assert audit["ranking_status"] == "verified_shared_splits"
    result = json.loads((tmp_path / "analysis" / "cv_aggregation_summary.json").read_text())
    assert result["code_version"] == "original"
    for name in ("cv_run_manifest.json", "a/repeat_1/cv_train_only.csv"):
        assert (
            result["source_sha256"][name]
            == hashlib.sha256((source / name).read_bytes()).hexdigest()
        )


def test_offline_corrupt_manifest_is_reported_and_training_version_stays_unknown(tmp_path):
    source = tmp_path / "old"
    write_history(source, {"a": repeats([[0.2, 0.4]] * 2)})
    (source / "cv_run_manifest.json").write_text("[]")
    audit = reaggregate(source, tmp_path / "analysis", 2, 2)
    assert "cv_run_manifest.json" in audit["input_errors"]
    result = json.loads((tmp_path / "analysis" / "cv_aggregation_summary.json").read_text())
    assert result["code_version"] == "unknown_historical"


@pytest.mark.parametrize(
    "trained_on,required,unknown",
    [
        ("all_samples", ["a"], []),
        ("holdout_training_partition", [], []),
        (None, [], ["a"]),
    ],
)
def test_offline_refit_list_uses_artifact_training_metadata(
    tmp_path, trained_on, required, unknown
):
    source = tmp_path / "old"
    write_history(source, {"a": repeats([[0.2, 0.4]] * 2)})
    artifact = source / "a" / "deployment_ensemble"
    artifact.mkdir()
    if trained_on is not None:
        (artifact / "info.json").write_text(json.dumps({"trained_on": trained_on}))
    audit = reaggregate(source, tmp_path / "analysis", 2, 2)
    assert audit["deployment_refit_required"] == required
    assert audit["deployment_refit_unknown"] == unknown
    assert len(audit["saved_artifact_audit"]) == 1
