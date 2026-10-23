"""Ensembling must use the fitted CV members, independently of persistence."""

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import LinearSVC

from gpse.tasks.classification import GenomicClassifier
from gpse.train._ensemble import _compute_ensemble_predictions
from gpse.train.predictor import GenomicPredictorV2


@pytest.mark.parametrize(
    "estimator_type", [RandomForestRegressor, RandomForestClassifier, LinearSVC]
)
def test_repeat_fits_each_fold_once_with_or_without_model_files(
    tmp_path, monkeypatch, estimator_type
):
    classification = estimator_type is not RandomForestRegressor
    rng = np.random.default_rng(176)
    X = pd.DataFrame(rng.integers(0, 4, size=(80, 8)).astype(float))
    signal = X[0] * 3 - X[1] + rng.normal(size=len(X))
    y = (signal > signal.median()).astype(int) if classification else signal
    calls = []
    original_fit = estimator_type.fit

    def fit(model, features, labels, *args, **kwargs):
        calls.append(len(features))
        return original_fit(model, features, labels, *args, **kwargs)

    monkeypatch.setattr(estimator_type, "fit", fit)
    outputs = []
    for save_models in (False, True):
        predictor = GenomicPredictorV2(
            results_dir=str(tmp_path / str(save_models)),
            task_type="classification" if classification else "regression",
            n_classes=2 if classification else None,
            n_splits=3,
            n_repeats=1,
            cv_stability_metric="cv_std_within",
            use_default_params=True,
            save_models=save_models,
            standardize_phenotype=not classification,
            genotype_imputation="mean",
            feature_selection="univariate",
            select_k=4,
        )
        options = {"random_state": 17}
        if estimator_type is not LinearSVC:
            options.update(n_estimators=12, max_depth=4, n_jobs=1)
        monkeypatch.setattr(predictor, "create_model", lambda *args: estimator_type(**options))
        before = len(calls)
        model_name = (
            "svm_clf" if estimator_type is LinearSVC else "rf_clf" if classification else "rf_reg"
        )
        result = predictor.train_and_evaluate_model_for_repeat(
            model_name,
            X,
            y,
            0,
            test_indices=np.arange(60, 80),
        )
        assert len(calls) - before == 3
        assert len(result["fold_results"]) == 3
        assert len(list(predictor.results_dir.rglob("fold_*_model.pkl"))) == (
            3 if save_models else 0
        )
        outputs.append(result["ensemble_metrics"])
    assert outputs[0] == outputs[1]


def test_cached_classification_averages_probabilities_rather_than_hard_votes(tmp_path):
    classifier = GenomicClassifier(n_classes=3, n_threads=1)
    predictor = SimpleNamespace(
        task_type="classification", n_splits=2, genomic_classifier=classifier
    )
    first = np.array([[0.40, 0.25, 0.35], [0.50, 0.49, 0.01], [0.80, 0.10, 0.10]])
    second = np.array([[0.01, 0.01, 0.98], [0.01, 0.98, 0.01], [0.90, 0.05, 0.05]])
    metrics = _compute_ensemble_predictions(
        predictor,
        "rf_clf",
        {},
        [],
        None,
        None,
        None,
        None,
        pd.Series([2, 1, 0]),
        tmp_path,
        logging.getLogger(__name__),
        fold_predictions=[(first.argmax(axis=1), first), (second.argmax(axis=1), second)],
    )
    assert metrics["accuracy"] == 1


def test_cached_regression_restores_original_phenotype_scale(tmp_path):
    predictor = SimpleNamespace(task_type="regression", n_splits=2)
    metrics = _compute_ensemble_predictions(
        predictor,
        "rf_reg",
        {},
        [],
        None,
        None,
        None,
        None,
        pd.Series([-1.0, 0.0, 1.0]),
        tmp_path,
        logging.getLogger(__name__),
        y_test_original=pd.Series([8.0, 10.0, 12.0]),
        phenotype_scaler={"applied": True, "mean": 10.0, "std": 2.0},
        fold_predictions=[(np.array([-0.5, -0.5, 0.5]), None), (np.array([-1.5, 0.5, 1.5]), None)],
    )
    assert metrics["mse"] == 0


def test_partial_fold_predictions_are_rejected(tmp_path, caplog):
    predictor = SimpleNamespace(
        task_type="regression", n_splits=2, _get_default_ensemble_metrics=lambda: {"invalid": True}
    )
    metrics = _compute_ensemble_predictions(
        predictor,
        "rf_reg",
        {},
        [],
        None,
        None,
        None,
        None,
        pd.Series([1.0, 2.0]),
        tmp_path,
        logging.getLogger(__name__),
        fold_predictions=[(np.array([1.0, 2.0]), None)],
    )
    assert metrics == {"invalid": True}
    assert "every configured CV fold" in caplog.text
