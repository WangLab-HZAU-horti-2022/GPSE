"""Dimension-aware kernels, strict solver policy and saved-model consistency."""

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd
import pytest
from scipy import linalg
from sklearn.base import clone
from sklearn.kernel_ridge import KernelRidge

from gpse.models.regression_model_optimizer import RegressionModelOptimizer
from gpse.models.safe_kernel_ridge import KernelNumericalError, SafeKernelRidge
from gpse.train._feature_selection import FeatureSelectionConfig
from gpse.train._model_pipeline import build_training_pipeline
from gpse.train.predictor import GenomicPredictorV2


@pytest.mark.parametrize("kernel", ["rbf", "laplacian", "polynomial"])
def test_duplicating_features_does_not_change_scaled_kernel_predictions(kernel):
    rng = np.random.default_rng(71)
    X, y, target = rng.normal(size=(45, 16)), rng.normal(size=45), rng.normal(size=(9, 16))
    params = dict(alpha=0.8, kernel=kernel, gamma_scale=0.7, numerical_policy="raise")
    a = SafeKernelRidge(**params).fit(X, y)
    b = SafeKernelRidge(**params).fit(np.tile(X, (1, 4)), y)
    assert b.gamma_ == a.gamma_ / 4
    np.testing.assert_allclose(a.predict(target), b.predict(np.tile(target, (1, 4))), atol=1e-10)


def test_scaling_uses_fold_selected_features_and_survives_clone_and_save():
    rng = np.random.default_rng(72)
    X = pd.DataFrame(rng.normal(size=(80, 32)))
    y = pd.Series(3 * X[0] - X[1] + rng.normal(size=len(X)))
    pipeline = build_training_pipeline(
        SafeKernelRidge(kernel="rbf", gamma_scale=0.8, numerical_policy="raise"),
        task_type="regression",
        model_name="kernelridge_reg",
        feature_selection_config=FeatureSelectionConfig(method="univariate", select_k=8),
    )
    pipeline.fit(X.iloc[:60], y.iloc[:60])
    fitted = pipeline.named_steps["model"]
    assert fitted.gamma_ == 0.1 and fitted.n_kernel_features_ == 8
    assert fitted.gamma is None and fitted.gamma_scale == 0.8
    preprocessing = pipeline.named_steps["preprocess"]
    reference = KernelRidge(kernel="rbf", gamma=0.1).fit(
        preprocessing.transform(X.iloc[:60]), y.iloc[:60]
    )
    expected = reference.predict(preprocessing.transform(X.iloc[60:]))
    np.testing.assert_allclose(pipeline.predict(X.iloc[60:]), expected)
    copied = clone(pipeline).fit(X.iloc[:60], y.iloc[:60])
    buffer = io.BytesIO()
    joblib.dump(copied, buffer)
    buffer.seek(0)
    restored = joblib.load(buffer)
    np.testing.assert_allclose(restored.predict(X.iloc[60:]), expected)
    assert restored.named_steps["model"].gamma_ == 0.1


def test_legacy_artifact_without_new_fields_keeps_absolute_gamma():
    rng = np.random.default_rng(73)
    X, y = rng.normal(size=(40, 8)), rng.normal(size=40)
    model = SafeKernelRidge(kernel="rbf", gamma=0.15).fit(X, y)
    expected = model.predict(X)
    for name in ("gamma_scale", "numerical_policy", "gamma_", "n_kernel_features_"):
        delattr(model, name)
    assert model.get_params()["gamma"] == 0.15
    assert model.get_params()["gamma_scale"] is None
    np.testing.assert_array_equal(model.predict(X), expected)
    np.testing.assert_allclose(clone(model).fit(X, y).predict(X), expected)


@pytest.mark.parametrize(
    "params",
    [
        {"gamma_scale": 0},
        {"gamma_scale": np.nan},
        {"gamma_scale": 1, "gamma": 0.1},
    ],
)
def test_invalid_or_ambiguous_gamma_is_rejected(params):
    with pytest.raises(ValueError, match="gamma"):
        SafeKernelRidge(kernel="rbf", **params).fit(np.eye(4), np.arange(4.0))


def test_non_positive_definite_sigmoid_never_falls_back_in_strict_mode(monkeypatch):
    rng = np.random.default_rng(20261023)
    X = rng.normal(size=(60, 256))
    y = rng.normal(size=60)
    params = dict(kernel="sigmoid", alpha=0.0651699061, gamma=0.0938480072, coef0=3.5675332669)
    original_solve, original_lstsq = linalg.solve, linalg.lstsq
    fallback_calls = []

    def fallback(*args, **kwargs):
        fallback_calls.append(1)
        return original_lstsq(*args, **kwargs)

    monkeypatch.setattr(linalg, "lstsq", fallback)
    with pytest.raises(KernelNumericalError, match="Unstable sigmoid"):
        SafeKernelRidge(**params, numerical_policy="raise").fit(X, y)
    assert not fallback_calls
    assert linalg.solve is original_solve
    # A user may explicitly retain historical behavior for reproducing old runs.
    with pytest.warns(UserWarning, match="Singular matrix"):
        legacy = SafeKernelRidge(**params, numerical_policy="legacy").fit(X, y)
    assert len(fallback_calls) == 1 and np.isfinite(legacy.predict(X)).all()


def test_ill_conditioning_warning_and_nonfinite_kernel_are_rejected(monkeypatch):
    original = linalg.solve

    def ill_conditioned(*args, **kwargs):
        warnings.warn("Ill-conditioned matrix", linalg.LinAlgWarning)
        return original(*args, **kwargs)

    monkeypatch.setattr(linalg, "solve", ill_conditioned)
    with pytest.raises(KernelNumericalError, match="Ill-conditioned"):
        SafeKernelRidge(numerical_policy="raise").fit(np.eye(4), np.arange(4.0))
    assert linalg.solve is ill_conditioned
    with pytest.raises(KernelNumericalError, match="Non-finite"):
        SafeKernelRidge(kernel="polynomial", gamma=1e308, degree=5, numerical_policy="raise").fit(
            np.eye(4) * 10, np.arange(4.0)
        )


def test_default_search_uses_relative_gamma_and_requires_explicit_sigmoid():
    optimizer = RegressionModelOptimizer(random_seed=42, n_threads=1)
    for kernel in ("linear", "rbf", "laplacian", "polynomial"):
        trial = optuna.trial.FixedTrial(
            dict(kernel=kernel, alpha=1, gamma_scale=0.5, degree=2, coef0=1)
        )
        params = optimizer.get_param_func("kernelridge_reg")(trial)
        model = optimizer.create_model("kernelridge_reg", params)
        assert model.numerical_policy == "raise"
        assert model.gamma is None
        assert model.gamma_scale == (None if kernel == "linear" else 0.5)
    with pytest.raises(ValueError, match="sigmoid"):
        optimizer.get_param_func("kernelridge_reg")(optuna.trial.FixedTrial({"kernel": "sigmoid"}))


def test_fixed_yaml_parameters_are_retained_in_winning_parameters(tmp_path):
    config = tmp_path / "models.yaml"
    config.write_text("""models:
- name: kernelridge_reg
  param_func: null
  search_space:
  - {name: kernel, type: fixed, value: rbf}
  - {name: alpha, type: fixed, value: 0.7}
  - {name: gamma_scale, type: fixed, value: 0.8}
""")
    predictor = GenomicPredictorV2(
        results_dir=str(tmp_path / "run"),
        n_repeats=2,
        n_splits=3,
        n_trials=1,
        model_config=str(config),
    )
    rng = np.random.default_rng(81)
    X = pd.DataFrame(rng.normal(size=(30, 6)))
    y = pd.Series(2 * X[0] + rng.normal(size=len(X)))
    result = predictor.optimize_model_parameters(X, y, "kernelridge_reg", 0)
    assert result["best_params"] == {
        "kernel": "rbf",
        "alpha": 0.7,
        "gamma": None,
        "gamma_scale": 0.8,
        "numerical_policy": "raise",
    }


@pytest.mark.parametrize("strict_sigmoid", [False, True])
def test_real_fold_pool_preserves_scaled_parameters_and_prunes_invalid_sigmoid(
    tmp_path, strict_sigmoid
):
    rng = np.random.default_rng(79)
    X = pd.DataFrame(
        rng.binomial(2, 0.35, size=(100, 2048)), index=[f"s{i}" for i in range(100)], dtype=float
    )
    y = pd.Series(3 * X[0] - 2 * X[1] + rng.normal(size=len(X)), index=X.index, name="yield")
    X.to_csv(tmp_path / "X.csv", index_label="ID")
    y.to_frame().to_csv(tmp_path / "y.csv", index_label="ID")
    flags = []
    if strict_sigmoid:
        config = tmp_path / "models.yaml"
        config.write_text("""models:
- name: kernelridge_reg
  param_func: null
  search_space:
  - {name: kernel, type: fixed, value: sigmoid}
  - {name: alpha, type: fixed, value: 0.01}
  - {name: gamma, type: fixed, value: 0.0938480072}
  - {name: coef0, type: fixed, value: 3.5675332669}
""")
        flags = ["--model_config", str(config)]
    else:
        flags = ["--feature_selection", "univariate", "--select_k", "32", "--save_models"]
    for mode, folds in [("repeat", "1"), ("fold", "3")]:
        out = tmp_path / mode
        command = [
            sys.executable,
            "-m",
            "gpse.train.cli",
            "--geno_file",
            str(tmp_path / "X.csv"),
            "--pheno_file",
            str(tmp_path / "y.csv"),
            "--target_trait",
            "yield",
            "--task_type",
            "regression",
            "--models",
            "kernelridge_reg",
            "--threads",
            "3",
            "--n_jobs",
            "1",
            "--n_repeats",
            "2",
            "--train_folds",
            "3",
            "--cv_folds",
            "3",
            "--trials",
            "2",
            "--fold_workers",
            folds,
            "--results_dir",
            str(out),
            "--no-use_stacking",
            *flags,
        ]
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=60,
            env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
        )
        histories = [
            json.loads(s)
            for s in (out / "logs/optuna/kernelridge_reg_repeat_001.jsonl").read_text().splitlines()
        ]
        if strict_sigmoid:
            assert result.returncode == 1
            assert all(
                t["state"] == "PRUNED" and "Unstable sigmoid" in t["prune_reason"]
                for t in histories
            )
            assert (
                result.stderr.count("T1 pruned:") == 2
            )  # Once per trial in each attempted repeat.
            assert "UserWarning: Singular matrix" not in result.stderr
            assert "ERROR kernelridge_reg 1 1" not in result.stderr
            assert not list(out.glob("models/*/representative_model/*model*"))
        else:
            assert result.returncode == 0, result.stderr
            assert all(t["state"] == "COMPLETE" for t in histories)
            repeat = json.loads(
                (out / "models/kernelridge_reg/repeat_1/repeat_results.json").read_text()
            )
            assert repeat["params"]["numerical_policy"] == "raise"
            assert "gamma_scale" in repeat["params"] and repeat["params"]["gamma"] is None
            for fold in repeat["fold_results"]:
                metadata = fold["kernel_parameters"]
                assert metadata["feature_count"] == 32
                assert metadata["effective_gamma"] == metadata["gamma_scale"] / 32
            saved = joblib.load(out / "models/kernelridge_reg/repeat_1/fold_1_model.pkl")
            assert saved["model"].gamma_ == repeat["params"]["gamma_scale"] / 32
    if not strict_sigmoid:
        for relative in (
            "evaluation/holdout_metrics.json",
            "models/kernelridge_reg/repeat_1/all_predictions.json",
        ):
            assert (tmp_path / "repeat" / relative).read_bytes() == (
                tmp_path / "fold" / relative
            ).read_bytes()
