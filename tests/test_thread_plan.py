"""ThreadPlan capabilities, legacy compatibility and real native-pool checks."""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
import os

import pytest

from gpse.train._thread_plan import (
    ThreadPlan,
    current_plan,
    estimator_thread_count,
    execution_plan,
    plan_for,
    training_thread_limits,
)


@pytest.mark.parametrize("cpu", [1, 4, 8, 16])
@pytest.mark.parametrize(
    "path",
    [
        "sklearn.ensemble.RandomForestRegressor",
        "sklearn.ensemble.RandomForestClassifier",
        "xgboost.XGBRegressor",
        "xgboost.XGBClassifier",
        "lightgbm.LGBMRegressor",
        "lightgbm.LGBMClassifier",
        "catboost.CatBoostRegressor",
        "catboost.CatBoostClassifier",
    ],
)
def test_direct_model_plans_do_not_exceed_lease(cpu, path):
    plan = plan_for("model", path, cpu)
    assert plan.estimator_threads == cpu
    assert plan.joblib_threads * max(plan.blas_threads, plan.openmp_threads) <= cpu
    assert estimator_thread_count("model", path, 9) == 9


def test_unknown_profile_is_explicit_and_conservative():
    plan = plan_for("custom", "unknown.Model", 8)
    assert plan.adapter == "conservative_unverified"
    assert plan.estimator_threads == plan.blas_threads == plan.openmp_threads == 1
    with pytest.raises(ValueError, match="Nested"):
        ThreadPlan("bad", "unknown.Model", 8, "bad", 8, 8, 8, 8)


def test_controller_thread_cannot_change_process_native_limits():
    with ThreadPoolExecutor(max_workers=1) as pool:

        def enter():
            with execution_plan(plan_for("rf", "sklearn.ensemble.RandomForestRegressor", 4)):
                pass

        with pytest.raises(RuntimeError, match="isolated"):
            pool.submit(enter).result()


def _native_sequence(counts):
    from threadpoolctl import threadpool_info

    # Import before capturing baseline, just as the execution context does.
    import sklearn.ensemble
    import gpse.models.safe_kernel_ridge

    def pools():
        return {entry["filepath"]: entry["num_threads"] for entry in threadpool_info()}

    baseline = pools()
    snapshots = []
    for count in counts:
        plan = plan_for("rf", "gpse.models.safe_kernel_ridge.SafeKernelRidge", count)
        with execution_plan(plan):
            assert current_plan() == plan
            assert estimator_thread_count("rf", plan.import_path, 99) == plan.estimator_threads
            with training_thread_limits(limits=99):
                snapshot = pools()
                assert snapshot
                assert all(
                    entry["num_threads"] == (count if entry["user_api"] == "blas" else 1)
                    for entry in threadpool_info()
                )
            with pytest.raises(ValueError, match="match"):
                estimator_thread_count("other", plan.import_path, 99)
            snapshots.append(snapshot)
        assert current_plan() is None
        assert pools() == baseline
    try:
        with execution_plan(plan):
            raise ValueError("fit failed")
    except ValueError:
        pass
    assert pools() == baseline
    return os.getpid(), snapshots


def test_reused_worker_restores_limits_after_16_then_4_and_failure():
    pytest.importorskip("sklearn")
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as pool:
        pid, snapshots = pool.submit(_native_sequence, [16, 4]).result(timeout=30)
    assert pid != os.getpid() and len(snapshots) == 2


def test_real_registry_clone_fit_predict_obey_plan():
    np = pytest.importorskip("numpy")
    from sklearn.base import clone
    from gpse.models.regression_model_optimizer import RegressionModelOptimizer

    optimizer = RegressionModelOptimizer(n_threads=99)
    plan = plan_for("rf_reg", "sklearn.ensemble.RandomForestRegressor", 2)
    X = np.arange(120).reshape(30, 4)
    with execution_plan(plan):
        model = optimizer.create_model("rf_reg", {"n_estimators": 3, "n_jobs": 80})
        assert model.n_jobs == 2
        copied = clone(model)
        copied.fit(X, np.arange(30))
        assert np.isfinite(copied.predict(X)).all()
        assert copied.n_jobs == 2
    assert optimizer.create_model("rf_reg", {"n_estimators": 3}).n_jobs == 99


_BARRIER = None


def _set_barrier(barrier):
    global _BARRIER
    _BARRIER = barrier


def _concurrent_native(count):
    from threadpoolctl import threadpool_info

    plan = plan_for("kernelridge_reg", "gpse.models.safe_kernel_ridge.SafeKernelRidge", count)
    with execution_plan(plan):
        _BARRIER.wait(timeout=10)
        pools = threadpool_info()
        _BARRIER.wait(timeout=10)
        return os.getpid(), count, pools


def test_two_heterogeneous_native_plans_overlap_in_separate_processes():
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=2, mp_context=context, initializer=_set_barrier, initargs=(context.Barrier(2),)
    ) as pool:
        futures = [pool.submit(_concurrent_native, count) for count in (2, 4)]
        results = [future.result(timeout=20) for future in futures]
    assert len({result[0] for result in results}) == 2
    for _, count, pools in results:
        assert any(entry["user_api"] == "blas" for entry in pools)
        assert all(
            entry["num_threads"] == (count if entry["user_api"] == "blas" else 1) for entry in pools
        )


@pytest.mark.parametrize("classification", [False, True])
def test_catboost_pool_prediction_and_importance_receive_lease_threads(monkeypatch, classification):
    import numpy as np
    import catboost
    from gpse.models import _lease_catboost
    from gpse.train._thread_plan import apply_estimator_plan

    calls = []
    original_pool = _lease_catboost.Pool

    class RecordingPool(original_pool):
        def __init__(self, *args, **kwargs):
            calls.append(("pool", kwargs.get("thread_count")))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(_lease_catboost, "Pool", RecordingPool)
    base = catboost.CatBoostClassifier if classification else catboost.CatBoostRegressor
    for name in ["predict", "get_feature_importance"] + (
        ["predict_proba"] if classification else []
    ):
        original = getattr(base, name)

        def recorded(self, *args, _original=original, _name=name, **kwargs):
            calls.append((_name, kwargs.get("thread_count")))
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(base, name, recorded)
    plan = plan_for("cat", "catboost." + base.__name__, 2)
    model = apply_estimator_plan(base(iterations=3, verbose=False, allow_writing_files=False), plan)
    X = np.arange(120).reshape(30, 4)
    y = np.arange(30) % 2 if classification else np.arange(30)
    with execution_plan(plan):
        model.fit(X, y)
        model.predict(X)
        model.get_feature_importance()
        if classification:
            model.predict_proba(X)
    assert calls and all(count == 2 for _, count in calls)


@pytest.mark.parametrize(
    "model_name",
    [
        "elasticnet_reg",
        "lasso_reg",
        "sgd_reg",
        "svr_reg",
        "adaboost_reg",
        "knn_reg",
        "rf_reg",
        "lightgbm_reg",
        "histgradientboost_reg",
        "gblup_reg",
        "gbdt_reg",
        "xgboost_reg",
        "catboost_reg",
        "kernelridge_reg",
        "mlp_reg",
        "rf_clf",
        "svm_clf",
        "xgboost_clf",
        "lightgbm_clf",
        "catboost_clf",
        "mlp_clf",
    ],
)
def test_all_registered_estimators_real_fit_and_predict(model_name):
    import numpy as np
    from sklearn.base import clone
    from gpse.models.regression_model_optimizer import RegressionModelOptimizer
    from gpse.models.classification_model_optimizer import ClassificationModelOptimizer

    classification = model_name.endswith("_clf")
    optimizer = (
        ClassificationModelOptimizer(n_threads=99, n_classes=2)
        if classification
        else RegressionModelOptimizer(n_threads=99)
    )
    registry = optimizer._registry
    entry = registry.model_configs[model_name]
    plan = plan_for(model_name, entry.import_path, 2)
    rng = np.random.default_rng(124)
    X = rng.normal(size=(40, 6))
    y = (X[:, 0] > 0).astype(int) if classification else X[:, 0] + X[:, 1] * 0.2
    params = registry.get_default_params(model_name)
    # A smoke workload only: performance comparisons will use identical full workloads.
    for name in ("n_estimators", "iterations", "max_iter"):
        if name in params:
            params[name] = 4
    if "hidden_layer_sizes" in params:
        params["hidden_layer_sizes"] = (5,)
    with execution_plan(plan):
        model = clone(registry.create_model(model_name, params))
        for field in entry.thread_strategy:
            assert model.get_params()[field] == plan.estimator_threads
        with training_thread_limits(99):
            model.fit(X, y)
            predicted = model.predict(X)
            assert np.isfinite(predicted).all()
            assert len(predicted) == len(y)
