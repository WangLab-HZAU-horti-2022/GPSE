#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Leak-free stacking ensemble based on train-side out-of-fold predictions."""

from gpse.utils.training_monitor import observed, observation
import os
import time
import multiprocessing as mp
from concurrent.futures import as_completed
from pathlib import Path
from gpse.train._thread_plan import training_thread_limits as threadpool_limits

from gpse.utils.parallel import graceful_process_pool
from gpse.utils.training_monitor import configure_worker, get_client_config, init_client
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gpse.utils.log_utils import logger
from gpse.train._feature_selection import (
    FeatureSelectionConfig,
    GenotypeImputationConfig,
)
from gpse.train._model_pipeline import build_training_pipeline

_STACK_DATA = None
_STACK_GATE = None


def _init_stack_worker(
    X, y, X_test, n_threads, task_type, log_dir, config, gate=None, checkpoint_run=None
):
    global _STACK_DATA, _STACK_GATE
    init_client(config)
    _STACK_DATA = (X, y, X_test, n_threads, task_type)
    _STACK_GATE = gate
    if checkpoint_run is not None:
        from gpse.train._task_store import hold_worker_lock

        hold_worker_lock(checkpoint_run)
    configure_worker(config, Path(log_dir) / f"stacking_{os.getpid()}.log")


def _stack_warmup():
    _STACK_GATE.wait()


def _stack_leased_job(kind, model, repeat, payload, lease):
    from dataclasses import asdict
    from threadpoolctl import threadpool_info
    from gpse.train._thread_plan import (
        apply_estimator_plan,
        execution_plan,
        plan_for,
        leased_worker_observation,
    )
    from gpse.utils.training_monitor import publish

    if (lease.key.model, lease.key.stage) != (model, kind):
        raise ValueError("Stacking task does not match lease")
    fold, train, val, pipeline, import_path = payload
    plan = plan_for(model, import_path, lease.request.cpu)
    pipeline.set_params(model=apply_estimator_plan(pipeline.named_steps["model"], plan))
    from gpse.train._task_store import task_artifacts

    with execution_plan(plan), leased_worker_observation(lease), task_artifacts(lease):
        publish(
            "thread_plan",
            "started",
            model=model,
            lease_id=lease.lease_id,
            attempt=lease.attempt,
            worker_generation=lease.generation,
            thread_plan=asdict(plan),
            native_pools=threadpool_info(),
        )
        result = _stack_job(pipeline, model, fold, train, val)
        if (Path(lease.key.run) / ".task_store" / "run.json").exists():
            from gpse.train._task_store import prepare_result

            return prepare_result(lease, result)
        return result


def _stack_job(pipeline, model_name, fold_idx, train_idx, val_idx):
    X, y, X_test, threads, task_type = _STACK_DATA
    stage = "stacking_refit" if fold_idx is None else "stacking_fold"
    with observation(stage, model=model_name, fold=None if fold_idx is None else fold_idx + 1):
        with threadpool_limits(limits=threads):
            if train_idx is None:
                pipeline.fit(X, y)
                target = X_test
            else:
                pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
                target = X.iloc[val_idx]
            prediction = None
            if target is not None:
                if task_type == "classification" and hasattr(pipeline, "predict_proba"):
                    probabilities = pipeline.predict_proba(target)
                    prediction = (
                        probabilities[:, 1]
                        if probabilities.shape[1] == 2
                        else probabilities.max(axis=1)
                    )
                else:
                    prediction = pipeline.predict(target)
    return prediction, pipeline if train_idx is None else None


class StackingEnsemble:
    """Train a low-capacity meta-model on strictly out-of-fold base predictions."""

    def __init__(
        self,
        base_models_dir: str,
        model_factory: Callable[[str], Any],
        top_n_models: int = 5,
        meta_model_type: str = "ridge",
        cv_folds: int = 5,
        random_seed: int = 42,
        output_dir: Optional[str] = None,
        n_threads: int = 1,
        use_default_params: bool = False,
        task_type: str = "regression",
        feature_selection_config=None,
        genotype_imputation_config=None,
        cpu_budget=None,
        thread_resolver: Optional[Callable[[str], int]] = None,
        execution_config=None,
        resource_manager=None,
        model_import_paths=None,
        checkpoint_fingerprint=None,
        resume_from=None,
    ):
        self.base_models_dir = base_models_dir
        self._model_factory = model_factory
        self.top_n_models = top_n_models
        self.meta_model_type = meta_model_type
        self.cv_folds = cv_folds
        self.random_seed = random_seed
        self.output_dir = output_dir or os.path.join(base_models_dir, "ensemble_stacking")
        self.n_threads = n_threads
        self.cpu_budget = cpu_budget
        self.execution_config = execution_config
        self._resource_manager = resource_manager
        self.model_import_paths = model_import_paths or {}
        self.checkpoint_fingerprint = checkpoint_fingerprint
        self.resume_from = resume_from
        self.use_default_params = use_default_params
        self.task_type = task_type
        self.feature_selection_config = feature_selection_config or FeatureSelectionConfig()
        self.genotype_imputation_config = genotype_imputation_config or GenotypeImputationConfig()
        self._thread_resolver = thread_resolver
        os.makedirs(self.output_dir, exist_ok=True)
        self.base_models: Dict[str, Any] = {}
        self.selected_models: List[str] = []
        self.meta_model = None
        self.meta_features_cols: List[str] = []

    def _effective_threads(self, model_name: str) -> int:
        if self._thread_resolver is not None:
            return self._thread_resolver(model_name)
        return self.n_threads

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_resource_manager", None)
        return state

    def _prediction_feature(self, model: Any, X: np.ndarray) -> np.ndarray:
        if self.task_type == "classification" and hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)
            return probabilities[:, 1] if probabilities.shape[1] == 2 else probabilities.max(axis=1)
        return model.predict(X)

    def _splitter(self, X: pd.DataFrame, y: pd.Series):
        if self.task_type == "classification":
            counts = y.value_counts()
            if counts.min() < self.cv_folds:
                raise ValueError(
                    "Stacking OOF CV requires every class to contain at least "
                    f"cv_folds={self.cv_folds} samples; class counts: {counts.to_dict()}"
                )
            return StratifiedKFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed
            ).split(X, y)
        return KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed).split(X)

    def create_meta_features(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_test: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Build OOF features; every training row is predicted by an unseen model."""
        meta_train = np.full((len(X_train), len(self.selected_models)), np.nan)
        meta_test = (
            np.full((len(X_test), len(self.selected_models)), np.nan)
            if X_test is not None
            else None
        )

        if self.execution_config is not None:
            return self._elastic_meta_features(X_train, y_train, X_test, meta_train, meta_test)

        if self.cpu_budget is not None:
            return self._parallel_meta_features(X_train, y_train, X_test, meta_train, meta_test)

        for column, model_name in enumerate(self.selected_models):
            logger.info(f"Generating strict OOF meta-feature for model {model_name}")
            for fold_idx, (train_idx, val_idx) in enumerate(self._splitter(X_train, y_train)):
                with observation("stacking_fold", model=model_name, fold=fold_idx + 1):
                    fold_pipeline = build_training_pipeline(
                        self._model_factory(model_name),
                        task_type=self.task_type,
                        model_name=model_name,
                        feature_selection_config=self.feature_selection_config,
                        genotype_imputation_config=self.genotype_imputation_config,
                    )
                    with threadpool_limits(limits=self._effective_threads(model_name)):
                        fold_pipeline.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
                    meta_train[val_idx, column] = self._prediction_feature(
                        fold_pipeline, X_train.iloc[val_idx]
                    )

            full_pipeline = build_training_pipeline(
                self._model_factory(model_name),
                task_type=self.task_type,
                model_name=model_name,
                feature_selection_config=self.feature_selection_config,
                genotype_imputation_config=self.genotype_imputation_config,
            )
            with threadpool_limits(limits=self._effective_threads(model_name)):
                full_pipeline.fit(X_train, y_train)
            self.base_models[model_name] = full_pipeline
            if X_test is not None:
                meta_test[:, column] = self._prediction_feature(
                    full_pipeline,
                    X_test,
                )

        return meta_train, meta_test

    def _parallel_meta_features(self, X_train, y_train, X_test, meta_train, meta_test):
        count = len(self.selected_models) * (self.cv_folds + 1)
        cap = max(1, min(count, self.cpu_budget // self.n_threads))
        with graceful_process_pool(
            max_workers=cap,
            mp_context=mp.get_context("spawn"),
            initializer=_init_stack_worker,
            initargs=(
                X_train,
                y_train,
                X_test,
                self.n_threads,
                self.task_type,
                Path(self.output_dir).parents[1] / "logs" / "workers",
                get_client_config(),
            ),
            logger=logger,
            native_threads=self.n_threads,
        ) as pool:
            futures = {}
            for column, name in enumerate(self.selected_models):
                splits = list(self._splitter(X_train, y_train))
                for fold_idx, (train, val) in enumerate(splits):
                    pipeline = build_training_pipeline(
                        self._model_factory(name),
                        task_type=self.task_type,
                        model_name=name,
                        feature_selection_config=self.feature_selection_config,
                        genotype_imputation_config=self.genotype_imputation_config,
                    )
                    future = pool.submit(_stack_job, pipeline, name, fold_idx, train, val)
                    futures[future] = (column, name, val)
                pipeline = build_training_pipeline(
                    self._model_factory(name),
                    task_type=self.task_type,
                    model_name=name,
                    feature_selection_config=self.feature_selection_config,
                    genotype_imputation_config=self.genotype_imputation_config,
                )
                future = pool.submit(_stack_job, pipeline, name, None, None, None)
                futures[future] = (column, name, None)
            for future in as_completed(futures):
                column, name, val = futures[future]
                prediction, fitted = future.result()
                if val is None:
                    self.base_models[name] = fitted
                    if meta_test is not None:
                        meta_test[:, column] = prediction
                else:
                    meta_train[val, column] = prediction
        return meta_train, meta_test

    def _elastic_meta_features(self, X_train, y_train, X_test, meta_train, meta_test):
        from gpse.train._elastic_dispatcher import ElasticDispatcher
        from gpse.train._resources import ResourceRequest
        from gpse.utils.training_monitor import publish

        config = self.execution_config
        manager = self._resource_manager
        if manager is None:
            raise ValueError("Elastic stacking requires the run resource manager")
        gate = mp.get_context("spawn").Event()
        dispatcher = None
        coverage = np.zeros(meta_train.shape, dtype=int)
        task_store = None
        if self.checkpoint_fingerprint:
            from gpse.train._task_store import TaskStore

            task_store = TaskStore(
                Path(self.output_dir).parents[1], self.checkpoint_fingerprint, self.resume_from
            )
        try:
            with graceful_process_pool(
                max_workers=config.worker_slots,
                mp_context=mp.get_context("spawn"),
                initializer=_init_stack_worker,
                initargs=(
                    X_train,
                    y_train,
                    X_test,
                    1,
                    self.task_type,
                    Path(self.output_dir).parents[1] / "logs" / "workers",
                    get_client_config(),
                    gate,
                    Path(self.output_dir).parents[1] if task_store else None,
                ),
                logger=logger,
                native_threads=1,
            ) as pool:
                try:
                    warmup = [pool.submit(_stack_warmup) for _ in range(config.worker_slots)]
                finally:
                    gate.set()
                for future in warmup:
                    future.result()
                dispatcher = ElasticDispatcher(
                    pool,
                    _stack_leased_job,
                    manager,
                    {
                        name: ResourceRequest(
                            config.resource_profiles[name].preferred_cpu_per_fit,
                            config.resource_profiles[name].memory_bytes,
                        )
                        for name in self.selected_models
                    },
                    {
                        name: (
                            config.resource_profiles[name].weight
                            if config.scheduler == "elastic_v1"
                            else 1
                        )
                        for name in self.selected_models
                    },
                    str(Path(self.output_dir).parents[1]),
                    config.aging_seconds,
                    on_event=publish,
                    task_store=task_store,
                )
                try:
                    futures = {}
                    for column, name in enumerate(self.selected_models):
                        for fold, (train, val) in enumerate(self._splitter(X_train, y_train)):
                            pipeline = self._stack_pipeline(name)
                            future = dispatcher.submit(
                                "stacking_fold",
                                name,
                                None,
                                (fold, train, val, pipeline, self.model_import_paths[name]),
                            )
                            futures[future] = column, name, val
                        future = dispatcher.submit(
                            "stacking_refit",
                            name,
                            None,
                            (
                                None,
                                None,
                                None,
                                self._stack_pipeline(name),
                                self.model_import_paths[name],
                            ),
                        )
                        futures[future] = column, name, None
                    for future in as_completed(futures):
                        column, name, val = futures[future]
                        prediction, fitted = future.result()
                        if val is None:
                            self.base_models[name] = fitted
                            if meta_test is not None:
                                meta_test[:, column] = prediction
                        else:
                            if np.any(coverage[val, column]):
                                raise ValueError("Duplicate stacking OOF sample assignment")
                            meta_train[val, column] = prediction
                            coverage[val, column] += 1
                except BaseException:
                    dispatcher.stop()
                    raise
        finally:
            if dispatcher is not None:
                dispatcher.stop()
                dispatcher.confirm_pool_stopped()
                manager.next_generation()
        if not np.all(coverage == 1) or not np.isfinite(meta_train).all():
            raise ValueError("Incomplete stacking OOF coverage")
        if meta_test is not None and not np.isfinite(meta_test).all():
            raise ValueError("Incomplete stacking holdout predictions")
        return meta_train, meta_test

    def _stack_pipeline(self, name):
        return build_training_pipeline(
            self._model_factory(name),
            task_type=self.task_type,
            model_name=name,
            feature_selection_config=self.feature_selection_config,
            genotype_imputation_config=self.genotype_imputation_config,
        )

    def fit_meta_model(self, meta_train: np.ndarray, y_train: pd.Series) -> None:
        if self.task_type == "classification":
            estimator = LogisticRegression(
                C=1.0, random_state=self.random_seed, max_iter=1000, n_jobs=self.n_threads
            )
            self.meta_model = Pipeline([("scaler", StandardScaler()), ("logistic", estimator)])
        else:
            estimator = Ridge(alpha=1.0, random_state=self.random_seed)
            self.meta_model = Pipeline([("scaler", StandardScaler()), ("ridge", estimator)])
        # The meta learner has only Top-N features; keep this tiny fit single-threaded.
        with threadpool_limits(limits=1):
            self.meta_model.fit(meta_train, y_train)

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.meta_model is None:
            raise ValueError("Meta-model is not trained; call fit first")
        meta_features = np.zeros((len(X), len(self.selected_models)))
        for column, model_name in enumerate(self.selected_models):
            model = self.base_models[model_name]
            meta_features[:, column] = self._prediction_feature(model, X)
        predictions = self.meta_model.predict(meta_features)
        probabilities = (
            self.meta_model.predict_proba(meta_features)
            if self.task_type == "classification" and hasattr(self.meta_model, "predict_proba")
            else None
        )
        return predictions, probabilities

    @observed("stacking")
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        model_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if not model_names:
            raise ValueError("Stacking requires training-side selected base models")
        self.selected_models = list(model_names[: self.top_n_models])
        self.meta_features_cols = self.selected_models.copy()
        start_time = time.time()
        meta_train, meta_test = self.create_meta_features(X_train, y_train, X_test)
        if not np.isfinite(meta_train).all() or (
            meta_test is not None and not np.isfinite(meta_test).all()
        ):
            raise ValueError("Stacking requires complete finite OOF and holdout features")
        from gpse.train._execution_config import coordinator_stage

        task_store = None
        if self.checkpoint_fingerprint:
            from dataclasses import asdict
            from gpse.train._task_store import TaskStore, task_signature
            from gpse.train._resources import TaskKey
            from gpse.utils.training_monitor import publish

            run_dir = Path(self.output_dir).parents[1]
            task_store = TaskStore(run_dir, self.checkpoint_fingerprint, self.resume_from)
            key = TaskKey(str(run_dir), "__meta__", None, None, None, "stacking_meta")
            signature = task_signature("stacking_meta", (meta_train, y_train, meta_test, y_test))
            restored, cached = task_store.restore(key, signature)
            if restored:
                self.meta_model, results = cached
                publish(
                    "resources", "task_restored", task_key=asdict(key), source="verified_checkpoint"
                )
                return results

        with coordinator_stage(
            getattr(self, "_resource_manager", None),
            self.execution_config,
            Path(self.output_dir).parents[1],
            "stacking_meta",
            "__meta__",
        ) as lease:
            self.fit_meta_model(meta_train, y_train)

            results: Dict[str, Any] = {
                "selected_models": self.selected_models,
                "train_metrics": self.calculate_metrics(
                    y_train,
                    self.meta_model.predict(meta_train),
                    self._meta_probabilities(meta_train),
                ),
                "training_time": time.time() - start_time,
            }
            if X_test is not None and y_test is not None and meta_test is not None:
                results["test_metrics"] = self.calculate_metrics(
                    y_test, self.meta_model.predict(meta_test), self._meta_probabilities(meta_test)
                )
            if task_store is not None:
                from gpse.train._task_store import prepare_result, stage_directory

                destination = stage_directory(lease) / "ensemble"
                destination.mkdir(parents=True, exist_ok=False)
                self._save_stacking_files(results, destination)
                receipt = prepare_result(lease, (self.meta_model, results))
                task_store.commit(lease, receipt, signature, self._resource_manager)
                return results

        self._save_stacking_files(results, self.output_dir)
        return results

    def _save_stacking_files(self, results, directory):
        factory = self._model_factory
        self._model_factory = None
        try:
            joblib.dump(self, os.path.join(directory, "stacking_ensemble_model.pkl"))
        finally:
            self._model_factory = factory
        joblib.dump(results, os.path.join(directory, "stacking_results.pkl"))

    def _meta_probabilities(self, meta_features: np.ndarray) -> Optional[np.ndarray]:
        if self.task_type == "classification" and hasattr(self.meta_model, "predict_proba"):
            return self.meta_model.predict_proba(meta_features)
        return None

    def calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        if self.task_type == "classification":
            metrics = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(
                    precision_score(y_true, y_pred, average="weighted", zero_division=0)
                ),
                "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            }
            if y_pred_proba is not None:
                try:
                    metrics["auc"] = float(
                        roc_auc_score(
                            y_true,
                            y_pred_proba[:, 1] if y_pred_proba.shape[1] == 2 else y_pred_proba,
                            multi_class="ovr" if y_pred_proba.shape[1] > 2 else "raise",
                            average="weighted" if y_pred_proba.shape[1] > 2 else "macro",
                        )
                    )
                except ValueError:
                    metrics["auc"] = 0.0
            return metrics

        from gpse.utils.genomic_utils import calculate_metrics

        return calculate_metrics(y_true, y_pred)
