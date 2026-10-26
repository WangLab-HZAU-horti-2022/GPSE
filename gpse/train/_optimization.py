#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hyperparameter Optimization Module
===================================
Optuna-based hyperparameter optimization with cross-validation.
"""

from gpse.utils.training_monitor import observed, publish, observation
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import balanced_accuracy_score
from optuna.pruners import MedianPruner
import optuna
from typing import Dict, Any

from loguru import logger as main_logger

from gpse.config import ModelConstants
from gpse.utils.genomic_utils import generate_optimization_seed
from threadpoolctl import threadpool_limits
from gpse.train._model_pipeline import build_training_pipeline
from gpse.models.safe_kernel_ridge import KernelNumericalError


class InvalidValidationScore(ValueError):
    """An expected invalid candidate, not an unexpected training failure."""


def evaluate_optimization_fold(self, X, y, model_name, params, train_idx, val_idx, fold_idx, log):
    """Evaluate one trial fold, shared by serial and budgeted process execution."""
    X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
    y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
    log.debug(f"Training fold {fold_idx + 1}...")
    try:
        fold_model = build_training_pipeline(
            self.create_model(model_name, params),
            task_type=self.task_type,
            model_name=model_name,
            feature_selection_config=self.feature_selection_config,
            genotype_imputation_config=self.genotype_imputation_config,
        )
        with threadpool_limits(limits=self.get_effective_threads(model_name)):
            fold_model.fit(X_fold_train, y_fold_train)
            estimator = fold_model.named_steps["model"]
            if hasattr(estimator, "get_kernel_metadata"):
                publish(
                    "kernel_parameters",
                    "record",
                    fold=fold_idx + 1,
                    **estimator.get_kernel_metadata(),
                )
            if self.task_type == "classification":
                y_fold_pred, y_fold_proba = (
                    self.genomic_classifier.generate_classification_labels_from_model(
                        fold_model, X_fold_val
                    )
                )
                self.genomic_classifier.calculate_classification_metrics(
                    y_fold_val, y_fold_pred, y_fold_proba
                )
                return balanced_accuracy_score(y_fold_val, y_fold_pred)
            y_fold_pred = fold_model.predict(X_fold_val)
        if len(y_fold_val) < 2:
            raise InvalidValidationScore("Validation Pearson requires at least two samples")
        if not np.isfinite(y_fold_val).all():
            raise InvalidValidationScore("Non-finite validation targets")
        if np.ptp(y_fold_val) == 0:
            raise InvalidValidationScore("Constant validation targets; Pearson is undefined")
        if not np.isfinite(y_fold_pred).all():
            raise InvalidValidationScore("Non-finite validation predictions")
        if np.ptp(y_fold_pred) == 0:
            raise InvalidValidationScore("Constant validation predictions; Pearson is undefined")
        score = pearsonr(y_fold_val, y_fold_pred)[0]
        if not np.isfinite(score):
            raise InvalidValidationScore("Non-finite validation Pearson")
        return score
    except (KernelNumericalError, InvalidValidationScore) as exc:
        log.debug(f"Pruning invalid CV fold {fold_idx + 1}: {exc}")
        raise optuna.TrialPruned(f"Invalid CV fold {fold_idx + 1}: {exc}") from exc
    except Exception as exc:
        log.error(f"Fold {fold_idx + 1} training failed: {exc}")
        raise optuna.TrialPruned(f"Invalid CV fold {fold_idx + 1}: {exc}") from exc


@observed("optimization")
def optimize_model_parameters(
    self,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    repeat_idx: int,
    task_logger=None,
    cv_pheno_data=None,
    fold_runner=None,
    cv_folds=None,
) -> Dict[str, Any]:
    """
    Optimize model parameters using Optuna.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    model_name : str
        Model name.
    repeat_idx : int
        Repeat index.
    task_logger
        Task-specific logger.
    cv_pheno_data : pd.DataFrame, optional
        Phenotype data containing CV groups.

    Returns
    -------
    Dict[str, Any]
        Dictionary of best parameters and optimization info.
    """
    log = task_logger if task_logger else main_logger

    if not self.optuna_per_repeat and model_name in self._optimization_cache:
        cached = self._optimization_cache[model_name].copy()
        cached["reused_from_repeat"] = 0
        log.info(f"Reusing frozen Optuna parameters from repeat 1 for {model_name}")
        return cached

    log.info(
        f"Starting parameter optimization for model {model_name} "
        f"(repeat {repeat_idx + 1}/{self.n_repeats})..."
    )

    # Get CV folds
    folds = (
        cv_folds
        if cv_folds is not None
        else self.generate_cv_folds_from_file(X, y, cv_pheno_data, repeat_idx, task_logger)
    )

    # Determinism comes from the TPESampler seed and per-model random_state
    # (merged from YAML default_params); avoid mutating global numpy RNG state
    # here because repeat workers run concurrently.
    opt_seed = generate_optimization_seed(self.random_seed, repeat_idx)

    # Define optimization objective
    @observed("trial")
    def objective(trial):
        param_func = self.get_param_func(model_name)
        params = param_func(trial)

        filtered_params = self.filter_model_params_unified(model_name, params)
        if model_name == "kernelridge_reg":
            resolved = self.get_default_params(model_name)
            resolved.update(filtered_params)
            # Fixed YAML parameters are absent from study.best_params. Retain
            # the exact constructor inputs used by the evaluated candidate.
            trial.set_user_attr("model_params", resolved)

        log.debug(f"Evaluating parameter set: {filtered_params}")
        try:
            if fold_runner is not None:
                cv_scores = fold_runner(filtered_params, folds, trial.number)
            else:
                cv_scores = []
                for fold_idx, (train_idx, val_idx) in enumerate(folds):
                    with observation("optimization_fold", fold=fold_idx + 1):
                        cv_scores.append(
                            evaluate_optimization_fold(
                                self,
                                X,
                                y,
                                model_name,
                                filtered_params,
                                train_idx,
                                val_idx,
                                fold_idx,
                                log,
                            )
                        )
        except optuna.TrialPruned as exc:
            trial.set_user_attr("prune_reason", str(exc))
            raise

        if len(cv_scores) != self.n_splits:
            raise optuna.TrialPruned("Incomplete CV trial; every configured fold is required")

        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        objective_value = mean_score - self.cv_stability_penalty * std_score
        trial.set_user_attr("cv_mean", float(mean_score))
        trial.set_user_attr("cv_std", float(std_score))
        trial.set_user_attr("objective", float(objective_value))
        if self.task_type == "classification":
            log.debug(f"Average balanced accuracy: {mean_score:.6f} (±{std_score:.6f})")
        else:
            log.debug(f"Average Pearson correlation: {mean_score:.6f} (±{std_score:.6f})")
        return objective_value

    # Create early stopping callback
    def early_stopping_callback(study, trial):
        if trial.number >= self.patience:
            if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
                return False
            best_value = study.best_value
            recent_trials = [t.value for t in study.trials[-self.patience :] if t.value is not None]
            if not recent_trials:
                return False
            recent_best_value = max(recent_trials)

            if best_value is None:
                return False

            if recent_best_value <= best_value - 1e-9:
                log.info("")
                log.info(
                    f"Early stopping: no improvement over best={best_value:.6f} "
                    f"in last {self.patience} trials (recent best={recent_best_value:.6f})"
                )
                study.stop()
        return False

    # Create Optuna study and optimize
    pruner = MedianPruner(
        n_startup_trials=ModelConstants.optuna_n_startup_trials,
        n_warmup_steps=ModelConstants.optuna_n_warmup_steps,
        interval_steps=ModelConstants.optuna_interval_steps,
    )
    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        sampler=optuna.samplers.TPESampler(seed=opt_seed),
    )

    history_path = (
        Path(self.logs_dir) / "optuna" / f"{model_name}_repeat_{repeat_idx + 1:03d}.jsonl"
    )
    history_warning = False

    def record_trial(study, trial):
        nonlocal history_warning
        record = {
            "run_id": self.paths.run_id,
            "model": model_name,
            "repeat": repeat_idx + 1,
            "trial": trial.number + 1,
            "optuna_trial_number": trial.number,
            "state": trial.state.name,
            "params": trial.params,
            "metrics": {
                "cv_mean": trial.user_attrs.get("cv_mean"),
                "cv_std": trial.user_attrs.get("cv_std"),
                "robust_objective": trial.value,
            },
            "elapsed_s": trial.duration.total_seconds() if trial.duration else None,
            "prune_reason": trial.user_attrs.get("prune_reason"),
        }
        if model_name == "kernelridge_reg":
            record["model_params"] = trial.user_attrs.get("model_params")
        if trial.state == optuna.trial.TrialState.PRUNED:
            log.warning(
                f"{model_name} R{repeat_idx + 1} T{trial.number + 1} pruned: "
                f"{record['prune_reason'] or 'incomplete/invalid CV trial'}"
            )
        publish("trial_history", "record", **record)
        try:
            history_path.parent.mkdir(parents=True, exist_ok=True)
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
        except Exception as exc:
            if not history_warning:
                log.warning(f"Optuna history file unavailable: {exc}")
                history_warning = True

    max_trials = self.n_trials
    log.info(
        "Starting robust parameter optimization, max trials: "
        f"{max_trials}, objective=mean(CV)-{self.cv_stability_penalty}*std(CV)"
    )
    study.optimize(
        objective,
        n_trials=max_trials,
        callbacks=[record_trial, early_stopping_callback],
        show_progress_bar=False,
        n_jobs=ModelConstants.optuna_n_jobs,
    )

    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        raise ValueError(
            "No valid complete CV trials; undefined Pearson/failed folds cannot be imputed"
        )
    best_params = study.best_params
    log.info(f"Best parameters: {best_params}")
    if self.task_type == "classification":
        log.info(f"Best robust CV balanced accuracy objective: {study.best_value:.6f}")
    else:
        log.info(f"Best robust CV Pearson objective: {study.best_value:.6f}")

    filtered_best_params = self.filter_model_params_unified(model_name, best_params)
    if model_name == "kernelridge_reg":
        # Persist the effective policy, including YAML overrides, rather than
        # relying on future defaults when deployment/stacking refits the model.
        filtered_best_params = dict(study.best_trial.user_attrs["model_params"])

    top_trials = sorted(
        (trial for trial in study.trials if trial.value is not None),
        key=lambda trial: trial.value,
        reverse=True,
    )[:5]
    result = {
        "best_params": filtered_best_params,
        "best_value": study.best_value,
        "n_trials": len(study.trials),
        "requested_n_trials": self.n_trials,
        "effective_n_trials": max_trials,
        "early_stopped": len(study.trials) < max_trials,
        "objective": "mean_cv_score - cv_stability_penalty * std_cv_score",
        "cv_stability_penalty": self.cv_stability_penalty,
        "feature_selection": self.feature_selection_config.as_dict(),
        "genotype_imputation": self.genotype_imputation_config.as_dict(),
        "top_trials": [
            {
                "number": trial.number,
                "objective": trial.value,
                "cv_mean": trial.user_attrs.get("cv_mean"),
                "cv_std": trial.user_attrs.get("cv_std"),
                "params": trial.params,
            }
            for trial in top_trials
        ],
    }
    if not self.optuna_per_repeat:
        self._optimization_cache[model_name] = result.copy()
    return result
