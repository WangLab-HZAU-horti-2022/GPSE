"""Parent-side repeat control, with every fit submitted to one bounded pool.

The controller threads never fit estimators or change native thread limits.
Workers keep a single holdout partition; tasks carry indices and parameters.
"""

from collections import deque
from concurrent.futures import CancelledError, ThreadPoolExecutor
from contextlib import contextmanager
from copy import copy
import os
import threading

import numpy as np
from loguru import logger

from gpse.train._optimization import evaluate_optimization_fold
from gpse.utils.genomic_utils import generate_fold_seed, generate_repeat_seed
from gpse.utils.process_control import uninterrupted_cleanup
from gpse.utils.training_monitor import configure_worker, get_client_config, observation, publish

_PARTITION = None


def reset_partition():
    global _PARTITION
    _PARTITION = None


def make_partition(X, y, indices):
    indices = np.asarray(indices)
    train_indices = np.setdiff1d(np.arange(len(X)), indices, assume_unique=True)
    return (
        train_indices,
        indices,
        X.iloc[train_indices],
        y.iloc[train_indices],
        X.iloc[indices],
        y.iloc[indices],
    )


def execute_fold(kind, model, repeat, payload, worker):
    global _PARTITION
    predictor, X, y, _, indices = worker
    if _PARTITION is None:
        partition = make_partition(X, y, indices)
        _, _, train_X, train_y, test_X, test_y = partition
        if predictor.task_type == "regression" and predictor.standardize_phenotype:
            train_y, scaler = predictor._standardize_phenotype(train_y)
            if scaler["applied"]:
                test_y = (test_y - scaler["mean"]) / scaler["std"]
        _PARTITION = train_X, train_y, test_X, test_y
    train_X, train_y, test_X, test_y = _PARTITION
    configure_worker(
        get_client_config(), predictor.logs_dir / "workers" / f"fold_worker_{os.getpid()}.log"
    )
    if kind == "cv_folds":
        with observation("cv_folds", model=model, repeat=repeat + 1):
            return predictor.generate_cv_folds_from_file(
                train_X, train_y, worker[3], repeat, logger
            )
    fold, train, val, params, extra = payload
    if kind == "optimization_fold":
        np.random.seed(generate_repeat_seed(predictor.random_seed, repeat))
        with observation(
            "optimization_fold", model=model, repeat=repeat + 1, fold=fold + 1, trial=extra + 1
        ):
            return evaluate_optimization_fold(
                predictor, train_X, train_y, model, params, train, val, fold, logger
            )
    np.random.seed(generate_fold_seed(predictor.random_seed, repeat, fold))
    predictions = {"train": [], "val": [], "test": []}
    ensemble = []
    result = predictor._train_single_fold(
        predictor.create_model(model, params),
        model,
        params,
        fold,
        train,
        val,
        train_X,
        train_y,
        test_X,
        test_y,
        indices,
        repeat,
        extra,
        predictions,
        logger,
        ensemble_predictions=ensemble,
    )
    return result, predictions, ensemble


class _Dispatcher:
    def __init__(self, pool, execute, capacity):
        self.pool, self.execute, self.capacity = pool, execute, capacity
        self.slots = threading.BoundedSemaphore(capacity)
        self.stopped = threading.Event()
        self.lock = threading.Lock()
        self.pending = set()

    def submit(self, kind, model, repeat, payload):
        while not self.stopped.is_set():
            if self.slots.acquire(timeout=0.1):
                break
        else:
            raise CancelledError("Training cancelled")
        try:
            if self.stopped.is_set():
                raise CancelledError("Training cancelled")
            future = self.pool.submit(self.execute, kind, model, repeat, payload)
            with self.lock:
                self.pending.add(future)
                publish(
                    "scheduler",
                    "submitted",
                    model=model,
                    repeat=None if repeat is None else repeat + 1,
                    job=kind,
                    active_tasks=len(self.pending),
                    worker_limit=self.capacity,
                    queue="compute",
                )
        except BaseException:
            self.slots.release()
            raise

        def finished(done):
            with self.lock:
                self.pending.discard(done)
            self.slots.release()

        future.add_done_callback(finished)
        return future

    def map_folds(self, kind, model, repeat, params, folds, extra, limit):
        """Bound both the per-repeat window and all in-flight compute tasks.

        Results are reduced in fold order. A failed trial drains/cancels its
        window before the same study is allowed to suggest another trial.
        """
        remaining = iter(enumerate(folds))
        pending = deque()

        def submit_next():
            item = next(remaining, None)
            if item is not None:
                fold, (train, val) = item
                pending.append(self.submit(kind, model, repeat, (fold, train, val, params, extra)))

        try:
            for _ in range(min(limit, len(folds))):
                submit_next()
            while pending:
                result = pending.popleft().result()
                submit_next()
                yield result
        finally:
            for future in pending:
                future.cancel()
            if not self.stopped.is_set():
                for future in pending:
                    try:
                        future.result()
                    except Exception:
                        pass

    def stop(self):
        self.stopped.set()
        with self.lock:
            pending = list(self.pending)
        for future in pending:
            future.cancel()


class _RepeatExecutor:
    def __init__(self, dispatcher, controllers, predictor, X, y, cv, indices, fold_limit):
        self.dispatcher, self.controllers = dispatcher, controllers
        self.predictor, self.X, self.y, self.cv, self.indices = predictor, X, y, cv, indices
        # All controllers share these read-only frames instead of making one
        # whole matrix copy per active repeat in the parent.
        self.partition = make_partition(X, y, indices)
        self.fold_limit = fold_limit
        self.cv_lock = threading.Lock()
        self.cv_futures = {}

    def submit(self, unused_function, kind, model, repeat, payload):
        if kind != "repeat":
            return self.dispatcher.submit(kind, model, repeat, payload)
        return self.controllers.submit(self.run_repeat, model, repeat, payload)

    def run_repeat(self, model, repeat, cache):
        if self.dispatcher.stopped.is_set():
            raise CancelledError("Training cancelled")
        predictor = copy(self.predictor)
        predictor._optimization_cache = dict(self.predictor._optimization_cache)
        if cache is not None:
            predictor._optimization_cache[model] = cache

        folds = None
        if predictor.split_strategy == "structure_aware" and predictor.cv_file is None:
            # PCA and clustering also belong in the bounded compute pool.
            # These assignments depend on the repeat seed, not on the model.
            with self.cv_lock:
                if repeat not in self.cv_futures:
                    self.cv_futures[repeat] = self.dispatcher.submit(
                        "cv_folds", model, repeat, None
                    )
                cv_future = self.cv_futures[repeat]
            folds = cv_future.result()

        def optimization(params, folds, trial):
            return list(
                self.dispatcher.map_folds(
                    "optimization_fold", model, repeat, params, folds, trial, self.fold_limit
                )
            )

        def training(params, folds, repeat_dir):
            return self.dispatcher.map_folds(
                "training_fold", model, repeat, params, folds, repeat_dir, self.fold_limit
            )

        return predictor.train_and_evaluate_model_for_repeat(
            model,
            self.X,
            self.y,
            repeat,
            self.indices,
            task_logger=logger,
            cv_pheno_data=self.cv,
            fold_runner=training,
            optimization_fold_runner=optimization,
            data_partition=self.partition,
            cv_folds=folds,
        )


@contextmanager
def fold_training_pool(
    pool_context,
    execute,
    predictor,
    X,
    y,
    cv,
    indices,
    capacity,
    controller_limit,
    fold_limit,
    warmup_gate,
):
    """Stop compute workers before joining controllers blocked on their futures."""
    controllers = None
    dispatcher = None
    try:
        with pool_context as pool:
            # Spawn in the signal-handling main thread, before controller
            # threads submit work. This keeps cancellation during large
            # initializer transfers out of background submit/shutdown locks.
            try:
                warmup = [pool.submit(execute, "warmup", None, None, None) for _ in range(capacity)]
            finally:
                warmup_gate.set()
            for future in warmup:
                future.result()
            dispatcher = _Dispatcher(pool, execute, capacity)
            controllers = ThreadPoolExecutor(
                max_workers=controller_limit, thread_name_prefix="gpse-repeat"
            )
            executor = _RepeatExecutor(
                dispatcher, controllers, predictor, X, y, cv, indices, fold_limit
            )
            try:
                yield executor
            except BaseException:
                dispatcher.stop()
                raise
    finally:
        if dispatcher is not None:
            dispatcher.stop()
        if controllers is not None:
            with uninterrupted_cleanup():
                controllers.shutdown(wait=True, cancel_futures=True)
