"""One CPU-budgeted pool for CV, Optuna folds and model-refit jobs.

The parent owns dependencies and limits. Frozen searches finish before later
repeats become eligible. No worker creates another training pool.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import wait, FIRST_COMPLETED
import multiprocessing as mp
import os
import time
import uuid

from loguru import logger
from threadpoolctl import threadpool_limits, threadpool_info

from gpse.config import ModelConstants
from gpse.train._repeat_training import aggregate_model_repeats
from gpse.utils.genomic_utils import create_model_result_directory
from gpse.utils.parallel import graceful_process_pool
from gpse.utils.training_monitor import (
    configure_worker,
    get_client_config,
    init_client,
    publish,
    result_metrics,
)

_WORKER = None
_WARMUP_GATE = None


def _initialize(predictor, X, y, cv_data, test_indices, config, warmup_gate=None):
    global _WORKER, _WARMUP_GATE
    from gpse.train._fold_scheduler import reset_partition

    reset_partition()
    init_client(config)
    for variable in ModelConstants.thread_env_vars:
        os.environ[variable] = str(predictor.n_threads)
    _WORKER = (predictor, X, y, cv_data, test_indices)
    _WARMUP_GATE = warmup_gate
    publish("worker", "ready", native_threads=predictor.n_threads, native_pools=threadpool_info())


def _execute(kind, model, repeat, payload):
    if kind == "warmup":
        _WARMUP_GATE.wait()
        return None
    predictor, X, y, cv_data, indices = _WORKER
    with threadpool_limits(limits=predictor.n_threads):
        if kind in {"training_fold", "optimization_fold", "cv_folds"}:
            from gpse.train._fold_scheduler import execute_fold

            return execute_fold(kind, model, repeat, payload, _WORKER)
        if kind == "repeat":
            if payload is not None:
                predictor._optimization_cache[model] = payload
            return predictor._run_repeat_task(model, X, y, repeat, indices, cv_data)
        configure_worker(
            get_client_config(), predictor.logs_dir / "workers" / f"{model}_finalize.log"
        )
        with logger.contextualize(model=model):
            return aggregate_model_repeats(
                predictor, model, X, y, payload, predictor.paths.model_dir(model)
            )


def run_budgeted_models(predictor, models, X, y, cv_data, indices):
    fold_limit = min(predictor.n_splits, predictor.fold_workers or predictor.n_splits)
    controller_limit = max(
        1,
        min(
            predictor.cpu_budget // predictor.n_threads,
            min(len(models), predictor.max_parallel_jobs)
            * min(predictor.n_repeats, predictor.repeat_workers),
        ),
    )
    cap = min(
        predictor.cpu_budget // predictor.n_threads,
        min(len(models), predictor.max_parallel_jobs)
        * min(predictor.n_repeats, predictor.repeat_workers)
        * fold_limit,
    )
    cap = max(1, cap)
    parallel_folds = cap > 1 and fold_limit > 1
    warmup_gate = mp.get_context("spawn").Event() if parallel_folds else None
    states = {
        name: {
            "next": 0,
            "active": 0,
            "done": 0,
            "results": [],
            "cache": None,
            "started": None,
            "execution_id": uuid.uuid4().hex,
            "finalizing": False,
            "finished": False,
        }
        for name in models
    }
    ready = deque(models)
    results, failures, futures = {}, {}, {}
    independent = predictor.use_default_params or predictor.optuna_per_repeat

    def emit(name, status, **fields):
        state = states[name]
        publish(
            "model",
            status,
            model=name,
            execution_id=state["execution_id"],
            elapsed_s=time.monotonic() - state["started"],
            **fields,
        )

    publish(
        "scheduler",
        "configured",
        cpu_budget=predictor.cpu_budget,
        worker_limit=cap,
        threads_per_worker=predictor.n_threads,
        scheduling=(
            "shared_fold_repeat_refit_pool" if parallel_folds else "shared_repeat_refit_pool"
        ),
        controller_limit=controller_limit,
        fold_workers=fold_limit if parallel_folds else 1,
    )

    # spawn avoids forking a process containing the Loguru/renderer/BLAS threads.
    pool_context = graceful_process_pool(
        max_workers=cap,
        mp_context=mp.get_context("spawn"),
        initializer=_initialize,
        initargs=(predictor, X, y, cv_data, indices, get_client_config(), warmup_gate),
        logger=logger,
        native_threads=predictor.n_threads,
    )
    if parallel_folds:
        from gpse.train._fold_scheduler import fold_training_pool

        pool_context = fold_training_pool(
            pool_context,
            _execute,
            predictor,
            X,
            y,
            cv_data,
            indices,
            cap,
            controller_limit,
            fold_limit,
            warmup_gate,
        )
    with pool_context as executor:
        while not all(state["finished"] for state in states.values()):
            while len(futures) < cap:
                submitted = False
                active_models = sum(
                    s["started"] is not None and not s["finished"] for s in states.values()
                )
                for _ in range(len(ready)):
                    name = ready[0]
                    ready.rotate(-1)
                    state = states[name]
                    if state["finished"] or state["finalizing"]:
                        continue
                    if state["started"] is None:
                        if active_models >= predictor.max_parallel_jobs:
                            continue
                        state["started"] = time.monotonic()
                        create_model_result_directory(predictor.paths.models_dir, name)
                        emit(name, "started")
                    if state["done"] == predictor.n_repeats:
                        future = executor.submit(_execute, "finalize", name, None, state["results"])
                        state["finalizing"] = True
                        futures[future] = (name, "finalize", None)
                        publish(
                            "scheduler",
                            "submitted",
                            model=name,
                            job="finalize",
                            active_tasks=len(futures),
                            worker_limit=cap,
                            queue="controller" if parallel_folds else "compute",
                        )
                        submitted = True
                        break
                    if (
                        state["next"] >= predictor.n_repeats
                        or state["active"] >= predictor.repeat_workers
                    ):
                        continue
                    if (
                        not independent
                        and name != "gblup_reg"
                        and state["cache"] is None
                        and state["active"]
                    ):
                        continue
                    repeat = state["next"]
                    future = executor.submit(_execute, "repeat", name, repeat, state["cache"])
                    state["next"] += 1
                    state["active"] += 1
                    futures[future] = (name, "repeat", repeat)
                    publish(
                        "scheduler",
                        "submitted",
                        model=name,
                        repeat=repeat + 1,
                        job="repeat",
                        active_tasks=len(futures),
                        worker_limit=cap,
                        queue="controller" if parallel_folds else "compute",
                    )
                    submitted = True
                    break
                if not submitted:
                    break
            if not futures:
                raise RuntimeError("No eligible training jobs; invalid scheduling limits")
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                name, kind, repeat = futures.pop(future)
                state = states[name]
                if kind == "repeat":
                    state["active"] -= 1
                    state["done"] += 1
                    try:
                        result = future.result()
                        state["results"].append(result)
                        if not predictor.optuna_per_repeat and state["cache"] is None:
                            state["cache"] = result.get("optimization_info")
                    except Exception as exc:
                        logger.error(f"{name} repeat {repeat + 1} failed: {exc}")
                else:
                    state["finished"] = True
                    try:
                        results[name] = future.result()
                        emit(name, "completed", metrics=result_metrics("model", results[name]))
                    except Exception as exc:
                        failures[name] = str(exc)
                        emit(name, "failed", error=str(exc))
                        logger.error(f"Model {name} failed: {exc}")
    return {name: results[name] for name in models if name in results}, failures
