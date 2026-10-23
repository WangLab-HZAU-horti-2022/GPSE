#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utilities for validating and running GPSE parallel execution."""

from __future__ import annotations

import contextlib
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Generator
from pathlib import Path
import math

from gpse.utils.process_control import (initialize_pool_worker, owned_processes, terminate_owned, uninterrupted_cleanup)


def _log(logger: Any | None, level: str, message: str) -> None:
    """Log through a loguru/logging-compatible object when available."""
    if logger is None:
        return
    log_func = getattr(logger, level, None)
    if callable(log_func):
        log_func(message)


def get_available_cpu_cores(logger: Any | None = None) -> int:
    """Respect affinity and cgroup CPU quotas, including fractional quotas."""
    available = os.cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        try:
            available = min(available, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    # Inspect the current cgroup and ancestors; an ancestor may impose the limit.
    candidates = {Path("/sys/fs/cgroup")}
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            _, controllers, relative = line.split(":", 2)
            root = Path("/sys/fs/cgroup")
            bases = [root]
            if controllers:
                if "cpu" not in controllers.split(","):
                    continue
                bases = [root / name for name in ("cpu", "cpu,cpuacct", "cpuacct,cpu")]
            for base in bases:
                current = base / relative.lstrip("/")
                while current.is_relative_to(base):
                    candidates.add(current)
                    if current == base:
                        break
                    current = current.parent
    except (OSError, ValueError):
        pass
    for directory in candidates:
        try:
            quota, period = (directory / "cpu.max").read_text().split()
            if quota != "max":
                available = min(available, max(1, math.ceil(int(quota) / int(period))))
        except (OSError, ValueError, ZeroDivisionError):
            try:
                quota = int((directory / "cpu.cfs_quota_us").read_text())
                period = int((directory / "cpu.cfs_period_us").read_text())
                if quota > 0 and period > 0:
                    available = min(available, max(1, math.ceil(quota / period)))
            except (OSError, ValueError):
                pass
    available = max(1, available)
    _log(logger, "debug", f"Available CPU budget (affinity/cgroup): {available}")
    return available


def resolve_cpu_budget(value):
    if value in {None, "auto", "all", -1, "-1"}:
        return get_available_cpu_cores()
    try:
        budget = int(value)
    except (TypeError, ValueError):
        raise ValueError("--threads must be auto, -1, or a positive integer") from None
    if budget < 1 or budget > get_available_cpu_cores():
        raise ValueError(f"--threads must be between 1 and {get_available_cpu_cores()}, or auto")
    return budget


def validate_parallelism(
    n_jobs: int,
    max_workers: int,
    repeat_workers: int = 1,
    *,
    logger: Any | None = None,
    available_cores: int | None = None,
) -> tuple[int, int, int]:
    """Validate GPSE process/thread parallelism.

    Parameters
    ----------
    n_jobs
        Number of threads used inside each training worker.
    max_workers
        Number of models trained in parallel.
    repeat_workers
        Number of repeats trained in parallel within each model.
    logger
        Optional logger initialized by ``gpse.utils.log_utils``.
    available_cores
        Optional explicit core count, mainly useful for tests.

    Returns
    -------
        tuple[int, int, int]
        The validated ``(n_jobs, max_workers, repeat_workers)`` tuple.

    Raises
    ------
    ValueError
        If either value is not positive, or if ``n_jobs * max_workers`` exceeds
        the cores available to the current process.
    """
    if n_jobs < 1:
        message = f"--n_jobs must be >= 1, got {n_jobs}"
        _log(logger, "error", message)
        raise ValueError(message)
    if max_workers < 1:
        message = f"--max_workers must be >= 1, got {max_workers}"
        _log(logger, "error", message)
        raise ValueError(message)
    if repeat_workers < 1:
        message = f"--repeat_workers must be >= 1, got {repeat_workers}"
        _log(logger, "error", message)
        raise ValueError(message)

    available = available_cores if available_cores is not None else get_available_cpu_cores(logger)
    if available < 1:
        message = f"Available CPU core count must be >= 1, got {available}"
        _log(logger, "error", message)
        raise ValueError(message)

    requested = n_jobs * max_workers * repeat_workers
    if requested > available:
        message = (
            "Requested parallelism n_jobs * max_workers * repeat_workers = "
            f"{requested}, but only {available} CPU core(s) are available. "
            "Please reduce --n_jobs, --max_workers, or --repeat_workers."
        )
        _log(logger, "error", message)
        raise ValueError(message)

    _log(
        logger,
        "info",
        (
            f"Parallelism check passed: n_jobs={n_jobs}, model_workers={max_workers}, "
            f"repeat_workers={repeat_workers}, cores={available}"
        ),
    )
    return n_jobs, max_workers, repeat_workers


def derive_parallelism_from_threads(
    threads: int,
    n_models: int,
    n_repeats: int,
    n_jobs: int = 1,
    max_workers: int = 1,
    repeat_workers: int = 1,
    explicit: set[str] | None = None,
) -> tuple[int, int, int]:
    """Derive ``(n_jobs, max_workers, repeat_workers)`` from a total thread budget.

    The strategy prefers to spend the budget on model-level parallelism first,
    then repeat-level parallelism, and finally recycles any leftover budget
    into ``n_jobs``. Values that are still at their argparse defaults (``1``)
    are treated as "not explicitly set" and are overridden; non-default values
    are preserved. ``explicit`` also preserves explicitly supplied values of 1.

    Parameters
    ----------
    threads
        Target total number of concurrent training units.
    n_models
        Number of models that will be trained.
    n_repeats
        Number of repeats configured for each model.
    n_jobs
        Current ``--n_jobs`` value.
    max_workers
        Current ``--max_workers`` value.
    repeat_workers
        Current ``--repeat_workers`` value.

    Returns
    -------
    tuple[int, int, int]
        The derived ``(n_jobs, max_workers, repeat_workers)`` tuple.

    Raises
    ------
    ValueError
        If any input is not positive.
    """
    if threads < 1:
        raise ValueError(f"--threads must be >= 1, got {threads}")
    if n_models < 1:
        raise ValueError(f"n_models must be >= 1, got {n_models}")
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")

    if min(n_jobs, max_workers, repeat_workers) < 1:
        raise ValueError("Parallel worker and thread counts must be positive")
    explicit = explicit or set()
    derived_n_jobs = n_jobs
    derived_max_workers = max_workers
    derived_repeat_workers = repeat_workers

    if max_workers == 1 and "max_workers" not in explicit:
        derived_max_workers = min(n_models, threads)

    # Budget left for repeats inside each model, accounting for n_jobs.
    per_model_budget = threads // derived_max_workers
    if repeat_workers == 1 and "repeat_workers" not in explicit:
        derived_repeat_workers = max(1, min(n_repeats, per_model_budget // derived_n_jobs))

    # Give any leftover budget to per-worker threads so the --threads target
    # is actually used (e.g. threads=80, 15 models, 2 repeats would otherwise
    # only use 15 * 2 = 30 workers and waste the remaining cores).
    if n_jobs == 1 and "n_jobs" not in explicit:
        used = derived_max_workers * derived_repeat_workers
        derived_n_jobs = max(1, threads // used)

    return derived_n_jobs, derived_max_workers, derived_repeat_workers


@contextlib.contextmanager
def native_thread_environment(count):
    if count is None:
        yield
        return
    names = ("GPSE_NATIVE_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
    previous = {name: os.environ.get(name) for name in names}
    os.environ.update({name: str(count) for name in names})
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextlib.contextmanager
def graceful_process_pool(
    max_workers: int,
    *,
    initializer=None,
    initargs=(),
    mp_context=None,
    logger=None,
    native_threads=None,
) -> Generator[ProcessPoolExecutor, None, None]:
    """ProcessPoolExecutor context manager that shuts down promptly on Ctrl+C.

    When the user presses Ctrl+C during a training run, the default
    ``ProcessPoolExecutor`` context manager waits for all worker processes to
    finish their current task before returning. This helper intercepts the
    resulting ``KeyboardInterrupt``, cancels pending futures, and terminates
    any worker processes that are still alive so the CLI exits quickly.

    Parameters
    ----------
    max_workers
        Maximum number of worker processes.
    initializer
        Optional callable run in each worker on start.
    initargs
        Arguments passed to ``initializer``.
    mp_context
        Optional multiprocessing context (e.g. ``mp.get_context("spawn")``).
        When ``None`` the interpreter default is used.
    logger
        Optional loguru/logging-compatible logger for shutdown messages.

    Yields
    ------
    ProcessPoolExecutor
        The underlying executor; submit work and iterate ``as_completed`` as
        usual inside the ``with`` block.
    """
    with native_thread_environment(native_threads):
        existing = {(info[0], info[1]) for info in owned_processes([os.getpid()]).values()}
        executor = ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp_context,
            initializer=initialize_pool_worker,
            initargs=(initializer, initargs),
        )
        try:
            yield executor
            # Keep normal shutdown inside the interrupt handler as well.
            executor.shutdown(wait=True, cancel_futures=True)
        except BaseException as exc:
            with uninterrupted_cleanup():
                if isinstance(exc, KeyboardInterrupt):
                    _log(logger, "warning", "Interrupt received (Ctrl+C); shutting down worker processes...")
                workers = list((getattr(executor, "_processes", None) or {}).values())
                identities = owned_processes([w.pid for w in workers])
                # Ctrl+C can arrive after Process.start but before Executor registers
                # its handle. Include newly owned children so that window cannot leak.
                for pid, info in owned_processes([os.getpid()]).items():
                    if (pid, info[1]) not in existing:
                        try:
                            command = Path(f"/proc/{pid}/cmdline").read_bytes()
                        except OSError:
                            command = b""
                        if b"multiprocessing.resource_tracker" not in command:
                            identities[pid] = info

                executor.shutdown(wait=False, cancel_futures=True)
                _terminate_executor_workers(executor, workers=workers, identities=identities)
            raise


def _terminate_executor_workers(executor: ProcessPoolExecutor, workers=None, identities=None) -> None:
    workers = workers if workers is not None else list((getattr(executor, "_processes", None) or {}).values())
    identities = identities if identities is not None else owned_processes([w.pid for w in workers])
    terminate_owned(identities)
    for worker in workers:
        try:
            if worker.is_alive():
                worker.kill()
            worker.join(timeout=1)
        except (OSError, ValueError):
            pass
