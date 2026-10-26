"""Lease-derived thread settings for isolated, sequential compute workers.

No heavy imports at module import time. Legacy callers keep their original
threadpoolctl behavior unless a worker explicitly enters execution_plan.
"""

from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import importlib
import threading
import time

_CURRENT = ContextVar("gpse_thread_plan", default=None)
_EXECUTION_LOCK = threading.Lock()


@dataclass(frozen=True)
class ThreadPlan:
    model: str
    import_path: str
    cpu: int
    adapter: str
    estimator_threads: int
    joblib_threads: int
    openmp_threads: int
    blas_threads: int

    def __post_init__(self):
        counts = (
            self.cpu,
            self.estimator_threads,
            self.joblib_threads,
            self.openmp_threads,
            self.blas_threads,
        )
        if any(type(value) is not int or value < 1 for value in counts):
            raise ValueError("Thread counts must be positive integers")
        if any(value > self.cpu for value in counts[1:]):
            raise ValueError("Thread plan exceeds lease CPU")
        if self.joblib_threads * max(self.openmp_threads, self.blas_threads) > self.cpu:
            raise ValueError("Nested joblib/native thread plan exceeds lease CPU")


def plan_for(model, import_path, cpu):
    """Select capability by actual estimator class, not a configurable model ID."""
    module, _, name = import_path.rpartition(".")
    if (
        module == "sklearn.ensemble"
        and name in {"RandomForestRegressor", "RandomForestClassifier"}
        or module == "sklearn.neighbors"
        and name == "KNeighborsRegressor"
    ):
        return ThreadPlan(model, import_path, cpu, "joblib_threads", cpu, cpu, 1, 1)
    if module in {"xgboost", "lightgbm"}:
        return ThreadPlan(model, import_path, cpu, "native_openmp", cpu, 1, cpu, 1)
    if module == "catboost":
        return ThreadPlan(model, import_path, cpu, "catboost_threads", cpu, 1, 1, 1)
    if name in {"HistGradientBoostingRegressor", "HistGradientBoostingClassifier"}:
        return ThreadPlan(model, import_path, cpu, "openmp", 1, 1, cpu, 1)
    if import_path in {
        "gpse.models.safe_kernel_ridge.SafeKernelRidge",
        "sklearn.neural_network.MLPRegressor",
        "sklearn.neural_network.MLPClassifier",
    }:
        return ThreadPlan(model, import_path, cpu, "blas", 1, 1, 1, cpu)
    return ThreadPlan(model, import_path, cpu, "conservative_unverified", 1, 1, 1, 1)


def current_plan():
    return _CURRENT.get()


def estimator_thread_count(model, import_path, legacy_count):
    plan = current_plan()
    if plan is None:
        return legacy_count
    if (plan.model, plan.import_path) != (model, import_path):
        raise ValueError("Estimator does not match the active lease thread plan")
    return plan.estimator_threads


def apply_estimator_plan(estimator, plan):
    """Reinject limits into cloned/loaded estimators before refit or prediction."""
    if plan.adapter == "catboost_threads":
        from gpse.models._lease_catboost import adapter_class

        cls = adapter_class(plan.import_path)
        if not isinstance(estimator, cls):
            adapted = cls()
            adapted.__setstate__(estimator.__getstate__())
            estimator = adapted
    params = estimator.get_params(deep=False)
    updates = {
        name: plan.estimator_threads
        for name in ("n_jobs", "nthread", "thread_count")
        if name in params
    }
    if updates:
        estimator.set_params(**updates)
    return estimator


@contextmanager
def execution_plan(plan):
    """Apply one plan around preparation, fit, predict and score; restore on exit.

    This is deliberately restricted to a process's main thread. Concurrent
    fits require separate processes: threadpoolctl controls process state.
    Estimator imports precede snapshot/limits so their native pools are seen.
    """
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Thread plans require isolated worker processes, not controller threads")
    if not _EXECUTION_LOCK.acquire(blocking=False):
        raise RuntimeError("A compute thread plan is already active in this process")
    token = None
    try:
        importlib.import_module(plan.import_path.rpartition(".")[0])
        from joblib import parallel_config
        from threadpoolctl import threadpool_limits

        with ExitStack() as stack:
            stack.enter_context(threadpool_limits(limits=plan.blas_threads, user_api="blas"))
            stack.enter_context(threadpool_limits(limits=plan.openmp_threads, user_api="openmp"))
            stack.enter_context(parallel_config(backend="threading", n_jobs=plan.joblib_threads))
            token = _CURRENT.set(plan)
            try:
                yield plan
            finally:
                _CURRENT.reset(token)
    finally:
        _EXECUTION_LOCK.release()


@contextmanager
def training_thread_limits(limits):
    """Keep legacy scopes; never let an inner legacy count override a lease."""
    if current_plan() is not None:
        yield
    else:
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=limits):
            yield


@contextmanager
def leased_worker_observation(lease):
    """Real process CPU time (includes native threads); no child-time summation."""
    import os
    import resource
    import sys
    from dataclasses import asdict
    from gpse.utils.training_monitor import publish

    started, cpu_started = time.monotonic(), time.process_time()
    usage_before = resource.getrusage(resource.RUSAGE_SELF)
    status = "completed"
    try:
        publish(
            "compute",
            "started",
            task_key=asdict(lease.key),
            lease_id=lease.lease_id,
            cpu_units=lease.request.cpu,
            attempt=lease.attempt,
            worker_generation=lease.generation,
        )
        yield
    except BaseException:
        status = "failed"
        raise
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        publish(
            "compute",
            status,
            task_key=asdict(lease.key),
            lease_id=lease.lease_id,
            cpu_units=lease.request.cpu,
            attempt=lease.attempt,
            worker_generation=lease.generation,
            pid=os.getpid(),
            task_wall_seconds=time.monotonic() - started,
            process_cpu_seconds=time.process_time() - cpu_started,
            process_lifetime_peak_rss_bytes=usage.ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
            minor_faults=usage.ru_minflt - usage_before.ru_minflt,
            major_faults=usage.ru_majflt - usage_before.ru_majflt,
        )
