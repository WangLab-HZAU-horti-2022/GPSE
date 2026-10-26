"""Deterministic admission and error tests; no actual model performance claims."""

from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import threading
import multiprocessing as mp
import os
import time

import pytest

from gpse.train._elastic_dispatcher import ElasticDispatcher
from gpse.train._resources import ResourceManager, ResourceRequest


class ControlledPool:
    def __init__(self):
        self.condition = threading.Condition()
        self.jobs = []
        self.error = None

    def submit(self, function, *args):
        if self.error:
            raise self.error
        future = Future()
        future.set_running_or_notify_cancel()
        with self.condition:
            self.jobs.append((args, future))
            self.condition.notify_all()
        return future

    def job(self, index):
        with self.condition:
            assert self.condition.wait_for(lambda: len(self.jobs) > index, timeout=3)
            return self.jobs[index]


def dispatcher(pool, cpu=2, model_limit=2):
    manager = ResourceManager(cpu, 100, cpu, {"slow": model_limit, "fast": cpu})
    return ElasticDispatcher(
        pool,
        None,
        manager,
        {"slow": ResourceRequest(1, 10), "fast": ResourceRequest(1, 10)},
        {"slow": 1, "fast": 2},
        "run",
    )


def submit(d, fold, model="slow"):
    return d.submit("training_fold", model, 0, (fold, (), (), {}, "unused"))


def test_nonblocking_queue_only_submits_admitted_tasks_and_reuses_release():
    pool = ControlledPool()
    d = dispatcher(pool)
    try:
        promises = [submit(d, i) for i in range(5)]
        pool.job(1)
        assert len(pool.jobs) == 2
        assert d.manager.snapshot()["cpu_used"] == 2
        for index in range(5):
            args, future = pool.job(index)
            future.set_result(args[3][0])
            assert promises[index].result(timeout=3) == index
        assert d.manager.snapshot()["active_tasks"] == 0
    finally:
        d.confirm_pool_stopped()


def test_cancel_queued_task_does_not_cancel_running_lease():
    pool = ControlledPool()
    d = dispatcher(pool, cpu=1, model_limit=1)
    try:
        first = submit(d, 0)
        _, running = pool.job(0)
        second = submit(d, 1)
        assert second.cancel()
        assert not first.cancel()
        assert d.manager.snapshot()["cpu_used"] == 1
        running.set_result(0)
        assert first.result(timeout=3) == 0
        assert len(pool.jobs) == 1
    finally:
        d.confirm_pool_stopped()


def test_submission_failure_rolls_back_lease():
    pool = ControlledPool()
    pool.error = RuntimeError("submission failed")
    d = dispatcher(pool)
    try:
        with pytest.raises(RuntimeError, match="submission failed"):
            submit(d, 0).result(timeout=3)
        assert d.manager.snapshot()["cpu_used"] == 0
    finally:
        d.confirm_pool_stopped()


def test_stopping_dispatcher_notifies_wait_for_cancelled_queued_futures():
    from concurrent.futures import wait

    pool = ControlledPool()
    d = dispatcher(pool, cpu=1, model_limit=1)
    try:
        first = submit(d, 0)
        pool.job(0)
        queued = submit(d, 1)
        d.stop()
        done, pending = wait([queued], timeout=3)
        assert done == {queued} and not pending
        assert queued.cancelled()
        assert d.manager.snapshot()["active_tasks"] == 1
        pool.job(0)[1].set_result(0)
        assert first.result(timeout=3) == 0
    finally:
        d.confirm_pool_stopped()


def test_broken_pool_keeps_leases_until_confirmed_cleanup():
    pool = ControlledPool()
    d = dispatcher(pool)
    promises = [submit(d, i) for i in range(2)]
    pool.job(1)
    pool.job(0)[1].set_exception(BrokenProcessPool("worker disappeared"))
    with pytest.raises(BrokenProcessPool):
        promises[0].result(timeout=3)
    assert d.manager.snapshot()["cpu_used"] == 2
    assert d.stopped.is_set()
    # Future failures do not constitute confirmation that a sibling has stopped.
    pool.job(1)[1].set_exception(BrokenProcessPool("pool failed"))
    with pytest.raises(BrokenProcessPool):
        promises[1].result(timeout=3)
    assert d.manager.snapshot()["cpu_used"] == 2
    d.confirm_pool_stopped()
    assert d.manager.snapshot()["cpu_used"] == 0


def test_broken_submission_resolves_all_queued_parents_without_worker_callback():
    pool = ControlledPool()
    d = dispatcher(pool, cpu=1, model_limit=1)
    try:
        first = submit(d, 0)
        _, running = pool.job(0)
        queued = [submit(d, index) for index in range(1, 4)]
        pool.error = BrokenProcessPool("pool broke before submit")
        running.set_result(0)
        assert first.result(timeout=3) == 0
        for promise in queued:
            with pytest.raises(BrokenProcessPool, match="before submit"):
                promise.result(timeout=3)
        assert d.stopped.is_set()
        assert len(pool.jobs) == 1
        assert d.manager.snapshot()["active_tasks"] == 0
    finally:
        d.confirm_pool_stopped()


def test_fold_refill_uses_completion_order_but_reduction_stays_in_fold_order():
    pool = ControlledPool()
    d = dispatcher(pool)
    with ThreadPoolExecutor(max_workers=1) as controller:
        result = controller.submit(
            lambda: list(d.map_folds("training_fold", "slow", 0, {}, [((), ())] * 4, "unused", 2))
        )
        try:
            pool.job(1)[1].set_result(1)
            # Fold 2 is admitted before slow fold 0 finishes.
            args, third = pool.job(2)
            assert args[3][0] == 2
            third.set_result(2)
            pool.job(3)[1].set_result(3)
            pool.job(0)[1].set_result(0)
            assert result.result(timeout=3) == [0, 1, 2, 3]
        finally:
            d.confirm_pool_stopped()


def test_fast_completion_automatically_expands_backlogged_slow_model_one_three_five():
    pool = ControlledPool()
    manager = ResourceManager(10, 1000, 7, {"slow": 5, "fast": 2})
    d = ElasticDispatcher(
        pool,
        None,
        manager,
        {"slow": ResourceRequest(2, 10), "fast": ResourceRequest(4, 10)},
        {"slow": 1, "fast": 2},
        "run",
    )
    try:
        fast = [submit(d, i, "fast") for i in range(2)]
        pool.job(1)
        slow = [submit(d, i) for i in range(5)]
        pool.job(2)
        assert manager.snapshot()["active_by_model"]["slow"] == 1
        pool.job(0)[1].set_result("fast0")
        assert fast[0].result(timeout=3) == "fast0"
        pool.job(4)
        assert manager.snapshot()["active_by_model"]["slow"] == 3
        pool.job(1)[1].set_result("fast1")
        assert fast[1].result(timeout=3) == "fast1"
        pool.job(6)
        assert manager.snapshot()["active_by_model"]["slow"] == 5
        assert manager.snapshot()["cpu_used"] == 10
        for index in range(2, 7):
            pool.job(index)[1].set_result(index - 2)
        assert [future.result(timeout=3) for future in slow] == list(range(5))
    finally:
        d.confirm_pool_stopped()


def _crash_compute(kind, model, repeat, payload, lease):
    if payload[0] == 0:
        os._exit(17)
    time.sleep(30)


def test_real_worker_exit_fences_pool_until_owner_has_joined_processes():
    from gpse.utils.parallel import graceful_process_pool

    d = None
    workers = []
    try:
        with graceful_process_pool(max_workers=2, mp_context=mp.get_context("spawn")) as pool:
            pool.submit(os.getpid).result(timeout=10)
            workers = list(pool._processes.values())
            manager = ResourceManager(2, 1000, 2, {"slow": 2})
            d = ElasticDispatcher(pool, _crash_compute, manager,
                                  {"slow": ResourceRequest(1, 10)}, {"slow": 1}, "run")
            first, second = submit(d, 0), submit(d, 1)
            with pytest.raises(BrokenProcessPool):
                first.result(timeout=15)
            assert manager.snapshot()["active_tasks"] > 0
    finally:
        # Returning from graceful_process_pool is the required stop evidence.
        assert all(not worker.is_alive() for worker in workers)
        if d is not None:
            d.confirm_pool_stopped()
            assert d.manager.snapshot()["active_tasks"] == 0
