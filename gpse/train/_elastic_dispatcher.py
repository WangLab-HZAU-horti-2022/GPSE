"""Nonblocking compute admission for the existing parent repeat controllers."""

from collections import deque
from concurrent.futures import CancelledError, FIRST_COMPLETED, Future, wait
from concurrent.futures.process import BrokenProcessPool
import threading
import time

from gpse.train._resources import ReadyQueue, TaskKey


class ElasticDispatcher:
    """Queue in the coordinator; submit only tasks holding a complete lease.

    Broken pools retain all outstanding leases until the pool owner confirms
    process cleanup. Future failure alone is not evidence that siblings stopped.
    """

    def __init__(
        self,
        pool,
        execute,
        manager,
        requests,
        weights,
        run_id,
        aging_seconds=5,
        on_event=None,
        completion_driven=True,
        task_store=None,
    ):
        self.pool, self.execute, self.manager = pool, execute, manager
        self.requests, self.run_id = dict(requests), run_id
        self.completion_driven = completion_driven
        self.task_store = task_store
        self._signatures = {}
        # System-only profiles may share the ledger without entering this queue.
        all_weights = {name: weights.get(name, 1) for name in manager.model_limits}
        self.queue = ReadyQueue(manager, all_weights, aging_seconds=aging_seconds)
        self.stopped = threading.Event()
        self._condition = threading.Condition()
        self._waiting = {}
        self._active = {}
        self._attempts = {}
        self._completed = deque()
        self._closed = False
        self._broken = None
        self._on_event = on_event
        self._thread = threading.Thread(target=self._run, name="gpse-admission", daemon=True)
        self._thread.start()

    def submit(self, kind, model, repeat, payload):
        fold = (
            payload[0] if kind in {"training_fold", "optimization_fold", "stacking_fold"} else None
        )
        trial = payload[4] if kind == "optimization_fold" else None
        key = TaskKey(self.run_id, model, repeat, trial, fold, kind)
        with self._condition:
            if self.stopped.is_set():
                raise CancelledError("Training admission stopped")
            if key in self._waiting or any(lease.key == key for lease, _ in self._active.values()):
                raise ValueError("Task is already waiting or running")
            if self.task_store is not None:
                from gpse.train._task_store import STAGES, task_signature

                if kind in STAGES:
                    signature = task_signature(kind, payload)
                    self._signatures[key] = signature
                    restored, result = self.task_store.restore(key, signature)
                    if restored:
                        promise = Future()
                        promise.set_result(result)
                        self._emit("task_restored", key=key, source="verified_checkpoint")
                        return promise
            attempt = self._attempts.get(key, -1) + 1
            self.queue.add(key, attempt, self.requests[model])
            self._attempts[key] = attempt
            promise = Future()
            self._waiting[key] = (promise, (kind, model, repeat, payload))
            self._emit("task_ready", key=key, attempt=attempt)
            promise.add_done_callback(lambda _: self._wake())
            self._condition.notify_all()
            return promise

    def _emit(self, status, *, key=None, attempt=None, lease=None, **fields):
        if self._on_event is None:
            return
        from dataclasses import asdict

        if lease is not None:
            key, attempt = lease.key, lease.attempt
            fields.update(
                lease_id=lease.lease_id,
                worker_generation=lease.generation,
                cpu_units=lease.request.cpu,
                task_memory_reserved_bytes=lease.request.memory_bytes,
            )
        self._on_event(
            "resources",
            status,
            task_key=asdict(key) if key else None,
            model=key.model if key else None,
            attempt=attempt,
            monotonic_s=time.monotonic(),
            **self.manager.snapshot(),
            **fields,
        )

    def _wake(self):
        with self._condition:
            self._condition.notify_all()

    def _completed_callback(self, future):
        with self._condition:
            self._completed.append(future)
            self._condition.notify_all()

    def _run(self):
        try:
            self._coordinate()
        except BaseException as exc:
            # A coordinator failure must unblock parents, while retaining any
            # compute leases until the owning pool has really been stopped.
            with self._condition:
                self._broken = exc
                self.stopped.set()
                for promise, _ in self._waiting.values():
                    if not promise.done():
                        promise.set_exception(exc)
                for _, promise in self._active.values():
                    if not promise.done():
                        promise.set_exception(exc)

    def _coordinate(self):
        with self._condition:
            while not self._closed:
                while self._completed:
                    future = self._completed.popleft()
                    lease, promise = self._active[future]
                    try:
                        result = future.result()
                    except BrokenProcessPool as exc:
                        self._broken = exc
                        self.stopped.set()
                        # Keep lease until confirm_pool_stopped, including siblings.
                        if not promise.done():
                            promise.set_exception(exc)
                        continue
                    except BaseException as exc:
                        result, error = None, exc
                    else:
                        error = None
                    if self._broken is not None:
                        if not promise.done():
                            promise.set_exception(self._broken)
                        continue
                    if not self.manager.accepts_result(lease):
                        error = RuntimeError("Stale compute result rejected")
                    if error is None and lease.key in self._signatures:
                        try:
                            result = self.task_store.commit(
                                lease, result, self._signatures[lease.key], self.manager
                            )
                            self._emit("task_committed", lease=lease)
                        except Exception as exc:
                            error = exc
                    self.manager.release(lease, computation_stopped=True)
                    self._emit(
                        "lease_released",
                        lease=lease,
                        reason="completed" if error is None else "failed",
                    )
                    del self._active[future]
                    if not promise.done():
                        if error is not None:
                            promise.set_exception(error)
                        else:
                            promise.set_result(result)

                for key, (promise, _) in list(self._waiting.items()):
                    if promise.cancelled() or self.stopped.is_set():
                        self.queue.cancel(key)
                        del self._waiting[key]
                        if not promise.done():
                            if self._broken is not None:
                                promise.set_exception(self._broken)
                            else:
                                promise.cancel()
                        if promise.cancelled():
                            # wait()/as_completed() require executor-style
                            # acknowledgement, not just Future.cancel(). These
                            # coordinator futures never reach a real executor.
                            promise.set_running_or_notify_cancel()

                while not self.stopped.is_set():
                    admitted = self.queue.pop_available()
                    if admitted is None:
                        break
                    task, lease = admitted
                    self._emit(
                        "lease_granted",
                        lease=lease,
                        ready_at=task.ready_at,
                        queue_seconds=time.monotonic() - task.ready_at,
                    )
                    promise, args = self._waiting.pop(task.key)
                    if not promise.set_running_or_notify_cancel():
                        self.manager.release(lease, computation_stopped=True)
                        self._emit("lease_released", lease=lease, reason="cancelled_before_submit")
                        continue
                    try:
                        future = self.pool.submit(self.execute, *args, lease)
                    except BaseException as exc:
                        # submit did not return a future: no task was accepted.
                        self.manager.release(lease, computation_stopped=True)
                        self._emit("lease_released", lease=lease, reason="submission_failed")
                        promise.set_exception(exc)
                        if isinstance(exc, BrokenProcessPool):
                            self._broken = exc
                            self.stopped.set()
                        continue
                    self._active[future] = lease, promise
                    future.add_done_callback(self._completed_callback)
                # A broken submit can stop admission inside the loop above.
                # Resolve queued parents before sleeping: no worker callback is
                # guaranteed to wake us when submit itself failed.
                if self._completed or (self.stopped.is_set() and self._waiting):
                    continue
                if not self.stopped.is_set():
                    reasons = set()
                    for key in self._waiting:
                        reasons.update(self.manager.blocking_reasons(key, self.requests[key.model]))
                    if not self._waiting:
                        reasons.add(
                            "only_running_task" if len(self._active) == 1 else "no_ready_work"
                        )
                    elif not reasons:
                        reasons.add("fairness_reservation")
                    self._emit("waiting", ready_tasks=len(self._waiting), reasons=sorted(reasons))
                self._condition.wait()

    def map_folds(self, kind, model, repeat, params, folds, extra, limit):
        """Refill on completion, but expose results in original fold order."""
        if not self.completion_driven:
            from gpse.train._fold_scheduler import _Dispatcher

            yield from _Dispatcher.map_folds(self, kind, model, repeat, params, folds, extra, limit)
            return
        remaining = iter(enumerate(folds))
        pending, buffered = {}, {}
        next_result = 0

        def submit_next():
            item = next(remaining, None)
            if item is not None:
                fold, (train, val) = item
                future = self.submit(kind, model, repeat, (fold, train, val, params, extra))
                pending[future] = fold

        try:
            for _ in range(min(limit, len(folds))):
                submit_next()
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                # Resolve all completed errors before submitting further work.
                for future in done:
                    fold = pending.pop(future)
                    buffered[fold] = future.result()
                for _ in done:
                    submit_next()
                while next_result in buffered:
                    yield buffered.pop(next_result)
                    next_result += 1
        finally:
            for future in pending:
                future.cancel()
            if not self.stopped.is_set():
                for future in pending:
                    try:
                        future.result()
                    except BaseException:
                        pass

    def stop(self):
        self.stopped.set()
        self._wake()

    def confirm_pool_stopped(self):
        """Call only after the pool context has joined/terminated all workers."""
        with self._condition:
            self.stopped.set()
            for future, (lease, promise) in list(self._active.items()):
                self.manager.release(lease, computation_stopped=True)
                self._emit("lease_released", lease=lease, reason="pool_stopped")
                if not promise.done():
                    promise.set_exception(CancelledError("Compute pool stopped"))
            self._active.clear()
            for key, (promise, _) in self._waiting.items():
                self.queue.cancel(key)
                promise.cancel()
                if promise.cancelled():
                    promise.set_running_or_notify_cancel()
            self._waiting.clear()
            self._closed = True
            self._condition.notify_all()
        self._thread.join()
