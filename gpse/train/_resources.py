"""Coordinator-owned admission ledger; tokens are not OS resource isolation.

Only compute tasks acquire leases. A timeout or cancellation request is not
completion: callers must establish that computation stopped before release.
"""

from dataclasses import dataclass
import threading
import time
import uuid


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class TaskKey:
    run: str
    model: str
    repeat: int | None
    trial: int | None
    fold: int | None
    stage: str


@dataclass(frozen=True)
class ResourceRequest:
    cpu: int
    memory_bytes: int

    def __post_init__(self):
        _positive(self.cpu, "cpu")
        _positive(self.memory_bytes, "memory_bytes")


@dataclass(frozen=True)
class FitLease:
    lease_id: str
    key: TaskKey
    attempt: int
    generation: int
    request: ResourceRequest


class ResourceManager:
    """Atomic, nonblocking CPU/memory/slot admission with attempt fencing.

    Memory reservations must include per-worker resident data; fixed memory
    covers coordinator/shared allocations. Estimates do not guarantee no OOM.
    Generation is the compute pool generation, not a claimed worker PID.
    """

    def __init__(self, cpu, memory_bytes, worker_slots, model_limits, fixed_memory_bytes=0):
        for name, value in (
            ("cpu", cpu),
            ("memory_bytes", memory_bytes),
            ("worker_slots", worker_slots),
        ):
            _positive(value, name)
        if type(fixed_memory_bytes) is not int or not 0 <= fixed_memory_bytes < memory_bytes:
            raise ValueError("fixed_memory_bytes must leave memory for computation")
        for model, limit in model_limits.items():
            _positive(limit, f"max_active_fits for {model}")
        self.cpu = cpu
        self.memory_bytes = memory_bytes
        self.worker_slots = worker_slots
        self.fixed_memory_bytes = fixed_memory_bytes
        self.model_limits = dict(model_limits)
        self._lock = threading.RLock()
        self._active = {}
        self._latest = {}
        self._generation = 0

    def validate(self, key, request):
        if key.model not in self.model_limits:
            raise ValueError(f"No resource profile for {key.model}")
        if request.cpu > self.cpu:
            raise ValueError("Task CPU request exceeds total budget")
        if request.memory_bytes + self.fixed_memory_bytes > self.memory_bytes:
            raise ValueError("Task memory request exceeds available budget")

    def try_acquire(self, key, attempt, request):
        """Return a lease or None for temporary contention; never wait for tokens."""
        self.validate(key, request)
        if type(attempt) is not int or attempt < 0:
            raise ValueError("attempt must be a nonnegative integer")
        with self._lock:
            for lease in self._active.values():
                if lease.key == key:
                    if lease.attempt == attempt and lease.request == request:
                        return lease
                    raise ValueError("Previous task attempt is still active")
            if attempt <= self._latest.get(key, -1):
                raise ValueError("Stale or already completed attempt")
            leases = tuple(self._active.values())
            if (
                len(leases) >= self.worker_slots
                or sum(l.request.cpu for l in leases) + request.cpu > self.cpu
                or sum(l.request.memory_bytes for l in leases)
                + request.memory_bytes
                + self.fixed_memory_bytes
                > self.memory_bytes
                or sum(l.key.model == key.model for l in leases) >= self.model_limits[key.model]
            ):
                return None
            lease = FitLease(uuid.uuid4().hex, key, attempt, self._generation, request)
            self._active[lease.lease_id] = lease
            self._latest[key] = attempt
            return lease

    def accepts_result(self, lease):
        with self._lock:
            return (
                self._active.get(lease.lease_id) == lease and lease.generation == self._generation
            )

    def blocking_reasons(self, key, request):
        with self._lock:
            snapshot = self.snapshot()
            reasons = []
            if snapshot["cpu_used"] + request.cpu > self.cpu:
                reasons.append("cpu_budget")
            if snapshot["memory_reserved_bytes"] + request.memory_bytes > self.memory_bytes:
                reasons.append("memory_budget")
            if snapshot["active_tasks"] >= self.worker_slots:
                reasons.append("worker_slots")
            if snapshot["active_by_model"].get(key.model, 0) >= self.model_limits[key.model]:
                reasons.append("model_max_active_fits")
            return reasons

    def validate_new_attempt(self, key, attempt):
        with self._lock:
            if any(lease.key == key for lease in self._active.values()):
                raise ValueError("Previous task attempt is still active")
            if attempt <= self._latest.get(key, -1):
                raise ValueError("Stale or already completed attempt")

    def release(self, lease, *, computation_stopped):
        """Release once after completion, failed submission, or verified process exit."""
        if computation_stopped is not True:
            raise ValueError("Cannot reclaim resources without confirmed computation stop")
        with self._lock:
            if self._active.get(lease.lease_id) != lease:
                return False
            del self._active[lease.lease_id]
            return True

    def next_generation(self):
        """Advance only after all old computation has stopped and leases released."""
        with self._lock:
            if self._active:
                raise ValueError("Cannot replace pool while leases remain active")
            self._generation += 1
            return self._generation

    def snapshot(self):
        with self._lock:
            leases = tuple(self._active.values())
            return {
                "cpu_used": sum(l.request.cpu for l in leases),
                "memory_reserved_bytes": self.fixed_memory_bytes
                + sum(l.request.memory_bytes for l in leases),
                "active_tasks": len(leases),
                "active_by_model": {
                    model: sum(l.key.model == model for l in leases) for model in self.model_limits
                },
                "generation": self._generation,
            }


@dataclass(frozen=True)
class ReadyTask:
    key: TaskKey
    attempt: int
    request: ResourceRequest
    ready_at: float
    sequence: int


class ReadyQueue:
    """Weighted model admission with bounded backfill and age reservation.

    Used by one coordinator. Call pop_available after every completion and
    arrival. Once a task ages, stop backfilling until it can start; finite
    running tasks therefore cannot indefinitely starve a large request.
    """

    def __init__(self, manager, weights, aging_seconds=5.0, clock=time.monotonic):
        if set(weights) != set(manager.model_limits):
            raise ValueError("Weights must cover every resource profile")
        for model, weight in weights.items():
            _positive(weight, f"weight for {model}")
        if not 0 < aging_seconds < float("inf"):
            raise ValueError("aging_seconds must be finite and positive")
        self.manager = manager
        self.weights = dict(weights)
        self.aging_seconds = aging_seconds
        self.clock = clock
        self._ready = {}
        self._sequence = 0
        self._service = {model: 0.0 for model in weights}

    def add(self, key, attempt, request):
        self.manager.validate(key, request)
        if type(attempt) is not int or attempt < 0:
            raise ValueError("attempt must be a nonnegative integer")
        if key in self._ready:
            task = self._ready[key]
            if (task.attempt, task.request) != (attempt, request):
                raise ValueError("Task already queued with different attempt or resources")
            return task
        self.manager.validate_new_attempt(key, attempt)
        task = ReadyTask(key, attempt, request, self.clock(), self._sequence)
        self._sequence += 1
        self._ready[key] = task
        return task

    def cancel(self, key):
        """Cancel only queued work; active leases require confirmed stop."""
        return self._ready.pop(key, None)

    def pop_available(self):
        if not self._ready:
            return None
        now = self.clock()
        active = self.manager.snapshot()["active_by_model"]
        eligible = [
            task
            for task in self._ready.values()
            if active[task.key.model] < self.manager.model_limits[task.key.model]
        ]
        oldest = min(eligible, key=lambda task: task.sequence) if eligible else None
        reserved = oldest is not None and now - oldest.ready_at >= self.aging_seconds
        candidates = (
            [oldest]
            if reserved
            else sorted(
                self._ready.values(),
                key=lambda task: (
                    self._service[task.key.model] / self.weights[task.key.model],
                    -self.weights[task.key.model],
                    task.sequence,
                ),
            )
        )
        for task in candidates:
            lease = self.manager.try_acquire(task.key, task.attempt, task.request)
            if lease is not None:
                del self._ready[task.key]
                self._service[task.key.model] += task.request.cpu
                return task, lease
        return None

    def __len__(self):
        return len(self._ready)
