"""Resource accounting tests use logical tokens, not claimed hardware load."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from gpse.train._resources import ReadyQueue, ResourceManager, ResourceRequest, TaskKey


def key(fold=0, model="slow", repeat=0):
    return TaskKey("run", model, repeat, 0, fold, "hpo")


@pytest.mark.parametrize("budget", [1, 4, 8, 16, 80])
def test_atomic_admission_and_exactly_once_release(budget):
    manager = ResourceManager(budget, 1000, budget, {"slow": budget})
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(
                lambda i: manager.try_acquire(key(i), 0, ResourceRequest(1, 1)),
                range(budget * 2),
            )
        )
    leases = [lease for lease in results if lease is not None]
    assert len(leases) == budget
    assert manager.snapshot()["cpu_used"] == budget
    for lease in leases:
        assert manager.release(lease, computation_stopped=True)
        assert not manager.release(lease, computation_stopped=True)
    assert manager.snapshot()["active_tasks"] == 0
    with pytest.raises(ValueError, match="CPU"):
        manager.try_acquire(key(999), 0, ResourceRequest(budget + 1, 1))


def test_memory_slots_and_cross_repeat_model_limit():
    manager = ResourceManager(80, 100, 3, {"slow": 2, "fast": 4}, fixed_memory_bytes=10)
    a = manager.try_acquire(key(), 0, ResourceRequest(1, 40))
    b = manager.try_acquire(key(repeat=1), 0, ResourceRequest(1, 40))
    assert a and b
    assert manager.try_acquire(key(repeat=2), 0, ResourceRequest(1, 1)) is None
    assert manager.try_acquire(key(model="fast"), 0, ResourceRequest(1, 11)) is None
    assert manager.try_acquire(key(model="fast"), 0, ResourceRequest(1, 10))
    assert manager.try_acquire(key(1, "fast"), 0, ResourceRequest(1, 1)) is None
    with pytest.raises(ValueError, match="memory"):
        manager.try_acquire(key(2), 0, ResourceRequest(1, 91))


def test_attempt_fencing_does_not_reclaim_on_timeout():
    manager = ResourceManager(4, 100, 1, {"slow": 1})
    request = ResourceRequest(4, 10)
    old = manager.try_acquire(key(), 0, request)
    assert manager.try_acquire(key(), 0, request) == old
    with pytest.raises(ValueError, match="still active"):
        manager.try_acquire(key(), 1, request)
    with pytest.raises(ValueError, match="confirmed"):
        manager.release(old, computation_stopped=False)
    with pytest.raises(ValueError, match="leases"):
        manager.next_generation()
    assert manager.accepts_result(old)
    manager.release(old, computation_stopped=True)
    manager.next_generation()
    with pytest.raises(ValueError, match="Stale"):
        manager.try_acquire(key(), 0, request)
    new = manager.try_acquire(key(), 1, request)
    assert new.generation == 1
    assert not manager.accepts_result(old)
    assert not manager.release(old, computation_stopped=True)
    assert manager.accepts_result(new)
    assert manager.snapshot()["cpu_used"] == 4


@pytest.mark.parametrize("scale", [1, 8])
def test_released_budget_allows_one_three_five_fits(scale):
    manager = ResourceManager(10 * scale, 100, 7, {"slow": 5, "fast": 2})
    fast = [
        manager.try_acquire(key(i, "fast"), 0, ResourceRequest(4 * scale, 10)) for i in range(2)
    ]
    slow = [manager.try_acquire(key(), 0, ResourceRequest(2 * scale, 10))]
    assert all(fast) and all(slow)
    assert manager.try_acquire(key(1), 0, ResourceRequest(2 * scale, 10)) is None
    for index, lease in enumerate(fast):
        manager.release(lease, computation_stopped=True)
        for _ in range(2):
            slow.append(manager.try_acquire(key(len(slow)), 0, ResourceRequest(2 * scale, 10)))
        assert all(slow)
        assert manager.snapshot()["active_by_model"]["slow"] == 3 + 2 * index
        assert manager.snapshot()["cpu_used"] == 10 * scale


def test_fast_weight_and_slow_service_share():
    manager = ResourceManager(1, 100, 1, {"slow": 1, "fast": 1})
    queue = ReadyQueue(manager, {"slow": 1, "fast": 2}, clock=lambda: 0)
    for model in ["slow", "fast"]:
        for fold in range(6):
            queue.add(key(fold, model), 0, ResourceRequest(1, 1))
    order = []
    for _ in range(6):
        task, lease = queue.pop_available()
        order.append(task.key.model)
        manager.release(lease, computation_stopped=True)
    assert order[0] == "fast"
    assert order.count("slow") == 2
    assert order.count("fast") == 4


def test_large_aged_task_reserves_budget_instead_of_endless_backfill():
    now = [0.0]
    manager = ResourceManager(4, 100, 4, {"slow": 1, "fast": 4})
    running = manager.try_acquire(key(99, "fast"), 0, ResourceRequest(2, 1))
    queue = ReadyQueue(manager, {"slow": 1, "fast": 2}, aging_seconds=5, clock=lambda: now[0])
    queue.add(key(), 0, ResourceRequest(4, 1))
    queue.add(key(1, "fast"), 0, ResourceRequest(1, 1))
    task, backfill = queue.pop_available()
    assert task.key.model == "fast"
    queue.add(key(2, "fast"), 0, ResourceRequest(1, 1))
    now[0] = 6
    assert queue.pop_available() is None
    manager.release(backfill, computation_stopped=True)
    assert queue.pop_available() is None
    manager.release(running, computation_stopped=True)
    task, lease = queue.pop_available()
    assert task.key.model == "slow"
    assert lease.request.cpu == 4
    assert queue.cancel(key(2, "fast"))
    assert len(queue) == 0
    assert manager.snapshot()["active_tasks"] == 1


def test_queue_cannot_dispatch_active_or_completed_attempt_twice():
    manager = ResourceManager(4, 100, 4, {"slow": 4})
    queue = ReadyQueue(manager, {"slow": 1})
    request = ResourceRequest(1, 1)
    task = queue.add(key(), 0, request)
    assert queue.add(key(), 0, request) == task
    assert len(queue) == 1
    _, lease = queue.pop_available()
    with pytest.raises(ValueError, match="active"):
        queue.add(key(), 0, request)
    manager.release(lease, computation_stopped=True)
    with pytest.raises(ValueError, match="Stale"):
        queue.add(key(), 0, request)
    queue.add(key(), 1, request)
    assert queue.pop_available()[1].attempt == 1


def test_aged_task_at_model_cap_does_not_freeze_unrelated_models():
    now = [0.0]
    manager = ResourceManager(4, 100, 4, {"slow": 1, "fast": 4})
    queue = ReadyQueue(manager, {"slow": 1, "fast": 2}, clock=lambda: now[0])
    queue.add(key(), 0, ResourceRequest(1, 1))
    assert queue.pop_available()
    queue.add(key(1), 0, ResourceRequest(1, 1))
    now[0] = 20
    queue.add(key(0, "fast"), 0, ResourceRequest(1, 1))
    assert queue.pop_available()[0].key.model == "fast"
