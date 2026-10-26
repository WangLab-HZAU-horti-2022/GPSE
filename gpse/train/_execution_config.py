"""Resource plans for default lease-based execution (all sizes in bytes)."""

from dataclasses import asdict, dataclass
from contextlib import contextmanager
import json
from pathlib import Path

from gpse.train._resources import ResourceManager, ResourceRequest, TaskKey, _positive


@dataclass(frozen=True)
class ResourceProfile:
    preferred_cpu_per_fit: int
    memory_bytes: int
    max_active_fits: int
    weight: int = 1

    def __post_init__(self):
        for name, value in asdict(self).items():
            _positive(value, name)


@dataclass(frozen=True)
class ExecutionConfig:
    cpu_budget: int
    worker_slots: int
    memory_budget_bytes: int
    fixed_memory_bytes: int
    worker_memory_bytes: int
    resource_profiles: dict[str, ResourceProfile]
    scheduler: str = "elastic_v1"
    aging_seconds: float = 5.0

    def __post_init__(self):
        if self.scheduler not in {"elastic_v1", "thread_plan"}:
            raise ValueError(
                "execution scheduler must be elastic_v1 or thread_plan; omit config for legacy"
            )
        for name in ("cpu_budget", "worker_slots", "memory_budget_bytes", "worker_memory_bytes"):
            _positive(getattr(self, name), name)
        if type(self.fixed_memory_bytes) is not int or self.fixed_memory_bytes < 0:
            raise ValueError("fixed_memory_bytes must be a nonnegative integer")
        if not isinstance(self.aging_seconds, (int, float)) or not 0 < self.aging_seconds < float(
            "inf"
        ):
            raise ValueError("aging_seconds must be finite and positive")
        if not self.resource_profiles:
            raise ValueError("At least one resource profile is required")
        if any(
            not isinstance(name, str) or name.startswith("__") for name in self.resource_profiles
        ):
            raise ValueError("Resource profile IDs must be strings; '__' IDs are reserved")
        if self.worker_slots > self.cpu_budget:
            raise ValueError("worker_slots must not exceed cpu_budget")
        manager = self.make_manager(list(self.resource_profiles))
        for model, profile in self.resource_profiles.items():
            manager.validate(
                TaskKey("validation", model, None, None, None, "fit"),
                ResourceRequest(profile.preferred_cpu_per_fit, profile.memory_bytes),
            )

    @classmethod
    def load(cls, path):
        import yaml

        raw = yaml.safe_load(Path(path).read_text())
        if not isinstance(raw, dict):
            raise ValueError("Execution config must be a mapping")
        raw = dict(raw)
        profiles = raw.pop("resource_profiles", None)
        if not isinstance(profiles, dict):
            raise ValueError("resource_profiles must be a mapping")
        try:
            return cls(
                resource_profiles={
                    name: ResourceProfile(**profile) for name, profile in profiles.items()
                },
                **raw,
            )
        except TypeError as exc:
            raise ValueError(f"Invalid execution config fields: {exc}") from exc

    @classmethod
    def automatic(cls, predictor, models, X, y):
        """Generate a bounded plan using loaded data and observable memory.

        Reservations are estimates, not OOM guarantees. Limit resident workers
        separately from the CPU budget; native model workspaces still vary.
        """
        from gpse.train._thread_plan import plan_for

        available = available_memory_bytes()
        if available is None:
            raise ValueError("Cannot detect available memory; provide --execution_config")
        budget = int(available * 0.75)
        data_bytes = int(X.memory_usage(index=True, deep=True).sum() + y.memory_usage(deep=True))
        fixed = 2 * data_bytes + 64 * 2**20
        resident = 2 * data_bytes + 128 * 2**20
        # Cover several transformed copies and dense kernel/BLAS workspaces.
        workspace = max(128 * 2**20, 4 * data_bytes + 3 * len(X)**2 * 8)
        slots = min(32, predictor.cpu_budget,
                    max(0, (budget - fixed) // (resident + workspace)))
        if slots < 1:
            raise ValueError("Insufficient memory for one training worker; provide a measured execution_config")
        optimizer = (predictor.model_optimizer if predictor.task_type == "regression"
                     else predictor.classification_optimizer)
        profiles = {}
        for name in models:
            entry = optimizer._registry.model_configs[name]
            plan = plan_for(name, entry.import_path, predictor.n_threads)
            cpu = predictor.n_threads if plan.adapter != "conservative_unverified" else 1
            heavy = plan.adapter in {"native_openmp", "catboost_threads"}
            if heavy and predictor.automatic_fit_threads:
                folds = min(predictor.n_splits, predictor.fold_workers or predictor.n_splits)
                cpu = max(1, min(16, predictor.cpu_budget // folds))
            profiles[name] = ResourceProfile(cpu, workspace, slots, 1 if heavy else 2)
        return cls(predictor.cpu_budget, slots, budget, fixed, resident, profiles,
                   scheduler=predictor.scheduler)

    def make_manager(self, models):
        missing = set(models) - self.resource_profiles.keys()
        if missing:
            raise ValueError(f"Missing resource profiles: {sorted(missing)}")
        limits = {model: self.resource_profiles[model].max_active_fits for model in models}
        limits.update({"__run__": 1, "__meta__": 1})
        return ResourceManager(
            self.cpu_budget,
            self.memory_budget_bytes,
            self.worker_slots,
            limits,
            self.fixed_memory_bytes + self.worker_slots * self.worker_memory_bytes,
        )

    def write_plan(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")

    def validate_environment(self, X, y):
        """Check declared resident-memory reservations before spawning workers.

        This floor accounts for two copies of the dataset and conservative
        interpreter overhead; transient algorithm memory still needs profiling.
        """
        data_bytes = int(X.memory_usage(index=True, deep=True).sum() + y.memory_usage(deep=True))
        if self.worker_memory_bytes < 2 * data_bytes + 128 * 2**20:
            raise ValueError("worker_memory_bytes is below dataset copies plus 128 MiB overhead")
        if self.fixed_memory_bytes < 2 * data_bytes + 64 * 2**20:
            raise ValueError("fixed_memory_bytes is below coordinator dataset copies plus 64 MiB")
        available = available_memory_bytes()
        if available is not None and self.memory_budget_bytes > available:
            raise ValueError(
                f"Requested memory budget {self.memory_budget_bytes} exceeds available {available} bytes"
            )


def available_memory_bytes():
    """Linux available memory and cgroup-v2 ancestor headroom, when observable."""
    limits = []
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                limits.append(int(line.split()[1]) * 1024)
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            _, controllers, relative = line.split(":", 2)
            if controllers:
                continue
            base = Path("/sys/fs/cgroup")
            current = base / relative.lstrip("/")
            while current.is_relative_to(base):
                try:
                    maximum = (current / "memory.max").read_text().strip()
                    if maximum != "max":
                        used = int((current / "memory.current").read_text())
                        limits.append(max(0, int(maximum) - used))
                except (OSError, ValueError):
                    pass
                if current == base:
                    break
                current = current.parent
    except (OSError, ValueError):
        pass
    return min(limits) if limits else None


@contextmanager
def coordinator_stage(manager, config, run_id, stage, model="__run__"):
    """Single-threaded compute between worker phases uses the same run ledger."""
    if manager is None:
        yield
        return
    from gpse.train._thread_plan import ThreadPlan, execution_plan
    from gpse.utils.training_monitor import publish

    key = TaskKey(str(run_id), model, None, None, None, stage)
    request = ResourceRequest(1, max(p.memory_bytes for p in config.resource_profiles.values()))
    lease = manager.try_acquire(key, 0, request)
    if lease is None:
        raise RuntimeError(f"Coordinator stage {stage} started before compute phase drained")
    try:
        publish(
            "resources",
            "lease_granted",
            task_key=asdict(key),
            lease_id=lease.lease_id,
            cpu_units=1,
            **manager.snapshot(),
        )
        with execution_plan(ThreadPlan(model, "numpy.linalg", 1, "coordinator", 1, 1, 1, 1)):
            yield lease
    finally:
        manager.release(lease, computation_stopped=True)
        publish(
            "resources",
            "lease_released",
            task_key=asdict(key),
            lease_id=lease.lease_id,
            cpu_units=1,
            reason="coordinator_stopped",
            **manager.snapshot(),
        )
