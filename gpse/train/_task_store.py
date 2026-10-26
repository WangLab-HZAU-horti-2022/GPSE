"""Coordinator commits for isolated fit outputs and verified cross-run recovery.

Workers write only their lease's private directory. A current lease and a
complete hash inventory are required before the coordinator publishes a pointer.
Recovery reads a stopped source run into a fresh destination; it never edits
the source. Joblib payloads are local GPSE artifacts, not untrusted input.
"""

from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

from gpse.output.manifest import atomic_write_json

STAGES = {
    "training_fold",
    "optimization_fold",
    "stacking_fold",
    "stacking_refit",
    "finalize",
    "stacking_meta",
}
_WORKER_LOCK = None
_ARTIFACT_DIRECTORY = ContextVar("gpse_task_artifact_directory", default=None)


@contextmanager
def task_artifacts(lease):
    token = _ARTIFACT_DIRECTORY.set(stage_directory(lease))
    try:
        yield
    finally:
        _ARTIFACT_DIRECTORY.reset(token)


def native_log_directory():
    directory = _ARTIFACT_DIRECTORY.get()
    if directory is None:
        return None
    directory = directory / "native_logs"
    directory.mkdir(parents=True, exist_ok=True)
    return tempfile.mkdtemp(prefix="fit_", dir=directory)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def task_identity(key):
    fields = asdict(key)
    fields.pop("run")
    return fields


def task_digest(key):
    return hashlib.sha256(json.dumps(task_identity(key), sort_keys=True).encode()).hexdigest()


def task_signature(kind, payload):
    from joblib import hash as joblib_hash

    # A training payload's last field is its destination directory. It has no
    # bearing on the fit; the source and destination necessarily differ.
    values = payload[:4] if kind == "training_fold" else payload
    if kind in {"stacking_fold", "stacking_refit"}:
        from sklearn.base import clone

        values = list(payload)
        pipeline = clone(payload[3])
        estimator = pipeline.named_steps["model"]
        if "train_dir" in estimator.get_params(deep=False):
            estimator.set_params(train_dir="__task_native_logs__")
        values[3] = pipeline
    if kind == "finalize":

        def scientific_values(value):
            if isinstance(value, dict):
                return {
                    name: scientific_values(item)
                    for name, item in value.items()
                    if name
                    not in {
                        "training_time",
                        "avg_training_time",
                        "all_predictions_file",
                        "cv_folds_file",
                    }
                }
            if isinstance(value, list):
                return [scientific_values(item) for item in value]
            return value

        values = scientific_values(sorted(payload, key=lambda r: r["repeat_idx"]))
    return joblib_hash((kind, values), hash_name="sha1")


def stage_directory(lease):
    return Path(lease.key.run) / ".task_store" / "pending" / task_digest(lease.key) / lease.lease_id


def relocate(value, source, destination):
    if isinstance(value, str):
        return (
            str(destination) + value[len(str(source)) :]
            if value.startswith(str(source) + "/")
            else value
        )
    if isinstance(value, dict):
        return {key: relocate(item, source, destination) for key, item in value.items()}
    if isinstance(value, list):
        return [relocate(item, source, destination) for item in value]
    if isinstance(value, tuple):
        return tuple(relocate(item, source, destination) for item in value)
    return value


@dataclass(frozen=True)
class PreparedResult:
    lease_id: str
    task_digest: str
    attempt: int
    generation: int
    manifest_sha256: str


def prepare_result(lease, result):
    """Called by a worker after fit/score; does not publish any final output."""
    import joblib

    stage = stage_directory(lease)
    stage.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, stage / "result.joblib")
    files = {}
    for path in sorted(stage.rglob("*")):
        if path.is_symlink():
            raise ValueError("Task artifacts must not contain symlinks")
        if path.is_file():
            files[str(path.relative_to(stage))] = file_hash(path)
    manifest = {
        "schema": 1,
        "task": task_identity(lease.key),
        "lease_id": lease.lease_id,
        "attempt": lease.attempt,
        "generation": lease.generation,
        "files": files,
        "origin_run": lease.key.run,
    }
    atomic_write_json(stage / "prepared.json", manifest)
    return PreparedResult(
        lease.lease_id,
        task_digest(lease.key),
        lease.attempt,
        lease.generation,
        file_hash(stage / "prepared.json"),
    )


def _verify_files(directory, manifest):
    files = manifest.get("files", {})
    if "result.joblib" not in files:
        raise ValueError("Task manifest lacks its result")
    for relative, expected in files.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Invalid task artifact path")
        target = directory / path
        if any(p.is_symlink() for p in [target, *target.parents] if p != directory.parent):
            raise ValueError("Task artifact symlink rejected")
        if not target.is_file() or file_hash(target) != expected:
            raise ValueError(f"Task artifact hash mismatch: {relative}")


class TaskStore:
    def __init__(self, run_dir, fingerprint, source=None):
        self.run_dir = Path(run_dir).resolve()
        self.root = self.run_dir / ".task_store"
        self.fingerprint = fingerprint
        self.source = Path(source).resolve() / ".task_store" if source else None
        self.root.mkdir(parents=True, exist_ok=True)
        metadata = self.root / "run.json"
        if metadata.exists():
            if json.loads(metadata.read_text())["fingerprint"] != fingerprint:
                raise ValueError("Checkpoint run fingerprint mismatch")
        else:
            from gpse.utils.process_control import process_identity

            atomic_write_json(
                metadata,
                {
                    "schema": 1,
                    "fingerprint": fingerprint,
                    "owner_pid": os.getpid(),
                    "owner_identity": process_identity(os.getpid()),
                },
            )
        if self.source:
            source_meta = json.loads((self.source / "run.json").read_text())
            if source_meta.get("schema") != 1 or source_meta.get("fingerprint") != fingerprint:
                raise ValueError(
                    "Recovery source input/code/config/environment fingerprint mismatch"
                )
            # Fail before submitting fits if any published source task is damaged.
            for pointer in self.source.glob("tasks/*/current.json"):
                self._read(pointer, signature=None)

    def _read(self, pointer, signature):
        record = json.loads(pointer.read_text())
        identifier = record.get("commit", "")
        if not identifier or not identifier.isalnum():
            raise ValueError("Invalid checkpoint commit identifier")
        directory = pointer.parent / identifier
        manifest_path = directory / "committed.json"
        if file_hash(manifest_path) != record.get("manifest_sha256"):
            raise ValueError("Checkpoint manifest hash mismatch")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != 1 or manifest.get("status") != "completed":
            raise ValueError("Checkpoint is not a committed completed task")
        identity_digest = hashlib.sha256(
            json.dumps(manifest.get("task"), sort_keys=True).encode()
        ).hexdigest()
        if identity_digest != pointer.parent.name:
            raise ValueError("Checkpoint task identity mismatch")
        if manifest.get("fingerprint") != self.fingerprint:
            raise ValueError("Checkpoint task fingerprint mismatch")
        if signature is not None and manifest.get("signature") != signature:
            raise ValueError("Checkpoint task parameters/splits mismatch")
        _verify_files(directory, manifest)
        return directory, manifest

    def _export(self, key, directory, manifest):
        if key.stage not in {"training_fold", "finalize", "stacking_meta"}:
            return
        if Path(key.model).name != key.model or key.model in {".", ".."}:
            raise ValueError("Invalid model artifact destination")
        model_dir = self.run_dir / "models" / key.model
        if key.stage == "stacking_meta":
            exports = {
                relative: self.run_dir / "ensembles/gpse" / Path(relative).relative_to("ensemble")
                for relative in manifest["files"]
                if relative.startswith("ensemble/")
            }
        elif key.stage == "training_fold":
            filename = f"fold_{key.fold + 1}_model.pkl"
            exports = {filename: model_dir / f"repeat_{key.repeat + 1}" / filename}
        else:
            exports = {
                relative: model_dir / Path(relative).relative_to("model")
                for relative in manifest["files"]
                if relative.startswith("model/")
            }
        for relative, destination in exports.items():
            if relative not in manifest["files"]:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", dir=destination.parent)
            os.close(fd)
            try:
                if relative.endswith(".json"):
                    contents = json.loads((directory / relative).read_text())
                    contents = relocate(contents, manifest["origin_run"], self.run_dir)
                    Path(temporary).write_text(json.dumps(contents, indent=2))
                else:
                    shutil.copyfile(directory / relative, temporary)
                os.replace(temporary, destination)
            finally:
                Path(temporary).unlink(missing_ok=True)

    def _publish(self, key, directory, manifest):
        self._export(key, directory, manifest)
        atomic_write_json(
            directory.parent / "current.json",
            {
                "commit": directory.name,
                "manifest_sha256": file_hash(directory / "committed.json"),
            },
        )

    def commit(self, lease, prepared, signature, manager):
        import joblib

        if Path(lease.key.run).resolve() != self.run_dir:
            raise ValueError("Compute lease belongs to a different checkpoint run")
        if not manager.accepts_result(lease):
            raise ValueError("Stale task attempt cannot publish artifacts")
        if not isinstance(prepared, PreparedResult) or (
            prepared.lease_id,
            prepared.task_digest,
            prepared.attempt,
            prepared.generation,
        ) != (lease.lease_id, task_digest(lease.key), lease.attempt, lease.generation):
            raise ValueError("Prepared result does not match the active lease")
        stage = stage_directory(lease)
        if file_hash(stage / "prepared.json") != prepared.manifest_sha256:
            raise ValueError("Prepared manifest hash mismatch")
        manifest = json.loads((stage / "prepared.json").read_text())
        if manifest["task"] != task_identity(lease.key) or (
            manifest.get("lease_id"),
            manifest.get("attempt"),
            manifest.get("generation"),
        ) != (lease.lease_id, lease.attempt, lease.generation):
            raise ValueError("Prepared task identity mismatch")
        _verify_files(stage, manifest)
        result = joblib.load(stage / "result.joblib")
        manifest.update(fingerprint=self.fingerprint, signature=signature, status="completed")
        atomic_write_json(stage / "committed.json", manifest)
        destination = self.root / "tasks" / task_digest(lease.key) / lease.lease_id
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage, destination)
        self._publish(lease.key, destination, manifest)
        return result

    def restore(self, key, signature):
        import joblib

        for root in (self.root, self.source):
            if root is None:
                continue
            pointer = root / "tasks" / task_digest(key) / "current.json"
            if not pointer.exists():
                continue
            directory, manifest = self._read(pointer, signature)
            if manifest.get("task") != task_identity(key):
                raise ValueError("Checkpoint task identity mismatch")
            if root != self.root:
                destination = self.root / "tasks" / task_digest(key) / uuid.uuid4().hex
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(destination.name + ".tmp")
                shutil.copytree(directory, temporary)
                os.replace(temporary, destination)
                directory = destination
            self._publish(key, directory, manifest)
            result = joblib.load(directory / "result.joblib")
            return True, relocate(result, manifest["origin_run"], self.run_dir)
        return False, None


@contextmanager
def _file_lock(path, shared=False, create=True):
    import fcntl

    with Path(path).open("a+" if create else "r") as handle:
        try:
            fcntl.flock(handle, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Checkpoint run still has an active coordinator or worker") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def checkpoint_run(function):
    @wraps(function)
    def run(self, *args, **kwargs):
        if (getattr(self, "execution_config", None) is None
                and getattr(self, "scheduler", "legacy") == "legacy"):
            return function(self, *args, **kwargs)
        validate_recovery_paths(self.results_dir, getattr(self, "resume_from", None))
        root = Path(self.results_dir).resolve() / ".task_store"
        root.mkdir(parents=True, exist_ok=True)
        with ExitStack() as stack:
            stack.enter_context(_file_lock(root / "owner.lock"))
            stack.enter_context(_file_lock(root / "live.lock", shared=True))
            source = getattr(self, "resume_from", None)
            if source:
                source = Path(source).resolve() / ".task_store"
                if root == source or root.is_relative_to(source) or source.is_relative_to(root):
                    raise ValueError("Recovery requires a separate new results directory")
                stack.enter_context(_file_lock(source / "owner.lock", create=False))
                stack.enter_context(_file_lock(source / "live.lock", create=False))
            return function(self, *args, **kwargs)

    return run


def validate_recovery_paths(destination, source):
    if source is None:
        return
    destination, source = Path(destination).resolve(), Path(source).resolve()
    if (
        destination == source
        or destination.is_relative_to(source)
        or source.is_relative_to(destination)
    ):
        raise ValueError("Recovery requires a separate new results directory")


def hold_worker_lock(run_dir):
    """Keep orphan workers visible after coordinator death until process exit."""
    global _WORKER_LOCK
    import fcntl
    from gpse.utils.process_control import process_identity

    root = Path(run_dir) / ".task_store"
    handle = (root / "live.lock").open("r")
    try:
        fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        owner = json.loads((root / "run.json").read_text())
        identity = process_identity(owner["owner_pid"])
        if os.getppid() != owner["owner_pid"] or (
            identity is not None and list(identity)[:2] != owner["owner_identity"][:2]
        ):
            raise ValueError("Compute coordinator exited before worker initialization")
    except BaseException:
        handle.close()
        raise
    _WORKER_LOCK = handle
