"""Checkpoint publication, integrity and stopped-run ownership invariants."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from gpse.train._resources import ResourceManager, ResourceRequest, TaskKey
from gpse.train._task_store import (
    TaskStore,
    _file_lock,
    checkpoint_run,
    prepare_result,
    stage_directory,
    task_digest,
)


def lease_for(root, fold=0):
    manager = ResourceManager(2, 100, 2, {"model": 2})
    key = TaskKey(str(root), "model", 0, None, fold, "training_fold")
    return manager, manager.try_acquire(key, 0, ResourceRequest(1, 10))


def prepared(lease, value=42):
    stage = stage_directory(lease)
    stage.mkdir(parents=True)
    (stage / f"fold_{lease.key.fold + 1}_model.pkl").write_bytes(b"model artifact")
    return prepare_result(lease, value)


def test_only_coordinator_commit_publishes_fold_model_and_result(tmp_path):
    store = TaskStore(tmp_path, "fingerprint")
    manager, lease = lease_for(tmp_path)
    receipt = prepared(lease)
    model = tmp_path / "models/model/repeat_1/fold_1_model.pkl"
    assert not model.exists()
    assert store.restore(lease.key, "signature") == (False, None)
    assert store.commit(lease, receipt, "signature", manager) == 42
    assert model.read_bytes() == b"model artifact"
    assert not stage_directory(lease).exists()
    assert store.restore(lease.key, "signature") == (True, 42)
    manager.release(lease, computation_stopped=True)


def test_stale_attempt_and_wrong_receipt_never_publish(tmp_path):
    store = TaskStore(tmp_path, "fingerprint")
    manager, old = lease_for(tmp_path)
    receipt = prepared(old)
    manager.release(old, computation_stopped=True)
    current = manager.try_acquire(old.key, 1, old.request)
    with pytest.raises(ValueError, match="Stale"):
        store.commit(old, receipt, "signature", manager)
    with pytest.raises(ValueError, match="active lease"):
        store.commit(current, receipt, "signature", manager)
    assert not list(store.root.glob("tasks/*/current.json"))
    assert not list(tmp_path.glob("models/**/*.pkl"))
    manager.release(current, computation_stopped=True)


def test_prepared_artifact_corruption_prevents_publication(tmp_path):
    store = TaskStore(tmp_path, "fingerprint")
    manager, lease = lease_for(tmp_path)
    receipt = prepared(lease)
    (stage_directory(lease) / "result.joblib").write_bytes(b"truncated")
    with pytest.raises(ValueError, match="hash mismatch"):
        store.commit(lease, receipt, "signature", manager)
    assert store.restore(lease.key, "signature") == (False, None)


def test_late_attempt_cannot_replace_newer_committed_results(tmp_path):
    store = TaskStore(tmp_path, "fingerprint")
    manager, old = lease_for(tmp_path)
    old_receipt = prepared(old, value=1)
    manager.release(old, computation_stopped=True)
    newer = manager.try_acquire(old.key, 1, old.request)
    newer_receipt = prepared(newer, value=2)
    store.commit(newer, newer_receipt, "signature", manager)
    pointer = next(store.root.glob("tasks/*/current.json"))
    contents = pointer.read_bytes()
    with pytest.raises(ValueError, match="Stale"):
        store.commit(old, old_receipt, "signature", manager)
    assert pointer.read_bytes() == contents
    assert store.restore(old.key, "signature") == (True, 2)


def test_recovery_checks_fingerprint_signature_and_every_artifact(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    store = TaskStore(source, "fingerprint")
    manager, lease = lease_for(source)
    store.commit(lease, prepared(lease), "signature", manager)
    old_contents = {
        str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()
    }
    recovered = TaskStore(target, "fingerprint", source)
    key = replace(lease.key, run=str(target))
    assert recovered.restore(key, "signature") == (True, 42)
    assert (target / "models/model/repeat_1/fold_1_model.pkl").read_bytes() == b"model artifact"
    assert old_contents == {
        str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()
    }
    with pytest.raises(ValueError, match="parameters/splits"):
        recovered.restore(key, "different")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        TaskStore(tmp_path / "bad", "different", source)
    (source / "models/model/repeat_1/fold_1_model.pkl").unlink()
    # Canonical checkpoint is independent of its materialized public export.
    assert TaskStore(tmp_path / "other", "fingerprint", source).restore(
        replace(key, run=str(tmp_path / "other")), "signature"
    )[0]
    pointer = next(store.root.glob("tasks/*/current.json"))
    record = json.loads(pointer.read_text())
    (pointer.parent / record["commit"] / "fold_1_model.pkl").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        TaskStore(tmp_path / "damaged", "fingerprint", source)


def test_manifest_tampering_and_symlinks_are_rejected(tmp_path):
    store = TaskStore(tmp_path, "fingerprint")
    manager, lease = lease_for(tmp_path)
    receipt = prepared(lease)
    path = stage_directory(lease) / "prepared.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="manifest hash"):
        store.commit(lease, receipt, "signature", manager)
    manager2, lease2 = lease_for(tmp_path, fold=1)
    stage = stage_directory(lease2)
    stage.mkdir(parents=True)
    (stage / "escape").symlink_to(path)
    with pytest.raises(ValueError, match="symlink"):
        prepare_result(lease2, 42)


def test_recovery_refuses_live_worker_lock(tmp_path):
    from types import SimpleNamespace

    source = tmp_path / "source"
    root = source / ".task_store"
    root.mkdir(parents=True)
    (root / "owner.lock").touch()
    (root / "live.lock").touch()
    predictor = SimpleNamespace(
        results_dir=tmp_path / "target", execution_config=True, resume_from=source
    )

    @checkpoint_run
    def run(self):
        return "started"

    with _file_lock(root / "live.lock", shared=True):
        with pytest.raises(ValueError, match="active coordinator or worker"):
            run(predictor)
    assert run(predictor) == "started"


@pytest.mark.parametrize("relationship", ["same", "child", "parent"])
def test_api_recovery_rejects_overlapping_directories_before_writing(tmp_path, relationship):
    from types import SimpleNamespace

    source = tmp_path / "source"
    destination = {"same": source, "child": source / "new", "parent": tmp_path}[relationship]
    predictor = SimpleNamespace(results_dir=destination, execution_config=True, resume_from=source)

    @checkpoint_run
    def run(self):
        pytest.fail("overlapping recovery must not run")

    with pytest.raises(ValueError, match="separate new"):
        run(predictor)
    assert not source.exists()


def test_finalize_signature_includes_splits_and_scores_but_ignores_wall_time():
    from copy import deepcopy
    from gpse.train._task_store import task_signature

    payload = [
        {
            "repeat_idx": 0,
            "params": {"alpha": 1},
            "test_indices": [4],
            "fold_results": [
                {"split_identifier": "one", "val_metrics": {"pearson": 0.4}, "training_time": 2}
            ],
            "avg_training_time": 2,
            "all_predictions_file": "old/path",
            "cv_folds_file": "old/cv",
        }
    ]
    changed = deepcopy(payload)
    changed[0]["fold_results"][0]["training_time"] = 7
    changed[0]["avg_training_time"] = 7
    changed[0]["all_predictions_file"] = "new/path"
    assert task_signature("finalize", payload) == task_signature("finalize", changed)
    changed[0]["fold_results"][0]["split_identifier"] = "two"
    assert task_signature("finalize", payload) != task_signature("finalize", changed)


def test_orphan_compute_worker_retains_live_lock_after_coordinator_is_killed(tmp_path):
    import os
    import select
    import signal
    import subprocess
    import sys
    import time
    from gpse.utils.process_control import process_identity

    script = tmp_path / "owner.py"
    script.write_text("""import subprocess, sys
from pathlib import Path
from gpse.train._task_store import TaskStore
root = Path(sys.argv[1])
TaskStore(root, "fingerprint")
(root / ".task_store/live.lock").touch()
code = "from gpse.train._task_store import hold_worker_lock; import sys,time; hold_worker_lock(sys.argv[1]); print('ready',flush=True); time.sleep(60)"
worker = subprocess.Popen([sys.executable, "-c", code, str(root)], stdout=subprocess.PIPE, text=True)
assert worker.stdout.readline().strip() == "ready"
print(worker.pid, flush=True)
worker.wait()
""")
    run_dir = tmp_path / "run"
    owner = subprocess.Popen(
        [sys.executable, str(script), str(run_dir)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
    )
    worker_pid = None
    try:
        assert select.select([owner.stdout], [], [], 10)[0], "worker did not initialize"
        line = owner.stdout.readline().strip()
        assert line.isdigit(), owner.stderr.read()
        worker_pid = int(line)
        identity = process_identity(worker_pid)
        assert identity is not None
        owner.kill()
        owner.wait(timeout=5)
        assert process_identity(worker_pid)[:2] == identity[:2]
        with pytest.raises(ValueError, match="active coordinator or worker"):
            with _file_lock(run_dir / ".task_store/live.lock", create=False):
                pytest.fail("orphan compute must block recovery")
        os.kill(worker_pid, signal.SIGTERM)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            current = process_identity(worker_pid)
            if current is None or current[3] == "Z":
                break
            time.sleep(0.01)
        else:
            pytest.fail("test worker failed to exit")
        with _file_lock(run_dir / ".task_store/live.lock", create=False):
            pass
    finally:
        if owner.poll() is None:
            owner.kill()
        owner.wait(timeout=5)
        if worker_pid is not None:
            try:
                os.kill(worker_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("classification", [False, True])
def test_catboost_fit_logs_are_private_and_in_the_committed_inventory(tmp_path, classification):
    import numpy as np
    from gpse.models._lease_catboost import adapter_class
    from gpse.train._task_store import task_artifacts, task_signature
    from gpse.train._thread_plan import execution_plan, plan_for
    from sklearn.pipeline import Pipeline

    path = "catboost.CatBoostClassifier" if classification else "catboost.CatBoostRegressor"
    cls = adapter_class(path)
    X = np.arange(120).reshape(30, 4)
    y = np.arange(30) % 2 if classification else np.arange(30)
    store = TaskStore(tmp_path, "fingerprint")
    directories = []
    for fold in (0, 1):
        manager, lease = lease_for(tmp_path, fold)
        model = cls(iterations=3, verbose=False, thread_count=1, train_dir=str(tmp_path / "shared"))
        with execution_plan(plan_for("model", path, 1)), task_artifacts(lease):
            model.fit(X, y)
        directory = Path(model.get_param("train_dir"))
        assert directory.is_relative_to(stage_directory(lease))
        assert (directory / "catboost_training.json").exists()
        directories.append(directory)
        receipt = prepare_result(lease, model.predict(X))
        store.commit(lease, receipt, "signature", manager)
        pointer = store.root / "tasks" / task_digest(lease.key) / "current.json"
        record = json.loads(pointer.read_text())
        manifest = json.loads((pointer.parent / record["commit"] / "committed.json").read_text())
        assert any(name.startswith("native_logs/") for name in manifest["files"])
        manager.release(lease, computation_stopped=True)
    assert directories[0] != directories[1]
    assert not (tmp_path / "shared").exists()
    a = Pipeline([("model", cls(train_dir="old/run"))])
    b = Pipeline([("model", cls(train_dir="new/run"))])
    assert task_signature("stacking_refit", (None, None, None, a, path)) == task_signature(
        "stacking_refit", (None, None, None, b, path)
    )
