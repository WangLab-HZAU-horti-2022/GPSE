"""Fold concurrency, deterministic reduction, failed trials and budget controls."""

from concurrent.futures import CancelledError, ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import numpy as np
import pandas as pd
import pytest

from gpse.train._fold_scheduler import _Dispatcher
from gpse.train.cli import _apply_threads_budget
from gpse.train.workflow import _build_parser


def test_fold_budget_includes_single_repeat_and_manual_caps(monkeypatch):
    monkeypatch.setattr("gpse.utils.parallel.get_available_cpu_cores", lambda *a: 90)
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--task_type",
            "regression",
            "--models",
            "rf_reg",
            "--threads",
            "90",
            "--n_repeats",
            "100",
            "--train_folds",
            "5",
            "--max_workers",
            "1",
            "--repeat_workers",
            "1",
        ]
    )
    _apply_threads_budget(args, log=False)
    assert args.cpu_budget == 90
    assert args.n_jobs == 18  # Five independent folds, rather than one 90-thread fit.
    args = parser.parse_args(
        [
            "--task_type",
            "regression",
            "--models",
            "rf_reg",
            "--fold_workers",
            "3",
            "--n_jobs",
            "1",
        ]
    )
    _apply_threads_budget(args, log=False)
    assert args.cpu_budget == 90  # Explicit fold concurrency enables the bounded pool.
    with pytest.raises(ValueError, match="fold_workers"):
        args.fold_workers = 0
        _apply_threads_budget(args, log=False)


def test_dispatcher_limits_global_work_and_preserves_fold_order():
    lock = threading.Lock()
    active = peak = 0

    def execute(kind, model, repeat, payload):
        nonlocal active, peak
        fold = payload[0]
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.015 * (4 - fold))
        with lock:
            active -= 1
        return fold

    with ThreadPoolExecutor(max_workers=8) as pool, ThreadPoolExecutor(max_workers=2) as controls:
        dispatcher = _Dispatcher(pool, execute, capacity=3)

        def run(repeat):
            return list(dispatcher.map_folds("fold", "test", repeat, {}, [((), ())] * 4, None, 3))

        futures = [controls.submit(run, repeat) for repeat in range(2)]
        assert [future.result(timeout=5) for future in futures] == [list(range(4))] * 2
        assert 2 <= peak <= 3


def test_failed_trial_drains_running_folds_before_returning():
    started = threading.Barrier(3)
    finished = threading.Event()

    def execute(kind, model, repeat, payload):
        started.wait(timeout=3)
        if payload[0] == 0:
            raise ValueError("invalid fold")
        time.sleep(0.05)
        finished.set()
        return payload[0]

    with ThreadPoolExecutor(max_workers=3) as pool:
        dispatcher = _Dispatcher(pool, execute, capacity=3)
        with pytest.raises(ValueError, match="invalid fold"):
            list(dispatcher.map_folds("optimization_fold", "test", 0, {}, [((), ())] * 5, 0, 3))
        assert finished.is_set()
        assert not dispatcher.pending


def test_stopping_dispatcher_unblocks_waiting_controllers():
    release = threading.Event()
    with ThreadPoolExecutor(max_workers=1) as pool, ThreadPoolExecutor(max_workers=1) as controls:
        dispatcher = _Dispatcher(pool, lambda *args: release.wait(timeout=5), capacity=1)
        dispatcher.submit("fold", "test", 0, None)
        blocked = controls.submit(dispatcher.submit, "fold", "test", 1, None)
        dispatcher.stop()
        try:
            with pytest.raises(CancelledError):
                blocked.result(timeout=2)
        finally:
            release.set()


def _peak(trace, stages):
    active, peak = set(), 0
    for event in sorted(trace, key=lambda e: e["timestamp"]):
        if event["stage"] in stages:
            if event["status"] == "started":
                active.add(event["execution_id"])
            elif event["status"] in {"completed", "failed", "pruned", "interrupted"}:
                active.discard(event["execution_id"])
            peak = max(peak, len(active))
    return peak


@pytest.mark.parametrize("case", ["pipeline_custom_cv", "structure", "classification_search"])
def test_one_repeat_controller_parallelizes_folds_without_changing_results(tmp_path, case):
    rng = np.random.default_rng(845)
    X = pd.DataFrame(rng.normal(size=(120, 10)), index=[f"s{i}" for i in range(120)])
    y = 2 * X[0] - X[1] + rng.normal(size=len(X)) * 0.2
    flags = []
    model, task = "kernelridge_reg", "regression"
    if case == "pipeline_custom_cv":
        model = "rf_reg"
        flags = [
            "--genotype_imputation",
            "mean",
            "--feature_selection",
            "pca",
            "--select_k",
            "4",
            "--standardize_phenotype",
            "--optuna_per_repeat",
            "--cv_file",
            str(tmp_path / "cv.csv"),
        ]
        cv = pd.DataFrame({f"cv{r}": rng.permutation(len(X)) % 3 for r in range(2)}, index=X.index)
        cv.to_csv(tmp_path / "cv.csv")
        X.iloc[::9, 2] = np.nan
    elif case == "structure":
        flags = ["--split_strategy", "structure_aware", "--structure_clusters", "6"]
    else:
        model, task = "svm_clf", "classification"
        y = pd.qcut(y, 3, labels=False)
        flags = ["--n_classes", "3"]
    X.to_csv(tmp_path / "X.csv", index_label="ID")
    y.to_frame("yield").to_csv(tmp_path / "y.csv", index_label="ID")
    traces = {}
    for mode, fold_workers in [("repeat", "1"), ("fold", "3")]:
        out = tmp_path / mode
        command = [
            sys.executable,
            "-m",
            "gpse.train.cli",
            "--geno_file",
            str(tmp_path / "X.csv"),
            "--pheno_file",
            str(tmp_path / "y.csv"),
            "--target_trait",
            "yield",
            "--task_type",
            task,
            "--models",
            model,
            "--n_repeats",
            "2",
            "--train_folds",
            "3",
            "--cv_folds",
            "3",
            "--trials",
            "3",
            "--threads",
            "4",
            "--n_jobs",
            "1",
            "--max_workers",
            "1",
            "--repeat_workers",
            "1",
            "--fold_workers",
            fold_workers,
            "--no-use_stacking",
            "--save_models",
            "--results_dir",
            str(out),
            *flags,
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=90,
            env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
        )
        assert result.returncode == 0, result.stderr
        traces[mode] = [
            json.loads(line) for line in (out / "logs/events.jsonl").read_text().splitlines()
        ]

    before, after = tmp_path / "repeat", tmp_path / "fold"
    for repeat in range(1, 3):
        relative = Path(f"models/{model}/repeat_{repeat}")
        a, b = [
            json.loads((p / relative / "repeat_results.json").read_text()) for p in (before, after)
        ]
        for key in [
            "params",
            "test_indices",
            "ensemble_metrics",
            "phenotype_scaler",
            "optimization_info",
        ]:
            assert a[key] == b[key], key
        for af, bf in zip(a["fold_results"], b["fold_results"], strict=True):
            for key in [
                "train_indices",
                "val_indices",
                "train_metrics",
                "val_metrics",
                "test_metrics",
            ]:
                assert af[key] == bf[key], key
        for name in ["all_predictions.json", "cv_train_only.csv"]:
            assert (before / relative / name).read_bytes() == (after / relative / name).read_bytes()
    assert json.loads((before / "evaluation/holdout_metrics.json").read_text()) == json.loads(
        (after / "evaluation/holdout_metrics.json").read_text()
    )

    histories = []
    for mode in ("repeat", "fold"):
        histories.append(
            [
                {k: event[k] for k in ("model", "repeat", "trial", "state", "params", "metrics")}
                for event in traces[mode]
                if event["stage"] == "trial_history"
            ]
        )
    assert histories[0] == histories[1]
    trace = traces["fold"]
    assert _peak(trace, {"fold"}) >= 2
    assert 2 <= _peak(trace, {"optimization_fold"}) <= 3
    assert _peak(trace, {"fold", "optimization_fold", "cv_folds"}) <= 3
    assert _peak(trace, {"repeat"}) == 1
    workers = {event["pid"] for event in trace if event["stage"] == "worker"}
    fits = {event["pid"] for event in trace if event["stage"] in {"fold", "optimization_fold"}}
    assert len(workers) == 3 and fits <= workers
    # An entire trial, including its running folds, finishes before the next suggestion.
    for repeat in (1, 2):
        trials = [e for e in trace if e["stage"] == "trial" and e["repeat"] == repeat]
        trial_ends = [e["timestamp"] for e in trials if e["status"] in {"completed", "pruned"}]
        trial_starts = [e["timestamp"] for e in trials if e["status"] == "started"]
        assert all(end <= start for end, start in zip(trial_ends, trial_starts[1:]))


@pytest.mark.parametrize("stage", ["warmup", "optimization_fold", "training_fold"])
def test_interrupt_cleans_compute_pool_and_blocked_controllers(tmp_path, stage):
    # Block real spawned workers independently of event logging. In particular,
    # interrupt the initial warmup and the first frozen search, before any repeat
    # has completed and while controller threads may be waiting for pool slots.
    script = tmp_path / "slow_fold.py"
    script.write_text("""import os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from gpse.train import _scheduler
from gpse.train.cli import main

original = _scheduler._execute
def slow(kind, model, repeat, payload):
    if kind == os.environ["BLOCK_STAGE"]:
        (Path(os.environ["PID_DIR"]) / str(os.getpid())).write_text("started")
        time.sleep(300)
    return original(kind, model, repeat, payload)
_scheduler._execute = slow

if __name__ == "__main__":
    root = Path(sys.argv[1])
    rng = np.random.default_rng(91)
    X = pd.DataFrame(rng.normal(size=(80, 5)), index=[f"s{i}" for i in range(80)])
    X.to_csv(root / "X.csv", index_label="ID")
    (X[0] + rng.normal(size=80)*.1).to_frame("yield").to_csv(root / "y.csv", index_label="ID")
    sys.exit(main([
        "--geno_file", str(root / "X.csv"), "--pheno_file", str(root / "y.csv"),
        "--target_trait", "yield", "--task_type", "regression", "--models", "kernelridge_reg",
        "--n_repeats", "4", "--train_folds", "3", "--threads", "4", "--n_jobs", "1",
        "--trials", "2", "--results_dir", str(root / "results"), "--no-use_stacking",
    ]))
""")
    markers = tmp_path / "pids"
    markers.mkdir()
    process = subprocess.Popen(
        [sys.executable, str(script), str(tmp_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "BLOCK_STAGE": stage,
            "PID_DIR": str(markers),
        },
    )
    try:
        deadline = time.monotonic() + 25
        while not list(markers.iterdir()):
            assert process.poll() is None
            assert time.monotonic() < deadline, f"{stage} did not start"
            time.sleep(0.025)
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=15) == 130
        journal = tmp_path / "results/logs/events.jsonl"
        trace = [json.loads(line) for line in journal.read_text().splitlines()]
        pids = {int(path.name) for path in markers.iterdir()}
        pids.update(e["pid"] for e in trace if e["stage"] == "worker")
        assert trace[-1]["status"] == "interrupted"
        deadline = time.monotonic() + 3
        while pids and time.monotonic() < deadline:
            for pid in list(pids):
                try:
                    if Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] == "Z":
                        pids.remove(pid)
                except OSError:
                    pids.remove(pid)
            if pids:
                time.sleep(0.025)
        assert not pids, f"Live compute workers remain: {pids}"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
