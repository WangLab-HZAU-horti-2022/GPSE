"""Budget invariants, actual frozen-repeat scheduling and process-tree cancellation."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pytest

from gpse.train.cli import _apply_threads_budget
from gpse.train.workflow import _build_parser
from gpse.utils.parallel import get_available_cpu_cores, resolve_cpu_budget


def test_auto_budget_and_explicit_caps(monkeypatch):
    monkeypatch.setattr("gpse.utils.parallel.get_available_cpu_cores", lambda *a: 12)
    args = _build_parser().parse_args(
        ["--task_type", "regression", "--models", "lasso_reg", "kernelridge_reg"]
    )
    _apply_threads_budget(args, log=False)
    assert args.cpu_budget == 12
    assert args.n_jobs == 1
    assert args.max_workers == 2
    assert args.repeat_workers == 12  # A trailing model may consume the whole pool.
    args = _build_parser().parse_args(
        [
            "--task_type",
            "regression",
            "--threads",
            "auto",
            "--models",
            "lasso_reg",
            "--repeat_workers",
            "1",
            "--n_jobs",
            "1",
        ]
    )
    _apply_threads_budget(args, log=False)
    assert args.cpu_budget == 12
    assert (args.n_jobs, args.max_workers, args.repeat_workers) == (1, 1, 1)
    args = _build_parser().parse_args(["--task_type", "regression", "--max_workers", "2"])
    _apply_threads_budget(args, log=False)
    assert args.cpu_budget is None  # Explicit manual mode is preserved.
    assert resolve_cpu_budget("-1") == 12
    with pytest.raises(ValueError):
        resolve_cpu_budget(13)


def test_cgroup_quota_limits_affinity(monkeypatch):
    original = Path.read_text

    def read(path, *args, **kwargs):
        if str(path) == "/sys/fs/cgroup/cpu.max":
            return "150000 100000"
        if str(path) == "/proc/self/cgroup":
            return "0::/\n"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(8)))
    assert get_available_cpu_cores() == 2


@pytest.fixture
def data(tmp_path):
    rng = np.random.default_rng(123)
    X = pd.DataFrame(rng.normal(size=(80, 6)), index=[f"s{i}" for i in range(80)])
    y = pd.DataFrame({"yield": X[0] * 2 + rng.normal(size=80) * 0.1}, index=X.index)
    X.to_csv(tmp_path / "X.csv", index_label="ID")
    y.to_csv(tmp_path / "y.csv", index_label="ID")
    return tmp_path / "X.csv", tmp_path / "y.csv"


def cli(data, out, *flags):
    return [
        sys.executable,
        "-m",
        "gpse.train.cli",
        "--geno_file",
        str(data[0]),
        "--pheno_file",
        str(data[1]),
        "--target_trait",
        "yield",
        "--task_type",
        "regression",
        "--models",
        "kernelridge_reg",
        "lasso_reg",
        "--n_repeats",
        "4",
        "--train_folds",
        "2",
        "--cv_folds",
        "2",
        "--trials",
        "2",
        "--results_dir",
        str(out),
        *flags,
    ]


@pytest.mark.parametrize("use_defaults", [False, True])
def test_shared_pool_matches_serial_and_reuses_optuna(data, tmp_path, use_defaults):
    extra = ["--use_default_params"] if use_defaults else []
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    for mode, flags in [
        ("serial", ["--n_jobs", "1", "--max_workers", "1", "--repeat_workers", "1"]),
        ("shared", ["--threads", "4", "--n_jobs", "1"]),
    ]:
        result = subprocess.run(
            cli(data, tmp_path / mode, *flags, *extra),
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr

    def events(mode):
        return [
            json.loads(s) for s in (tmp_path / mode / "logs/events.jsonl").read_text().splitlines()
        ]

    trace = events("shared")
    active = set()
    peak = 0
    # Independently reconstruct simultaneous fits from the worker stage events.
    for event in sorted(trace, key=lambda e: e["timestamp"]):
        if event["stage"] in {"fold", "optimization_fold", "refit"}:
            if event["status"] == "started":
                active.add(event["execution_id"])
            else:
                active.discard(event["execution_id"])
            peak = max(peak, len(active))
    assert 2 <= peak <= 4
    fit_pids = {e["pid"] for e in trace if e["stage"] == "fold" and e["status"] == "started"}
    assert 2 <= len(fit_pids) <= 4
    for model in ["kernelridge_reg", "lasso_reg"]:
        trials = [e for e in trace if e["model"] == model and e["stage"] == "trial_history"]
        assert len(trials) == (0 if use_defaults else 2)
        if not use_defaults:
            first_end = max(
                e["timestamp"]
                for e in trace
                if e["model"] == model
                and e["stage"] == "repeat"
                and e["repeat"] == 1
                and e["status"] == "completed"
            )
            later_starts = [
                e["timestamp"]
                for e in trace
                if e["model"] == model
                and e["stage"] == "repeat"
                and e["repeat"] > 1
                and e["status"] == "started"
            ]
            assert min(later_starts) >= first_end
        for mode in ["serial", "shared"]:
            params = [
                json.loads(
                    (tmp_path / mode / f"models/{model}/repeat_{r}/repeat_results.json").read_text()
                )["params"]
                for r in range(1, 5)
            ]
            assert all(p == params[0] for p in params)
        before = json.loads((tmp_path / f"serial/models/{model}/summary_results.json").read_text())
        after = json.loads((tmp_path / f"shared/models/{model}/summary_results.json").read_text())
        for metric in ["cv_mean", "cv_std_repeat", "cv_std_within", "cv_std_all_folds"]:
            assert before["cv_aggregation"][metric] == after["cv_aggregation"][metric]
    assert json.loads(
        (tmp_path / "serial/evaluation/holdout_metrics.json").read_text()
    ) == json.loads((tmp_path / "shared/evaluation/holdout_metrics.json").read_text())
    assert (
        tmp_path / "serial/evaluation/predictions/topsis_top1_holdout_predictions.csv"
    ).read_bytes() == (
        tmp_path / "shared/evaluation/predictions/topsis_top1_holdout_predictions.csv"
    ).read_bytes()
    stacking = [e for e in trace if e["stage"] == "stacking_fold" and e["status"] == "started"]
    assert len(stacking) == 4
    assert len({e["pid"] for e in stacking}) >= 2


def wait_for_file(path, process, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        assert process.poll() is None
        if path.exists() and path.stat().st_size:
            return
        time.sleep(0.025)
    pytest.fail(f"{path} not produced")


def assert_stopped(pids):
    deadline = time.monotonic() + 3
    remaining = []
    while time.monotonic() < deadline:
        remaining = []
        for pid in pids:
            path = Path(f"/proc/{pid}/stat")
            try:
                if path.read_text().rsplit(")", 1)[1].split()[0] != "Z":
                    remaining.append(pid)
            except OSError:
                pass
        if not remaining:
            return
        time.sleep(0.025)
    assert not remaining, f"Live descendant processes remain: {remaining}"


@pytest.mark.parametrize("outer", ["pool", "shutdown", "external", "adaptive"])
def test_stubborn_worker_and_detached_grandchild_are_stopped(tmp_path, outer):
    # The worker ignores TERM; its grandchild starts a separate session.
    # Both write PIDs independently of GPSE's logging/event system.
    script = tmp_path / "target.py"
    script.write_text("""import os,sys,time,signal,subprocess
from pathlib import Path
from gpse.utils.parallel import graceful_process_pool
from gpse.utils.process_control import run_subprocess

def work(path):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    child=subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(300)'],start_new_session=True)
    Path(path).write_text(str(os.getpid())+' '+str(child.pid))
    while True: time.sleep(.1)

if __name__=='__main__':
    if sys.argv[1]=='worker': work(sys.argv[2])
    elif sys.argv[1] in ('pool','shutdown'):
        try:
            with graceful_process_pool(max_workers=1) as pool:
                future=pool.submit(work,sys.argv[2])
                if sys.argv[1]=='pool': future.result()
        except KeyboardInterrupt: sys.exit(130)
    elif sys.argv[1]=='external':
        try: run_subprocess([sys.executable,__file__,'worker',sys.argv[2]])
        except KeyboardInterrupt: sys.exit(130)
    else:
        from gpse.batch.adaptive_scheduler import TaskRunner, TraitTask
        from loguru import logger
        runner=TaskRunner(logger)
        from types import SimpleNamespace
        p=Path(sys.argv[2])
        task=SimpleNamespace(key='test',log_file=p.with_suffix('.log'),results_dir=p.parent,pid_file=p.with_suffix('.pid'),command=[sys.executable,__file__,'worker',str(p)],attempt=0,threads=1,max_workers=1)
        runner.start(task)
        try:
            while True: time.sleep(.1)
        except KeyboardInterrupt:
            runner.shutdown_all()
            sys.exit(130)
""")
    pids_file = tmp_path / "pids"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    process = subprocess.Popen(
        [sys.executable, str(script), outer, str(pids_file)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env=env,
    )
    try:
        wait_for_file(pids_file, process)
        pids = list(map(int, pids_file.read_text().split()))
        process.send_signal(signal.SIGINT)
        time.sleep(0.15)
        process.send_signal(signal.SIGINT)  # Repeated Ctrl+C must not abort cleanup.
        _, err = process.communicate(timeout=12)
        assert process.returncode == 130, err.decode()
        assert_stopped(pids)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
def test_shared_pool_real_signal(data, tmp_path, sig):
    out = tmp_path / "interrupt"
    process = subprocess.Popen(
        cli(
            data, out, "--threads", "4", "--n_jobs", "1", "--optuna_per_repeat", "--trials", "10000"
        ),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1"},
    )
    try:
        journal = out / "logs/events.jsonl"
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            assert process.poll() is None
            if journal.exists() and '"stage": "trial"' in journal.read_text():
                break
            time.sleep(0.05)
        else:
            pytest.fail("Trials never started")
        process.send_signal(sig)
        assert process.wait(timeout=15) == 130
        trace = [json.loads(s) for s in journal.read_text().splitlines()]
        assert trace[-1]["stage"] == "run" and trace[-1]["status"] == "interrupted"
        assert_stopped({e["pid"] for e in trace if e["stage"] == "worker"})
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_interrupt_during_worker_start_before_registration(tmp_path):
    script = tmp_path / "startup.py"
    marker = tmp_path / "worker"
    script.write_text("""import multiprocessing as mp,sys,time
from pathlib import Path
from gpse.utils.parallel import graceful_process_pool
if __name__=='__main__':
    ctx=mp.get_context('fork')
    original=ctx.Process.start
    def slow_registration(process):
        original(process)
        Path(sys.argv[1]).write_text(str(process.pid))
        time.sleep(300)
    ctx.Process.start=slow_registration
    try:
        with graceful_process_pool(max_workers=1,mp_context=ctx) as pool:
            pool.submit(pow,2,2).result()
    except KeyboardInterrupt:
        sys.exit(130)
""")
    process = subprocess.Popen(
        [sys.executable, str(script), str(marker)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
    )
    try:
        wait_for_file(marker, process)
        pid = int(marker.read_text())
        process.send_signal(signal.SIGINT)
        _, err = process.communicate(timeout=10)
        assert process.returncode == 130, err.decode()
        assert_stopped([pid])
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_adaptive_command_preserves_training_flags_without_pinning_repeats(tmp_path):
    from gpse.batch.adaptive_scheduler import _build_tasks_from_config, AdaptiveScheduler

    tasks = _build_tasks_from_config(
        {
            "defaults": {
                "geno_file": "X.csv",
                "pheno_file": "y.csv",
                "use_default_params": True,
                "use_stacking": False,
            },
            "traits": [{"name": "yield"}],
        },
        tmp_path,
    )
    task = tasks[0]
    assert task.command[:3] == [sys.executable, "-m", "gpse.train.cli"]
    assert "--repeat_workers" not in task.command
    parsed = _build_parser().parse_args(task.command[3:])
    assert parsed.use_default_params and not parsed.use_stacking
    scheduler = AdaptiveScheduler(
        tasks,
        max_threads="auto",
        threads_per_trait=10000,
        min_threads_per_trait=2,
        min_free_mem_gb=0,
        mem_per_trait_gb=1,
        max_concurrent_traits=1,
        status_path=tmp_path / "status.json",
        tsv_path=tmp_path / "status.tsv",
        log_path=tmp_path / "log.txt",
        dry_run=True,
    )
    assert scheduler.threads_per_trait == get_available_cpu_cores()
    scheduler._apply_runtime_params(task)
    parsed = _build_parser().parse_args(task.command[3:])
    _apply_threads_budget(parsed, log=False)
    assert parsed.cpu_budget == get_available_cpu_cores()
    assert parsed.repeat_workers > 1


def test_classification_shared_pool_matches_serial(data, tmp_path):
    X = pd.read_csv(data[0], index_col="ID")
    pd.DataFrame({"yield": (X.iloc[:, 0] > 0).astype(int)}, index=X.index).to_csv(
        data[1], index_label="ID"
    )
    for mode, flags in [
        ("serial", ["--n_jobs", "1", "--max_workers", "1"]),
        ("shared", ["--threads", "4", "--n_jobs", "1"]),
    ]:
        result = subprocess.run(
            cli(
                data,
                tmp_path / mode,
                "--models",
                "rf_clf",
                "svm_clf",
                "--task_type",
                "classification",
                "--n_classes",
                "2",
                "--use_default_params",
                *flags,
            ),
            capture_output=True,
            text=True,
            env={**os.environ, "OPENBLAS_NUM_THREADS": "1"},
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
    a = json.loads((tmp_path / "serial/evaluation/holdout_metrics.json").read_text())
    b = json.loads((tmp_path / "shared/evaluation/holdout_metrics.json").read_text())
    assert a == b


def test_spawn_bootstrap_obeys_resolved_native_threads(data, tmp_path):
    result = subprocess.run(
        cli(
            data,
            tmp_path / "native",
            "--threads",
            "4",
            "--n_jobs",
            "2",
            "--use_default_params",
            "--no-use_stacking",
        ),
        capture_output=True,
        text=True,
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1"},
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    events = [
        json.loads(line)
        for line in (tmp_path / "native/logs/events.jsonl").read_text().splitlines()
    ]
    workers = [e for e in events if e["stage"] == "worker" and e["status"] == "ready"]
    assert workers
    assert all(e["native_threads"] == 2 for e in workers)
    assert all(1 <= pool["num_threads"] <= 2 for e in workers for pool in e["native_pools"])
    submissions = [e for e in events if e["stage"] == "scheduler" and e["status"] == "submitted"]
    assert max(e["active_tasks"] for e in submissions) <= 2


def test_interrupt_during_stacking(data, tmp_path):
    script = tmp_path / "slow_stacking.py"
    out = tmp_path / "stack-stop"
    markers = tmp_path / "stack-pids"
    markers.mkdir()
    script.write_text("""import os,sys,time
from pathlib import Path
from gpse.train import stacking

def slow_stack_job(*args):
    Path(sys.argv[3],str(os.getpid())).write_text(str(os.getpid()))
    time.sleep(300)

if __name__=='__main__':
    stacking._stack_job=slow_stack_job
    from gpse.train.cli import main
    sys.exit(main(['--geno_file',sys.argv[1],'--pheno_file',sys.argv[2],
                   '--results_dir',sys.argv[4],'--target_trait','yield','--task_type','regression',
                   '--models','lasso_reg','kernelridge_reg','--use_default_params',
                   '--n_repeats','2','--train_folds','2','--cv_folds','2',
                   '--threads','2','--n_jobs','1']))
""")
    process = subprocess.Popen(
        [sys.executable, str(script), str(data[0]), str(data[1]), str(markers), str(out)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
            "OPENBLAS_NUM_THREADS": "1",
        },
    )
    try:
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            assert process.poll() is None
            if list(markers.iterdir()):
                break
            time.sleep(0.05)
        else:
            pytest.fail("Stacking jobs did not start")
        process.send_signal(signal.SIGINT)
        _, err = process.communicate(timeout=15)
        assert process.returncode == 130, err.decode()
        assert_stopped([int(path.name) for path in markers.iterdir()])
        events = [json.loads(s) for s in (out / "logs/events.jsonl").read_text().splitlines()]
        assert any(e["stage"] == "stacking" and e["status"] == "interrupted" for e in events)
        assert events[-1]["stage"] == "run" and events[-1]["status"] == "interrupted"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_cgroup_v1_combined_cpu_mount_and_ancestor_limit(monkeypatch):
    def read(path, *args, **kwargs):
        files = {
            "/proc/self/cgroup": "2:cpu,cpuacct:/jobs/train\n",
            "/sys/fs/cgroup/cpu,cpuacct/jobs/train/cpu.cfs_quota_us": "150000",
            "/sys/fs/cgroup/cpu,cpuacct/jobs/train/cpu.cfs_period_us": "100000",
            "/sys/fs/cgroup/cpu,cpuacct/jobs/cpu.cfs_quota_us": "100000",
            "/sys/fs/cgroup/cpu,cpuacct/jobs/cpu.cfs_period_us": "100000",
        }
        if str(path) in files:
            return files[str(path)]
        raise FileNotFoundError(str(path))

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(8)))
    assert get_available_cpu_cores() == 1


def test_adaptive_interrupt_status_and_signal_handlers_are_restored(tmp_path):
    from gpse.batch.adaptive_scheduler import AdaptiveScheduler, TraitTask

    task = TraitTask(
        species="test",
        trait="yield",
        command=[],
        results_dir=tmp_path / "run",
        log_file=tmp_path / "child.log",
    )
    scheduler = AdaptiveScheduler(
        [task],
        max_threads="auto",
        threads_per_trait=2,
        min_threads_per_trait=1,
        min_free_mem_gb=0,
        mem_per_trait_gb=1,
        max_concurrent_traits=1,
        status_path=tmp_path / "status.json",
        tsv_path=tmp_path / "status.tsv",
        log_path=tmp_path / "log.txt",
    )
    task.status = "RUNNING"
    scheduler._stop.set()
    before = {sig: signal.getsignal(sig) for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]}
    assert scheduler.run() == 130
    assert task.status == "INTERRUPTED"
    assert task.exit_code == 130
    assert all(signal.getsignal(sig) == handler for sig, handler in before.items())
    assert "INTERRUPTED" in (tmp_path / "status.json").read_text()


def test_entry_cancellation_restores_handlers_and_returns_130():
    from gpse.utils.process_control import interruptible_entry

    before = {sig: signal.getsignal(sig) for sig in [signal.SIGTERM, signal.SIGHUP]}

    @interruptible_entry
    def entry():
        raise KeyboardInterrupt

    assert entry() == 130
    assert all(signal.getsignal(sig) == handler for sig, handler in before.items())
