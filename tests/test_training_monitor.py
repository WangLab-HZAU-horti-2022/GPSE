"""Real training and terminal acceptance coverage for the observation layer."""

import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from loguru import logger

from gpse.train.cli import _apply_threads_budget
from gpse.train.workflow import _build_parser, _configure_logging
from gpse.train.predictor import GenomicPredictorV2
from gpse.utils import training_monitor as monitor
from gpse.utils.log_utils import logger_init


@pytest.fixture
def inputs(tmp_path):
    rng = np.random.default_rng(123)
    X = pd.DataFrame(
        rng.normal(size=(60, 5)),
        columns=[f"snp{i}" for i in range(5)],
        index=[f"s{i}" for i in range(60)],
    )
    y = pd.DataFrame({"yield": 2 * X.iloc[:, 0] + rng.normal(size=60) * 0.1}, index=X.index)
    geno, pheno = tmp_path / "geno.csv", tmp_path / "pheno.csv"
    X.to_csv(geno, index_label="ID")
    y.to_csv(pheno, index_label="ID")
    return geno, pheno


def command(inputs, output, *extra):
    return [
        sys.executable,
        "-m",
        "gpse.train.cli",
        "--geno_file",
        str(inputs[0]),
        "--pheno_file",
        str(inputs[1]),
        "--target_trait",
        "yield",
        "--task_type",
        "regression",
        "--n_repeats",
        "2",
        "--train_folds",
        "2",
        "--cv_folds",
        "2",
        "--results_dir",
        str(output),
        *extra,
    ]


def run_cli(inputs, output, *extra):
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    completed = subprocess.run(
        command(inputs, output, *extra), text=True, capture_output=True, timeout=60, env=env
    )
    assert completed.returncode == 0, completed.stderr
    assert "\x1b" not in completed.stdout + completed.stderr
    assert "\r" not in completed.stdout + completed.stderr
    events = [json.loads(line) for line in (output / "logs/events.jsonl").read_text().splitlines()]
    assert len({e["event_id"] for e in events}) == len(events)
    assert {e["run_id"] for e in events} == {output.name}
    assert events[0]["stage"] == "run" and events[0]["status"] == "started"
    assert events[-1]["stage"] == "run" and events[-1]["status"] == "completed"
    return completed, events


def test_explicit_one_wins_over_threads_and_yaml(tmp_path, monkeypatch):
    # Project YAML is not an input to training worker options.
    (tmp_path / "gpse.yaml").write_text("repeat_workers: 9\n")
    monkeypatch.chdir(tmp_path)
    args = _build_parser().parse_args(
        [
            "--task_type",
            "regression",
            "--models",
            "lasso_reg",
            "--threads",
            "8",
            "--repeat_workers=1",
            "--n_jobs",
            "1",
        ]
    )
    _apply_threads_budget(args)
    assert (args.n_jobs, args.max_workers, args.repeat_workers) == (1, 1, 1)


@pytest.mark.parametrize("verbosity", ["NORMAL", "VERBOSE", "DEBUG"])
def test_single_model_levels_and_redirect(inputs, tmp_path, verbosity):
    out = tmp_path / verbosity
    completed, events = run_cli(
        inputs,
        out,
        "--models",
        "lasso_reg",
        "--use_default_params",
        "--no-use_stacking",
        "--verbosity",
        verbosity,
    )
    for stage, count in (("model", 1), ("repeat", 2), ("fold", 4)):
        done = [e for e in events if e["stage"] == stage and e["status"] == "completed"]
        assert len(done) == count
        for event in done:
            assert event["elapsed_s"] >= 0
            assert event["model"] == "lasso_reg"
    assert completed.stderr.count("model completed model=lasso_reg") == 1
    assert ("repeat started" in completed.stderr) == (verbosity != "NORMAL")
    assert "CV Mean=" in completed.stderr or verbosity == "DEBUG"
    if verbosity == "DEBUG":
        assert "DEBUG" in completed.stderr
    assert any(e["stage"] == "log" and e["level"] == "DEBUG" for e in events)
    manifest = json.loads((out / "run_manifest.json").read_text())
    log_artifacts = [a for a in manifest["artifact_index"] if a["artifact_type"] == "execution_log"]
    assert any(a["relative_path"] == "logs/events.jsonl" for a in log_artifacts)
    assert all(a["file_size"] == (out / a["relative_path"]).stat().st_size for a in log_artifacts)


def test_nested_parallel_repeats_optuna_and_stacking(inputs, tmp_path):
    out = tmp_path / "nested"
    completed, events = run_cli(
        inputs,
        out,
        "--models",
        "kernelridge_reg",
        "lasso_reg",
        "--trials",
        "2",
        "--optuna_per_repeat",
        "--repeat_workers",
        "2",
        "--max_workers",
        "2",
        "--verbosity",
        "VERBOSE",
    )
    assert "Trial 0 finished" not in completed.stderr
    assert "it/s" not in completed.stderr
    done = lambda stage: [e for e in events if e["stage"] == stage and e["status"] == "completed"]
    assert len(done("model")) == 2
    assert len(done("repeat")) == 4
    assert len(done("fold")) == 8
    terminal_trials = [
        e for e in events if e["stage"] == "trial" and e["status"] in {"completed", "pruned"}
    ]
    assert len(terminal_trials) == 8
    assert 8 <= len(done("optimization_fold")) <= 16
    assert len(done("stacking_fold")) == 4
    assert len(done("stacking")) == 1
    histories = list((out / "logs/optuna").glob("*.jsonl"))
    assert len(histories) == 4
    assert all(len(p.read_text().splitlines()) == 2 for p in histories)
    trials = [e for e in events if e["stage"] == "trial_history"]
    assert len(trials) == 8
    assert all(e["model"] and e["repeat"] and e["trial"] for e in trials)
    assert all(set(e["metrics"]) == {"cv_mean", "cv_std", "robust_objective"} for e in trials)
    complete_trials = [e for e in trials if e["state"] == "COMPLETE"]
    assert len(complete_trials) == len(done("trial"))
    assert all(
        e["metrics"]["robust_objective"]
        == pytest.approx(e["metrics"]["cv_mean"] - 0.5 * e["metrics"]["cv_std"])
        for e in complete_trials
    )
    assert len({e["pid"] for e in done("repeat")}) >= 2
    assert (out / "ensembles/gpse/stacking_ensemble_model.pkl").exists()


def test_duplicate_model_not_scheduled_twice(inputs, tmp_path):
    completed, events = run_cli(
        inputs,
        tmp_path / "duplicates",
        "--models",
        "lasso_reg",
        "lasso_reg",
        "--use_default_params",
        "--no-use_stacking",
        "--threads",
        "4",
        "--repeat_workers",
        "1",
        "--n_jobs",
        "1",
    )
    assert "Duplicate model names removed" in completed.stderr
    assert events[0]["config"]["repeat_workers"] == 1
    assert events[0]["cli_config"]["repeat_workers"] == 1
    assert len([e for e in events if e["stage"] == "model" and e["status"] == "started"]) == 1


@pytest.mark.parametrize("use_defaults", [True, False])
def test_monitor_does_not_change_cv_holdout_or_stacking(inputs, tmp_path, use_defaults):
    def train(name, observed):
        _configure_logging("INFO")
        predictor = GenomicPredictorV2(
            results_dir=str(tmp_path / name),
            n_repeats=2,
            n_splits=2,
            use_default_params=use_defaults,
            n_trials=2,
        )
        method = (
            predictor.run_all_models
            if observed
            else lambda *a, **k: GenomicPredictorV2.run_all_models.__wrapped__(predictor, *a, **k)
        )
        return method(
            str(inputs[0]),
            str(inputs[1]),
            "yield",
            models=["lasso_reg", "kernelridge_reg"],
            use_stacking=True,
            cv_folds=2,
            top_n_models=2,
        )

    before, after = train("baseline", False), train("monitored", True)
    for name in ("lasso_reg", "kernelridge_reg"):
        assert before[name]["cv_aggregation"] == after[name]["cv_aggregation"]
        for metric in ("avg_ensemble_pearson", "avg_ensemble_mse", "avg_ensemble_spearman"):
            assert before[name][metric] == after[name][metric]
    assert before["stacking_ensemble"] == after["stacking_ensemble"]
    for path in (
        "evaluation/holdout_metrics.json",
        "evaluation/predictions/topsis_top1_holdout_predictions.csv",
    ):
        assert (tmp_path / "baseline" / path).read_bytes() == (
            tmp_path / "monitored" / path
        ).read_bytes()


def test_render_and_journal_failure_do_not_change_results(tmp_path, monkeypatch):
    predictor = SimpleNamespace(
        paths=SimpleNamespace(run_id="io-failure"),
        logs_dir=tmp_path,
        n_repeats=1,
        n_splits=2,
        results_dir=tmp_path,
    )
    # events.jsonl cannot be opened because it is a directory.
    (tmp_path / "events.jsonl").mkdir()

    def broken(*args):
        raise OSError("terminal disconnected")

    monkeypatch.setattr(monitor.ConsoleRenderer, "accept", broken)
    tracker = monitor.TrainingMonitor(predictor, ["a"])
    tracker.start()

    @monitor.observed("model")
    def compute(model_name):
        logger.warning("still observed")
        return {"value": 42}

    try:
        assert compute("a") == {"value": 42}
        with pytest.raises(ValueError, match="original computation failure"):

            @monitor.observed("fold")
            def fail():
                raise ValueError("original computation failure")

            fail()
    finally:
        tracker.close()
    assert not monitor.terminal_active()


def test_training_failure_is_visible_and_journaled(inputs, tmp_path):
    out = tmp_path / "failed"
    completed = subprocess.run(
        command(inputs, out, "--models", "does_not_exist"),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode != 0
    events = [json.loads(line) for line in (out / "logs/events.jsonl").read_text().splitlines()]
    assert events[-1]["stage"] == "run" and events[-1]["status"] == "failed"
    assert "failed" in completed.stderr


def test_keyboard_interrupt_journal_and_cleanup(tmp_path):
    predictor = SimpleNamespace(
        paths=SimpleNamespace(run_id="interrupt"),
        logs_dir=tmp_path,
        n_repeats=1,
        n_splits=2,
        results_dir=tmp_path,
        available_models=["a"],
    )

    @monitor.monitored_run
    def run(self):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run(predictor)
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert events[-1]["status"] == "interrupted"
    assert not monitor.terminal_active()


@pytest.mark.parametrize("environment", [{}, {"CI": "1"}, {"TERM": "dumb"}])
def test_live_and_warning_share_console(monkeypatch, environment):
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("TERM", raising=False)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)

    class Terminal(io.StringIO):
        def isatty(self):
            return True

    stream = Terminal()
    renderer = monitor.ConsoleRenderer(["a"], 2, 2, stream=stream)
    assert not monitor.ConsoleRenderer(["a"], 2, 2, progress="off", stream=stream).dynamic
    assert renderer.dynamic == (not environment)
    renderer.start()
    renderer.accept({"event_id": "1", "stage": "model", "status": "started", "model": "a"})
    renderer.accept(
        {
            "event_id": "2",
            "stage": "log",
            "status": "record",
            "level": "WARNING",
            "message": "warning stays visible",
            "model": "a",
        }
    )
    renderer.close()
    assert stream.getvalue().count("warning stays visible") == 1
    if environment:
        assert "\x1b" not in stream.getvalue()


@pytest.mark.parametrize("models", [["kernelridge_reg"], ["kernelridge_reg", "lasso_reg"]])
def test_real_cli_interrupt_with_repeat_workers(inputs, tmp_path, models):
    out = tmp_path / "sigint"
    capture = tmp_path / "interrupt.txt"
    with capture.open("w") as stream:
        process = subprocess.Popen(
            command(
                inputs,
                out,
                "--models",
                *models,
                "--max_workers",
                "2",
                "--trials",
                "10000",
                "--optuna_per_repeat",
                "--repeat_workers",
                "2",
            ),
            stdout=stream,
            stderr=stream,
            env={**os.environ, "OPENBLAS_NUM_THREADS": "1"},
        )
        try:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                assert process.poll() is None, capture.read_text()
                journal = out / "logs/events.jsonl"
                if journal.exists() and '"stage": "trial"' in journal.read_text():
                    break
                time.sleep(0.05)
            else:
                pytest.fail("Repeat workers did not start trials")
            process.send_signal(signal.SIGINT)
            assert process.wait(timeout=15) == 130
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
    events = [json.loads(line) for line in (out / "logs/events.jsonl").read_text().splitlines()]
    assert events[-1]["stage"] == "run" and events[-1]["status"] == "interrupted"
    if Path("/proc").exists():
        # Nested model cancellation must also stop the repeat grandchildren.
        worker_pids = {
            e["pid"] for e in events if e["stage"] == "repeat" and e["status"] == "started"
        }
        for pid in worker_pids:
            stat = Path(f"/proc/{pid}/stat")
            assert not stat.exists() or stat.read_text().split()[2] == "Z"
    assert "Training interrupted by user" in capture.read_text()
    assert json.loads((out / "run_manifest.json").read_text())["run_status"] == "failed"


def test_non_tty_overrides_force_color(monkeypatch):
    monkeypatch.setenv("FORCE_COLOR", "1")
    stream = io.StringIO()
    renderer = monitor.ConsoleRenderer(["a"], 1, 2, stream=stream)
    assert not renderer.dynamic
    renderer.start()
    renderer.accept(
        {
            "event_id": "a",
            "stage": "log",
            "status": "record",
            "level": "ERROR",
            "message": "explicit error",
        }
    )
    renderer.close()
    assert "\x1b" not in stream.getvalue()


def test_detailed_file_failure_keeps_training_logger_available(tmp_path):
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    configured = logger_init(str(blocked))
    configured.warning("logging continues without the file")


def test_failed_queue_does_not_change_compute_result(monkeypatch):
    class BrokenQueue:
        def put(self, *args, **kwargs):
            raise OSError("queue closed")

    monkeypatch.setattr(
        monitor, "_CLIENT", (os.getpid(), {"queue": BrokenQueue(), "run_id": "failure"})
    )

    @monitor.observed("model")
    def compute(model_name):
        return 42

    assert compute("a") == 42


def test_worker_python_output_is_routed_to_log(tmp_path):
    log_file = tmp_path / "worker.log"
    script = (
        "from gpse.utils.training_monitor import configure_worker; import sys; "
        + f"configure_worker(None, {str(log_file)!r}); "
        + "print('worker stdout'); print('worker warning', file=sys.stderr)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], text=True, capture_output=True, timeout=20
    )
    assert completed.returncode == 0
    assert completed.stdout == completed.stderr == ""
    assert "worker stdout" in log_file.read_text()
    assert "worker warning" in log_file.read_text()


def test_real_training_persists_results_when_render_and_event_file_fail(
    inputs, tmp_path, monkeypatch
):
    _configure_logging("INFO")
    predictor = GenomicPredictorV2(
        results_dir=str(tmp_path / "io-run"),
        n_repeats=2,
        n_splits=2,
        use_default_params=True,
        max_parallel_jobs=2,
    )
    (predictor.logs_dir / "events.jsonl").mkdir()

    def disconnected(*args):
        raise OSError("terminal disconnected")

    monkeypatch.setattr(monitor.ConsoleRenderer, "accept", disconnected)
    result = predictor.run_all_models(
        str(inputs[0]),
        str(inputs[1]),
        "yield",
        models=["lasso_reg", "kernelridge_reg"],
        use_stacking=True,
        cv_folds=2,
        top_n_models=2,
    )
    assert set(result) == {"lasso_reg", "kernelridge_reg", "stacking_ensemble"}
    assert (predictor.results_dir / "evaluation/holdout_metrics.json").exists()
    assert (predictor.results_dir / "ensembles/gpse/stacking_ensemble_model.pkl").exists()
