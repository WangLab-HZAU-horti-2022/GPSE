"""Configuration admission and complete legacy/elastic training comparisons."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from gpse.train._execution_config import ExecutionConfig, ResourceProfile
from gpse.train.cli import _apply_threads_budget
from gpse.train.workflow import _build_parser


def config(models=("kernelridge_reg", "elasticnet_reg")):
    return ExecutionConfig(
        cpu_budget=4,
        worker_slots=3,
        memory_budget_bytes=4 * 2**30,
        fixed_memory_bytes=256 * 2**20,
        worker_memory_bytes=256 * 2**20,
        resource_profiles={
            name: ResourceProfile(2 if i == 0 else 1, 128 * 2**20, 3, 1 if i == 0 else 2)
            for i, name in enumerate(models)
        },
    )


def test_config_rejects_impossible_memory_cpu_and_unknown_fields(tmp_path):
    plan = config()
    raw = asdict(plan)
    for changes, pattern in [
        ({"worker_slots": 5}, "worker_slots"),
        ({"cpu_budget": 1, "worker_slots": 1}, "CPU"),
        ({"memory_budget_bytes": 100}, "memory"),
        ({"fixed_memory_bytes": -1}, "fixed_memory"),
        ({"aging_seconds": float("nan")}, "aging"),
        ({"unknown": True}, "fields"),
    ]:
        path = tmp_path / "execution.yaml"
        path.write_text(yaml.safe_dump({**raw, **changes}))
        with pytest.raises(ValueError, match=pattern):
            ExecutionConfig.load(path)
    with pytest.raises(ValueError, match="Missing"):
        plan.make_manager(["rf_reg"])


def test_cli_conflicts_are_explicit_and_default_is_elastic(tmp_path, monkeypatch):
    monkeypatch.setattr("gpse.utils.parallel.get_available_cpu_cores", lambda *a: 16)
    path = tmp_path / "execution.yaml"
    path.write_text(yaml.safe_dump(asdict(config())))
    parser = _build_parser()
    base = ["--task_type", "regression", "--models", "kernelridge_reg", "elasticnet_reg"]
    for flags, pattern in [
        (["--threads", "8"], "threads conflicts"),
        (["--n_jobs", "4"], "n_jobs conflicts"),
    ]:
        args = parser.parse_args(base + ["--execution_config", str(path)] + flags)
        with pytest.raises(ValueError, match=pattern):
            _apply_threads_budget(args, log=False)
    args = parser.parse_args(base + ["--execution_config", str(path)])
    _apply_threads_budget(args, log=False)
    assert args.cpu_budget == 4
    assert parser.parse_args(base).execution_config is None
    automatic = parser.parse_args(base)
    _apply_threads_budget(automatic, log=False)
    assert automatic.cpu_budget == 16
    for flags in (["--scheduler", "legacy", "--execution_config", str(path)],
                  ["--scheduler", "thread_plan", "--execution_config", str(path)]):
        with pytest.raises(ValueError, match="conflicts"):
            _apply_threads_budget(parser.parse_args(base + flags), log=False)


def test_dataset_reservations_are_checked_before_workers_start():
    from dataclasses import replace

    X = pd.DataFrame(np.ones((5, 3)))
    y = pd.Series(np.ones(5))
    with pytest.raises(ValueError, match="worker_memory_bytes"):
        replace(config(), worker_memory_bytes=1).validate_environment(X, y)
    with pytest.raises(ValueError, match="fixed_memory_bytes"):
        replace(config(), fixed_memory_bytes=1).validate_environment(X, y)


def test_automatic_plan_bounds_memory_and_gives_heavy_models_fit_threads(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        "gpse.train._execution_config.available_memory_bytes", lambda: 2 * 2**30
    )
    predictor = SimpleNamespace(
        cpu_budget=80, n_threads=1, n_splits=5, fold_workers=None,
        automatic_fit_threads=True, task_type="regression", scheduler="elastic_v1",
        model_optimizer=SimpleNamespace(_registry=SimpleNamespace(model_configs={
            "xgboost_reg": SimpleNamespace(import_path="xgboost.XGBRegressor"),
            "lasso_reg": SimpleNamespace(import_path="sklearn.linear_model.Lasso"),
        })),
    )
    X, y = pd.DataFrame(np.ones((100, 5))), pd.Series(np.arange(100))
    plan = ExecutionConfig.automatic(predictor, ["xgboost_reg", "lasso_reg"], X, y)
    assert plan.scheduler == "elastic_v1"
    assert plan.resource_profiles["xgboost_reg"].preferred_cpu_per_fit == 16
    assert plan.resource_profiles["lasso_reg"].preferred_cpu_per_fit == 1
    assert plan.worker_slots < 80
    assert (plan.fixed_memory_bytes + plan.worker_slots * plan.worker_memory_bytes
            + max(p.memory_bytes for p in plan.resource_profiles.values())
            <= plan.memory_budget_bytes)
    predictor.automatic_fit_threads = False
    predictor.n_threads = 4
    explicit = ExecutionConfig.automatic(predictor, ["xgboost_reg"], X, y)
    assert explicit.resource_profiles["xgboost_reg"].preferred_cpu_per_fit == 4


def test_default_elastic_cli_generates_plan_and_recovers_without_yaml(tmp_path):
    from scripts.profile_fold_scheduler import compare_outputs

    rng = np.random.default_rng(81)
    X = pd.DataFrame(rng.normal(size=(80, 5)), index=[f"s{i}" for i in range(80)])
    X.to_csv(tmp_path / "X.csv", index_label="ID")
    (X[0] + .2 * X[1]).to_frame("trait").to_csv(tmp_path / "y.csv", index_label="ID")
    common = [sys.executable, "-m", "gpse.train.cli",
              "--geno_file", str(tmp_path / "X.csv"),
              "--pheno_file", str(tmp_path / "y.csv"),
              "--target_trait", "trait", "--task_type", "regression",
              "--models", "kernelridge_reg", "elasticnet_reg", "--threads", "4",
              "--n_repeats", "2", "--train_folds", "3", "--cv_folds", "3",
              "--use_default_params", "--save_models", "--use_stacking"]
    environment = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    for mode, flags in [("default", []), ("recovered", ["--resume_from", str(tmp_path / "default")]),
                        ("legacy", ["--scheduler", "legacy"])]:
        result = subprocess.run(common + flags + ["--results_dir", str(tmp_path / mode)],
                                capture_output=True, text=True, timeout=90, env=environment)
        assert result.returncode == 0, result.stdout + result.stderr
        summary = json.loads((tmp_path / mode / "reports/run_summary.json").read_text())
        assert summary["run"]["failed_models"] == 0
    plan = json.loads((tmp_path / "default/resource_plan.json").read_text())
    assert plan["scheduler"] == "elastic_v1" and plan["cpu_budget"] == 4
    assert plan == json.loads((tmp_path / "recovered/resource_plan.json").read_text())
    assert not (tmp_path / "legacy/resource_plan.json").exists()
    events = [json.loads(line) for line in
              (tmp_path / "default/logs/events.jsonl").read_text().splitlines()]
    assert any(e["stage"] == "resources" and e["status"] == "lease_granted" for e in events)
    assert compare_outputs(tmp_path / "default", tmp_path / "recovered",
                           tolerance=1e-9, exact_predictions=False)["passed"]
    assert compare_outputs(tmp_path / "legacy", tmp_path / "default",
                           tolerance=1e-9, exact_predictions=False)["passed"]


def _assert_nested_close(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _assert_nested_close(a[key], b[key])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for av, bv in zip(a, b):
            _assert_nested_close(av, bv)
    elif isinstance(a, (float, int)):
        assert a == pytest.approx(b, rel=1e-9, abs=1e-10)
    else:
        assert a == b


@pytest.mark.parametrize(
    "case", ["defaults_stacking", "thread_plan", "regression_hpo", "classification_hpo"]
)
def test_complete_elastic_run_preserves_cv_oof_selection_and_reports(tmp_path, case):
    rng = np.random.default_rng(983)
    X = pd.DataFrame(rng.normal(size=(100, 6)), index=[f"s{i}" for i in range(100)])
    y = X[0] + 0.3 * X[1] + rng.normal(size=len(X)) * 0.1
    classification = case == "classification_hpo"
    models = ("svm_clf", "rf_clf") if classification else ("kernelridge_reg", "elasticnet_reg")
    if classification:
        y = (y > y.median()).astype(int)
    X.to_csv(tmp_path / "X.csv", index_label="ID")
    y.to_frame("trait").to_csv(tmp_path / "y.csv", index_label="ID")
    path = tmp_path / "execution.yaml"
    raw_plan = asdict(config(models))
    if case == "thread_plan":
        raw_plan["scheduler"] = "thread_plan"
    path.write_text(yaml.safe_dump(raw_plan))
    defaults = case in {"defaults_stacking", "thread_plan"}
    flags = ["--use_default_params"] if defaults else ["--trials", "2"]
    if not defaults:
        flags += ["--no-use_stacking"]
    if classification:
        flags += ["--n_classes", "2"]
    common = [
        sys.executable,
        "-m",
        "gpse.train.cli",
        "--geno_file",
        str(tmp_path / "X.csv"),
        "--pheno_file",
        str(tmp_path / "y.csv"),
        "--target_trait",
        "trait",
        "--task_type",
        "classification" if classification else "regression",
        "--models",
        *models,
        "--n_repeats",
        "2",
        "--train_folds",
        "3",
        "--cv_folds",
        "3",
        "--threads",
        "4",
        "--n_jobs",
        "1",
        "--repeat_workers",
        "1",
        "--max_workers",
        "2",
        "--save_models",
        *flags,
    ]
    for mode in ["legacy", "elastic"]:
        out = tmp_path / mode
        # Explicit n_jobs=1 would contradict heterogeneous elastic profiles.
        command = list(common)
        if mode == "elastic":
            index = command.index("--n_jobs")
            del command[index : index + 2]
            command += ["--execution_config", str(path)]
        else:
            command += ["--scheduler", "legacy"]
        command += ["--results_dir", str(out)]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=100,
            env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
        )
        assert result.returncode == 0, result.stdout + result.stderr
        summary = json.loads((out / "reports/run_summary.json").read_text())
        assert summary["run"]["failed_models"] == 0
        assert (out / "reports/training_report.html").exists()
        if defaults:
            assert (out / "ensembles/gpse/stacking_ensemble_model.pkl").exists()

    for name in models:
        for repeat in (1, 2):
            relative = Path(f"models/{name}/repeat_{repeat}")
            a, b = [
                json.loads((tmp_path / mode / relative / "repeat_results.json").read_text())
                for mode in ("legacy", "elastic")
            ]
            for key in ("params", "test_indices", "optimization_info", "ensemble_metrics"):
                _assert_nested_close(a[key], b[key])
            for af, bf in zip(a["fold_results"], b["fold_results"]):
                for key in (
                    "split_identifier",
                    "train_indices",
                    "val_indices",
                    "train_metrics",
                    "val_metrics",
                    "test_metrics",
                ):
                    _assert_nested_close(af[key], bf[key])
            for file in ("all_predictions.json", "cv_train_only.csv"):
                aa, bb = [
                    (tmp_path / mode / relative / file).read_text()
                    for mode in ("legacy", "elastic")
                ]
                if file.endswith("json"):
                    _assert_nested_close(json.loads(aa), json.loads(bb))
                else:
                    assert aa == bb

    for mode in ("legacy", "elastic"):
        ranking = pd.read_csv(tmp_path / mode / "selection/model_comparison_cv_topsis_simple.csv")
        if mode == "legacy":
            expected_order = ranking.sort_values("TOPSIS_Rank")["Model"].tolist()
        else:
            assert ranking.sort_values("TOPSIS_Rank")["Model"].tolist() == expected_order
    a, b = [
        json.loads((tmp_path / mode / "evaluation/holdout_metrics.json").read_text())
        for mode in ("legacy", "elastic")
    ]
    _assert_nested_close(a, b)

    trace = [
        json.loads(line)
        for line in (tmp_path / "elastic/logs/events.jsonl").read_text().splitlines()
    ]
    grants = [e for e in trace if e["stage"] == "resources" and e["status"] == "lease_granted"]
    releases = [e for e in trace if e["stage"] == "resources" and e["status"] == "lease_released"]
    assert {e["lease_id"] for e in grants} == {e["lease_id"] for e in releases}
    assert len(grants) == len(releases) == len({e["lease_id"] for e in grants})
    assert all(0 <= e["cpu_used"] <= 4 and e["active_tasks"] <= 3 for e in grants + releases)
    assert releases[-1]["cpu_used"] == 0
    stages = {e["task_key"]["stage"] for e in grants}
    assert {"training_fold", "finalize", "holdout_split", "repeat_ensemble"} <= stages
    if defaults:
        assert {"stacking_fold", "stacking_refit", "stacking_meta"} <= stages
    else:
        assert "optimization_fold" in stages

    recovered = tmp_path / "recovered"
    resumed = list(command)
    resumed[resumed.index("--results_dir") + 1] = str(recovered)
    resumed += ["--resume_from", str(tmp_path / "elastic")]
    result = subprocess.run(
        resumed,
        capture_output=True,
        text=True,
        timeout=100,
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    from scripts.profile_fold_scheduler import compare_outputs

    assert compare_outputs(
        tmp_path / "elastic", recovered, tolerance=1e-9, exact_predictions=False
    )["passed"]
    recovery_events = [
        json.loads(line) for line in (recovered / "logs/events.jsonl").read_text().splitlines()
    ]
    restored_stages = {
        event["task_key"]["stage"]
        for event in recovery_events
        if event["stage"] == "resources" and event["status"] == "task_restored"
    }
    assert {"training_fold", "finalize"} <= restored_stages
    if defaults:
        assert {"stacking_fold", "stacking_refit", "stacking_meta"} <= restored_stages
    else:
        assert "optimization_fold" in restored_stages
    from gpse.train._task_store import STAGES

    assert not [
        event
        for event in recovery_events
        if event["stage"] == "compute"
        and event["status"] == "started"
        and event["task_key"]["stage"] in STAGES
    ]
    for model in models:
        summary = json.loads((recovered / f"models/{model}/summary_results.json").read_text())
        for key in ("representative_model", "deployment_ensemble"):
            item = summary.get(key, {})
            path = item.get("model_path", item.get("path"))
            if path:
                assert Path(path).is_relative_to(recovered) and Path(path).exists()
    if defaults:
        import joblib

        source = tmp_path / "elastic"
        hidden_source = tmp_path / "source_moved"
        source.rename(hidden_source)
        try:
            saved = joblib.load(recovered / "ensembles/gpse/stacking_ensemble_model.pkl")
            predictions, probabilities = saved.predict(X.iloc[:5])
            assert len(predictions) == 5 and np.isfinite(predictions).all()
        finally:
            hidden_source.rename(source)


def test_failed_model_is_reported_as_partial_without_fabricated_metrics(tmp_path):
    rng = np.random.default_rng(17)
    X = pd.DataFrame(rng.normal(size=(80, 5)), index=[f"s{i}" for i in range(80)])
    X.to_csv(tmp_path / "X.csv", index_label="ID")
    X[0].to_frame("trait").to_csv(tmp_path / "y.csv", index_label="ID")
    (tmp_path / "execution.yaml").write_text(yaml.safe_dump(asdict(config())))
    script = tmp_path / "fail_fold.py"
    script.write_text("""import sys
from gpse.train import _scheduler
from gpse.train.cli import main
original = _scheduler._execute
def execute(kind, model, repeat, payload):
    if kind == "training_fold" and model == "elasticnet_reg" and payload[0] == 1:
        raise ValueError("injected fold failure")
    return original(kind, model, repeat, payload)
_scheduler._execute = execute
if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
""")
    output = tmp_path / "result"
    command = [
        sys.executable,
        str(script),
        "--geno_file",
        str(tmp_path / "X.csv"),
        "--pheno_file",
        str(tmp_path / "y.csv"),
        "--target_trait",
        "trait",
        "--task_type",
        "regression",
        "--models",
        "kernelridge_reg",
        "elasticnet_reg",
        "--n_repeats",
        "2",
        "--train_folds",
        "3",
        "--use_default_params",
        "--no-use_stacking",
        "--execution_config",
        str(tmp_path / "execution.yaml"),
        "--results_dir",
        str(output),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=30,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((output / "reports/run_summary.json").read_text())
    assert summary["run"]["completed_models"] == 1
    assert summary["run"]["failed_models"] == 1
    failed = next(row for row in summary["leaderboard"] if row["model"] == "elasticnet_reg")
    assert failed["status"] == "failed" and failed["pearson"] is None
    assert json.loads((output / "run_manifest.json").read_text())["run_status"] == "partial"
    ranking = pd.read_csv(output / "selection/model_comparison_cv_topsis_simple.csv")
    assert ranking["Model"].tolist() == ["kernelridge_reg"]

    from gpse.train._task_store import file_hash

    before = {str(p.relative_to(output)): file_hash(p) for p in output.rglob("*") if p.is_file()}
    refused = subprocess.run(
        [sys.executable, "-m", "gpse.train.cli", *command[2:]],
        capture_output=True,
        text=True,
        timeout=15,
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
    )
    assert refused.returncode != 0
    assert before == {
        str(p.relative_to(output)): file_hash(p) for p in output.rglob("*") if p.is_file()
    }
    recovered = tmp_path / "recovered"
    resumed = [sys.executable, "-m", "gpse.train.cli", *command[2:]]
    resumed[resumed.index("--results_dir") + 1] = str(recovered)
    resumed += ["--resume_from", str(output)]
    environment = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    result = subprocess.run(resumed, capture_output=True, text=True, timeout=45, env=environment)
    assert result.returncode == 0, result.stdout + result.stderr
    assert before == {
        str(p.relative_to(output)): file_hash(p) for p in output.rglob("*") if p.is_file()
    }
    recovered_summary = json.loads((recovered / "reports/run_summary.json").read_text())
    assert recovered_summary["run"]["completed_models"] == 2
    assert recovered_summary["run"]["failed_models"] == 0
    events = [
        json.loads(line) for line in (recovered / "logs/events.jsonl").read_text().splitlines()
    ]
    restored = [e for e in events if e["stage"] == "resources" and e["status"] == "task_restored"]
    assert restored
    restored_ids = {json.dumps(e["task_key"], sort_keys=True) for e in restored}
    computed_ids = {
        json.dumps(e["task_key"], sort_keys=True)
        for e in events
        if e["stage"] == "compute" and e["status"] == "started"
    }
    assert restored_ids.isdisjoint(computed_ids)
    assert not list((recovered / ".task_store/pending").rglob("result.joblib"))
    clean = tmp_path / "clean"
    clean_command = resumed[:-2]
    clean_command[clean_command.index("--results_dir") + 1] = str(clean)
    result = subprocess.run(
        clean_command, capture_output=True, text=True, timeout=45, env=environment
    )
    assert result.returncode == 0, result.stdout + result.stderr
    from scripts.profile_fold_scheduler import compare_outputs

    assert compare_outputs(clean, recovered, tolerance=1e-9, exact_predictions=False)["passed"]

    def rejected_recovery(label, pattern, extra=()):
        destination = tmp_path / label
        invocation = list(resumed)
        invocation[invocation.index("--results_dir") + 1] = str(destination)
        result = subprocess.run(
            invocation + list(extra), capture_output=True, text=True, timeout=15, env=environment
        )
        assert result.returncode != 0, result.stdout + result.stderr
        assert pattern in result.stdout + result.stderr
        trace = destination / "logs/events.jsonl"
        assert not [
            event
            for event in map(json.loads, trace.read_text().splitlines())
            if event["stage"] == "compute" and event["status"] == "started"
        ]

    rejected_recovery("changed_seed", "fingerprint mismatch", ["--random_seed", "999"])
    genotype = tmp_path / "X.csv"
    original_input = genotype.read_bytes()
    genotype.write_bytes(original_input + b"\n")
    try:
        rejected_recovery("changed_input", "fingerprint mismatch")
    finally:
        genotype.write_bytes(original_input)
    pointer = next((output / ".task_store").glob("tasks/*/current.json"))
    record = json.loads(pointer.read_text())
    payload = pointer.parent / record["commit"] / "result.joblib"
    original_result = payload.read_bytes()
    payload.write_bytes(b"corrupt")
    try:
        rejected_recovery("corrupt_result", "hash mismatch")
    finally:
        payload.write_bytes(original_result)
