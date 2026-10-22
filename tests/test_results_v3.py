"""Regression tests for the v3 results layout (output layer only)."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import pandas as pd
import pytest

from gpse.output.index import append_run_index, rebuild_runs_summary
from gpse.output.manifest import (
    finalize_run_manifest,
    read_run_manifest,
    start_run_manifest,
)
from gpse.output.paths import (
    FILE_MAP,
    RunPaths,
    create_run,
    detect_layout,
)
from gpse.output.readme import write_root_readme, write_run_readme
from gpse.output.registry import collect_artifacts, find_artifact
from gpse.output.summary import write_selection_summary


# -- path management --------------------------------------------------------


def test_create_run_with_root_makes_unique_run_dirs(tmp_path):
    first = create_run(trait="height", root=tmp_path)
    second = create_run(trait="height", root=tmp_path)
    assert first.run_dir != second.run_dir
    assert first.run_dir.parent == tmp_path / "runs"
    assert first.run_id.startswith("")
    assert "height" in first.run_id
    for name in ("data", "models", "selection", "evaluation", "reports", "logs"):
        assert (first.run_dir / name).is_dir()
    assert (first.run_dir / "ensembles" / "gpse").is_dir()


def test_create_run_explicit_dir_keeps_single_run_semantics(tmp_path):
    target = tmp_path / "my_run"
    paths = create_run(results_dir=target, trait="y")
    assert paths.run_dir == target
    assert paths.root_dir is None
    assert (target / "selection").is_dir()


def test_create_run_concurrent_never_collides(tmp_path):
    with ThreadPoolExecutor(max_workers=8) as pool:
        runs = list(pool.map(lambda _: create_run(trait="y", root=tmp_path), range(16)))
    run_dirs = {paths.run_dir for paths in runs}
    assert len(run_dirs) == 16
    assert len(list((tmp_path / "runs").iterdir())) == 16


def test_detect_layout_v3_and_legacy(tmp_path):
    v3_dir = tmp_path / "v3"
    (v3_dir / "selection").mkdir(parents=True)
    assert detect_layout(v3_dir) == "v3"

    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    (legacy_dir / "model_comparison_cv.csv").write_text("a\n", encoding="utf-8")
    assert detect_layout(legacy_dir) == "v2"


def test_resolve_falls_back_to_legacy_paths(tmp_path):
    legacy_dir = tmp_path / "run"
    legacy_dir.mkdir()
    (legacy_dir / "split_manifest.json").write_text("{}\n", encoding="utf-8")
    paths = RunPaths(legacy_dir, layout="v2")
    assert paths.resolve("split_manifest") == legacy_dir / "split_manifest.json"
    # v3 location wins when both exist
    (legacy_dir / "data").mkdir()
    (legacy_dir / "data" / "split_manifest.json").write_text("{}\n", encoding="utf-8")
    v3_paths = RunPaths(legacy_dir, layout="v3")
    assert v3_paths.resolve("split_manifest") == legacy_dir / "data" / "split_manifest.json"


def test_every_file_map_entry_has_a_v3_path():
    for logical, (v3_rel, _) in FILE_MAP.items():
        assert v3_rel, logical


# -- manifest / registry ----------------------------------------------------


def test_manifest_start_finalize_roundtrip(tmp_path):
    paths = create_run(results_dir=tmp_path / "run", trait="height")
    start_run_manifest(paths, trait="height", task_type="regression")
    manifest = read_run_manifest(paths)
    assert manifest["run_status"] == "running"
    assert manifest["trait"] == "height"

    finalize_run_manifest(
        paths,
        run_status="completed",
        selection_status="completed",
        stacking_status="not_requested",
        artifact_index=[],
    )
    manifest = read_run_manifest(paths)
    assert manifest["run_status"] == "completed"
    assert manifest["selection_status"] == "completed"
    assert manifest["stacking_status"] == "not_requested"
    assert manifest["completed_at"]


def test_registry_registers_existing_files_and_checksums(tmp_path):
    paths = create_run(results_dir=tmp_path / "run", trait="y")
    paths.ensure("split_manifest").write_text('{"a": 1}\n', encoding="utf-8")
    big = paths.models_dir / "rf"
    big.mkdir(parents=True)
    (big / "fold_1_model.pkl").write_bytes(b"\x00" * 128)

    artifacts = collect_artifacts(paths)
    by_id = {record["artifact_id"]: record for record in artifacts}
    assert "data/split_manifest.json" in by_id
    assert by_id["data/split_manifest.json"]["checksum"]
    assert "models/rf/fold_1_model.pkl" in by_id
    assert by_id["models/rf/fold_1_model.pkl"]["artifact_type"] == "model"

    manifest = {"artifact_index": artifacts}
    found = find_artifact(manifest, paths.run_dir, "data/split_manifest.json")
    assert found == paths.run_dir / "data" / "split_manifest.json"
    assert find_artifact(manifest, paths.run_dir, "nope.csv") is None


# -- selection summary -------------------------------------------------------


def _seed_selection_files(paths: RunPaths, pearson: float = 0.5):
    selection_dir = paths.selection_dir
    selection_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "Model": ["a", "b"],
            "TOPSIS_Score": [0.9, 0.1],
            "TOPSIS_Rank": [1, 2],
        }
    ).to_csv(selection_dir / "model_comparison_cv_topsis_simple.csv", index=False)
    (selection_dir / "stacking_selected_models.json").write_text(
        json.dumps({"selected_models": ["a", "b"], "weights": [0.8, 0.2]}),
        encoding="utf-8",
    )
    ensembles = paths.ensembles_gpse_dir
    ensembles.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"selected_models": ["a", "b"], "test_metrics": {"pearson": pearson}},
        ensembles / "stacking_results.pkl",
    )
    evaluation = paths.evaluation_dir
    evaluation.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"model": ["a", "stacking_ensemble"], "pearson": [0.4, pearson]}
    ).to_csv(evaluation / "model_comparison_holdout.csv", index=False)


def test_selection_summary_aggregates_without_recompute(tmp_path):
    paths = create_run(results_dir=tmp_path / "run", trait="y")
    _seed_selection_files(paths, pearson=0.5)
    target = write_selection_summary(paths, "regression")
    summary = json.loads(target.read_text(encoding="utf-8"))
    assert summary["topsis"]["models"][0]["Model"] == "a"
    assert summary["stacking_selection"]["selected_models"] == ["a", "b"]
    assert summary["gpse_stacking"]["test_metrics"]["pearson"] == 0.5
    assert summary["conflicts"] == []
    # per-sample GPSE predictions are honestly flagged as missing
    assert any("gpse_holdout_predictions" in item for item in summary["missing"])


def test_selection_summary_records_conflicts(tmp_path):
    paths = create_run(results_dir=tmp_path / "run", trait="y")
    _seed_selection_files(paths, pearson=0.5)
    # Diverge the two existing sources: both values must be preserved.
    frame = pd.read_csv(paths.evaluation_dir / "model_comparison_holdout.csv")
    frame.loc[frame["model"] == "stacking_ensemble", "pearson"] = 0.7
    frame.to_csv(paths.evaluation_dir / "model_comparison_holdout.csv", index=False)

    target = write_selection_summary(paths, "regression")
    summary = json.loads(target.read_text(encoding="utf-8"))
    assert len(summary["conflicts"]) == 1
    conflict = summary["conflicts"][0]
    assert conflict["stacking_results_pkl"] == 0.5
    assert conflict["holdout_comparison_csv"] == 0.7


# -- readme / index ----------------------------------------------------------


def test_run_readme_points_at_existing_and_missing_outputs(tmp_path):
    paths = create_run(results_dir=tmp_path / "run", trait="height")
    paths.ensure("topsis_simple").write_text("Model\na\n", encoding="utf-8")
    target = write_run_readme(paths, trait="height", task_type="regression")
    text = target.read_text(encoding="utf-8")
    assert "selection/model_comparison_cv_topsis_simple.csv" in text
    assert "not produced" in text  # stacking model absent


def test_root_readme_and_concurrent_index_appends(tmp_path):
    write_root_readme(tmp_path)
    assert "layout v3" in (tmp_path / "README.md").read_text(encoding="utf-8")

    manifests = []
    for idx in range(6):
        paths = create_run(trait="y", root=tmp_path)
        manifests.append(
            {"run_id": paths.run_id, "run_status": "completed", "run_dir": paths.run_dir}
        )

    def append(entry):
        append_run_index(tmp_path, entry, entry["run_dir"])

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(append, manifests))
    index = pd.read_csv(tmp_path / "runs_index.csv")
    assert len(index) == 6
    assert index["run_id"].nunique() == 6

    # rebuild from on-disk manifests
    for entry in manifests:
        run_dir = entry["run_dir"]
        (run_dir / "run_manifest.json").write_text(
            json.dumps({"run_id": entry["run_id"], "run_status": "completed"}),
            encoding="utf-8",
        )
    target = rebuild_runs_summary(tmp_path)
    summary = pd.read_csv(target)
    assert len(summary) == 6
    assert set(summary["run_status"]) == {"completed"}


# -- downstream consumers ----------------------------------------------------


def test_merge_trait_results_supports_v3_and_legacy(tmp_path):
    from gpse.batch.merge import merge_trait_results

    v3_dir = tmp_path / "v3_run"
    (v3_dir / "selection").mkdir(parents=True)
    pd.DataFrame({"Model": ["a"], "TOPSIS_Score": [0.9]}).to_csv(
        v3_dir / "selection" / "model_comparison_cv_topsis_simple.csv", index=False
    )
    legacy_dir = tmp_path / "legacy_run"
    legacy_dir.mkdir()
    pd.DataFrame({"Model": ["b"], "TOPSIS_Score": [0.8]}).to_csv(
        legacy_dir / "model_comparison_cv_topsis_simple.csv", index=False
    )

    written = merge_trait_results(
        {"new": v3_dir, "old": legacy_dir}, tmp_path / "merged"
    )
    merged = pd.read_csv(tmp_path / "merged" / "model_comparison_cv_topsis_simple.csv")
    assert set(merged["Trait"]) == {"new", "old"}
    assert written


def test_resolve_model_path_v3_layout(tmp_path):
    from gpse.predict.core import resolve_model_path

    run_dir = tmp_path / "run"
    ensemble = run_dir / "models" / "rf_reg" / "deployment_ensemble"
    ensemble.mkdir(parents=True)
    (ensemble / "info.json").write_text("{}\n", encoding="utf-8")
    (ensemble / "member_0.pkl").write_bytes(b"\x00")
    assert resolve_model_path(run_dir) == ensemble

    stacking_dir = tmp_path / "run2"
    stacking_model = (
        stacking_dir / "ensembles" / "gpse" / "stacking_ensemble_model.pkl"
    )
    stacking_model.parent.mkdir(parents=True)
    stacking_model.write_bytes(b"\x00")
    assert resolve_model_path(stacking_dir) == stacking_model
