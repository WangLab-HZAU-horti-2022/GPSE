#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Artifact registry (v3 output layer).

Registers files a run already produced. Records reference real file
paths only — large model files are never copied, and their checksum is
skipped (``None``) above a size threshold to keep registration cheap.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Optional

CHECKSUM_MAX_BYTES = 64 * 1024 * 1024

# Well-known logical files and who writes them.
_PRODUCERS = {
    "split_manifest": "train._pipeline._save_split_manifest",
    "train_ids": "train._pipeline._save_split_manifest",
    "test_ids": "train._pipeline._save_split_manifest",
    "cv_decision_matrix": "train._cv_statistics.export_cv",
    "topsis_full": "train.topsis.TOPSISEvaluator",
    "topsis_simple": "train.topsis.TOPSISEvaluator",
    "stacking_selected_models": "train._pipeline.run_all_models",
    "selection_summary": "output.summary.write_selection_summary",
    "model_comparison": "utils.genomic_utils.create_comparison_table",
    "holdout_comparison": "train._pipeline._write_holdout_reports",
    "holdout_metrics_csv": "train._pipeline._write_holdout_reports",
    "holdout_metrics_json": "train._pipeline._write_holdout_reports",
    "topsis_top1_predictions": "train._pipeline._write_topsis_final_predictions",
    "leaderboard": "train._results.write_result_bundle",
    "run_summary": "train._results.write_result_bundle",
    "training_report_md": "train._results.write_result_bundle",
    "training_report_html": "train._results.write_result_bundle",
    "run_manifest": "output.manifest",
    "run_readme": "output.readme",
}

_ARTIFACT_TYPES = {
    "split_manifest": "data_manifest",
    "train_ids": "data_manifest",
    "test_ids": "data_manifest",
    "cv_decision_matrix": "selection_table",
    "topsis_full": "selection_table",
    "topsis_simple": "selection_table",
    "stacking_selected_models": "selection_record",
    "selection_summary": "selection_record",
    "model_comparison": "evaluation_table",
    "holdout_comparison": "evaluation_table",
    "holdout_metrics_csv": "evaluation_table",
    "holdout_metrics_json": "evaluation_table",
    "topsis_top1_predictions": "predictions",
    "leaderboard": "report",
    "run_summary": "report",
    "training_report_md": "report",
    "training_report_html": "report",
    "run_manifest": "run_metadata",
    "run_readme": "report",
}


def _checksum(path: Path) -> Optional[str]:
    try:
        if path.stat().st_size > CHECKSUM_MAX_BYTES:
            return None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _record(run_dir: Path, path: Path, artifact_type: str, producer: str,
            source_reference: Optional[str] = None) -> dict[str, Any]:
    relative = str(path.relative_to(run_dir))
    exists = path.exists()
    return {
        "artifact_id": relative,
        "artifact_type": artifact_type,
        "relative_path": relative,
        "producer": producer,
        "source_reference": source_reference,
        "file_size": path.stat().st_size if exists else None,
        "checksum": _checksum(path) if exists else None,
        "status": "present" if exists else "missing",
    }


def collect_artifacts(paths) -> list[dict[str, Any]]:
    """Register every artifact this run produced (existing files only).

    Covers the well-known logical files plus model/ensemble files found
    on disk. Never raises for a missing file — it is simply skipped.
    """
    from gpse.output.paths import FILE_MAP

    run_dir = paths.run_dir
    records: list[dict[str, Any]] = []
    seen: set[str] = set()

    for logical in FILE_MAP:
        try:
            target = paths.path(logical)
        except KeyError:
            continue
        if not target.exists():
            continue
        records.append(
            _record(
                run_dir,
                target,
                _ARTIFACT_TYPES.get(logical, "other"),
                _PRODUCERS.get(logical, "unknown"),
            )
        )
        seen.add(str(target))

    # Model and ensemble artifacts discovered on disk (never copied).
    discoveries = [
        (paths.logs_dir, "execution_log", "utils.training_monitor / utils.log_utils"),
        (paths.models_dir, "model", "train._repeat_training / train._ensemble"),
        (paths.ensembles_gpse_dir, "ensemble", "train.stacking.StackingEnsemble"),
    ]
    for base, artifact_type, producer in discoveries:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or str(path) in seen:
                continue
            records.append(_record(run_dir, path, artifact_type, producer))

    # Run-root operational files kept outside the categorized layout.
    for name in (
        "cv_run_manifest.json",
        "cv_model_failures.json",
        "feature_manifest.json",
        "label_encoder.pkl",
    ):
        path = run_dir / name
        if path.exists() and str(path) not in seen:
            records.append(
                _record(run_dir, path, "run_metadata", "train pipeline")
            )

    return records


def find_artifact(
    manifest: dict[str, Any], run_dir: str | Path, selector: str
) -> Optional[Path]:
    """Locate an artifact by id or relative path from a manifest dict."""
    run_dir = Path(run_dir)
    for record in manifest.get("artifact_index", []):
        if selector in (record.get("artifact_id"), record.get("relative_path")):
            if record.get("status") != "present":
                return None
            candidate = run_dir / record["relative_path"]
            return candidate if candidate.exists() else None
    return None
