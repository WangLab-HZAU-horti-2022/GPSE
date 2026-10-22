#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unified output path management for GPSE results (v3 layout).

This module only decides *where* files live. It never computes metrics,
never re-runs any analysis, and never copies model artifacts.

Layout modes
------------
- ``v3``: categorized run directory (data/, models/, selection/, ...).
- ``legacy`` (v2): flat run directory as produced by older GPSE versions.

CLI semantics (see docs/results_v3_io_inventory.md §0)
------------------------------------------------------
- Explicit ``--results_dir`` keeps its historical meaning: it *is* the
  single-run directory. Its internal layout is upgraded to v3.
- When ``--results_dir`` is not supplied, a v3 root directory is used and
  each training run creates a unique ``runs/<run_id>/`` directory inside it.
"""

from __future__ import annotations

import datetime
import re
import uuid
from pathlib import Path
from typing import Optional

V3_ROOT_DIRNAME = "optimization_results_v3"
LEGACY_ROOT_DIRNAME = "optimization_results_v2"
LAYOUT_V3 = "v3"
LAYOUT_LEGACY = "v2"
RESULTS_LAYOUT_VERSION = "3.0"

# logical name -> (v3 relative path, legacy relative path or None)
FILE_MAP = {
    # data
    "split_manifest": ("data/split_manifest.json", "split_manifest.json"),
    "train_ids": ("data/train_ids.txt", "train_ids.txt"),
    "test_ids": ("data/test_ids.txt", "test_ids.txt"),
    # selection (TOPSIS)
    "cv_decision_matrix": (
        "selection/model_comparison_cv.csv",
        "model_comparison_cv.csv",
    ),
    "topsis_full": (
        "selection/model_comparison_cv_topsis.csv",
        "model_comparison_cv_topsis.csv",
    ),
    "topsis_simple": (
        "selection/model_comparison_cv_topsis_simple.csv",
        "model_comparison_cv_topsis_simple.csv",
    ),
    "stacking_selected_models": (
        "selection/stacking_selected_models.json",
        "stacking_selected_models.json",
    ),
    "selection_summary": ("selection/selection_summary.json", None),
    # evaluation
    "model_comparison": ("evaluation/model_comparison.csv", "model_comparison.csv"),
    "holdout_comparison": (
        "evaluation/model_comparison_holdout.csv",
        "reports/model_comparison_holdout.csv",
    ),
    "holdout_metrics_csv": ("evaluation/holdout_metrics.csv", "reports/holdout_metrics.csv"),
    "holdout_metrics_json": ("evaluation/holdout_metrics.json", "reports/holdout_metrics.json"),
    "topsis_top1_predictions": (
        "evaluation/predictions/topsis_top1_holdout_predictions.csv",
        "reports/final_predictions.csv",
    ),
    # reports
    "leaderboard": ("reports/model_leaderboard.csv", "reports/model_leaderboard.csv"),
    "run_summary": ("reports/run_summary.json", "reports/run_summary.json"),
    "training_report_md": ("reports/training_report.md", "reports/training_report.md"),
    "training_report_html": ("reports/training_report.html", "reports/training_report.html"),
    # run-level metadata (v3 only)
    "run_manifest": ("run_manifest.json", None),
    "run_readme": ("README.md", None),
}

_SUBDIRS_V3 = {
    "data": "data",
    "models": "models",
    "selection": "selection",
    "ensembles_gpse": "ensembles/gpse",
    "evaluation": "evaluation",
    "predictions": "evaluation/predictions",
    "reports": "reports",
    "logs": "logs",
}

# Legacy layout: model directories sit at the run root, holdout summaries
# under reports/, everything else flat.
_SUBDIRS_LEGACY = {
    "data": ".",
    "models": ".",
    "selection": ".",
    "ensembles_gpse": "ensemble_stacking",
    "evaluation": ".",
    "predictions": "reports",
    "reports": "reports",
    "logs": "logs",
}

_RUN_ID_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


class RunPaths:
    """Resolve every well-known output location for one training run."""

    def __init__(
        self,
        run_dir: str | Path,
        layout: str = LAYOUT_V3,
        root_dir: Optional[Path] = None,
        run_id: Optional[str] = None,
    ):
        if layout not in (LAYOUT_V3, LAYOUT_LEGACY):
            raise ValueError(f"Unknown results layout: {layout}")
        self.run_dir = Path(run_dir)
        self.layout = layout
        self.root_dir = Path(root_dir) if root_dir else None
        self.run_id = run_id or self.run_dir.name

    # -- logical file access ------------------------------------------------

    def path(self, logical: str) -> Path:
        """Write path for a logical file in this run's layout."""
        v3_rel, legacy_rel = FILE_MAP[logical]
        if self.layout == LAYOUT_V3:
            return self.run_dir / v3_rel
        if legacy_rel is None:
            raise KeyError(f"{logical!r} has no legacy layout path")
        return self.run_dir / legacy_rel

    def ensure(self, logical: str) -> Path:
        """Write path with the parent directory created."""
        target = self.path(logical)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def resolve(self, logical: str) -> Optional[Path]:
        """Existing file for a logical name; v3 location wins, legacy falls back."""
        v3_rel, legacy_rel = FILE_MAP[logical]
        candidate = self.run_dir / v3_rel
        if candidate.exists():
            return candidate
        if legacy_rel is not None:
            candidate = self.run_dir / legacy_rel
            if candidate.exists():
                return candidate
        return None

    # -- categorized subdirectories ----------------------------------------

    def subdir(self, name: str, create: bool = False) -> Path:
        table = _SUBDIRS_V3 if self.layout == LAYOUT_V3 else _SUBDIRS_LEGACY
        target = self.run_dir / table[name]
        if create:
            target.mkdir(parents=True, exist_ok=True)
        return target

    @property
    def data_dir(self) -> Path:
        return self.subdir("data")

    @property
    def models_dir(self) -> Path:
        return self.subdir("models")

    @property
    def selection_dir(self) -> Path:
        return self.subdir("selection")

    @property
    def ensembles_gpse_dir(self) -> Path:
        return self.subdir("ensembles_gpse")

    @property
    def evaluation_dir(self) -> Path:
        return self.subdir("evaluation")

    @property
    def predictions_dir(self) -> Path:
        return self.subdir("predictions")

    @property
    def reports_dir(self) -> Path:
        return self.subdir("reports")

    @property
    def logs_dir(self) -> Path:
        return self.subdir("logs")

    def model_dir(self, model_name: str) -> Path:
        return self.models_dir / model_name

    def ensure_model_dir(self, model_name: str) -> Path:
        target = self.model_dir(model_name)
        target.mkdir(parents=True, exist_ok=True)
        return target

    def ensure_subdirs(self) -> None:
        for name in _SUBDIRS_V3 if self.layout == LAYOUT_V3 else ():
            self.subdir(name, create=True)

    # -- introspection ------------------------------------------------------

    def detect_layout(self) -> str:
        return detect_layout(self.run_dir)

    def relative(self, path: str | Path) -> str:
        return str(Path(path).resolve().relative_to(self.run_dir.resolve()))


def detect_layout(run_dir: str | Path) -> str:
    """Best-effort layout detection for an existing run directory."""
    run_dir = Path(run_dir)
    if (run_dir / "run_manifest.json").exists():
        return LAYOUT_V3
    v3_markers = ("data", "selection", "ensembles", "evaluation", "models")
    legacy_markers = ("model_comparison_cv.csv", "ensemble_stacking", "split_manifest.json")
    if any((run_dir / marker).exists() for marker in v3_markers):
        return LAYOUT_V3
    if any((run_dir / marker).exists() for marker in legacy_markers):
        return LAYOUT_LEGACY
    return LAYOUT_V3


def open_run(run_dir: str | Path) -> RunPaths:
    """Open an existing run directory (v2 or v3) for reading."""
    run_dir = Path(run_dir)
    return RunPaths(run_dir, layout=detect_layout(run_dir))


def generate_run_id(trait: Optional[str] = None) -> str:
    """Unique, concurrency-safe run id: timestamp + optional trait + uuid."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [timestamp]
    if trait:
        cleaned = _RUN_ID_SAFE.sub("_", str(trait)).strip("_")[:40]
        if cleaned:
            parts.append(cleaned)
    parts.append(uuid.uuid4().hex[:6])
    return "_".join(parts)


def create_run(
    results_dir: Optional[str | Path] = None,
    trait: Optional[str] = None,
    root: Optional[str | Path] = None,
) -> RunPaths:
    """Create the run directory for one training run.

    - ``results_dir`` given: it is the run directory itself (historical CLI
      semantics), upgraded to the v3 internal layout.
    - ``results_dir`` omitted: create ``<root>/runs/<run_id>/`` where root
      defaults to ``optimization_results_v3/``.
    """
    if results_dir is not None:
        paths = RunPaths(Path(results_dir), layout=LAYOUT_V3)
        paths.run_dir.mkdir(parents=True, exist_ok=True)
        paths.ensure_subdirs()
        return paths

    root_dir = Path(root) if root else Path(V3_ROOT_DIRNAME)
    runs_dir = root_dir / "runs"
    for _ in range(5):
        run_id = generate_run_id(trait)
        run_dir = runs_dir / run_id
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            continue
    else:  # pragma: no cover - uuid collision is practically impossible
        raise RuntimeError("Could not allocate a unique run directory")
    paths = RunPaths(run_dir, layout=LAYOUT_V3, root_dir=root_dir, run_id=run_id)
    paths.ensure_subdirs()
    return paths
