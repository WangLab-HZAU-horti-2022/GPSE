#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Selection summary (v3 output layer).

Builds ``selection/selection_summary.json`` purely by extracting fields
from files the run already produced. It never recomputes TOPSIS, CV
metrics, Pearson, stacking, or predictions, and it never merges metrics
from different evaluation scopes into one number. Conflicting source
values are recorded side by side, never silently resolved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from gpse.output.manifest import atomic_write_json

SCHEMA_VERSION = "1.0"


def _read_json(path: Optional[Path]) -> Optional[dict[str, Any]]:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _load_stacking_results(paths) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Read the persisted stacking results without re-fitting anything."""
    candidates = [
        paths.ensembles_gpse_dir / "stacking_results.pkl",
        paths.run_dir / "ensemble_stacking" / "stacking_results.pkl",  # legacy
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            import joblib

            results = joblib.load(candidate)
            if isinstance(results, dict):
                return results, str(candidate.relative_to(paths.run_dir))
            return None, str(candidate.relative_to(paths.run_dir))
        except Exception:
            return None, str(candidate.relative_to(paths.run_dir))
    return None, None


def write_selection_summary(paths, task_type: str) -> Path:
    """Aggregate existing selection/ensemble/evaluation outputs into one JSON."""
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": paths.run_id,
        "task_type": task_type,
        "sources": {},
        "topsis": None,
        "stacking_selection": None,
        "gpse_stacking": None,
        "holdout_metrics": None,
        "conflicts": [],
        "missing": [],
    }

    # -- TOPSIS decision matrix and ranking (already computed) -----------
    topsis_simple_path = paths.resolve("topsis_simple")
    if topsis_simple_path is not None:
        frame = pd.read_csv(topsis_simple_path)
        summary["topsis"] = {
            "criteria": ["cv_mean", "cv_std_repeat"],
            "weights": [0.8, 0.2],
            "source_file": str(topsis_simple_path.relative_to(paths.run_dir)),
            "models": frame.to_dict(orient="records"),
        }
        summary["sources"]["topsis_simple"] = str(
            topsis_simple_path.relative_to(paths.run_dir)
        )
    else:
        summary["missing"].append("selection/model_comparison_cv_topsis_simple.csv")

    topsis_full_path = paths.resolve("topsis_full")
    if topsis_full_path is not None:
        summary["sources"]["topsis_full"] = str(
            topsis_full_path.relative_to(paths.run_dir)
        )

    # -- Stacking base-model selection (already recorded) ----------------
    selection = _read_json(paths.resolve("stacking_selected_models"))
    if selection is not None:
        summary["stacking_selection"] = selection
        summary["sources"]["stacking_selected_models"] = str(
            paths.resolve("stacking_selected_models").relative_to(paths.run_dir)
        )

    # -- GPSE stacking metrics (already computed at fit time) ------------
    stacking_results, stacking_source = _load_stacking_results(paths)
    if stacking_source is not None:
        summary["sources"]["stacking_results"] = stacking_source
    if stacking_results is not None:
        test_metrics = stacking_results.get("test_metrics") or {}
        summary["gpse_stacking"] = {
            "source_file": stacking_source,
            "selected_models": stacking_results.get("selected_models"),
            "test_metrics": test_metrics,
        }
        # GPSE per-sample holdout predictions are not persisted by the
        # current training pipeline; record the gap instead of refitting.
        summary["missing"].append(
            "gpse_holdout_predictions (per-sample GPSE/stacking hold-out "
            "predictions are not saved by the training pipeline)"
        )

    # -- Holdout metrics table (already computed) ------------------------
    holdout_path = paths.resolve("holdout_comparison")
    if holdout_path is not None:
        frame = pd.read_csv(holdout_path)
        summary["holdout_metrics"] = {
            "source_file": str(holdout_path.relative_to(paths.run_dir)),
            "models": frame.to_dict(orient="records"),
        }
        summary["sources"]["holdout_comparison"] = str(
            holdout_path.relative_to(paths.run_dir)
        )

        # Cross-check stacking metrics coming from two existing sources;
        # keep both values when they disagree.
        if summary["gpse_stacking"] is not None and task_type == "regression":
            pkl_pearson = (summary["gpse_stacking"]["test_metrics"] or {}).get("pearson")
            rows = frame[frame["model"] == "stacking_ensemble"]
            if pkl_pearson is not None and not rows.empty:
                csv_pearson = rows.iloc[0].get("pearson")
                if pd.notna(csv_pearson) and float(csv_pearson) != float(pkl_pearson):
                    summary["conflicts"].append(
                        {
                            "field": "stacking_ensemble pearson",
                            "stacking_results_pkl": float(pkl_pearson),
                            "holdout_comparison_csv": float(csv_pearson),
                        }
                    )
    else:
        summary["missing"].append("evaluation/model_comparison_holdout.csv")

    target = paths.ensure("selection_summary")
    atomic_write_json(target, summary)
    return target
