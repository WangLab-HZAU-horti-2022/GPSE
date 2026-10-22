#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Cross-run index and aggregates for the v3 results root.

``runs_index.csv`` is appended once per run (fcntl-flocked so parallel
runs do not interleave). ``aggregates/runs_summary.csv`` is rebuilt from
each run's ``run_manifest.json`` — it only collects already-saved
metadata and never computes new statistics.
"""

from __future__ import annotations

import csv
import fcntl
import json
from pathlib import Path
from typing import Any, Optional

INDEX_FIELDS = (
    "run_id",
    "started_at",
    "completed_at",
    "trait",
    "task_type",
    "run_status",
    "selection_status",
    "stacking_status",
    "run_dir",
)


def append_run_index(root_dir: str | Path, manifest: dict[str, Any],
                     run_dir: str | Path) -> Path:
    """Append one row to ``runs_index.csv`` (concurrency-safe)."""
    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    target = root_dir / "runs_index.csv"
    row = {field: manifest.get(field, "") for field in INDEX_FIELDS}
    row["run_id"] = manifest.get("run_id", Path(run_dir).name)
    try:
        row["run_dir"] = str(Path(run_dir).relative_to(root_dir))
    except ValueError:
        row["run_dir"] = str(run_dir)
    with target.open("a+", encoding="utf-8", newline="") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        is_empty = handle.read(1) == ""
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
        if is_empty:
            writer.writeheader()
        writer.writerow(row)
        fcntl.flock(handle, fcntl.LOCK_UN)
    return target


def rebuild_runs_summary(root_dir: str | Path) -> Optional[Path]:
    """Rebuild ``aggregates/runs_summary.csv`` from run manifests.

    Returns ``None`` when the root has no runs yet. Existing manifests
    that cannot be parsed contribute their run id with ``unknown`` fields.
    """
    root_dir = Path(root_dir)
    runs_dir = root_dir / "runs"
    if not runs_dir.exists():
        return None
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted(runs_dir.glob("*/run_manifest.json")):
        run_id = manifest_path.parent.name
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {"run_id": run_id}
        row = {field: manifest.get(field, "unknown") or "unknown"
               for field in INDEX_FIELDS}
        row["run_id"] = manifest.get("run_id", run_id)
        row["run_dir"] = f"runs/{run_id}"
        rows.append(row)
    if not rows:
        return None
    aggregates_dir = root_dir / "aggregates"
    aggregates_dir.mkdir(parents=True, exist_ok=True)
    target = aggregates_dir / "runs_summary.csv"
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return target
