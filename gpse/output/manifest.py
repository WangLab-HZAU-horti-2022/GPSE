#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run manifest writer (v3 output layer).

The manifest is pure metadata: it records *that* a run happened, which
layout it used, and where its artifacts are. It never computes metrics
and never influences training control flow. All writers are atomic
(tmp file + rename) and all callers are expected to treat manifest
failures as non-fatal.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from gpse.output.paths import RESULTS_LAYOUT_VERSION

SCHEMA_VERSION = "1.0"
UNKNOWN = "unknown"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically so readers never see a truncated file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
            handle.write("\n")
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _code_version() -> str:
    try:
        from importlib.metadata import version

        return version("gpse")
    except Exception:
        return UNKNOWN


def start_run_manifest(
    paths,
    *,
    trait: Optional[str] = None,
    task_type: Optional[str] = None,
    species: Optional[str] = None,
    config_reference: Optional[dict[str, Any]] = None,
) -> Path:
    """Create ``run_manifest.json`` at the start of a run (status=running)."""
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "results_layout": RESULTS_LAYOUT_VERSION,
        "layout_mode": paths.layout,
        "run_id": paths.run_id,
        "started_at": _utc_now(),
        "completed_at": None,
        "species": species or UNKNOWN,
        "trait": trait or UNKNOWN,
        "task_type": task_type or UNKNOWN,
        "code_version": _code_version(),
        "config_reference": config_reference or {},
        "run_status": "running",
        "selection_status": UNKNOWN,
        "stacking_status": UNKNOWN,
        "artifact_index": [],
    }
    target = paths.ensure("run_manifest")
    atomic_write_json(target, manifest)
    return target


def read_run_manifest(paths_or_dir) -> Optional[dict[str, Any]]:
    """Read an existing manifest; ``None`` when absent or unreadable."""
    run_dir = getattr(paths_or_dir, "run_dir", None)
    run_dir = Path(run_dir) if run_dir else Path(paths_or_dir)
    target = run_dir / "run_manifest.json"
    if not target.exists():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def finalize_run_manifest(
    paths,
    *,
    run_status: str = "completed",
    selection_status: Optional[str] = None,
    stacking_status: Optional[str] = None,
    artifact_index: Optional[list[dict[str, Any]]] = None,
) -> Optional[Path]:
    """Update the manifest at the end of a run.

    Missing pieces keep their previous value or ``unknown``; the writer
    never raises for a missing earlier manifest (it simply starts one).
    """
    existing = read_run_manifest(paths) or {}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "results_layout": RESULTS_LAYOUT_VERSION,
        "layout_mode": paths.layout,
        "run_id": paths.run_id,
        "started_at": existing.get("started_at") or UNKNOWN,
        "completed_at": _utc_now(),
        "species": existing.get("species", UNKNOWN),
        "trait": existing.get("trait", UNKNOWN),
        "task_type": existing.get("task_type", UNKNOWN),
        "code_version": existing.get("code_version", UNKNOWN),
        "config_reference": existing.get("config_reference", {}),
        "run_status": run_status,
        "selection_status": selection_status
        or existing.get("selection_status", UNKNOWN),
        "stacking_status": stacking_status
        or existing.get("stacking_status", UNKNOWN),
        "artifact_index": artifact_index
        if artifact_index is not None
        else existing.get("artifact_index", []),
    }
    target = paths.ensure("run_manifest")
    atomic_write_json(target, manifest)
    return target
