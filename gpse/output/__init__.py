#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Output-layer helpers: results layout, manifests, registries, summaries."""

from gpse.output.paths import (
    FILE_MAP,
    LAYOUT_LEGACY,
    LAYOUT_V3,
    RESULTS_LAYOUT_VERSION,
    V3_ROOT_DIRNAME,
    RunPaths,
    create_run,
    detect_layout,
    generate_run_id,
    open_run,
)
from gpse.output.manifest import (
    finalize_run_manifest,
    read_run_manifest,
    start_run_manifest,
)
from gpse.output.registry import collect_artifacts, find_artifact
from gpse.output.summary import write_selection_summary
from gpse.output.readme import write_root_readme, write_run_readme
from gpse.output.index import append_run_index, rebuild_runs_summary

__all__ = [
    "FILE_MAP",
    "LAYOUT_LEGACY",
    "LAYOUT_V3",
    "RESULTS_LAYOUT_VERSION",
    "V3_ROOT_DIRNAME",
    "RunPaths",
    "create_run",
    "detect_layout",
    "generate_run_id",
    "open_run",
    "start_run_manifest",
    "finalize_run_manifest",
    "read_run_manifest",
    "collect_artifacts",
    "find_artifact",
    "write_selection_summary",
    "write_root_readme",
    "write_run_readme",
    "append_run_index",
    "rebuild_runs_summary",
]
