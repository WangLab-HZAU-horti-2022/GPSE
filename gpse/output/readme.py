#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Human-readable README files for the v3 results layout.

Both the results root and every run directory get a README that points
directly at where each key result lives. Content is derived from files
that actually exist, so missing optional outputs are stated as such.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

ROOT_README = """# GPSE Results (layout v3)

Each training run lives in its own directory under `runs/<run_id>/`.
No computation happens in this root directory; it only indexes runs.

- `runs_index.csv` — one row per run (id, trait, status, path).
- `aggregates/runs_summary.csv` — metadata collected from each run's
  `run_manifest.json` (no new statistics are computed here).
- `runs/<run_id>/README.md` — where every result of that run lives.

Historical results produced by GPSE <= v2 stay untouched in
`optimization_results_v2/`; this tree never modifies them.
"""


def write_root_readme(root_dir: str | Path) -> Path:
    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    target = root_dir / "README.md"
    if not target.exists():
        target.write_text(ROOT_README, encoding="utf-8")
    return target


def _line(label: str, path: Optional[Path], run_dir: Path, missing_note: str) -> str:
    if path is not None and path.exists():
        return f"- **{label}**: `{path.relative_to(run_dir)}`"
    return f"- **{label}**: _not produced — {missing_note}_"


def write_run_readme(paths, *, trait: Optional[str] = None,
                     task_type: Optional[str] = None) -> Path:
    """Write ``README.md`` inside one run directory.

    Every pointer is checked against the filesystem; optional results
    that were not produced are listed explicitly.
    """
    run_dir = paths.run_dir
    ensembles = paths.ensembles_gpse_dir
    stacking_model = ensembles / "stacking_ensemble_model.pkl"
    legacy_stacking = run_dir / "ensemble_stacking" / "stacking_ensemble_model.pkl"
    if not stacking_model.exists() and legacy_stacking.exists():
        stacking_model = legacy_stacking

    lines = [
        f"# GPSE Run `{paths.run_id}`",
        "",
        f"- Trait: {trait or 'unknown'}",
        f"- Task type: {task_type or 'unknown'}",
        f"- Layout: results v3 (`data/ models/ selection/ ensembles/ evaluation/ reports/ logs/`)",
        "",
        "## Where each key result lives",
        "",
        _line(
            "TOPSIS scores (CV model ranking)",
            paths.resolve("topsis_simple"),
            run_dir,
            "TOPSIS ranking was not written",
        ),
        _line(
            "TOPSIS decision matrix (full)",
            paths.resolve("topsis_full"),
            run_dir,
            "TOPSIS detail table was not written",
        ),
        _line(
            "Selected base learners for stacking",
            paths.resolve("stacking_selected_models"),
            run_dir,
            "stacking was not requested",
        ),
        _line(
            "GPSE (stacking) hold-out metrics",
            paths.resolve("holdout_comparison"),
            run_dir,
            "hold-out evaluation was not written",
        ),
        (
            f"- **Final GPSE model**: `{stacking_model.relative_to(run_dir)}`"
            if stacking_model.exists()
            else "- **Final GPSE model**: _not produced — stacking was not "
            "requested or failed; see logs/_"
        ),
        _line(
            "TOPSIS rank-1 single-model hold-out predictions",
            paths.resolve("topsis_top1_predictions"),
            run_dir,
            "rank-1 model had no saved fold predictions",
        ),
        _line(
            "Data split manifest (train/test ids)",
            paths.resolve("split_manifest"),
            run_dir,
            "split manifest was not written",
        ),
        _line(
            "Per-model hold-out comparison",
            paths.resolve("model_comparison"),
            run_dir,
            "comparison table was not written",
        ),
        _line(
            "Aggregated selection summary",
            paths.resolve("selection_summary"),
            run_dir,
            "selection summary was not written",
        ),
        "",
        "## Notes",
        "",
        "- GPSE (stacking) per-sample hold-out predictions are not saved by",
        "  the current training pipeline; only aggregate metrics exist.",
        "- `models/<model>/` holds per-model repeats, fold models, the",
        "  deployment ensemble and the representative model.",
        "- `run_manifest.json` indexes every artifact with checksums for",
        "  small files.",
        "",
    ]
    target = paths.ensure("run_readme")
    target.write_text("\n".join(lines), encoding="utf-8")
    return target
