"""Compare a pre-fold-scheduling source snapshot against the current workspace.

Run with: python -m scripts.profile_fold_scheduler --baseline /tmp/source-snapshot
Measurements run sequentially, alternating version order between repetitions.
"""

from datetime import datetime, timezone
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys

import yaml

from scripts.profile_single_server import ROOT, measure, summarize_events, write_inputs

CASES = {
    "frozen_search": dict(
        samples=700, features=128, models=["rf_reg"], repeats=6, folds=3, trials=6
    ),
    "one_repeat_at_a_time": dict(
        samples=1000,
        features=192,
        models=["rf_reg"],
        repeats=2,
        folds=4,
        trials=6,
        flags=["--repeat_workers", "1"],
    ),
    "default_parameters": dict(
        samples=1000,
        features=192,
        models=["rf_reg"],
        repeats=2,
        folds=4,
        trials=6,
        flags=["--repeat_workers", "1", "--use_default_params"],
    ),
    "tiny": dict(
        samples=80,
        features=24,
        models=["lasso_reg", "kernelridge_reg"],
        repeats=4,
        folds=2,
        trials=2,
        flags=["--use_default_params"],
    ),
}


def compare_outputs(before, after):
    maximum = 0.0

    def compare(a, b, where):
        nonlocal maximum
        if isinstance(a, (float, int)) and not isinstance(a, bool):
            if math.isfinite(a) and math.isfinite(b):
                maximum = max(maximum, abs(a - b))
                assert math.isclose(a, b, abs_tol=1e-12, rel_tol=1e-12), (where, a, b)
            else:
                assert a == b, (where, a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys(), where
            for key in a:
                compare(a[key], b[key], f"{where}/{key}")
        elif isinstance(a, list):
            assert len(a) == len(b), where
            for index, (x, y) in enumerate(zip(a, b)):
                compare(x, y, f"{where}/{index}")
        else:
            assert a == b, (where, a, b)

    for relative in ["data/train_ids.txt", "data/test_ids.txt"]:
        assert (before / relative).read_bytes() == (after / relative).read_bytes(), relative
    compare(
        json.loads((before / "evaluation/holdout_metrics.json").read_text()),
        json.loads((after / "evaluation/holdout_metrics.json").read_text()),
        "holdout",
    )
    for relative in [
        "evaluation/predictions/topsis_top1_holdout_predictions.csv",
        "selection/model_comparison_cv_topsis_simple.csv",
    ]:
        import pandas as pd

        a, b = [pd.read_csv(p / relative) for p in (before, after)]
        columns = [
            c for c in a.columns if not any(s in c.lower() for s in ("time", "path", "version"))
        ]
        compare(a[columns].to_dict("records"), b[columns].to_dict("records"), relative)
    for old in before.glob("models/*/repeat_*/repeat_results.json"):
        relative = old.relative_to(before)
        a, b = json.loads(old.read_text()), json.loads((after / relative).read_text())
        for key in [
            "params",
            "test_indices",
            "ensemble_metrics",
            "phenotype_scaler",
            "optimization_info",
        ]:
            compare(a[key], b[key], f"{relative}/{key}")
        assert len(a["fold_results"]) == len(b["fold_results"])
        for af, bf in zip(a["fold_results"], b["fold_results"]):
            for key in [
                "fold_idx",
                "train_indices",
                "val_indices",
                "split_identifier",
                "train_metrics",
                "val_metrics",
                "test_metrics",
            ]:
                compare(af[key], bf[key], f"{relative}/fold/{key}")
        for name in ["all_predictions.json", "cv_train_only.csv"]:
            assert (before / relative.parent / name).read_bytes() == (
                after / relative.parent / name
            ).read_bytes(), name
    before_histories = sorted(before.glob("logs/optuna/*.jsonl"))
    assert len(before_histories) == len(list(after.glob("logs/optuna/*.jsonl")))
    for history in before_histories:
        paired = after / history.relative_to(before)
        a, b = [
            [json.loads(line) for line in p.read_text().splitlines()] for p in (history, paired)
        ]
        assert len(a) == len(b)
        for x, y in zip(a, b):
            for key in ["model", "repeat", "trial", "state", "params", "metrics"]:
                compare(x[key], y[key], f"{history.name}/{key}")
    return {"passed": True, "max_numeric_absolute_difference": maximum, "tolerance": 1e-12}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("/tmp/gpse-fold-scheduling-profile"))
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    args = parser.parse_args()
    if args.repetitions < 1 or args.threads < 1:
        parser.error("repetitions and threads must be positive")
    args.baseline = args.baseline.resolve()
    if not (args.baseline / "gpse/train/cli.py").is_file():
        parser.error("baseline must contain gpse/train/cli.py")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load((ROOT / "gpse/config/models.yaml").read_text())
    forest = next(m for m in config["models"] if m["name"] == "rf_reg")
    forest["default_params"].update(n_estimators=80, max_depth=8, max_features=0.5)
    forest["search_space"] = [
        {"name": "n_estimators", "type": "categorical", "choices": [60, 80]},
        {"name": "max_depth", "type": "int", "low": 4, "high": 8},
        {"name": "max_features", "type": "fixed", "value": 0.5},
    ]
    config_path = args.output / "models.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    sources = {"before": args.baseline, "after": ROOT}
    state = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "platform": platform.platform(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "threads": args.threads,
        "repetitions": args.repetitions,
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "scipy",
                "pandas",
                "scikit-learn",
                "optuna",
                "threadpoolctl",
            )
        },
        "sources": {
            name: {
                "root": str(root),
                "sha256": {
                    str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted((root / "gpse").rglob("*.py"))
                },
            }
            for name, root in sources.items()
        },
        "cases": {name: CASES[name] for name in args.cases},
        "runs": [],
        "checks": [],
    }
    for case in args.cases:
        spec = CASES[case]
        inputs = args.output / case
        inputs.mkdir()
        write_inputs(inputs, spec)
        for repetition in range(1, args.repetitions + 1):
            modes = ["before", "after"] if repetition % 2 else ["after", "before"]
            for mode in modes:
                directory = inputs / f"{mode}_{repetition}"
                directory.mkdir()
                command = [
                    sys.executable,
                    "-c",
                    "import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); "
                    "runpy.run_module('gpse.train.cli',run_name='__main__')",
                    str(sources[mode]),
                    "--geno_file",
                    str(inputs / "genotype.csv"),
                    "--pheno_file",
                    str(inputs / "phenotype.csv"),
                    "--target_trait",
                    "yield",
                    "--task_type",
                    "regression",
                    "--models",
                    *spec["models"],
                    "--n_repeats",
                    str(spec["repeats"]),
                    "--train_folds",
                    str(spec["folds"]),
                    "--cv_folds",
                    str(spec["folds"]),
                    "--trials",
                    str(spec["trials"]),
                    "--model_config",
                    str(config_path),
                    "--results_dir",
                    str(directory / "results"),
                    "--threads",
                    str(args.threads),
                    "--n_jobs",
                    "1",
                    *spec.get("flags", []),
                ]
                print(f"START {case}/{mode}/{repetition}", flush=True)
                result = measure(command, directory, timeout=300)
                result.update(
                    case=case,
                    mode=mode,
                    repetition=repetition,
                    events=summarize_events(directory, result["started_epoch"]),
                )
                state["runs"].append(result)
                (args.output / "summary.json").write_text(json.dumps(state, indent=2) + "\n")
                print(f"DONE exit={result['exit_code']} wall={result['wall_s']:.2f}s", flush=True)
                assert result["exit_code"] == 0, directory
            check = compare_outputs(
                inputs / f"before_{repetition}/results", inputs / f"after_{repetition}/results"
            )
            state["checks"].append(dict(case=case, repetition=repetition, **check))
            (args.output / "summary.json").write_text(json.dumps(state, indent=2) + "\n")
    state["aggregate"] = []
    for case in args.cases:
        row = {"case": case}
        for mode in sources:
            runs = [r for r in state["runs"] if r["case"] == case and r["mode"] == mode]
            for key in ["wall_s", "cpu_s", "average_cores", "peak_rss_sum_mib"]:
                row[f"{mode}_{key}"] = statistics.mean(r[key] for r in runs)
            row[f"{mode}_optimization_s"] = statistics.mean(
                r["events"]["stage_total_s"].get("optimization", 0) for r in runs
            )
        row["time_reduction_percent"] = 100 * (1 - row["after_wall_s"] / row["before_wall_s"])
        state["aggregate"].append(row)
        print(json.dumps(row), flush=True)
    (args.output / "summary.json").write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    main()
