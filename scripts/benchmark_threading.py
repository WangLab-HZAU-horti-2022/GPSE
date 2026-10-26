"""A/B/C fixed-work or real-HPO benchmark; all runs use independent directories.

python -m scripts.benchmark_threading --output /tmp/gpse-threading-benchmark
A: legacy. B: thread plans with original fold-window refill. C: elastic refill.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import sys

import yaml

from gpse.train._execution_config import ExecutionConfig, ResourceProfile
from gpse.utils.parallel import get_available_cpu_cores
from scripts.profile_fold_scheduler import compare_outputs
from scripts.profile_single_server import ROOT, measure, summarize_events, write_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=4)
    parser.add_argument("--per-fit", type=int, default=2)
    parser.add_argument("--samples", type=int, default=160)
    parser.add_argument("--features", type=int, default=24)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--hpo", action="store_true")
    parser.add_argument("--trials", type=int, default=2)
    args = parser.parse_args()
    if min(args.cpu, args.per_fit, args.repetitions, args.trials) < 1:
        parser.error("CPU, per-fit CPU, repetitions and trials must be positive")
    if args.cpu > get_available_cpu_cores() or args.per_fit > args.cpu:
        parser.error("CPU budget exceeds affinity/quota/job allocation, or per-fit exceeds budget")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    spec = dict(
        samples=args.samples,
        features=args.features,
        models=["rf_reg", "lasso_reg"],
        repeats=2,
        folds=3,
        trials=args.trials,
    )
    write_inputs(output, spec)
    slots = args.cpu // args.per_fit
    plan = ExecutionConfig(
        args.cpu,
        slots,
        8 * 2**30,
        512 * 2**20,
        512 * 2**20,
        {
            name: ResourceProfile(args.per_fit, 256 * 2**20, slots, 1 if name == "rf_reg" else 2)
            for name in spec["models"]
        },
    )
    for mode, scheduler in [("B", "thread_plan"), ("C", "elastic_v1")]:
        path = output / f"{mode}.yaml"
        path.write_text(yaml.safe_dump({**asdict(plan), "scheduler": scheduler}))
    state = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "platform": platform.platform(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "effective_cpu_budget": args.cpu,
        "per_fit_cpu": args.per_fit,
        "workload": spec,
        "hpo": args.hpo,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "scikit-learn", "optuna", "threadpoolctl", "joblib")
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT / "gpse").rglob("*.py"))
        },
        "input_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in output.glob("*.csv")
        },
        "measurement": "Full subprocess wall; wait/rusage CPU including reaped children. "
        "RSS is sampled process-tree sum and can double-count shared pages. "
        "Runs sequential, warmed OS cache, no forced cache eviction; other host load uncontrolled.",
        "runs": [],
        "checks": [],
    }
    for repetition in range(1, args.repetitions + 1):
        for mode in (["A", "B", "C"] if repetition % 2 else ["C", "B", "A"]):
            directory = output / f"{mode}_{repetition}"
            directory.mkdir()
            command = [
                sys.executable,
                "-m",
                "gpse.train.cli",
                "--geno_file",
                str(output / "genotype.csv"),
                "--pheno_file",
                str(output / "phenotype.csv"),
                "--target_trait",
                "yield",
                "--task_type",
                "regression",
                "--models",
                *spec["models"],
                "--n_repeats",
                "2",
                "--train_folds",
                "3",
                "--cv_folds",
                "3",
                "--repeat_workers",
                "1",
                "--max_workers",
                "2",
                "--threads",
                str(args.cpu),
                "--save_models",
                "--no-use_stacking",
                "--results_dir",
                str(directory / "results"),
            ]
            if mode == "A":
                command += ["--scheduler", "legacy", "--n_jobs", str(args.per_fit)]
            else:
                command += ["--execution_config", str(output / f"{mode}.yaml")]
            command += ["--trials", str(args.trials)] if args.hpo else ["--use_default_params"]
            print(f"START {mode}_{repetition}", flush=True)
            result = measure(command, directory, timeout=180)
            result.update(
                mode=mode,
                repetition=repetition,
                normalized_cpu_utilization=result["cpu_s"] / (args.cpu * result["wall_s"]),
                events=summarize_events(directory, result["started_epoch"]),
            )
            state["runs"].append(result)
            (output / "summary.json").write_text(json.dumps(state, indent=2) + "\n")
            print(f"DONE exit={result['exit_code']} wall={result['wall_s']:.3f}s", flush=True)
            if result["exit_code"]:
                raise RuntimeError(f"Benchmark failed; inspect {directory / 'console.log'}")
        for mode in ("B", "C"):
            check = compare_outputs(
                output / f"A_{repetition}/results",
                output / f"{mode}_{repetition}/results",
                tolerance=1e-9,
                exact_predictions=False,
            )
            state["checks"].append(dict(repetition=repetition, mode=mode, **check))
    state["aggregate"] = {}
    for mode in ("A", "B", "C"):
        selected = [run for run in state["runs"] if run["mode"] == mode]
        state["aggregate"][mode] = {
            metric: {
                "median": statistics.median(run[metric] for run in selected),
                "min": min(run[metric] for run in selected),
                "max": max(run[metric] for run in selected),
            }
            for metric in ("wall_s", "cpu_s", "normalized_cpu_utilization", "peak_rss_sum_mib")
        }
    final_sources = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted((ROOT / "gpse").rglob("*.py"))}
    state["source_unchanged_during_measurement"] = final_sources == state["source_sha256"]
    (output / "summary.json").write_text(json.dumps(state, indent=2) + "\n")
    if not state["source_unchanged_during_measurement"]:
        raise RuntimeError("Source changed during benchmark; these measurements are invalid")
    print(json.dumps(state["aggregate"], indent=2))


if __name__ == "__main__":
    main()
