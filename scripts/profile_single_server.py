#!/usr/bin/env python3
"""Reproduce GPSE single-host scheduling measurements without changing training.

Run from the repository root with the GPSE Python environment. All generated
matrices, models, logs, and /proc samples go under --output (default: /tmp).
Cases run sequentially so they do not compete with one another. CPU seconds
come from wait/rusage; /proc samples separately measure the live process tree.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
THREAD_ENV = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def snapshot():
    result = {}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            result[int(directory.name)] = {
                "ppid": int(fields[1]),
                "ticks": int(fields[11]) + int(fields[12]),
                "threads": int(fields[17]),
                "start": int(fields[19]),
                "rss": max(0, int(fields[21])) * os.sysconf("SC_PAGE_SIZE"),
            }
        except (OSError, ValueError, IndexError):
            pass
    return result


def measure(command, directory, timeout):
    environment = {**os.environ, **dict.fromkeys(THREAD_ENV, "1")}
    environment.pop("GPSE_NATIVE_THREADS", None)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started_wall = time.time()
    started = time.monotonic()
    identities, consumed = {}, {}
    samples = []
    timed_out = False
    with (directory / "console.log").open("w") as console:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=console,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                entries = snapshot()
                owned = {process.pid}
                owned.update(
                    pid
                    for pid, start in identities.items()
                    if pid in entries and entries[pid]["start"] == start
                )
                while True:
                    children = {pid for pid, value in entries.items() if value["ppid"] in owned}
                    if children <= owned:
                        break
                    owned.update(children)
                live = []
                for pid in owned:
                    if pid in entries:
                        entry = entries[pid]
                        identities[pid] = entry["start"]
                        consumed[pid, entry["start"]] = entry["ticks"]
                        live.append(entry)
                samples.append(
                    {
                        "elapsed_s": time.monotonic() - started,
                        "cpu_s_sampled": sum(consumed.values()) / os.sysconf("SC_CLK_TCK"),
                        "rss_sum_mib": sum(p["rss"] for p in live) / 2**20,
                        "processes": len(live),
                        "resident_threads": sum(p["threads"] for p in live),
                    }
                )
                if time.monotonic() - started > timeout:
                    timed_out = True
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        # Use only the identities from this benchmark's process tree.
                        from gpse.utils.process_control import owned_processes, terminate_owned

                        terminate_owned(owned_processes([process.pid]))
                        process.kill()
                        process.wait()
                    break
                time.sleep(0.1)
        except BaseException:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                from gpse.utils.process_control import owned_processes, terminate_owned

                terminate_owned(owned_processes([process.pid]))
                process.kill()
                process.wait()
            raise
    elapsed = time.monotonic() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime)
    (directory / "resources.json").write_text(json.dumps(samples, indent=2) + "\n")
    return {
        "command": command,
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "wall_s": elapsed,
        "cpu_s": cpu,
        "average_cores": cpu / elapsed,
        "peak_rss_sum_mib": max((s["rss_sum_mib"] for s in samples), default=0),
        "peak_processes": max((s["processes"] for s in samples), default=0),
        "peak_resident_threads": max((s["resident_threads"] for s in samples), default=0),
        "started_epoch": started_wall,
    }


def summarize_events(directory, started_epoch):
    paths = list(directory.glob("**/logs/events.jsonl"))
    if not paths:
        return {}
    events = [json.loads(line) for line in paths[0].read_text().splitlines()]
    events.sort(key=lambda event: event["timestamp"])
    starts, intervals = {}, []
    for event in events:
        key = (event["stage"], event.get("execution_id"))
        if event["status"] == "started":
            starts[key] = event
        elif key in starts and event["status"] in {"completed", "failed", "pruned", "interrupted"}:
            begin = datetime.fromisoformat(starts[key]["timestamp"]).timestamp()
            end = datetime.fromisoformat(event["timestamp"]).timestamp()
            intervals.append(
                {
                    "stage": event["stage"],
                    "model": event.get("model"),
                    "repeat": event.get("repeat"),
                    "pid": event["pid"],
                    "start_s": begin - started_epoch,
                    "end_s": end - started_epoch,
                }
            )
    (directory / "intervals.json").write_text(json.dumps(intervals, indent=2) + "\n")

    def occupancy(stages):
        selected = [item for item in intervals if item["stage"] in stages]
        edges = sorted(
            [(item["start_s"], 1) for item in selected] + [(item["end_s"], -1) for item in selected]
        )
        active, peak = 0, 0
        for _, delta in edges:
            active += delta
            peak = max(active, peak)
        span = max((i["end_s"] for i in selected), default=0) - min(
            (i["start_s"] for i in selected), default=0
        )
        work = sum(i["end_s"] - i["start_s"] for i in selected)
        return {
            "jobs": len(selected),
            "peak_active": peak,
            "span_s": span,
            "mean_active": work / span if span > 0 else 0,
        }

    pools = [
        event
        for event in events
        if event["stage"] == "scheduler" and event["status"] == "configured"
    ]
    first_repeat = min((i["start_s"] for i in intervals if i["stage"] == "repeat"), default=None)
    worker_pools = [
        event.get("native_pools")
        for event in events
        if event["stage"] == "worker" and event["status"] == "ready"
    ]
    return {
        "scheduler": pools[0] if pools else None,
        "worker_native_pools": worker_pools,
        "first_repeat_start_s": first_repeat,
        "repeats": occupancy({"repeat"}),
        "stacking": occupancy({"stacking_fold", "stacking_refit"}),
        "trial_count": sum(e["stage"] == "trial_history" for e in events),
        "stage_total_s": {
            stage: sum(i["end_s"] - i["start_s"] for i in intervals if i["stage"] == stage)
            for stage in {i["stage"] for i in intervals}
        },
    }


def workloads():
    return {
        "tiny": dict(
            samples=80,
            features=24,
            models=["lasso_reg", "kernelridge_reg"],
            repeats=4,
            folds=2,
            trials=2,
            defaults=True,
        ),
        "forest": dict(
            samples=1000,
            features=192,
            models=["rf_reg"],
            repeats=8,
            folds=3,
            trials=6,
            defaults=True,
        ),
        "frozen": dict(
            samples=700,
            features=128,
            models=["rf_reg"],
            repeats=6,
            folds=3,
            trials=6,
            defaults=False,
        ),
        "mixed": dict(
            samples=700,
            features=128,
            models=["rf_reg", "gbdt_reg", "lightgbm_reg"],
            repeats=4,
            folds=3,
            trials=4,
            defaults=False,
        ),
        "svm": dict(
            samples=800,
            features=128,
            models=["svm_clf"],
            repeats=4,
            folds=3,
            trials=2,
            defaults=True,
            classification=True,
        ),
    }


def settings():
    return {
        "serial": ["--n_jobs", "1", "--max_workers", "1", "--repeat_workers", "1"],
        "shared1": ["--threads", "1", "--n_jobs", "1"],
        "shared2": ["--threads", "2", "--n_jobs", "1"],
        "shared4": ["--threads", "4", "--n_jobs", "1"],
        "shared8": ["--threads", "8", "--n_jobs", "1"],
        "shared16": ["--threads", "16", "--n_jobs", "1"],
        "auto": ["--threads", "auto"],
        "inner4": ["--threads", "4", "--n_jobs", "4"],
        "hybrid8": ["--threads", "8", "--n_jobs", "2"],
        "save4": ["--threads", "4", "--n_jobs", "1", "--save_models"],
        "cap1": ["--threads", "4", "--n_jobs", "1", "--repeat_workers", "1"],
        "manual_repeats4": ["--n_jobs", "1", "--max_workers", "1", "--repeat_workers", "4"],
    }


def write_inputs(directory, spec):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(20261023)
    # Synthetic dosage matrix with a reproducible, nonconstant signal.
    X = pd.DataFrame(
        rng.binomial(2, 0.35, size=(spec["samples"], spec["features"])),
        index=[f"sample_{i}" for i in range(spec["samples"])],
        columns=[f"snp_{i}" for i in range(spec["features"])],
        dtype=float,
    )
    signal_value = X.iloc[:, 0] * 3 - X.iloc[:, 1] * 2 + X.iloc[:, 2] + rng.normal(size=len(X))
    y = (
        (signal_value > signal_value.median()).astype(int)
        if spec.get("classification")
        else signal_value
    )
    X.to_csv(directory / "genotype.csv", index_label="ID")
    pd.DataFrame({"yield": y}, index=X.index).to_csv(directory / "phenotype.csv", index_label="ID")


def audit_fit_counts(output, config_path):
    """Count real estimator fits on the inline API path, without timing claims."""
    import cProfile
    import pstats
    from gpse.train.predictor import GenomicPredictorV2

    directory = output / "fit_counts"
    directory.mkdir()
    write_inputs(directory, dict(samples=80, features=24))
    records = []
    for frozen in (False, True):
        for saved in (False, True):
            run_dir = directory / f"frozen_{frozen}_saved_{saved}"
            predictor = GenomicPredictorV2(
                results_dir=str(run_dir),
                model_config=str(config_path),
                random_seed=42,
                n_trials=2,
                n_threads=1,
                max_parallel_jobs=1,
                repeat_workers=1,
                n_repeats=4,
                n_splits=2,
                cpu_budget=None,
                use_default_params=not frozen,
                save_models=saved,
            )
            profiler = cProfile.Profile()
            profiler.runcall(
                predictor.run_all_models,
                geno_file=str(directory / "genotype.csv"),
                pheno_file=str(directory / "phenotype.csv"),
                target_trait="yield",
                models=["rf_reg", "lasso_reg"],
                use_stacking=True,
                cv_folds=2,
            )
            profiler.dump_stats(str(run_dir / "fit_counts.prof"))
            counts = [
                {"file": key[0], "line": key[1], "function": key[2], "calls": value[1]}
                for key, value in pstats.Stats(profiler).stats.items()
                if key[0].endswith("/ensemble/_forest.py") and key[2] == "fit"
            ]
            events = [
                json.loads(line)
                for line in (run_dir / "logs/events.jsonl").read_text().splitlines()
            ]
            # Invalid trials may stop before all folds: count attempted search
            # folds, including a fit whose predictions caused TrialPruned.
            search_fits = sum(
                event["stage"] == "optimization_fold"
                and event["status"] == "started"
                and event["model"] == "rf_reg"
                for event in events
            )
            records.append(
                {
                    "frozen_optuna": frozen,
                    "save_models": saved,
                    "search_fold_attempts": search_fits,
                    # Persistence must not add a second round of CV fitting.
                    "expected_fits": 8 + 1 + 2 + 3 + search_fits,
                    "actual_fits": sum(item["calls"] for item in counts),
                    "profile_entries": counts,
                }
            )
    (output / "fit_counts.json").write_text(json.dumps(records, indent=2) + "\n")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/gpse-single-server-profile"))
    parser.add_argument(
        "--source-root",
        type=Path,
        default=ROOT,
        help="Source snapshot containing gpse/, for before/after comparisons",
    )
    parser.add_argument(
        "--workloads", nargs="+", choices=list(workloads()), default=list(workloads())
    )
    parser.add_argument(
        "--settings",
        nargs="+",
        choices=list(settings()),
        default=["serial", "shared1", "shared4", "inner4", "save4"],
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument(
        "--audit-fit-counts",
        action="store_true",
        help="Also profile actual forest fit counts on the serial Python API",
    )
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    args.source_root = args.source_root.resolve()
    if not (args.source_root / "gpse/train/cli.py").is_file():
        parser.error("--source-root must contain gpse/train/cli.py")
    if args.audit_fit_counts and args.source_root != ROOT:
        parser.error("--audit-fit-counts requires the current workspace source")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(dict.fromkeys(THREAD_ENV, "1"))
    import yaml
    from gpse.utils.parallel import get_available_cpu_cores

    # Bound tree counts/depths. Named search-space overrides merge with the
    # packaged search space, retaining its other forest parameters.
    registry = yaml.safe_load((args.source_root / "gpse/config/models.yaml").read_text())
    forest = next(entry for entry in registry["models"] if entry["name"] == "rf_reg")
    forest["default_params"].update(n_estimators=80, max_depth=8, max_features=0.5)
    forest["search_space"] = [
        {"name": "n_estimators", "type": "categorical", "choices": [60, 80]},
        {"name": "max_depth", "type": "int", "low": 4, "high": 8},
        {"name": "max_features", "type": "fixed", "value": 0.5},
    ]
    config_path = args.output / "models.yaml"
    config_path.write_text(yaml.safe_dump(registry, sort_keys=False))
    state = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "source_root": str(args.source_root),
        "platform": platform.platform(),
        "available_cpus": get_available_cpu_cores(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "load_at_start": os.getloadavg(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in [
                "numpy",
                "scipy",
                "pandas",
                "scikit-learn",
                "optuna",
                "joblib",
                "threadpoolctl",
            ]
        },
        "source_sha256": {
            str(path.relative_to(args.source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((args.source_root / "gpse").rglob("*.py"))
        },
        "workloads": {name: workloads()[name] for name in args.workloads},
        "settings": {name: settings()[name] for name in args.settings},
        "runs": [],
    }
    for workload in args.workloads:
        spec = workloads()[workload]
        inputs = args.output / workload
        inputs.mkdir()
        write_inputs(inputs, spec)
        for repetition in range(1, args.repetitions + 1):
            # Reverse every other repetition to expose order/thermal effects.
            modes = args.settings if repetition % 2 else list(reversed(args.settings))
            for mode in modes:
                directory = inputs / f"{mode}_{repetition}"
                directory.mkdir()
                command = [
                    sys.executable,
                    "-c",
                    "import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); "
                    "runpy.run_module('gpse.train.cli',run_name='__main__')",
                    str(args.source_root),
                    "--geno_file",
                    str(inputs / "genotype.csv"),
                    "--pheno_file",
                    str(inputs / "phenotype.csv"),
                    "--target_trait",
                    "yield",
                    "--task_type",
                    "classification" if spec.get("classification") else "regression",
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
                    *settings()[mode],
                ]
                if spec["defaults"]:
                    command.append("--use_default_params")
                if spec.get("classification"):
                    command.extend(["--n_classes", "2"])
                print(f"START {workload}/{mode}/{repetition}", flush=True)
                result = measure(command, directory, args.timeout)
                result.update(
                    workload=workload,
                    setting=mode,
                    repetition=repetition,
                    events=summarize_events(directory, result["started_epoch"]),
                )
                state["runs"].append(result)
                (args.output / "summary.json").write_text(json.dumps(state, indent=2) + "\n")
                print(
                    f"DONE exit={result['exit_code']} wall={result['wall_s']:.2f}s "
                    f"avg_cores={result['average_cores']:.2f} "
                    f"rss_sum={result['peak_rss_sum_mib']:.0f}MiB",
                    flush=True,
                )
    if args.audit_fit_counts:
        state["fit_counts"] = audit_fit_counts(args.output, config_path)
        (args.output / "summary.json").write_text(json.dumps(state, indent=2) + "\n")
    return int(any(run["exit_code"] for run in state["runs"]))


if __name__ == "__main__":
    raise SystemExit(main())
