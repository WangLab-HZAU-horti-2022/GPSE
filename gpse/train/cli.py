#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Command-line interface for the GPSE train module.

Defines the `gpse train` argument parser and runs the training workflow.
The top-level `gpse.cli` module only routes to this module.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

# Bootstrap before NumPy/scipy imports; per-job threadpool_limits enforce the resolved budget.
_bootstrap_parser = argparse.ArgumentParser(add_help=False)
_bootstrap_parser.add_argument("--n_jobs", type=int, default=1)
_bootstrap_args, _ = _bootstrap_parser.parse_known_args()
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_variable] = os.environ.get("GPSE_NATIVE_THREADS", str(max(1, _bootstrap_args.n_jobs)))
del _bootstrap_parser, _bootstrap_args, _variable

from gpse.convert.processor import GenomicDataProcessor
from gpse.train.predictor import GenomicPredictorV2, main_logger
from gpse.train.workflow import (
    _build_parser,
    _configure_logging,
    _log_stage,
    _show_version,
)
from gpse.utils.parallel import resolve_cpu_budget, validate_parallelism
from gpse.utils.process_control import interruptible_entry

try:
    from rich_argparse import RichHelpFormatter
except ImportError:  # pragma: no cover
    RichHelpFormatter = argparse.HelpFormatter


def _short_path(value: object) -> str:
    if value is None:
        return "None"
    text = str(value)
    cwd = os.getcwd()
    home = os.path.expanduser("~")
    if text.startswith(cwd):
        return "..." + text[len(cwd):]
    if text.startswith(home):
        return "~" + text[len(home):]
    return text


def _write_config_to_log(args) -> None:
    ns = vars(args)
    main_logger.debug("=== GPSE Configuration ===")
    for key in sorted(ns):
        value = ns[key]
        if value is not None:
            main_logger.debug(f"{key}: {value}")


def _log_config(args) -> None:
    """Pretty-print the active GPSE configuration."""
    summary = (
        f"GPSE | {args.target_trait} ({args.task_type}) | "
        f"models={', '.join(args.models) if args.models else 'all'} | "
        f"training CV={args.n_repeats} repeats × {args.n_splits} folds | "
        f"Holdout={args.test_size:g} | Stacking={args.use_stacking}\n"
        f"n_jobs={args.n_jobs}, model_workers={args.max_workers}, repeat_workers={args.repeat_workers}"
    )
    main_logger.bind(console_summary=True).info(summary)


def _resolve_model_count(args: argparse.Namespace) -> int:
    """Return the number of models that will be trained for the current task."""
    if args.models:
        return len(args.models)
    model_config = getattr(args, "model_config", None)
    if args.task_type == "regression":
        from gpse.models.regression_model_optimizer import RegressionModelOptimizer

        return len(RegressionModelOptimizer(
            random_seed=42, n_threads=1, model_config_path=model_config
        ).model_configs)
    # Classification
    from gpse.models.classification_model_optimizer import ClassificationModelOptimizer

    return len(
        ClassificationModelOptimizer(
            random_seed=42, n_threads=1, n_classes=args.n_classes or 2,
            model_config_path=model_config,
        ).model_configs
    )


def _apply_threads_budget(args: argparse.Namespace, *, log: bool = True) -> None:
    """One budget for the shared pool; explicit legacy knobs remain respected."""
    explicit = getattr(args, "_explicit_parallel", set())
    if args.fold_workers is not None and args.fold_workers < 1:
        raise ValueError("--fold_workers must be a positive integer")
    args.cpu_budget = None
    if args.threads is None and explicit and args.fold_workers in {None, 1}:
        return
    budget = resolve_cpu_budget(args.threads)
    count = _resolve_model_count(args)
    folds = min(args.n_splits, args.fold_workers or args.n_splits)
    model_slots = min(count, args.max_workers) if "max_workers" in explicit else count
    repeats = min(args.n_repeats, args.repeat_workers) if "repeat_workers" in explicit else args.n_repeats
    if "n_jobs" not in explicit:
        args.n_jobs = max(1, budget // max(1, min(budget, model_slots * repeats * folds)))
    if min(args.n_jobs, args.max_workers, args.repeat_workers) < 1 or args.n_jobs > budget:
        raise ValueError("Worker counts must be positive and --n_jobs must not exceed the CPU budget")
    slots = max(1, budget // args.n_jobs)
    if "max_workers" not in explicit:
        args.max_workers = min(count, slots)
    if "repeat_workers" not in explicit:
        args.repeat_workers = min(args.n_repeats, slots)
    args.cpu_budget = budget
    if log:
        tasks = min(slots, min(count, args.max_workers) * min(args.n_repeats, args.repeat_workers) * folds)
        main_logger.bind(console_summary=True).info(
            f"CPU budget={budget}; shared pool ceiling={tasks} training tasks × "
            f"{args.n_jobs} native thread(s). Idle slots are reused across models. "
            "Models with min_threads use more native threads per fit; the shared pool "
            "caps total threads at the budget. "
            "Actual CPU usage depends on the estimator and available independent work."
        )


@interruptible_entry
def main(
    argv: list[str] | None = None,
    *,
    formatter_class=None,
    prog: str | None = None,
    help_action=None,
    parents: list[argparse.ArgumentParser] | None = None,
) -> int:
    """Run the GPSE training CLI."""
    raw_args = list(sys.argv[1:] if argv is None else argv)

    parser = _build_parser(
        formatter_class=formatter_class or RichHelpFormatter,
        prog=prog or "python -m gpse.train.cli",
        help_action=help_action,
        parents=parents,
    )
    if not raw_args:
        parser.error("train requires arguments. Use 'gpse train -h' for help.")

    # Version is a standalone flag: handle it before full parsing so required
    # training arguments (e.g. --task_type) are not enforced for `-v`.
    if raw_args[0] in {"-v", "--version"}:
        _show_version()
        return 0

    args = parser.parse_args(raw_args)

    try:
        if args.version:
            _show_version()
            return 0

        _configure_logging(args.log_level, args.verbosity)

        if args.models:
            unique = list(dict.fromkeys(args.models))
            if len(unique) != len(args.models):
                main_logger.warning("Duplicate model names removed; each model will execute once")
            args.models = unique
        _apply_threads_budget(args)

        try:
            if args.cpu_budget is None:
                args.n_jobs, args.max_workers, args.repeat_workers = validate_parallelism(
                    args.n_jobs,
                    args.max_workers,
                    args.repeat_workers,
                    logger=main_logger,
                )
        except ValueError:
            return 1

        if args.preprocess_only:
            if not args.preprocess_prefix:
                main_logger.error("--preprocess_prefix is required when --preprocess_only is set")
                return 1
            if args.standardize_phenotype:
                main_logger.error(
                    "--standardize_phenotype cannot be used with --preprocess_only because "
                    "its parameters must be fitted on the hold-out training partition"
                )
                return 1
        elif args.enable_preprocess:
            if not args.preprocess_prefix:
                main_logger.error("--preprocess_prefix is required when --enable_preprocess is set")
                return 1
            if not args.target_trait:
                main_logger.error("--target_trait is required for training")
                return 1
        else:
            if not args.geno_file or not args.pheno_file:
                main_logger.error("--geno_file and --pheno_file are required in normal training mode")
                return 1
            if not args.target_trait:
                main_logger.error("--target_trait is required for training")
                return 1

        main_logger.info(
            f"Stacking ensemble: {'enabled' if args.use_stacking else 'disabled'}"
        )
        _log_config(args)

        processed_geno_file = args.geno_file
        processed_pheno_file = args.pheno_file

        if args.enable_preprocess or args.preprocess_only:
            _log_stage("Starting data preprocessing")

            processor = GenomicDataProcessor(logger=main_logger, plink_path=args.plink_path)
            preprocess_kwargs = {
                "out_prefix": args.preprocess_prefix,
                "vcf": args.vcf_file,
                "bfile": args.bfile,
                "ped_file": args.ped_file,
                "map_file": args.map_file,
                "extract": args.extract_file,
                "snp_dir": args.snp_dir,
                "direct": args.direct_convert,
                "matrix_file": args.matrix_file,
                "pheno": args.raw_pheno_file,
                "plink_out": args.plink_out,
                "preserve_vcf_snp_ids": args.preserve_vcf_snp_ids,
                "skip_matrix": args.skip_matrix_conversion,
                "skip_match": args.skip_phenotype_match,
                "skip_clean": args.skip_data_clean,
                "load": args.load_matrix_info,
                "trait_name": args.target_trait,
                "standardize_phenotype": False,
                "threads": args.cpu_budget or (args.n_jobs * args.max_workers * args.repeat_workers),
            }

            if args.standardize_phenotype:
                main_logger.info(
                    "Deferring phenotype standardization until after hold-out splitting; "
                    "preprocessing will preserve raw phenotype values"
                )
            if args.preserve_vcf_snp_ids:
                main_logger.warning(
                    "Compatibility mode enabled: preserving VCF default SNP IDs. "
                    "Use --preserve-vcf-snp-ids when predicting from a VCF."
                )

            try:
                result = processor.process_genomic_data(**preprocess_kwargs)
                if result != 0:
                    main_logger.error("Data preprocessing failed")
                    return result

                main_logger.info("Data preprocessing completed successfully!")
                if not args.skip_phenotype_match and args.raw_pheno_file:
                    for ext in [".parquet", ".feather", ".csv"]:
                        test_file = f"{args.preprocess_prefix}_genotype{ext}"
                        if os.path.exists(test_file):
                            processed_geno_file = test_file
                            break
                    else:
                        processed_geno_file = f"{args.preprocess_prefix}_genotype.csv"
                    processed_pheno_file = f"{args.preprocess_prefix}_phenotype.csv"
                else:
                    for ext in [".parquet", ".feather", ".csv"]:
                        test_file = args.preprocess_prefix + ext
                        if os.path.exists(test_file):
                            processed_geno_file = test_file
                            break
                    else:
                        processed_geno_file = args.preprocess_prefix + ".csv"
                    processed_pheno_file = args.raw_pheno_file


            except Exception as e:
                main_logger.error(f"Error during data preprocessing: {e}")
                main_logger.error(traceback.format_exc())
                return 1

        if args.preprocess_only:
            main_logger.info("Preprocessing finished; exiting without training")
            return 0

        if not os.path.exists(processed_geno_file):
            main_logger.error(f"Genotype file not found: {processed_geno_file}")
            return 1
        if not os.path.exists(processed_pheno_file):
            main_logger.error(f"Phenotype file not found: {processed_pheno_file}")
            return 1

        # ---- Validate task_type / n_classes (user-specified, no auto-detect) ----
        if args.task_type == "classification":
            if args.n_classes is None:
                main_logger.error(
                    "--n_classes is required when --task_type=classification"
                )
                return 1
            if args.n_classes < 2:
                main_logger.error("--n_classes must be >= 2 for classification")
                return 1
        elif args.n_classes is not None:
            main_logger.warning("--n_classes is ignored for regression tasks")

        _log_stage("Starting model training")
        main_logger.info(f"Genotype file: {processed_geno_file}")
        main_logger.info(f"Phenotype file: {processed_pheno_file}")
        if not args.preserve_vcf_snp_ids:
            manifest_path = os.path.splitext(processed_geno_file)[0] + ".features.json"
            if os.path.exists(manifest_path):
                try:
                    import json
                    with open(manifest_path, encoding="utf-8") as manifest_handle:
                        feature_id_mode = json.load(manifest_handle).get("feature_id_mode")
                except (OSError, ValueError):
                    feature_id_mode = None
                if feature_id_mode == "vcf":
                    main_logger.warning(
                        "Training with VCF default SNP IDs detected in the input feature manifest. "
                        "Keep --preserve-vcf-snp-ids for matching VCF inputs during prediction."
                    )

        training_standardize = args.standardize_phenotype

        from gpse.output.paths import create_run

        output_paths = create_run(results_dir=args.results_dir, trait=args.target_trait)
        try:
            from gpse.output.manifest import start_run_manifest
            from gpse.output.readme import write_root_readme

            if output_paths.root_dir is not None:
                write_root_readme(output_paths.root_dir)
            start_run_manifest(
                output_paths,
                trait=args.target_trait,
                task_type=args.task_type,
            )
        except Exception as manifest_exc:  # metadata must never break training
            main_logger.warning(f"Run manifest initialization skipped: {manifest_exc}")

        predictor = GenomicPredictorV2(
            random_seed=args.random_seed,
            results_dir=str(output_paths.run_dir),
            output_paths=output_paths,
            n_trials=args.trials,
            n_threads=args.n_jobs,
            cpu_budget=args.cpu_budget or (args.n_jobs * args.max_workers * args.repeat_workers),
            max_parallel_jobs=args.max_workers,
            repeat_workers=args.repeat_workers,
            fold_workers=args.fold_workers,
            test_size=args.test_size,
            n_splits=args.n_splits,
            n_repeats=args.n_repeats,
            cv_stability_metric=args.cv_stability_metric,
            patience=args.patience,
            use_default_params=args.use_default_params,
            save_models=args.save_models,
            save_representative=args.save_representative,
            cv_file=args.cv_file,
            force_new_cv=args.force_new_cv,
            cv_id_column=args.cv_id_column,
            task_type=args.task_type,
            n_classes=args.n_classes,
            standardize_phenotype=training_standardize,
            cv_stability_penalty=args.cv_stability_penalty,
            optuna_per_repeat=args.optuna_per_repeat,
            split_strategy=args.split_strategy,
            structure_clusters=args.structure_clusters,
            feature_selection=args.feature_selection,
            select_k=args.select_k,
            variance_threshold=args.variance_threshold,
            select_percentile=args.select_percentile,
            genotype_imputation=args.genotype_imputation,
            missing_genotype_code=args.missing_genotype_code,
            topsis_config=args.topsis_config,
            model_config=args.model_config,
            feature_id_mode="vcf" if args.preserve_vcf_snp_ids else None,
        )

        predictor.cli_config = {key: sorted(value) if isinstance(value, set) else value
                                for key, value in vars(args).items()}
        _write_config_to_log(args)
        predictor.console_verbosity = args.verbosity
        predictor.progress = args.progress
        main_logger.bind(console_summary=True).info(
            f"Run: {output_paths.run_id} | Output: {_short_path(output_paths.run_dir)} | "
            "Logs: logs/ (DEBUG text + events.jsonl)"
        )
        try:
            predictor.run_all_models(
                geno_file=processed_geno_file,
                pheno_file=processed_pheno_file,
                target_trait=args.target_trait,
                models=args.models,
                use_stacking=args.use_stacking,
                top_n_models=args.top_n_models,
                cv_folds=args.cv_folds,
                use_same_test_set=args.use_same_test_set,
            )
        except BaseException:
            try:
                from gpse.output.manifest import finalize_run_manifest

                finalize_run_manifest(output_paths, run_status="failed")
            except Exception:
                pass
            raise
        return 0
    except ValueError as exc:
        main_logger.error(str(exc))
        return 1
    except KeyboardInterrupt:
        main_logger.warning("Training interrupted by user (Ctrl+C).")
        return 130
    finally:
        from gpse.utils.log_utils import set_training_verbosity
        set_training_verbosity(None)


if __name__ == "__main__":
    sys.exit(main())
