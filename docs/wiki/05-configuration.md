# 05. Configuration

## Project-level `gpse.yaml`

Placed in the working directory (or `gpse.local.yaml`, or given explicitly via `--config PATH`); declares external tools and convert settings:

```yaml
external_tools:
  - name: plink
    cmd: plink            # absolute paths are allowed
  - name: java
    cmd: java
    required: false
convert:
  beagle_jar_path: /path/to/beagle.jar   # only used with --impute
```

Loading logic (`gpse/utils/configuration.py`): deep-merged in the order **package defaults → project `gpse.yaml` → `gpse.local.yaml` → CLI `--config`**; named lists are overridden by `name`. `--no-project-config` disables automatic project config loading.

## Package Configs

| File | Contents |
|------|----------|
| `gpse/config/software.yaml` | Software metadata + external tool declarations (plink: `--version` probe, min_version 1.9, required; java: optional) |
| `gpse/config/default.yaml` | Software metadata + logging config (log_level, more_info, Label) |
| `gpse/config/topsis.yaml` | TOPSIS multi-criteria model ranking configuration (see below) |
| `gpse/config/models.yaml` | Model registry: all available ML models with import paths, thread strategies, default params, and Optuna search spaces (see below) |

## TOPSIS Configuration (`topsis.yaml`)

Formal selection uses **training-only aggregated repeated CV**, with exactly two
criteria and fixed manual weights `[0.8, 0.2]`. Holdout metrics cannot participate.
The default protocol is 50×5; stability uses `ddof=1`, not population SD.

```yaml
tasks:
  regression:  # same configuration for classification (validation accuracy)
    criteria:
      - {name: cv_mean, type: max, weight: 0.8}
      - {name: cv_std_repeat, type: min, weight: 0.2}
```

Use `--cv_stability_metric cv_std_within` or `cv_std_all_folds` to explicitly
change stability definition. If supplying `--topsis_config`, its two active
criteria must agree with that flag and retain the fixed weights. Legacy test-set
criteria, extra active criteria and changed weights are rejected before training.

All repeats and folds must be complete and validation scores finite. Zero SD uses
`-log(SD + 1e-12)` followed by L2 normalization. Constant criteria contribute no
separation; if all weighted criteria are constant, scores are 0.5 and ranks tie.
Alphabetical order resolves tied output order for deterministic Top-N selection.
Invalid inputs stop TOPSIS; no alternate ranking is silently substituted.

`cv_std_repeat` describes sensitivity to repeat splits on overlapping samples;
it is neither a standard error nor a generalization confidence interval.
See [CV audit, recovery and formulas](../cv-statistics.md).

## Model Registry (`models.yaml`)

All available ML models are declared in `gpse/config/models.yaml`. This YAML-driven registry eliminates the need to modify Python code when adding simple models — just add a YAML entry.

### Schema

```yaml
models:
  - name: my_model_reg          # unique key (used with --models)
    task: regression             # regression | classification
    import_path: sklearn.ensemble.BaggingRegressor  # lazy-imported at creation
    thread_strategy: n_jobs      # none | n_jobs | nthread | thread_count | [list]
    default_params:              # fallback hyperparams (--use_default_params)
      n_estimators: 100
      random_state: "{random_seed}"
    search_space:                # inline Optuna DSL (mutually exclusive with param_func)
      - {name: n_estimators, type: int, low: 10, high: 500}
      - {name: random_state, type: fixed, value: "{random_seed}"}
```

### Field Reference

| Field | Required | Description |
|-------|----------|-------------|
| `name` | yes | Unique model key |
| `task` | yes | `regression` or `classification` |
| `import_path` | yes | Fully-qualified class path (e.g. `sklearn.ensemble.RandomForestRegressor`) |
| `thread_strategy` | yes | How to inject thread count: `none`, `n_jobs`, `nthread`, `thread_count`, or a list |
| `default_params` | yes | Fallback params; supports `{random_seed}`, `{n_threads}`, `{n_classes}` placeholders |
| `search_space` | conditional | Inline Optuna DSL (for flat search spaces) |
| `param_func` | conditional | Python reference for complex search spaces: `"builtin:_xgboost_reg_params"` or `"my_module:my_func"` |
| `extra` | no | Special-handling flags (e.g. `catboost_train_dir: true`) |

### Search Space DSL

For models with flat hyperparameter spaces (no conditional branching), declare the search space inline:

```yaml
search_space:
  - {name: alpha, type: float, low: 0.001, high: 100.0, log: true}
  - {name: n_estimators, type: int, low: 10, high: 500, step: 10}
  - {name: criterion, type: categorical, choices: [gini, entropy]}
  - {name: random_state, type: fixed, value: "{random_seed}"}
```

| Type | Fields | Maps to |
|------|--------|---------|
| `int` | `low`, `high`, `step?`, `log?` | `trial.suggest_int(...)` |
| `float` | `low`, `high`, `log?` | `trial.suggest_float(...)` |
| `categorical` | `choices` | `trial.suggest_categorical(...)` |
| `fixed` | `value` | constant (supports placeholders) |

For models with conditional logic (e.g. XGBoost booster branching, MLP dynamic layers), use `param_func` to reference a Python function:

```yaml
param_func: "builtin:_xgboost_reg_params"   # built-in method
param_func: "my_package.search_spaces:my_func"  # external module
```

### Adding a New Model (Zero Code)

Create a file `my_models.yaml` with just the new model entry:

```yaml
models:
  - name: bagging_reg
    task: regression
    import_path: sklearn.ensemble.BaggingRegressor
    thread_strategy: n_jobs
    default_params:
      n_estimators: 100
      max_samples: 1.0
      random_state: "{random_seed}"
    search_space:
      - {name: n_estimators, type: int, low: 10, high: 500}
      - {name: max_samples, type: float, low: 0.5, high: 1.0}
      - {name: max_features, type: float, low: 0.5, high: 1.0}
      - {name: random_state, type: fixed, value: "{random_seed}"}
```

Then run:

```bash
gpse train \
    --geno_file data/genotype.csv \
    --pheno_file data/phenotype.csv \
    --target_trait Fruit_Weight \
    --task_type regression \
    --model_config my_models.yaml \
    --models bagging_reg
```

The user file is **deep-merged** with the built-in registry: entries with matching `name` keys are merged field-by-field, new entries are appended. You can override just one field of an existing model without re-declaring the rest.

### Built-in Models

**Regression** (15): `elasticnet_reg`, `lasso_reg`, `sgd_reg`, `svr_reg`, `adaboost_reg`, `knn_reg`, `rf_reg`, `lightgbm_reg`, `histgradientboost_reg`, `gblup_reg`, `gbdt_reg`, `xgboost_reg`, `catboost_reg`, `kernelridge_reg`, `mlp_reg`

**Classification** (6): `rf_clf`, `svm_clf`, `xgboost_clf`, `lightgbm_clf`, `catboost_clf`, `mlp_clf`

## Batch Config (`gpse batch`)

`gpse batch` reads its own YAML config, separate from `gpse.yaml`. It has two sections: `defaults` (any `gpse train` option plus the batch-only `results_root`) and `traits` (each entry requires `name` = target trait column and may override any option, or be skipped with `enabled: false`). See [03. `gpse train` → Batch Multi-Trait Training](03-cli-train.md#batch-multi-trait-training-gpse-batch) for the full schema. A ready-to-edit template lives at `batch/batch_config.example.yaml`; real-world validation configs are under `tests/validation/*_batch_config.yaml`.

## Manifest Files in Training Artifacts

- `feature_manifest.json`: canonical SNP ID list of the training features, used by `gpse predict` for alignment (`gpse/utils/feature_manifest.py`);
- `split_manifest.json`: hold-out / CV split record, guaranteeing reproducibility.
