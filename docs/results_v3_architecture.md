# GPSE Results Architecture v3

> 纯输出层重构。本文档描述 v3 结果目录的结构、路径管理与读取接口。
> 计算逻辑（训练、CV、TOPSIS、Stacking、指标）零改动，见
> `results_v3_acceptance.md` 的 diff 审查结论。

## 1. 目录结构

```text
optimization_results_v3/
├── README.md                     # 根目录说明（v3 布局、与 v2 的关系）
├── runs_index.csv                # 跨 Run 索引（每 run 一行，fcntl 锁并发追加）
├── runs/
│   └── <run_id>/                 # 例：20261022_142215_y_41bb63
│       ├── README.md             # 指路：每类关键结果在哪里、哪些未产生
│       ├── run_manifest.json     # run 元数据 + artifact_index
│       ├── data/
│       │   ├── split_manifest.json
│       │   ├── train_ids.txt / test_ids.txt
│       │   └── cv_folds/<trait>_cv_<r>x<k>.csv
│       ├── models/<model>/
│       │   ├── summary_results.json
│       │   ├── repeat_<i>/{repeat_results,all_predictions}.json
│       │   ├── repeat_<i>/fold_<j>_model.pkl        # --save_models 时
│       │   ├── repeat_<i>/{optuna_best_params.json,cv_train_only.csv}
│       │   ├── deployment_ensemble/                 # info.json + member_*.pkl
│       │   └── representative_model/model.pkl
│       ├── selection/
│       │   ├── cv_fold_metrics.csv / cv_repeat_metrics.csv
│       │   ├── cv_aggregation_summary.json
│       │   ├── model_comparison_cv.csv              # TOPSIS 决策矩阵
│       │   ├── model_comparison_cv_topsis.csv       # TOPSIS 完整结果
│       │   ├── model_comparison_cv_topsis_simple.csv# TOPSIS Score/Rank
│       │   ├── stacking_selected_models.json        # 入选基学习器
│       │   └── selection_summary.json               # 汇总（只提取不重算）
│       ├── ensembles/gpse/
│       │   ├── stacking_ensemble_model.pkl          # 最终 GPSE 模型
│       │   └── stacking_results.pkl                 # GPSE hold-out 指标
│       ├── evaluation/
│       │   ├── model_comparison.csv
│       │   ├── model_comparison_holdout.csv         # 各模型 hold-out 指标
│       │   ├── holdout_metrics.csv / .json
│       │   └── predictions/
│       │       └── topsis_top1_holdout_predictions.csv  # 原 final_predictions.csv
│       ├── reports/
│       │   ├── model_leaderboard.csv
│       │   ├── run_summary.json
│       │   └── training_report.md / .html
│       ├── logs/
│       ├── cv_run_manifest.json  # 复用守卫标记（run 根目录保留）
│       ├── cv_model_failures.json
│       ├── feature_manifest.json # predict 向上遍历查找，必须留在 run 根
│       └── label_encoder.pkl     # 同上（分类任务）
└── aggregates/
    └── runs_summary.csv          # 由各 run_manifest.json 汇总，不算新统计
```

## 2. CLI 语义（关键决策）

- **显式 `--results_dir <dir>`**：语义不变，`<dir>` 就是单次运行目录本身，
  其内部布局升级为 v3。批量/pipeline 调用方不受影响。
- **不传 `--results_dir`**（默认 `None`）：自动创建
  `optimization_results_v3/runs/<run_id>/`，run_id = 时间戳 + 表型 +
  uuid6，并发安全（`mkdir(exist_ok=False)` 分配）。

## 3. 统一路径管理（`gpse/output/`）

| 模块 | 职责 |
| --- | --- |
| `paths.py` | `FILE_MAP`（逻辑名→v3/legacy 相对路径）、`RunPaths`（path/ensure/resolve/subdir）、`create_run`、`detect_layout`、`open_run` |
| `manifest.py` | `run_manifest.json` 原子写（tmp+rename）；start/finalize；失败不影响训练控制流 |
| `registry.py` | Artifact 记录（id/type/relative_path/producer/size/checksum/status）；>64MB 文件 checksum 记 null；不复制模型文件 |
| `summary.py` | `selection/selection_summary.json`：只从已有文件提取汇总，禁止重算；来源冲突双值记录 |
| `readme.py` | 根目录 README + 每 run README（按文件实际存在情况指路） |
| `index.py` | `runs_index.csv`（fcntl.flock 追加）、`aggregates/runs_summary.csv`（扫描 manifest 重建） |

`RunPaths.resolve(logical)` 是 v2/v3 读取兼容的核心：v3 位置优先，
legacy 位置兜底。`detect_layout` 依据 marker 文件判断既有目录布局。

## 4. 新旧路径映射表

| 文件 | v2（旧） | v3（新） |
| --- | --- | --- |
| split_manifest.json | 根 | `data/` |
| train_ids.txt / test_ids.txt | 根 | `data/` |
| cv_folds/ | 根 | `data/cv_folds/` |
| 模型目录 `<model>/` | 根 | `models/<model>/` |
| cv_fold/repeat_metrics.csv、cv_aggregation_summary.json、model_comparison_cv*.csv | 根 | `selection/` |
| stacking_selected_models.json | 根 | `selection/` |
| ensemble_stacking/ | 根 | `ensembles/gpse/`（文件名不变） |
| model_comparison.csv | 根 | `evaluation/` |
| model_comparison_holdout.csv、holdout_metrics.* | `reports/` | `evaluation/` |
| final_predictions.csv | `reports/` | `evaluation/predictions/topsis_top1_holdout_predictions.csv`（更名，语义=TOPSIS rank-1 单模型 hold-out 预测） |
| model_leaderboard.csv、run_summary.json、training_report.* | `reports/` | `reports/`（不变） |
| cv_run_manifest.json、cv_model_failures.json、feature_manifest.json、label_encoder.pkl | 根 | 根（不变，predict 依赖向上遍历） |

## 5. 下游消费者适配

- `gpse/predict/core.py: resolve_model_path`：新增 v3 候选
  `ensembles/gpse/stacking_ensemble_model.pkl`、`models/*/deployment_ensemble`、
  `models/*/representative_model/model.pkl`；v2 glob 全部保留。
- `gpse/batch/merge.py`：`SUMMARY_FILES` 改为 (v3, legacy) 二元组，v3 优先，
  legacy 兜底；合并输出文件名不变。
- `gpse/batch/adaptive_scheduler.py`：读 `reports/run_summary.json`，v3 中
  路径不变，无需修改。
- `gpse/pipeline/cli.py`：显式传 `--results_dir`，语义保留，无需修改。

## 6. 不做什么（边界）

- 不修复 50×5 CV best-repeat 选择问题、不调熵权、不加运行时间准则
  （属于 docs/todo_v3.md 附录 C 的独立任务）。
- 不生成 GPSE 逐样本 hold-out 预测（现有管线未保存；
  `selection_summary.json` 中如实标记 `gpse_holdout_predictions` missing）。
- 不改动 `optimization_results_v2/` 历史数据。
