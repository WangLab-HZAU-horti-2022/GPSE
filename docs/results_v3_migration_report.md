# GPSE Results v3 — 迁移报告

> 配套文档：`results_v3_io_inventory.md`（I/O 调查）、
> `results_v3_architecture.md`（目标架构）、`results_v3_acceptance.md`（验收）。

## 1. 迁移范围

本次只迁移**输出层**：文件保存路径、目录结构、命名、序列化输出调用及
对应读取接口。所有数值计算代码未动（见验收文档的 diff 审查）。

## 2. 修改清单（按文件）

### 新增（纯输出层）

| 文件 | 内容 |
| --- | --- |
| `gpse/output/paths.py` | `FILE_MAP`、`RunPaths`、`create_run`、`detect_layout`、`open_run`、`generate_run_id` |
| `gpse/output/manifest.py` | run_manifest 原子写入、start/finalize/read |
| `gpse/output/registry.py` | artifact 记录与定位（大文件不算 checksum） |
| `gpse/output/summary.py` | `selection/selection_summary.json`（只提取不重算） |
| `gpse/output/readme.py` | 根目录 + 每 run README |
| `gpse/output/index.py` | `runs_index.csv`（flock 追加）、`aggregates/runs_summary.csv` |
| `tests/test_results_v3.py` | 14 个 v3 回归测试 |

### 只改路径/序列化

| 文件 | 改动性质 |
| --- | --- |
| `gpse/train/_pipeline.py` | 三个写文件 helper 改收 `RunPaths`；`export_cv`/TOPSIS/选型/holdout 写点改到 `selection/`、`evaluation/`；Stacking 加 `output_dir=ensembles/gpse`；末尾新增 `_finalize_run_outputs`（best-effort，try/except 吞掉自身异常，不改控制流） |
| `gpse/train/_repeat_training.py` | 2 行：`create_*_result_directory(self.paths.models_dir, ...)` |
| `gpse/train/_cv_manager.py` | 1 行：`cv_folds` 目录改到 `data/` |
| `gpse/train/_results.py` | `write_result_bundle` 加 `paths=None` 参数；三个读文件点走 `paths.resolve`（v3 优先、legacy 兜底） |
| `gpse/utils/genomic_utils.py` | `create_comparison_table` 增加 `mkdir(parents=True)`（输出层） |
| `gpse/train/predictor.py` | 构造函数加 `output_paths=None`，注入/自建 `RunPaths` |
| `gpse/train/cli.py` | `create_run` 接线；run 开始写 manifest（best-effort）；异常时标 `failed` 后原样 re-raise |
| `gpse/train/workflow.py` | `--results_dir` 默认 `optimization_results_v2` → `None`（省略时自动建 `optimization_results_v3/runs/<run_id>/`） |
| `gpse/predict/core.py` | `resolve_model_path` 增加 v3 候选与 glob；v2 路径全部保留 |
| `gpse/batch/merge.py` | `SUMMARY_FILES` 改 (v3, legacy) 二元组，双布局解析 |
| `gpse/mcp/server.py`、`gpse/pipeline/cli.py`、`README.md`、`docs/AGENTS.md` | 文档串/注释同步 |

### 测试更新

- `tests/test_train_pipeline_optimization.py`：holdout 测试改用 `RunPaths`，
  断言路径 `reports/` → `evaluation/`。
- `tests/test_cv_statistics.py`：真实管线测试断言路径改到 `selection/`。

## 3. 语义保留说明

- 显式 `--results_dir` 仍是"单次运行目录本身"，内部布局升 v3；
  批量与 pipeline 调用方零感知。
- `feature_manifest.json` / `label_encoder.pkl` 留在 run 根目录
  （`predict` 靠向上遍历 parents 查找，移动会破坏兼容）。
- `cv_run_manifest.json` / `cv_model_failures.json` 留在 run 根目录
  （复用守卫标记）。
- `final_predictions.csv` 更名为
  `evaluation/predictions/topsis_top1_holdout_predictions.csv`，
  **内容语义不变**：TOPSIS rank-1 单模型的 hold-out 预测，不是 GPSE 集成预测。
- GPSE 逐样本 hold-out 预测当前管线本就不保存；v3 在
  `selection_summary.json` 的 `missing` 字段如实标记，未为此重跑预测。

## 4. 迁移中修复的输出层问题

- `read_run_manifest` 误收文件路径导致根索引跳过的接线 bug（开发期发现即修）。
- `create_comparison_table` 不建目录，v3 下 `evaluation/` 可能不存在 → 补 mkdir。
- manifest `stacking_status` 区分 `not_requested` / `skipped_insufficient_models` /
  `failed` / `completed`，避免单模型运行被误标 failed。

## 5. 未迁移/保留项

- `optimization_results_v2/` 历史数据未动；读取走 legacy 兜底。
- `phenotype_scaler.json`：现管线把 scaler 存进 `data/split_manifest.json`，
  旧文件仅作 predict 的 legacy 兼容读取。
- catboost_info、logs 等运行期目录保持原位置。
