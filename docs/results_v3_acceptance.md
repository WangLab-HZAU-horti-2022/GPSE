# GPSE Results v3 — 验收报告

日期：2026-10-22

## 1. 验收结论

**通过。** 在计算、训练、TOPSIS、Stacking 与指标评估行为完全不变的前提下，
所有输出自动进入 `optimization_results_v3/` 下分类、独立、可定位、可追溯的
run 目录；v2 历史目录保持可读、不被修改。

## 2. 测试结果

### 全量测试

```
197 passed, 2 failed（末次运行）
```

失败均为**既有问题，与本次改动无关**（失败文件均不在本次修改范围内）：

| 失败测试 | 原因 | 证据 |
| --- | --- | --- |
| `test_audit_fixes.py::test_threadpoolctl_declared_in_pyproject` | pyproject.toml 缺 threadpoolctl 声明 | 本次未改 pyproject.toml |
| `test_audit_fixes.py::test_find_result_file_depth_limit` | `_find_result_file` 深度限制在外部重构中被移除 | 本次对 predict/core.py 的 diff 只含 `resolve_model_path` |

另：`test_convert_perf.py::test_multi_trait_genotype_files_are_hardlinked`
为时序敏感的性能测试，多次运行结果不稳定（flake），本次未改 gpse/convert/。

### 新增 v3 测试（tests/test_results_v3.py，14 项全过）

- 路径：默认 v3 根 + 独立 run 目录；显式目录语义保留；16 并发创建无冲突；
  v3/legacy 布局检测；resolve legacy 兜底与 v3 优先。
- 文件：merge 同时支持 v3/legacy 布局；predict 的 `resolve_model_path`
  识别 v3 的 `models/*/deployment_ensemble` 与 `ensembles/gpse/`。
- Manifest/Registry：start→finalize 往返、原子写、checksum、find_artifact。
- Selection Summary：只提取不重算；stacking pkl 与 holdout csv 冲突时双值记录；
  GPSE 逐样本预测如实标 missing。
- README/Index：指路内容按文件实际存在生成；flock 并发追加 6 行无交错；
  aggregates 从 manifest 重建。

### 端到端冒烟

1. **真实管线（pytest 内）**：`test_real_pipeline_selection_stacking_and_training_only_refit`
   在 v3 布局下完成 CV→聚合→TOPSIS→OOF stacking→两种 refit，全部断言通过。
2. **CLI 默认目录**：`gpse train`（不传 `--results_dir`）自动创建
   `optimization_results_v3/runs/<run_id>/`，根 README、run README、
   run_manifest、selection_summary、runs_index.csv、aggregates/runs_summary.csv
   全部生成。
3. **CLI 失败路径**：模型名错误导致全模型失败时，`run_manifest.json` 正确
   标记 `run_status=failed`，异常按原语义传播。

## 3. 数值一致性

固定数据 + 固定种子（合成 60×8，Ridge 双模型，3 repeats × 2 folds，
stacking 开启）运行两次独立 run：

| 校验项 | 结果 |
| --- | --- |
| TOPSIS_Score（逐模型） | 两次完全一致（DataFrame assert_frame_equal） |
| TOPSIS Rank / 入选模型 | 一致（`["a", "b"]`） |
| stacking_selected_models.json | 字节级一致 |
| holdout 指标（含 stacking_ensemble pearson） | 完全一致 |
| manifest 状态 | completed / completed / completed，33-51 个 artifact 登记 |

另：既有测试 `test_cv_statistics.py` 中"同种子两次运行 CV 指标逐项相等"
断言在 v3 布局下继续通过。

## 4. Git diff 静态审查结论

逐文件审查 `git diff` 后确认：

**只改路径/序列化的文件**：`_pipeline.py`、`_repeat_training.py`（2 行）、
`_cv_manager.py`（1 行）、`_results.py`、`predictor.py`（output_paths 注入）、
`cli.py`、`workflow.py`、`predict/core.py`、`batch/merge.py`、
`genomic_utils.py`（仅 `create_comparison_table` 补 mkdir）。

**新增纯元数据文件**：`gpse/output/` 全部、`tests/test_results_v3.py`。

**文档串**：README.md、docs/AGENTS.md、mcp/server.py docstring、
pipeline/cli.py 注释。

**计算逻辑改动：无。** 本任务未触碰任何模型训练、数学计算、指标聚合或
超参数相关代码。

**注意（非本任务改动）**：工作区在进入本任务前已存在外部重构（如
`_cv_statistics.py` 聚合、`aggregate_cv` 接入、NaN 指标策略、
`exist_ok=False`、holdout 训练分区 refit 等）。这些改动与本任务无关，
其 diff 与我的改动在同一批文件中并存，审查时已逐块区分；本验收只对
"输出层改动不引入计算差异"负责。

## 5. 已知限制 / 后续建议

1. GPSE 逐样本 hold-out 预测仍未持久化（如实标记 missing）；若论文需要
   GPSE 逐样本预测，需在后续方法学任务中新增（属计算层，不在本次范围）。
2. TOPSIS 熵权法/多准则可配置化按计划 2026-10-23 在独立任务中进行
   （docs/todo_v3.md 附录 C），本次未动。
3. 3 个既有失败测试建议另行处理（pyproject 声明、`_find_result_file`
   深度限制、convert hardlink）。
4. `species` 字段当前恒为 `unknown`——CLI 无此概念；若 batch 层后续提供，
   可在 `start_run_manifest` 传入。
