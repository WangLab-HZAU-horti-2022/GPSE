# GPSE Results v3 — I/O 清单（阶段 0 源码调查）

> 日期：2026-10-22
> 方法：对 `gpse/`（train/predict/batch/mcp/pipeline/utils）、`tests/`、`scripts/`、`README.md` 全量 grep + 关键文件逐行阅读。所有路径以当前源码为准。
> 范围：仅输出层。本文件是 v3 重构的映射依据。

---

## 0. `--results_dir` 语义调查结论（关键）

当前 `--results_dir`（`gpse/train/workflow.py:158-161`，默认 `"optimization_results_v2"`）的语义是**单次运行目录本身**：所有产物（模型子目录、TOPSIS csv、reports/、logs/）直接写进该目录，没有 run 级嵌套。

调用方现状：

| 调用方 | 行为 | 位置 |
|---|---|---|
| `gpse train` CLI | 直接透传给 `GenomicPredictorV2(results_dir=...)` | `gpse/train/cli.py`、`gpse/train/predictor.py:240` |
| `gpse pipeline` | 用户未提供时自动生成 `{out_prefix}_results`（避免并发互相覆盖） | `gpse/pipeline/cli.py:167-175` |
| batch adaptive_scheduler | 每性状一个独立 `results_dir`，用 `reports/run_summary.json` 判断完成 | `gpse/batch/adaptive_scheduler.py:247-250` |
| `gpse-mcp` | 透传 `results_dir`，docstring 写默认值 | `gpse/mcp/server.py:368-380, 514-554` |

**v3 语义决策**：显式 `--results_dir` 保持"单次运行目录"语义不变（仅内部布局升级为 v3 分类目录）；未提供时（argparse 默认改为 `None`）启用 v3 根目录模式，自动创建 `optimization_results_v3/runs/<run_id>/`。此决策不改变任何显式调用的行为。

---

## 1. 产物清单（生产者/消费者/时机/依赖/映射）

### A. 数据划分文件

| 文件 | 生产者 | 消费者 | 写时机 | 读时机 | V3 映射 | 兼容风险 |
|---|---|---|---|---|---|---|
| `split_manifest.json` | `_pipeline.py:43-63` `_save_split_manifest` | `_pipeline.py:167`（topsis_top1 预测反标准化）、`_results.py:229`（run_summary） | holdout 划分后，训练前 | 运行内 + bundle 生成时 | `data/split_manifest.json` | 低：两处读者同步改 |
| `train_ids.txt` / `test_ids.txt` | `_pipeline.py:57-62` | 无代码消费者（审计用） | 同上 | — | `data/` | 无 |
| `cv_folds/{trait}_cv_{R}x{K}.csv` | `_cv_manager.py:61-79`（`prepare_cv_data`） | `_cv_manager.py:65-68`（重复运行复用） | 首次运行生成 | 后续运行开头 | `data/cv_folds/` | 中：复用逻辑按固定相对路径找，需同步 |
| `{model}/repeat_{i}/cv_train_only.csv` | `_cv_manager.py:223-234` | 无（审计用） | 每 repeat | — | `models/{model}/repeat_{i}/`（随模型目录走） | 无 |

### B. 单模型结果（`create_model_result_directory`，`genomic_utils.py:410-420`：`results_dir/{model}`）

| 文件 | 生产者 | 消费者 | V3 映射 | 兼容风险 |
|---|---|---|---|---|
| `{model}/summary_results.json` | `_repeat_training.py:615-616` | 无代码消费者（figure 数据包离线取数） | `models/{model}/` | 低 |
| `{model}/repeat_{i}/fold_{k}_model.pkl` | `_fold_training.py:174` | `_ensemble.py:136-143`（折集成加载）；predict 间接 | 随模型目录 | 低：读者用 repeat_dir 变量，随动 |
| `{model}/repeat_{i}/repeat_results.json` | `_repeat_training.py:359-361` | 无 | 随模型目录 | 无 |
| `{model}/repeat_{i}/all_predictions.json` | `_repeat_training.py:313-315` | `_pipeline.py:139-148`（topsis_top1 预测，**按 `results_dir/{model}/repeat_{i}` 拼路径**） | 随模型目录 | **中：此读取点需改为 models/ 下** |
| `{model}/repeat_{i}/optuna_best_params.json` | `_repeat_training.py:279-280` | 无 | 随模型目录 | 无 |
| `{model}/deployment_ensemble/member_*.pkl` + `info.json` | `_ensemble.py:32-101` | `predict/core.py:41,51`（glob `*/deployment_ensemble`） | `models/{model}/deployment_ensemble/` | **高：predict glob 深度从 1 级变 2 级，必须加 v3 候选** |
| `{model}/representative_model/model.pkl` | `_save_representative_model` | `predict/core.py:43,52` | `models/{model}/representative_model/` | 同上 |

### C. TOPSIS / 选型

| 文件 | 生产者 | 消费者 | V3 映射 | 兼容风险 |
|---|---|---|---|---|
| `model_comparison_cv.csv` | `_pipeline.py:357-360` | `call_topsis_evaluator` 输入（同函数内）；`batch/merge.py:24` | `selection/` | 中：merge 需双布局解析 |
| `model_comparison_cv_topsis.csv` | `topsis.py:155` | 无 | `selection/` | 低 |
| `model_comparison_cv_topsis_simple.csv` | `topsis.py:163` | `_pipeline.py:372`（选型读回）、`_pipeline.py:127-135`（topsis_top1 预测读回） | `selection/` | 低：两处读者同步改 |
| `stacking_selected_models.json` | `_pipeline.py:383-393` | `_results.py:230`（run_summary） | `selection/` | 低 |

### D. Stacking（GPSE）

| 文件 | 生产者 | 消费者 | V3 映射 | 兼容风险 |
|---|---|---|---|---|
| `ensemble_stacking/stacking_ensemble_model.pkl` | `stacking.py:186` | `predict/core.py:42,53`（直接候选 + glob `*/ensemble_stacking/`） | `ensembles/gpse/` | **高：predict 两处路径都必须加 v3 候选** |
| `ensemble_stacking/stacking_results.pkl` | `stacking.py:188` | 无代码消费者 | `ensembles/gpse/` | 低 |
| 序列化内容路径引用检查 | — | — | — | 已核实：`stacking.py` dump 的是对象本身 + results dict，**不含路径引用**；`output_dir` 由构造参数传入，不参与 unpickle |

### E. 评估

| 文件 | 生产者 | 消费者 | V3 映射 | 兼容风险 |
|---|---|---|---|---|
| `model_comparison.csv` | `genomic_utils.py:230-231`（`_pipeline.py:341` 集成前 + `:468` 集成后**覆盖重写**） | `batch/merge.py:23`；`tests/test_batch_runner.py` | `evaluation/` | 中：merge 双布局 |
| `reports/model_comparison_holdout.csv` | `_pipeline.py:100-106`（按 split_strategy 累积追加） | `batch/merge.py:27` | `evaluation/` | 中 |
| `reports/holdout_metrics.csv` / `.json` | `_pipeline.py:107-109` | `batch/merge.py:28` | `evaluation/` | 中 |
| `reports/final_predictions.csv`（⚠️ 实为 TOPSIS rank-1 单模型预测；注意当前在 `reports/` 下而非根目录） | `_pipeline.py:115-189`（`:175`） | 无代码消费者 | `evaluation/predictions/topsis_top1_holdout_predictions.csv` | 低：仅改名+移动；语义不变 |

### F. 报告

| 文件 | 生产者 | 消费者 | V3 映射 | 兼容风险 |
|---|---|---|---|---|
| `reports/model_leaderboard.csv` | `_results.py:225-226` | 无 | `reports/`（不变） | 无 |
| `reports/run_summary.json` | `_results.py:249-252` | `batch/adaptive_scheduler.py:250`（完成判定） | `reports/`（不变） | **无：调度器依赖此路径，v3 保持不变** |
| `reports/training_report.md` / `.html` | `_results.py:253-256` | 无 | `reports/`（不变） | 无 |

### G. 日志

| 文件 | 生产者 | 消费者 | V3 映射 | 兼容风险 |
|---|---|---|---|---|
| `logs/` 主日志 + 子进程日志 | `predictor.py:243-256`、`_repeat_training.py:637-642` | 无 | `logs/`（不变） | 无 |

### H. 其他提及

| 位置 | 内容 | 处理 |
|---|---|---|
| `README.md:875` | "The default result directory is `optimization_results_v2/`" | 文档更新为 v3 |
| `scripts/genomic_prediction_v2.py:1604` | 旧脚本默认值 | 不动（旧脚本冻结） |
| `gpse/mcp/server.py:539` | docstring 默认值 | 文档更新 |
| `tests/test_batch_runner.py:215-286` | merge 测试 fixture 用 legacy 布局 | merge 双布局解析后测试应保持通过，另补 v3 布局测试 |
| `tests/test_cv_statistics.py:244-263` | 自造 tmp 路径直接调 evaluator | 不受影响 |
| `tests/test_train_pipeline_optimization.py:262` | 读 `reports/model_comparison_holdout.csv` | **需更新为 evaluation/ 路径或经兼容层** |

---

## 2. 需要同步修改的读取点汇总（实施清单依据）

**运行内读回（路径随写点同步）：**
1. `_pipeline.py:372` 读 topsis_simple（选型）
2. `_pipeline.py:127` 读 topsis_simple（topsis_top1 预测）
3. `_pipeline.py:139-140` 读 `{model}/repeat_{i}/all_predictions.json` → `models/`
4. `_pipeline.py:167` 读 `split_manifest.json` → `data/`
5. `_results.py:229-230` 读 `split_manifest.json` / `stacking_selected_models.json`
6. `_cv_manager.py:61-68` cv_folds 目录 → `data/cv_folds/`
7. `_ensemble.py:139` 读 fold pkl（repeat_dir 变量传递，随 `create_*_directory` 自动迁移）

**跨进程/外部消费者（需 legacy+v3 双布局解析）：**
8. `predict/core.py:40-53` `resolve_model_path`（deployment_ensemble / representative_model / stacking glob）
9. `batch/merge.py:22-29` SUMMARY_FILES 相对路径
10. `batch/adaptive_scheduler.py:250`（路径不变，仅验证）
11. `tests/test_train_pipeline_optimization.py:262`

**拼接模型目录的写点：**
12. `genomic_utils.py:410-420` `create_model_result_directory` / `create_repeat_result_directory`（`results_dir/{model}` → `results_dir/models/{model}`）
13. `stacking.py` 构造参数 `base_models_dir`/`output_dir`（`_pipeline.py` 调用处）→ `ensembles/gpse/`

---

## 3. 布局映射总表（V2 → V3）

```text
optimization_results_v2/                          optimization_results_v3/  （默认根）
                                                     ├── README.md
                                                     ├── runs_index.csv
                                                     ├── aggregates/runs_summary.csv
                                                     └── runs/<run_id>/
<run_dir>/                                             ├── README.md
├── split_manifest.json                          →     ├── run_manifest.json
├── train_ids.txt / test_ids.txt                 →     ├── data/{split_manifest.json,train_ids.txt,test_ids.txt}
├── cv_folds/                                    →     ├── data/cv_folds/
├── <model>/...                                  →     ├── models/<model>/...
├── model_comparison_cv.csv                    →     ├── selection/model_comparison_cv.csv
├── model_comparison_cv_topsis.csv             →     ├── selection/model_comparison_cv_topsis.csv
├── model_comparison_cv_topsis_simple.csv      →     ├── selection/model_comparison_cv_topsis_simple.csv
├── stacking_selected_models.json              →     ├── selection/stacking_selected_models.json
│                                              →     ├── selection/selection_summary.json   (新增)
├── ensemble_stacking/                         →     ├── ensembles/gpse/
├── model_comparison.csv                       →     ├── evaluation/model_comparison.csv
├── reports/model_comparison_holdout.csv       →     ├── evaluation/model_comparison_holdout.csv
├── reports/holdout_metrics.{csv,json}         →     ├── evaluation/holdout_metrics.{csv,json}
├── reports/final_predictions.csv              →     ├── evaluation/predictions/topsis_top1_holdout_predictions.csv
├── reports/{leaderboard,run_summary,report}   →     ├── reports/（不变）
└── logs/                                      →     └── logs/（不变）
```

---

*调查人：Kimi Code；2026-10-22。下一阶段（P1）据此实现 `gpse/output/paths.py`。*
