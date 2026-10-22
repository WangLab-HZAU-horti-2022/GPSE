# GPSE 训练结果目录：现状盘点与优化方案

> 整理日期：2026-10-22
> 范围：`gpse train` 一次运行（新代码 `gpse/`）产生的全部文件，逐处从源码核实（含产生位置）。
> 目的：① 说清"重要的结果分别在哪"；② 列出"重要但难找/易混淆"的清单；③ 给出目录结构优化方案。
> 关联文档：`topsis_investigation_report.md`（第十节 GPSE 分数定位、第十一节目录改进提案）。

---

## 一、结果目录全貌

默认结果根目录为 `optimization_results_v2/`（`workflow.py:158-161`，可用 `--results_dir` 改）。一次运行的完整产物树：

```
optimization_results_v2/
│
│─── ① 全局对比与 TOPSIS（根目录）
│    ├── model_comparison.csv                  # 全模型对比表（stacking 后会被重写）
│    ├── model_comparison_cv.csv               # TOPSIS 输入：决策矩阵
│    ├── model_comparison_cv_topsis.csv        # TOPSIS 完整结果（含中间量）
│    ├── model_comparison_cv_topsis_simple.csv # TOPSIS 精简结果（选型依据）
│    ├── stacking_selected_models.json         # GPSE 基模型名单
│    ├── final_predictions.csv                 # ⚠️ 不是 GPSE！TOPSIS rank-1 单模型的预测
│    ├── split_manifest.json                   # holdout 划分清单（含表型 scaler）
│    ├── train_ids.txt / test_ids.txt          # 划分样本 ID
│
│─── ② 每个候选模型一个子目录（如 kernelridge/、xgboost/）
│    ├── summary_results.json                  # 该模型全部 repeat 汇总 + training_selection
│    ├── deployment_ensemble/                  # 部署用折集成（全数据重训）
│    │    ├── member_1.pkl ... member_K.pkl
│    │    └── info.json
│    ├── representative_model/                 # 代表性 repeat 模型（若开启保存）
│    └── repeat_1/ ... repeat_R/               # 每个 repeat（论文协议 R=50）
│         ├── fold_1_model.pkl ... fold_K_model.pkl  # 折模型 artifact
│         ├── cv_train_only.csv                # 该 repeat 的折分配（可审计）
│         ├── optuna_best_params.json          # 若该 repeat 做了调参
│         ├── all_predictions.json             # 全部折的逐样本 train/val/test 预测
│         └── repeat_results.json              # 该 repeat 的指标汇总
│
│─── ③ GPSE（stacking）
│    └── ensemble_stacking/
│         ├── stacking_ensemble_model.pkl      # GPSE 模型本体
│         └── stacking_results.pkl             # GPSE 指标（test_metrics.pearson 等）
│
│─── ④ reports/（人读报告层）
│    ├── model_comparison_holdout.csv          # 最终 holdout 指标（按 split_strategy 累积）
│    ├── holdout_metrics.csv / .json           # 同内容的当前快照
│    ├── model_leaderboard.csv                 # 规范化 leaderboard（CV + holdout 指标合一）
│    ├── run_summary.json                      # run 元信息 + leaderboard
│    ├── training_report.md / .html            # 自动生成的图文报告
│
└─── ⑤ logs/（ModelConstants.default_logs_dir）
     ├── 主日志（时间戳命名）
     └── 每模型/每 repeat 的子进程独立日志
```

---

## 二、逐文件说明与"重要性/易找性"评级

评级含义：★★★ = 论文取数/答辩必用；★★ = 审计与复现需要；★ = 调试/存档。

### ① 根目录：对比与 TOPSIS

| 文件 | 内容 | 代码出处 | 评级 | 备注 |
|---|---|---|---|---|
| `model_comparison.csv` | 全模型 Test/Val/Ensemble Pearson±std、训练时长；stacking 完成后**重写**，追加一行显示名为 `Ensemble` | `genomic_utils.py:153-231`；`_pipeline.py:341, 468` | ★★ | 注意：stacking 行在这里叫 **Ensemble**，不叫 GPSE/stacking_ensemble；旧口径（test-based）遗留格式 |
| `model_comparison_cv.csv` | TOPSIS 决策矩阵：Model × [CV Pearson, CV Std]（训练侧） | `_pipeline.py:344-360` | ★★★ | **TOPSIS 的输入**，Panel f 消融的直接改造对象 |
| `model_comparison_cv_topsis.csv` | TOPSIS 完整输出：原始值 + `_positive`/`_norm`/`_weight` 中间量 + `TOPSIS_Score` + `TOPSIS_Rank` | `topsis.py:147-156` | ★★★ | **每模型 TOPSIS 分数在这里**；含全部中间量，可逐步核对 |
| `model_comparison_cv_topsis_simple.csv` | 精简版：Model、两准则、Score、Rank | `topsis.py:159-164` | ★★★ | stacking 选型实际读的文件（`_pipeline.py:372`） |
| `stacking_selected_models.json` | 入选基模型名单 + 选型来源（`train_only_cv`） | `_pipeline.py:383-393` | ★★★ | **"该表型选了哪几个模型"的唯一权威记录** |
| `final_predictions.csv` | TOPSIS **rank-1 单模型**的 holdout 逐样本预测（反标准化后） | `_pipeline.py:113-196` | ★★ | **⚠️ 命名误导重灾区**：不是 GPSE 的预测！ |
| `split_manifest.json` | holdout 划分 + 表型 scaler + 划分策略元数据 | `_pipeline.py:61` | ★★ | 复现与 predict 反标准化的关键 |
| `train_ids.txt` / `test_ids.txt` | 划分样本 ID 清单 | `_pipeline.py:55-59` | ★★ | 审计用 |

### ② 每模型子目录

| 文件 | 内容 | 代码出处 | 评级 | 备注 |
|---|---|---|---|---|
| `{model}/summary_results.json` | 该模型 R 次 repeat 的均值/SD 汇总 + `training_selection`（cv_mean/cv_std/best_params/selected_repeat） | `_repeat_training.py:595-616` | ★★★ | **TOPSIS 决策矩阵两个数的来源**；Panel d/e 离线实验的原料 |
| `{model}/repeat_{i}/repeat_results.json` | 单 repeat 的折平均指标 + 折集成指标 | `_repeat_training.py:356-361` | ★★ | |
| `{model}/repeat_{i}/all_predictions.json` | 该 repeat 全部折的逐样本 train/val/**test** 预测 | `_repeat_training.py:313-315`；`genomic_utils.py:552-588` | ★★★ | **50×5 逐 repeat 离线对比的原料**（stacking 重复协议评估、panel c 更严谨口径都靠它）；体量大 |
| `{model}/repeat_{i}/fold_{j}_model.pkl` | 折模型 artifact（含 scaler/selector/imputer） | `_fold_training.py:174` | ★ | stacking 折集成复用（`_ensemble.py:136-143`） |
| `{model}/repeat_{i}/cv_train_only.csv` | 该 repeat 训练侧折分配 | `_cv_manager.py:223-234` | ★★ | 可审计性设计，很好 |
| `{model}/repeat_{i}/optuna_best_params.json` | 该 repeat 调参结果（若启用） | `_repeat_training.py:279-280` | ★ | |
| `{model}/deployment_ensemble/` | 全数据重训的 K 成员折集成 + `info.json` | `_ensemble.py:32-101` | ★★ | **三个"最终模型"之一**（见第四节混淆点） |
| `{model}/representative_model/` | 表现最接近均值的代表 repeat 模型 | `_save_representative_model` | ★ | 默认不保存 |

### ③ GPSE（stacking）

| 文件 | 内容 | 代码出处 | 评级 | 备注 |
|---|---|---|---|---|
| `ensemble_stacking/stacking_results.pkl` | GPSE 完整指标：test_metrics（pearson/spearman/r²/rmse/mae）+ train_metrics + selected_models | `stacking.py:188` | ★★★ | **GPSE 分数最全的来源，但是 pkl 二进制**，需 `joblib.load` 才能看 |
| `ensemble_stacking/stacking_ensemble_model.pkl` | 训练好的 GPSE 模型（含元学习器与全部基模型） | `stacking.py:186` | ★★ | 部署用 |

### ④ reports/

| 文件 | 内容 | 代码出处 | 评级 | 备注 |
|---|---|---|---|---|
| `model_comparison_holdout.csv` | 最终 holdout 指标表，含 `model == "stacking_ensemble"` 行；按 split_strategy 累积追加 | `_pipeline.py:65-106` | ★★★ | **GPSE 分数最推荐的取数处** |
| `holdout_metrics.csv` / `.json` | 同上内容的当前快照 | `_pipeline.py:107-109` | ★★ | 与上一份重复 |
| `model_leaderboard.csv` | 规范化 leaderboard：cv_mean/cv_std、selected_repeat、holdout pearson±std、部署路径等，按主指标排名 | `_results.py:38-120, 225` | ★★★ | **信息最全的一张表**（CV 选型 + holdout 表现合一），但不含 TOPSIS_Score |
| `run_summary.json` | run 元信息 + leaderboard 全量 | `_results.py:231-252` | ★★ | 机器可读入口 |
| `training_report.md` / `.html` | 自动图文报告 | `_results.py:253-256` | ★ | 内容 = leaderboard + 元数据，无新增信息 |

### ⑤ logs/

主日志（时间戳命名，`predictor.py:243-256`）+ 每模型/repeat 子进程日志（`_repeat_training.py:637-642`，并行结束后归并进主日志）。★ 调试用。

---

## 三、快速定位表（FAQ）

| 我要找… | 去哪里 | 备注 |
|---|---|---|
| **GPSE 最终分数** | `reports/model_comparison_holdout.csv` 的 `stacking_ensemble` 行；或 `ensemble_stacking/stacking_results.pkl`（最全） | `model_comparison.csv` 里叫 `Ensemble` 行；std 恒 0.0（单次评估） |
| **每个模型的 TOPSIS 分数** | `model_comparison_cv_topsis.csv`（含中间量）/ `_simple.csv`（精简） | |
| **该表型选了哪几个基模型** | `stacking_selected_models.json` | |
| **TOPSIS 的输入是什么** | `model_comparison_cv.csv`；其数值来源是各模型 `summary_results.json` 的 `training_selection` | |
| **GPSE 的预测值（逐样本）** | ❌ **当前不落盘**（只有指标）；需用 `stacking_ensemble_model.pkl` 重新 predict | `final_predictions.csv` 是 rank-1 单模型的，别用错 |
| **50×5 逐 repeat 原始预测** | `{model}/repeat_{i}/all_predictions.json` | 离线重做严谨对比的原料 |
| **划分是否可复现** | `split_manifest.json` + `train_ids.txt`/`test_ids.txt` + 各 `cv_train_only.csv` | |
| **部署用模型** | `{model}/deployment_ensemble/`（单模型）；`ensemble_stacking/stacking_ensemble_model.pkl`（GPSE） | |

---

## 四、现状问题清单（重要但难找/易混淆）

按严重程度排序：

1. **`final_predictions.csv` 命名误导（P0）**：名字像"最终结果"，实为 TOPSIS rank-1 **单模型**的 holdout 预测，与 GPSE 无关（`_pipeline.py:113-119`）。已经造成"找不到 GPSE 分数"的实际困扰。
2. **GPSE 分数无单一权威入口（P0）**：散落于 4 处（holdout csv 行 / model_comparison.csv 的 `Ensemble` 行 / pkl / leaderboard），且名字不统一（`stacking_ensemble` vs `Ensemble` vs 口语"GPSE"），最全的那份是二进制 pkl。
3. **"选型 → 集成"链条断裂（P0）**：TOPSIS 分数（topsis csv）、入选名单（json）、GPSE 分数（pkl/holdout csv）分属三类文件三种格式，没有任何一个文件能回答"该表型：候选谁→各多少分→选了谁→集成后多少分"。
4. **GPSE 逐样本预测不落盘（P1）**：只有指标没有预测值，画图（拟合散点、Y=X 参考线，论文图 8/17 那种）必须重新加载模型 predict。
5. **stacking 单次评估，std 恒 0.0（P1）**：与单模型"均值±SD"口径不对称（详见 `topsis_investigation_report.md` 第十节）；`reports/` 各表里这个 0.0 没有任何注释说明。
6. **stacking 静默失败（P1）**：失败只 log 不中断（`_pipeline.py:470-472`），事后无法从结果文件区分"没开 stacking"和"开了但失败"。
7. **根目录与 reports/ 功能重叠（P2）**：holdout 指标存 3 份（`model_comparison_holdout.csv`、`holdout_metrics.csv/.json`），`model_comparison.csv` 与 `model_leaderboard.csv` 内容大量重叠但列名/口径不同（前者旧 test-based 格式，后者新规范化格式）。
8. **`model_comparison.csv` 会被重写（P2）**：stacking 前的版本被覆盖，若 stacking 未运行则永远没有 Ensemble 行——同一个文件名两种内容状态。
9. **三个"最终模型"易混（P2）**：`deployment_ensemble/`（单模型全数据折集成）、`representative_model/`（代表 repeat）、`stacking_ensemble_model.pkl`（GPSE）——部署时该用哪个，目录本身不说话。

---

## 五、优化方案

原则：**不改任何计算逻辑，只重组/汇总/命名**；历史结果目录可离线回填。

### P0-1 新增 `reports/selection_summary.json` + `.csv`（核心）

每个 run 一份，把"选型→集成"链条合上（schema 草案见 `topsis_investigation_report.md` 第十一节）：

- run 元信息：trait、task_type、top_n 实参、split 策略；
- TOPSIS 明细：criteria、weights、min_transform、每模型 cv_mean/cv_std/topsis_score/topsis_rank/selected；
- GPSE 结果：base_models、meta_learner、pearson/spearman/r²/rmse/mae、evaluation 口径（single_holdout）、status（成功/失败原因——同时解决问题 6）。

数据来源全部是现成文件（topsis csv + selected_models json + stacking_results.pkl），可在 `_pipeline.py` 末尾统一生成，也可写独立回填脚本处理历史目录。

### P0-2 消除 `final_predictions.csv` 误导

改名为 `topsis_top1_holdout_predictions.csv`（同步改 `_write_topsis_final_predictions` 与引用处），并给 GPSE 增加对应的 `gpse_holdout_predictions.csv`（在 `stacking.py:fit` 里已有 `meta_test`，顺手落盘即可——同时解决问题 4）。

### P1-1 统一命名与入口

- GPSE 在所有表格中统一显示名：`GPSE_Stacking`（替换 `Ensemble`/`stacking_ensemble` 混用）；
- `reports/` 作为唯一人读入口；根目录的 `model_comparison.csv` 标注为 legacy 或移入 `reports/legacy_`；
- `model_comparison.csv` 只在 stacking 完成后再写 Ensemble 行，或拆分"stacking 前/后"两个文件名（如 `model_comparison_base.csv` / `model_comparison_with_gpse.csv`），消除同文件名两态（问题 8）。

### P1-2 每目录一个 README

结果根目录放一份自动生成的 `README.md`：目录树 + 每个文件一句话说明 + "GPSE 分数看这里"指路（内容可由 `write_result_bundle` 生成，成本极低）。

### P2-1 batch 层跨 run 聚合

`batch/` 增加聚合脚本：遍历各 run 的 `reports/selection_summary.json`，产出跨表型总表（每行：物种/表型、top_n、入选模型、各模型 TOPSIS 分数、GPSE pearson、CV 可部署最优单模型 pearson、Δr）——直接对接 figure_2 数据包的取数格式。

### P2-2 stacking 纳入重复评估协议（口径修正，非纯目录问题）

用各 `{model}/repeat_{i}/all_predictions.json` 离线重算 stacking 的逐 repeat 表现，在汇总中报告 `pearson_mean ± pearson_std`，消除与单模型的口径不对称（问题 5）。此项目前数据已在盘上，无需重训。

### 优先级汇总

| 优先级 | 事项 | 解决的问题 | 风险 |
|---|---|---|---|
| P0-1 | `selection_summary.json/.csv` | 1(部分)、2、3、6 | 零（纯增量） |
| P0-2 | 改名 final_predictions + GPSE 预测落盘 | 1、4 | 低（需同步改引用） |
| P1-1 | 命名统一、model_comparison 拆分 | 2、7、8 | 低 |
| P1-2 | 结果目录 README | 全部（导航层） | 零 |
| P2-1 | batch 聚合脚本 | 3（跨 run 层） | 零 |
| P2-2 | stacking 重复协议评估 | 5 | 中（需离线重算验证） |

---

*整理人：Kimi Code；所有文件路径与代码行号均于 2026-10-22 从当前源码核实。*
