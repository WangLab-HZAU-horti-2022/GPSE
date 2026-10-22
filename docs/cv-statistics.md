# GPSE 重复 CV 统计修复与验收记录

日期：2026-10-22。适用新版 `gpse/` 训练入口；不回写旧 `scripts/` 或历史实验。

## 调查结果与修复前数据流

源码调查完成后实施。本次未修改用户已有的 `.gitignore`、`docs/todo_v3.md` 和
`docs/26.10.22/` 工作区内容；后者是本修复之前的调查快照。

```text
run_all_models
  → run_model_multiple_repeats
    → train_and_evaluate_model_for_repeat
      → generate_cv_folds_from_file（固定 holdout 训练分区上生成）
      → _train_single_fold → validation Pearson
      → _calculate_fold_average_metrics
      → repeat_results.json（fold_results + avg_val_pearson/std_val_pearson）
    → calculate_repeat_statistics（已有跨重复描述统计，但不是排名来源）
    → select_repeat_by_train_cv（验证均值最大一次，SD/编号破同分）
    → training_selection.cv_mean/cv_std = 最佳 repeat 的均值/折内总体 SD
  → model_comparison_cv.csv → TOPSIS → Top-N → Stacking
```

| 调查项 | 修复前证据 / 结论 |
|---|---|
| 逐折 Pearson 是否持久化 | 是，`repeat_results.json.fold_results[*].val_metrics.pearson` |
| 单次 repeat 均值 / SD | 是，`avg_val_pearson`、`std_val_pearson`；SD 用 `ddof=0` |
| 全部 50 次原始指标 | 成功 repeat 都保留 JSON；失败折/重复可被跳过，不能保证 250 折完整 |
| 最佳 repeat 主调用 | `_repeat_training.run_model_multiple_repeats → select_repeat_by_train_cv` |
| 其他用途 | 参数用于代表模型、部署集成和 Stacking 工厂；选中 repeat 的折预测用于 TOPSIS Top-1 holdout 预测 |
| 是否直接部署最佳折检查点 | 否，代表模型和部署集成重新拟合；但传入了包含 holdout 的完整 X/y |
| TOPSIS 的读取字段 | `training_selection.cv_mean/cv_std`；当前新版主链未读取 holdout 排名；异常时曾回退 CV 字典序排名 |
| 模型间 CV 划分 | 相同训练样本、`random_seed + repeat_idx`，不含模型名；原代码未强制核对实际分配 |
| Stacking 是否加载最佳 repeat 模型 | 否，按选中模型及参数重建 Pipeline，严格 OOF，再在完整训练分区拟合基学习器 |
| 历史可恢复性 | 完整折级 JSON 可以重新聚合；仅最佳 repeat / 总结 CSV 不能恢复全部定义 |
| Pearson 风险 | 工具函数将常数/计算异常/NaN 静默改 0；Optuna 失败折也可能填 0；Stacking 将 NaN 改 0 |

## 新数据流与实现位置

```text
相同 50×5 train-only CV → 原始逐折 / 逐 repeat JSON（保留）
  → aggregate_cv：完整性、编号、有限性、取值范围检查
  → 全部重复聚合 → training_selection（正式准确性/稳定性）
  → validate_shared_splits：逐 repeat/折比较真实样本划分摘要
  → 4 个可追溯聚合文件 → 固定 0.8/0.2 TOPSIS → Stacking Top-N

最佳 repeat → 诊断字段 / train-only 参数来源
  → 在完整 holdout 训练分区重新拟合模型与预处理（不传 holdout 标签）
holdout 指标 → 评估报告；不反向进入聚合、TOPSIS 或参数选择
```

| 文件 | 核心修改 |
|---|---|
| `gpse/train/_cv_statistics.py` | `aggregate_cv`、`validate_shared_splits`、`runtime_provenance`、`export_cv` |
| `gpse/train/_fold_training.py` | 真实样本划分 SHA-256；单次 Pearson SD 改样本 SD |
| `gpse/train/_repeat_training.py` | 跨重复聚合后才允许选型；按 repeat ID 排序；最佳 repeat 保留作诊断/参数来源；训练分区重拟合；失败审计 |
| `gpse/train/_cv_manager.py` | CV 分区完整性/不重叠验证；外部 CV 生成使用配置随机种子 |
| `gpse/train/_pipeline.py` | 写入来源、核对共享划分、正式 TOPSIS、记录 Stacking 入选名单；禁止覆盖旧结果和静默替代排名 |
| `gpse/train/topsis.py` | 只接受训练侧聚合列；固定权重；非有限/负 SD/归一化异常报错；0 SD、恒定准则、同分明确处理；修正排序后中间量错行 |
| `gpse/train/_ensemble.py` | 部署模型使用传入的训练分区；明确评估指标不属于新保存的模型实例 |
| `gpse/train/_optimization.py` | 未定义 Pearson / 失败折剪枝当前 trial，不填 0 |
| `gpse/utils/genomic_utils.py` | 未定义相关系数保留 NaN、记录 status 并警告；新建模型/repeat 目录禁止覆盖 |
| `gpse/train/stacking.py` | 统一相关系数异常语义；Ridge 元学习器、OOF 算法不变 |
| `gpse/train/_results.py` | 展示所有 SD 和指标定义；`rank` 标注 holdout 展示顺序，另列正式 `TOPSIS_Rank/Score` |
| `gpse/train/{predictor,workflow,cli}.py` | 默认 50×5；显式稳定性参数；单 repeat 默认拒绝 repeat SD |
| `gpse/config/topsis.yaml` | 修正配置口径，只有两个训练侧准则和固定权重 |
| `gpse/tools/reaggregate_cv.py` | 独立离线重聚合与新旧排名对照 |
| README、wiki、MCP 参数说明、单 repeat smoke 脚本 | 更新默认值、配置与迁移语义 |

模型列表、插补/特征选择/标准化算法、Stacking 的 Ridge 超参数未改变。

## 新旧公式

令验证折 Pearson 为 r[t,k]，R=50、K=5；m[t]=mean_k r[t,k]。

旧正式排名：t*=argmax_t m[t]，`cv_mean=m[t*]`，
`cv_std=sqrt(sum_k (r[t*,k]-m[t*])² / K)`（总体 SD）。

新定义：

| 字段 | 定义 |
|---|---|
| `cv_mean` | sum_t m[t] / R |
| `cv_std_repeat` | sqrt(sum_t (m[t]-cv_mean)² / (R-1)) |
| `cv_std_within` | sqrt(mean_t s[t]²)，s[t]=sqrt(sum_k (r[t,k]-m[t])² / (K-1)) |
| `cv_std_all_folds` | 全部 R×K 个 r 的样本 SD，分母 R×K-1 |
| `cv_best_repeat_mean` | max_t m[t]；诊断用途 |

默认 `cv_std` 是 `cv_std_repeat` 的兼容别名，始终记录 `stability_metric`。
`legacy_cv_mean/legacy_cv_std` 保留旧定义供对照，不作为正式准则。
历史旧方法的 SD 从原始折值重新计算 `ddof=0`，避免将新版 `ddof=1` 冒充旧口径。

重复 CV 共享样本，repeat 间不独立；这里的 SD 描述划分波动，不除以 sqrt(50)，
不当作泛化置信区间。关于重叠样本造成的方差解释风险，参见
[Bengio & Grandvalet, JMLR 2004](https://jmlr.org/papers/v5/grandvalet04a.html)。

## TOPSIS 与边界策略

输入明确记录：`selection_source=train_only_cv`、`accuracy_metric=cv_mean`、
`stability_metric=cv_std_repeat`、`weight_method=manual`、`weights=[0.8,0.2]`。
默认 SD 先 `-log(SD+1e-12)`，然后 L2 向量归一化，算法因素保持原样；不启用熵权。

- 缺失/重复 repeat 或折、非有限/越界验证指标：模型无选型资格，不静默减小 R/K，不填 0。
- 运行失败原因保留日志和可用的原始 JSON；聚合不合格写 `cv_failure.json`；主链写 `cv_model_failures.json` 明示排除模型。
- 不同候选模型实际 CV 划分不同：停止正式排名。
- 单次 repeat：默认报错，因为 repeat 样本 SD 未定义；只有显式 `--cv_stability_metric cv_std_within` 或 `cv_std_all_folds` 才可运行，repeat SD 记为 null。
- 常数输入/不足两样本 Pearson：NaN + status + 警告，验证侧 NaN 阻止选型；holdout 异常只用于异常评估报告。
- SD=0：按明确记录的 epsilon=1e-12 处理；不删除模型。
- 一个恒定准则：各模型贡献相同，不重新分配权重；L2 零列用分母 1 保持零列。
- 所有加权准则恒定或只有一个候选：分数 0.5、并列排名；同分输出按模型名排序，Top-N 使用该确定性顺序。
- 非有限准则、负 SD、无效权重、L2 溢出：报错；禁止抛弃异常行后输出看似成功的排名。
- TOPSIS 失败：停止；不降级为另一个未记录的排序算法。

## 输出与版本

结果根目录写：`cv_fold_metrics.csv`、`cv_repeat_metrics.csv`、
`model_comparison_cv.csv`、`cv_aggregation_summary.json`。沿用原始 repeat JSON，
不再额外复制每模型一套折 CSV。

包括 model、从 0 开始的 repeat/fold ID、validation Pearson、split seed/identifier、
聚合方式、预期/完成计数、指标定义和版本。分区 identifier 摘要包含训练样本 ID 顺序和折索引，
因此可核对模型间划分，而不只是核对随机种子。
`code_version` 是实际 Python 源码集合的 SHA-256；`config_version` 是有效参数和配置内容摘要，
包括运行模型清单、trait、Stacking 配置以及外部配置文件内容。完整有效配置在 `cv_run_manifest.json`。

新训练拒绝复用已有结果目录；离线工具只允许另建位于历史输入树以外的新目录。
历史缺失版本写 `unknown_historical`；单独记录离线计算代码版本和输入文件摘要，
不把历史训练产物标为已经用新代码重训。

## 离线恢复范围和待重训项

```bash
python -m gpse.tools.reaggregate_cv /path/to/old_results \
  --output /path/to/new_cv_analysis --expected-repeats 50 --expected-folds 5
```

输出 `reaggregation_audit.json`，完整数据且共享划分可核实时再输出
`cv_ranking_comparison.csv` 及新 TOPSIS 文件。无法完成的恢复返回非零 CLI 状态。损坏 JSON、错误数据结构、预测/折数不符都会写入
`input_errors` 和逐模型恢复失败原因；其他模型可导出统计，但不对缺失候选集生成正式排名。
应先从备份恢复损坏文件，无法恢复时才补跑 CV。
保存模型的重拟合清单按 `info.json.trained_on` 判定：`all_samples` 明确列入重拟合，
`holdout_training_partition` 不重复要求重训；缺失信息列入 `deployment_refit_unknown`，
不会武断地把全部历史模型标为需要重训。输入文件摘要覆盖实际读取的预测、划分和模型元数据。

| 历史数据 | 能恢复 / 不能恢复 |
|---|---|
| 所有 50×5 折指标完整、有限 | 可恢复全部聚合统计，不必重训 250 个折模型 |
| 另有 split_identifier 或 `cv_train_only.csv` + 折索引 | 可验证跨模型共享划分，输出新旧排名对照 |
| 只有完整每次均值但无折值 | 理论可算总均值和 repeat SD；本严格工具拒绝宣称完整恢复，不能恢复 within/all-fold SD |
| 只有最佳 repeat、汇总 CSV 或缺失折/重复 | 无法恢复完整 50×5，需补跑/重训缺失实验 |
| 旧版 Pearson=0，原始预测仍在 | 检查是否常数/非有限输入；未定义值会被拒绝 |
| 旧版 Pearson=0，原始预测丢失 | 无法分辨真 0 和历史异常填 0；离线摘要明确记录此限制 |
| 保存模型曾使用完整数据（含 holdout） | 重聚合不能修复模型权重；需在训练分区重拟合部署/代表模型 |
| 新排名改变了 Stacking 成员 | 必须重拟合 Stacking 的 OOF 基模型和元学习器；不能把旧集成重命名为新结果 |

**当前可核查的真实历史实验：未提供。** 工作区（含忽略文件）未找到 `repeat_results.json`、
`summary_results.json` 或实验 comparison CSV。因此尚不能给出具体物种/表型的新旧排名差异或
逐实验重训清单；已向用户询问历史目录路径。工具会对实际输入列出
`recovered_models`、`requires_cv_retraining`、`deployment_refit_required`，不会编造结果。

续轮核查定位到配置/文档中 12 个明确的历史结果路径，当前环境全部不可访问，
工作区仍没有原始 repeat JSON。逐路径证据见
[cv-history-availability.json](cv-history-availability.json)。这表示当前无法核查，
不等于这些实验数据丢失或必须重训；需要服务器目录的可访问副本或挂载路径。

## 已执行的新旧排名对照（合成 50×5）

此表是回归验收数据，**不代表用户真实实验**。模型 lucky 的 49 次均值为 0.2，
第 50 次为 0.9；stable 每次均值为 0.5。两者每次五折总体 SD 相同。
使用相同 0.8/0.2、neglog、L2，只改变聚合来源：

| 模型 | 旧 cv_mean | 新 cv_mean | repeat 样本 SD | 旧排名 | 新排名 |
|---|---:|---:|---:|---:|---:|
| synthetic_lucky | 0.900 | 0.214 | 0.098995 | 1 | 2 |
| synthetic_stable | 0.500 | 0.500 | 0 | 2 | 1 |

实际离线工具生成的数据见 [cv-ranking-example.csv.example](cv-ranking-example.csv.example)。

## 验收与风险

`tests/test_cv_statistics.py` 覆盖完整 50×5 合成公式、最佳 repeat 异常高分、ddof=1、
候选共享划分、TOPSIS 训练侧来源、Stacking 排名传递、
单 repeat/缺失/重复/NaN/Inf/零 SD/常数 Pearson、版本和历史不覆盖、新旧方法对照、
TOPSIS 中间量行对应和 L2 溢出。

端到端测试实际训练两个 Ridge 候选各 50×5；经过正式 TOPSIS、Stacking、代表模型和部署集成。
记录所有 fit 收到的标签，验证它们均来自固定训练分区；同一 seed 的第二次 50×5 运行仅改变
holdout 标签，要求全部 CV 聚合指标逐值相同。

测试命令（本机有依赖的 Python 环境）：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/zj/.pyenv/versions/gpse/bin/python -m pytest tests -q
```

主修复阶段完整测试结果：**172 passed，3 failed**（20.49 秒，含两次真实 50×5 流水线）。
新增 CV 验收用例全部通过；失败为下列原有问题。全仓这三项失败均已在 `git archive HEAD` 的独立副本
中复现：`test_threadpoolctl_declared_in_pyproject`、`test_find_result_file_depth_limit`、
`test_multi_trait_genotype_files_are_hardlinked`；最后一项在多轮运行中有时通过，存在波动。
这些与 CV 统计改动无关，本任务未顺带修改。

剩余方法学风险：

1. 最佳 repeat 仍可提供训练侧参数。固定参数是默认；若显式逐 repeat 调参，聚合衡量的是调参程序，
   而不是同一组固定超参数。消除调参/模型选择的全部乐观偏差需要嵌套 CV，属于后续独立工作。
2. repeat SD 描述相同样本上重复划分的稳定性；不能据此构造独立实验的 SE/CI。
3. holdout 仍按历史架构在各 repeat 计算并报告，但没有进入正式排名；人工反复观看后调整方案仍可能产生选择偏差。
4. Top-1 `reports/final_predictions.csv` 仍是选中模型的诊断 repeat 折平均 holdout 预测，
   不是新保存的训练分区重拟合模型预测，也不是 Stacking 预测；保存模型明确标注指标不属于该实例。
5. 新排名不能自动赋予旧 Stacking 结果新版语义；真实历史数据的恢复和重训清单须待目录可访问后完成。


续轮离线工具验收：`pytest tests/test_cv_statistics.py -k 'offline or export' -q`，
**13 passed，16 deselected**。新增 10 个参数化用例验证损坏历史文件仍产生审计、
已有 split identifier 无需重构缺失索引、历史清单损坏不伪造版本、读取输入的摘要以及
基于模型训练分区元数据生成重拟合清单。未重新运行无关训练或改变 50×5 协议。
