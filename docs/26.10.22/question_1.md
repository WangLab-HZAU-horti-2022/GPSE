# Figure 2 数据包：为什么需要 TOPSIS 引导的集成（而不是固定单模型）

## 目的和范围

本目录支撑论文的核心方法论证：**为什么不能固定用一个模型，而要 TOPSIS 决定集成什么、怎么集成**。
共 7 条论点，每条对应一个 panel 子目录。目录中只含数据与统计脚本，不含绘图脚本和图形文件。

> 整理日期：2026-10-21。结构对齐 `GPSE/tests/figure_1_delivery/` 的交付规范。
> 所有统计量由 `scripts/compute_claim_stats.py` 从目录内 CSV 重算，可复现：
> ```bash
> python3 scripts/compute_claim_stats.py
> ```

## 目录结构

```text
data/
├── panel_a_no_universal_best/            # 论点 1 ✅ 数据齐全
├── panel_b_topsis_ranking_validity/      # 论点 2 ⚠️ 数据齐全，但实测一致性为中等
├── panel_c_ensemble_vs_best_single/      # 论点 3 ✅ 公平口径下成立（9/11，+3.3%）
├── panel_d_subset_selection_vs_baselines/# 论点 4 ❌ 需补实验（见 NEEDS_EXPERIMENT.md）
├── panel_e_oracle_subset/                # 论点 5 ❌ 需补实验（见 NEEDS_EXPERIMENT.md）
├── panel_f_ablation_criteria_weights/    # 论点 6 ❌ 需补实验 + 有合规风险须先修复
└── panel_g_full_pool_scaling/            # 论点 7 ⚠️ 池数据有，抽样/穷举需补
scripts/
├── compute_claim_stats.py                # 重算论点 1/2/3 的基础统计量
└── compute_claim3_deployable.py          # 论点 3 公平口径（vs CV 可部署最优单模型）+ 选型损失
```

## 各 panel 数据与预填数字

### Panel a｜论点 1：无跨物种/性状普适最优单模型 ✅

| 文件 | 内容 | 来源 |
|---|---|---|
| `fig1_top1_base_models.csv` | 低维 11 任务的 Top-1 归属（CV 口径） | figure_1_delivery |
| `fig1_model_metrics_long.csv` | 11 任务 × 11 模型 CV 表现长表（热图直接数据源） | figure_1_delivery |
| `fig1_model_stability.csv` | 各模型跨任务获胜次数 | figure_1_delivery |
| `six_species_all_traits_combined_metrics.csv` | 6 物种 18 组合 × 14 模型 test 表现矩阵 | xiaoliu_result |
| `six_species_heatmap_matrix.csv` | 6 物种热图矩阵 | xiaoliu_result |
| `claim1_top1_distribution.csv` | **预填数字**：第一名漂移统计 | 本目录脚本计算 |

**预填数字**：
- 低维 11 任务（CV）：Top-1 分属 **6 个模型**——KernelRidge 4 次、CatBoost 3 次，XGBoost/SVR/HistGB/GBDT 各 1 次
- 6 物种 18 组合（test）：第一名分属 **5 个模型**——kernelridge 7、svr 5、xgboost 3、catboost 2、rf 1

### Panel b｜论点 2：TOPSIS 排序可信 ⚠️ 数字比预期弱

| 文件 | 内容 |
|---|---|
| `fig1_topsis_ranking.csv` | 11 任务 × 11 模型的 CV Pearson mean/sd + TOPSIS score/rank（CV 口径，**无泄漏，主证据**） |
| `claim2_topsis_vs_cv_per_task.csv` / `claim2_topsis_vs_cv_summary.csv` | **预填数字**：逐任务 Spearman ρ 与 Top-1 命中 |
| `six_species_complete_ranking_report_TESTBASED.csv` | 6 物种 TOPSIS 排名（**基于 test 指标，与 test 排名比较属循环论证，仅作参考**） |
| `claim2_six_species_TESTBASED_summary.csv` | 上述参考口径的统计（ρ=0.98、命中 14/18，勿直接引用） |

**预填数字（CV 口径，建议引用这个）**：
- 平均 Spearman ρ = **0.72**（11 任务；最低 0.28）
- Top-1 命中率 = **5/11**

⚠️ 写作建议：原论证稿中"高度一致、平均 ρ = X"的表述与实测（0.72、5/11）有差距。
0.72 可表述为"中高度一致"；TOPSIS 的价值本来就在于它不复制纯精度排名
（稳定性准则会主动重排），建议把论证重心放在"选出的 top-3 子集质量好"（panel d/e），
而不是"排名复刻精度排名"。

### Panel c｜论点 3：集成 > 最优单模型 ✅ 公平口径下成立

| 文件 | 内容 | 口径 |
|---|---|---|
| `fig1_gpse_holdout_comparison.csv` | 11 任务 GPSE stacking vs best base | manuscript 协议，报告性 holdout |
| `six_species_result_baseline.tsv` | 18 组合 ensemble vs best base | 6 物种 test |
| `three_species_summary_all_traits.csv` | 24 性状 × {GBLUP, rrBLUP, 10 单模型, Stacking} 宽表 | 历史汇总（协议与现行不同） |
| `three_species_GPSE_vs_traditional_table.csv` | 三物种 vs GBLUP/rrBLUP 简表 | 历史汇总 |
| `claim3_winrate_summary.csv` | 预填数字：三个来源的胜率（vs 测试集 oracle 最优单模型） | 本目录脚本计算 |
| `claim3_vs_deployable_single.csv` | **预填数字（推荐引用）**：stacking vs CV 选出的可部署最优单模型 | 本目录脚本计算 |
| `claim2b_topsis_selection_regret.csv` | TOPSIS top-1 相对 CV 最优的选型损失 | 本目录脚本计算 |

**预填数字（vs 测试集 oracle 最优单模型，偏严苛的下界）**：

| 数据来源 | 胜率 | 平均 Δr |
|---|---|---|
| 低维 11 任务 holdout（manuscript 协议） | 5/11 | -0.003 |
| 6 物种 18 组合 test | 14/18 | +0.013（+3.1%） |
| 三物种 24 性状（历史） | 10/24 | -0.024 |

**预填数字（vs CV 选出的可部署最优单模型，公平口径，建议引用）**：

| 数据来源 | 胜率 | 平均 Δr |
|---|---|---|
| 低维 11 任务 holdout（manuscript 协议） | **9/11** | **+0.0149（+3.3%）** |

**口径说明**：`fig1_gpse_holdout_comparison.csv` 里的 best single 是按**测试集**表现挑的
（oracle，实际部署时无法做到），5/11 是过严的下界。公平的对比是 CV 选出的最优单模型
（部署前不看测试集就能确定），此口径下 TOPSIS 引导的 top-3 stacking 9/11 胜出、
平均 +3.3%。由于该 stacking 的成员正是 TOPSIS 选出的，这也是 TOPSIS 选型有效的间接证据。

**TOPSIS 选型损失（panel b 的补充证据）**：TOPSIS top-1 相对 CV 最优模型的
平均选型损失仅 **0.0068** 个 Pearson 单位（最大 0.031），5/11 任务零损失——
双准则重排造成的精度损失很小，换来的是稳定性约束。

⚠️ 剩余注意：holdout 是单次划分，方差大；更可靠的做法是基于 50×5 CV 逐 repeat
比较（stacking 的逐 repeat 结果在各原始运行目录，尚未汇总）。历史三物种 24 性状
口径与现行协议不同，仅作参考。

### Panels d–g｜论点 4–7：需补实验

各目录下 `NEEDS_EXPERIMENT.md` 写明：论点、缺失内容、实验设计、可复用的现成设施。
关键利好：manuscript_runs 已落盘全部 50×5 逐 fold OOF 预测，
**论点 4/5/7 的对照实验都可纯离线完成，无需重训模型**。

**必须优先处理的风险（Panel f）**：熵权法代码存在但未接入主训练路径
（`_pipeline.py:367` 硬编码 0.8:0.2，`genomic_utils.py:134` 硬编码关闭熵权），
论文稿"Entropy-TOPSIS"表述与实现不符（`GPSE/docs/methods_code_audit.md` 已标记）。
跑消融前必须先统一论文表述与代码实现。

## 待回填占位 → 数字来源对照

| 论证稿占位 | 现状 |
|---|---|
| 论点 1 "第一名随组合漂移" | ✅ 6/11 任务、5/18 组合（panel a） |
| 论点 2 "平均 ρ = X，命中 X/X" | ⚠️ ρ=0.72、5/11（CV 口径，panel b） |
| 论点 3 "X/X 组合胜出，平均提升 X%" | ✅ 公平口径：9/11、+3.3%（vs CV 可部署最优单模型）；6 物种 14/18、+3.1%（panel c） |
| 论点 4–7 全部 X | ❌ 待补实验（panels d–g） |
 jzhang on  /data/jzhang/project/Temp/gpse_26.6.4_test/gpse_paper_data_res_figure/figur
 jzhang on  /data/jzhang/project/Temp/gpse_26.6.4_test/gpse_paper_data_res_figure/figure_2