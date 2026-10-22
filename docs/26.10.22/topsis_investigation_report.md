# TOPSIS / 熵权问题核查报告

> 整理日期：2026-10-22
> 起因：`docs/26.10.22/question_1.md` Panel f 标记的风险——"熵权法代码存在但未接入主训练路径，论文稿 Entropy-TOPSIS 表述与实现不符"。
> 范围：旧代码 `scripts/`、新代码 `gpse/`、答辩论文稿 `毕业论文答辩_刘晓.pdf` 三方对照核查。

---

## 一、结论摘要

1. **熵权法从未在任何一代代码的主训练路径中启用。** 新旧代码均硬编码手工固定权重 0.8/0.2，熵权函数存在但只能通过独立 CLI 手动调用。论文"基于熵权的 TOPSIS 客观赋权"的表述与实际产生结果的代码不符。
2. **新旧两代代码的 TOPSIS 权重逻辑完全一致**（0.8/0.2、neglog、熵权关闭）；不一致之处是刻意的防泄漏重构（准则来源从测试集指标改为训练侧 CV 指标、Stacking 改为严格 OOF），不是回归。
3. **论文稿与实现的不一致共 4 处**：熵权表述、标准化公式、准则数据口径、Top-N 默认值（详见第四节）。
4. **建议方向：改论文措辞，不要开启熵权重跑。** 理由见第六节。

---

## 二、风险核实：熵权确实未接入（question_1.md Panel f 属实）

| 位置 | 事实 |
|---|---|
| `gpse/train/_pipeline.py:362-371` | 主训练路径调用 `call_topsis_evaluator(..., manual_weights="0.8,0.2", min_transform="neglog")` |
| `gpse/utils/genomic_utils.py:134` | `use_entropy_weights=False` 硬编码（`scripts/genomic_utils.py:109` 同样如此） |
| `gpse/train/topsis.py:34-44, 117-123` | 熵权函数 `entropy_weight_method()` 存在且可用，但主路径走 `else` 分支（解析手工权重），熵权分支从不执行；仅独立 CLI `--use_entropy_weights` 可启用 |
| `docs/methods_code_audit.md:13, 75, 156` | 已标记"不符合"，给出两个选项：改论文措辞 或 启用熵权重跑 |

两种赋权方式互斥（`topsis.py:117-123` 的 if/else）：

- **手工权重（现状）**：人事先指定"准确性 0.8 / 稳定性 0.2"，每个任务、每次运行恒定。
- **熵权法（论文声称的）**：按各准则在数据中的离散程度自动计算权重，逐任务不同，一般恰不等于 0.8/0.2。

另注意熵权分支自身有实现瑕疵：权重在**原始数据**上计算（`topsis.py:119` 在正向化之前），却应用于变换+归一化后的矩阵，严格说与论文式 (2.25) 也不自洽。

---

## 三、新旧代码一致性核查（`scripts/` vs `gpse/`）

### 完全一致的部分

- **TOPSIS 算法本体**：`gpse/train/topsis.py` 与 `scripts/topsis_evaluator.py` 逐行 diff 仅注释英文化，算法、熵权函数、默认值完全相同。
- **调用参数**：两边均为 `criteria_types=[max, min]`、`manual_weights="0.8,0.2"`、`min_transform="neglog"`、熵权关闭。
- **工具函数**：`gpse/utils/genomic_utils.py` 与旧版基本一致，仅小的健壮性改进（常量输入 Pearson 返回 0、`prepare_cv_data` 增加 `seed` 参数）。

### 有意改变的部分（防泄漏重构）

| 点 | 旧 `scripts/` | 新 `gpse/` |
|---|---|---|
| TOPSIS 准则来源 | **测试集**指标：`Test Pearson` / `Test Pearson (std)`（`genomic_prediction_v2.py:289`） | **训练侧 CV** 指标：`training_selection.cv_mean/cv_std`（`_pipeline.py:344-360`） |
| 触发条件 | 仅 `use_stacking` 时运行（`genomic_prediction_v2.py:1454`） | ≥2 个模型即运行；排名还决定 `final_predictions.csv` 的 rank-1 模型 |
| TOPSIS 失败回退 | 按**测试集**性能选模型（`:1499-1507`） | 按 CV mean/std 排序，仍不看测试集（`_pipeline.py:373-380`） |
| Stacking 元特征 | 从磁盘加载已在全部训练数据上训好的基模型做 KFold OOF（`ensemble_stacking.py:246-296`） | 严格 OOF：每折内用 `model_factory` 重建完整预处理 pipeline（折内插补/特征选择/标准化），分类用 StratifiedKFold（`gpse/train/stacking.py:91-128`） |
| 选型留痕 | 无 | 新增 `stacking_selected_models.json`（`selection_source=train_only_cv`） |

**复现警示**：论文 6 物种 18 组合的 TOPSIS 排名由旧代码产生（测试集口径，figure_2 数据中相应文件已标注 TESTBASED、"循环论证、仅作参考"）。用新代码复现不会得到完全相同的选型，跑 Panel f 消融前需确认每组结果的产出口径。

---

## 四、论文稿（答辩稿）与实现的不一致

对照 `毕业论文答辩_刘晓.pdf`（88 页）逐节核查，共 4 处不一致，按严重程度排序：

### 4.1 熵权表述（最严重）

论文从摘要到结论反复声称"基于熵权的 TOPSIS 客观赋权"，并以"避免人为赋权带来的主观偏差"为动机（§1.4.2、§1.4.4、§2.7、§3.1、§4.2、§4.3、中英文摘要）。但产生结果的两代代码均为**手工固定权重 0.8/0.2**——论文宣称要避免的主观赋权恰恰是实际使用的方法。答辩中最容易被问穿的一点。

### 4.2 标准化公式与实现不符（本次核查新发现，audit 未标记）

- 论文 §2.7 式 (2.23)/(2.24)：**极差标准化**（min-max）。
- 代码实际：min 型准则（SD）先做 `neglog`（`-log(x+1e-12)`）正向化，然后所有列做 **L2 向量归一化**（`x/√Σx²`，`topsis.py:54-65`）。
- 注意论文 §1.4.3 文献综述自己写了"熵权 TOPSIS 分析中应优先采用向量标准化"——该句与代码一致、与 §2.7 公式矛盾，论文内部不自洽。

### 4.3 准则数据口径

论文 §2.10.3/2.10.4 描述的"50 次重复的均值/SD"与旧代码测试集口径吻合（`use_same_test_set=True`，故 SD 在 0.0004~0.003 量级）；新代码已改为训练侧 CV 口径（见下节），复现时需分清。

### 4.4 Top-N

论文写"严格筛选出综合排名**前三**"；新旧代码 `top_n_models` 默认均为 **5**（`genomic_prediction_v2.py:1610`；新代码 `workflow.py` 同）。论文实验必须是显式传了 top_n=3——建议核实运行配置，确认没有结果是用默认值 5 跑的。

### 与论文一致、无需修改的部分

- 评价准则本身：准确性（Pearson r）+ 稳定性（SD）✅
- 50×5 重复五折交叉验证协议 ✅
- TOPSIS 核心算法（理想解、欧氏距离、贴近度）✅
- Stacking 架构：K 折折外预测生成元特征、Ridge/Logistic 元学习器 ✅（新代码更严格，折内重建预处理流水线）
- "所有模型基础评估阶段采用默认参数" ✅（Optuna 仅为系统能力描述）

---

## 五、核心逻辑全链条（当前新代码 `gpse/`）

**第 1 步：每个模型跑重复 K 折 CV**
数据先切出 hold-out 测试集；训练分区上做 R 次重复 × K 折 CV（论文协议 50×5）。每折在训练折拟合模型、验证折算 Pearson。每个 repeat 得到 `avg_val_pearson`（5 折验证 Pearson 均值）与 `std_val_pearson`（5 折间标准差）（`_fold_training.py:284-287`）。

**第 2 步：每个模型提炼"准确性"和"稳定性"**
在 50 个 repeat 中选验证均值最大的代表 repeat（并列取波动小者），其 `(avg_val_pearson, std_val_pearson)` 记为该模型的 `(cv_mean, cv_std)`（`_repeat_training.py:142-170, 595-602`）。注意："稳定性"是**折间**波动，不是 50 次重复间波动。

**第 3 步：构建决策矩阵**
行 = 候选模型，两列 = [CV Pearson（max）、CV Std（min）]，写出 `model_comparison_cv.csv`（`_pipeline.py:344-360`）。全程只用训练侧数据。

**第 4 步：TOPSIS 评分（0.8/0.2 在此进入）**（`topsis.py:46-76`）
1. CV Std 正向化：`-log(x + 1e-12)`；
2. 每列 L2 向量归一化：`x / √Σx²`；
3. 乘固定权重 [0.8, 0.2]（`_pipeline.py:367`；熵权分支不执行）；
4. 正理想解 = 各列最大，负理想解 = 各列最小；
5. 欧氏距离 D⁺、D⁻，贴近度 `C = D⁻/(D⁺+D⁻)`，按 C 降序排名。

**第 5 步：Top-N 进 Stacking**
排名前 N 的模型作为基学习器（默认 5，论文实验 3；`_pipeline.py:381-382`）；rank-1 模型输出 `final_predictions.csv`。TOPSIS 异常时回退为按 CV 均值降序、std 升序直接排序（`_pipeline.py:373-380`）。

**旧代码（论文结果来源）口径**：仅第 1–3 步不同——准则为 50 次重复在同一 hold-out 测试集上的 Test Pearson 均值/SD。权重、变换、TOPSIS 算法、Top-N 逻辑两版相同。

---

## 六、TOPSIS_Score 的计算与解读

### 计算公式（与 `topsis.py` 逐步对应）

输入每模型两数：`x₁` = CV Pearson（w₁=0.8）、`x₂` = CV Std（w₂=0.2），共 m 个模型：

```
① 正向化:  y_i1 = x_i1 ;  y_i2 = -ln(x_i2 + 1e-12)
② 归一化:  r_ij = y_ij / √(Σ_i y_ij²)          （L2 向量归一化，每列分别做）
③ 加权:    z_i1 = 0.8·r_i1 ;  z_i2 = 0.2·r_i2
④ 理想解:  Z⁺ = 各列最大值 ;  Z⁻ = 各列最小值
⑤ 贴近度:  D⁺_i = √(Σ_j (z_ij − Z⁺_j)²)
           D⁻_i = √(Σ_j (z_ij − Z⁻_j)²)
           TOPSIS_Score_i = C_i = D⁻_i / (D⁺_i + D⁻_i)   ∈ [0,1]，越大越好
```

**注意：C 不是 `0.8×准确性 + 0.2×稳定性` 的加权和**，而是加权后到正/负理想解的距离比。

### 结果文件

- `model_comparison_cv_topsis.csv`：完整版，含中间量（`*_positive`、`*_norm`、`*_weight`）+ `TOPSIS_Score` + `TOPSIS_Rank`；
- `model_comparison_cv_topsis_simple.csv`：精简版，stacking 选型读此文件。

### 离线复算

```bash
python -m gpse.train.topsis \
  --input model_comparison_cv.csv \
  --output recheck.csv \
  --criteria "CV Pearson,CV Std" \
  --criteria_types max,min \
  --manual_weights 0.8,0.2 \
  --min_transform neglog
```

或手算验证：

```python
import numpy as np, pandas as pd

df = pd.read_csv("model_comparison_cv.csv")
y1 = df["CV Pearson"].to_numpy()
y2 = -np.log(df["CV Std"].to_numpy() + 1e-12)
r1 = y1 / np.sqrt((y1**2).sum())
r2 = y2 / np.sqrt((y2**2).sum())
z = np.column_stack([0.8*r1, 0.2*r2])
D_pos = np.sqrt(((z - z.max(axis=0))**2).sum(axis=1))
D_neg = np.sqrt(((z - z.min(axis=0))**2).sum(axis=1))
C = D_neg / (D_pos + D_neg)   # 与 TOPSIS_Score 一致
```

### TOPSIS_Score 与常规模型分数的区别

| 维度 | Pearson r（常规分数） | TOPSIS_Score（贴近度 C） |
|---|---|---|
| 度量对象 | 预测值 vs 真实值的拟合程度，**绝对表现** | 当前候选池内离"理想模型"的相对位置 |
| 可比性 | 跨任务、跨数据集可比较 | 仅在**同一次运行、同一批候选模型**内有效；换一个模型入池，所有人的分数都会变 |
| 含义 | r=0.85 有明确物理意义 | C=1.0 只表示"本池最优"，C=0.5 不表示"达到最优的一半" |
| 用途 | 报告预测精度 | 组内排名与 Top-N 筛选 |

论文中跨任务比较应使用原始 Pearson，TOPSIS_Score 只用于组内排名——现有数据包（figure_2）正是这么用的。

---

## 七、建议

1. **论文改措辞，代码不动**（首选）。把"基于熵权的 TOPSIS"全部改为固定权重 TOPSIS 的准确描述：
   - 需修改位置：中英文摘要、§1.4 整节、§2.7 标题与步骤 (1)(2)、§3.1 与 §4.2 标题、§4.3；
   - §2.7 标准化公式改为与代码一致的 neglog 正向化 + L2 向量归一化；
   - 替换句式可参考 `docs/methods_code_audit.md` §6；
   - 答辩安全表述："采用双准则 TOPSIS 综合评价：以重复交叉验证验证折 Pearson 均值为准确性准则（权重 0.8）、以其标准差为稳定性准则（权重 0.2），按贴近度排名筛选前 N 模型进入 Stacking。"
2. **不建议开启熵权重跑**，理由：
   - 现有全部论文数据（figure_2 panel a/b/c、manuscript_runs、6 物种）都是 0.8/0.2 产出，改权重会改变排名与 stacking 成员，结果全部作废需重训；
   - 仅 2 个准则、约 10–14 个备选模型的决策矩阵上，熵权逐任务漂移，方法学上不如显式固定权重好辩护；
   - Panel f 消融（权重敏感性）本身以固定权重为基线，且可作为"权重选择稳健"的支撑证据写入讨论；
   - 熵权分支实现有瑕疵（权重在原始数据上计算、应用于变换后矩阵）。
3. **可选工程改进**：把 `use_entropy_weights` / `manual_weights` 参数化接入 `_pipeline.py`（或接通目前未生效的 `topsis.yaml`），供 Panel f 消融传入不同权重组；熵权代码保留，可作为消融中"数据驱动赋权"的对照组。
4. **跑 Panel f 消融前**：核实每组历史结果的产出口径（旧测试集口径 vs 新训练侧 CV 口径）与 top_n 实参（3 还是默认 5），分开引用。

---

## 八、文献调研：集成方法的"可靠性"该如何论证

问题：论文中用"TOPSIS_Score 与 Pearson r 的一致性"来论证 TOPSIS 方法可靠，是否成立？

### 8.1 该论证方式不成立（循环论证）

- Pearson r（准确性）本身就是 TOPSIS 的输入准则且权重 0.8，两个排名高度一致是**数学构造的必然**，只能反映"0.8 权重足够大"，不能证明排序方法正确；不一致处（Reversal 模式）又是设计使然。一致与否都无法作为可靠性证据。
- 更根本地，"综合优劣"不存在客观真值排名，任何排名方法的内部自证都不算验证。

### 8.2 GP/GS 领域集成方法的验证惯例：下游预测性能 vs 基线

- **Liang et al. 2021, SELF stacking（Front. Genet.，被引 124，[PMC7969712](https://pmc.ncbi.nlm.nih.gov/articles/PMC7969712/)）**：基学习器（SVR/KRR/ENET）凭文献经验与多样性原则先验选定（RF 因算力剔除），无打分选型步骤；验证 = 20 折 CV 下 SELF 预测精度 vs 基学习器 / GBLUP / BayesB（3 数据集 9 性状，平均 +7.70% vs GBLUP）。
- **Nascimento et al. 2024, 咖啡 stacking（Front. Plant Sci.，被引 30，[PMC11288849](https://pmc.ncbi.nlm.nih.gov/articles/PMC11288849/)）**：基学习器 GBLUP/MARS/QRF/RF 直接集成；验证 = CV 方案下集成 vs 各基学习器与 GBLUP 的预测精度对比。

模式：**集成/选型的价值 = 留出数据上的预测增益 vs 可部署基线**，无人用内部选型分数与精度分数的一致性做论证。

### 8.3 MCDM/TOPSIS 领域的验证惯例：横向一致性 + 敏感性分析

- 用 Spearman 秩相关比较所提方法与其他**独立** MCDM 方法（VIKOR、GRA、EDAS、WASPAS）的排名（如 [MDPI Information](https://www.mdpi.com/2078-2489/17/6/553)、[Sci. Rep.](https://www.nature.com/articles/s41598-025-15775-2)）——注意是"方法 vs 方法"，不是"方法 vs 自己的输入指标"；
- 权重敏感性分析（如 [Entropy-TOPSIS 用于预测性维护模型选择](https://www.mdpi.com/2076-3417/16/17/8666)）；
- 育种材料评价中的 entropy-TOPSIS（[小麦](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2026.1907156/full)、[油茶](https://www.sciencedirect.com/science/article/pii/S0926669026018455)）不做可靠性验证——无真值，TOPSIS 即评价终点。

### 8.4 本论文的正确论证路径（数据均已在 figure_2 数据包中）

1. **下游验证（主证据）**：TOPSIS 选出的 top-3 stacking vs CV 选出的可部署最优单模型——9/11 胜出、平均 +3.3%（panel c 公平口径）；
2. **选型损失（辅助）**：TOPSIS top-1 相对 CV 最优的平均选型损失仅 0.0068（panel b claim2b）；
3. **排名一致性降级为描述性证据**：ρ=0.72 表述为"与精度排名中高度一致，同时在精度相近时引入稳定性约束"，不作可靠性支柱；
4. **权重敏感性（Panel f 消融）**：回应"0.8/0.2 是否拍脑袋"，对应 MCDM 惯例。

---

## 九、正确流程与各阶段指标口径

确认的完整流程（每阶段用什么指标严格分开）：

```
50×5 重复 CV（训练分区）→ 各模型验证折 r 均值/SD
        ↓
TOPSIS（固定权重 0.8/0.2）排名 → TOPSIS_Score（仅内部选型）
        ↓
选 Top-N 基模型（代码默认 5，论文实验 3）
        ↓
Stacking：K 折 OOF 元特征 → Ridge 元学习器 → 测试集预测
        ↓
GPSE 分数 = Pearson r（与单模型同一测试集、同一指标）
        ↓
对外比较：GPSE r (均值±SD) vs 单模型 r (均值±SD) vs GBLUP/rrBLUP
```

**一句话原则：TOPSIS 决定"谁上场"，Pearson 决定"谁更强"。** TOPSIS_Score 只解释基学习器怎么选出来的；所有"比常规模型好多少"的结论只能用 Pearson r（±SD）在相同 CV 划分下得出。

对外比较的报告组合建议：

- **主表**：每个物种×性状一行，GPSE / 各单模型 / GBLUP / rrBLUP 的 r 均值±SD，最优加粗（Liang 2021 Table 2 格式）；
- **汇总**：胜率（X/N 组合最高 r）、平均 Δr 与相对提升 %（vs CV 可部署最优单模型，不用测试集 oracle 口径）；
- **TOPSIS 价值证据**：top-3 stacking 胜率、选型损失 0.0068、ρ=0.72（描述性）、Panel f 权重敏感性。

---

## 十、GPSE（stacking）分数的计算与落盘位置

### 10.1 计算过程（`gpse/train/stacking.py`，N=3 或 5 机制相同，仅元特征列数不同）

1. **OOF 元特征**（`stacking.py:91-128`）：每个入选基模型在训练分区 K 折切分，每折重建完整预处理 pipeline 拟合并预测折外样本 → `meta_train`（n_训练 × N）；再在全部训练数据重拟合 → 预测测试集 → `meta_test`（n_测试 × N）。
2. **元学习器**（`stacking.py:130-139`）：回归用 `StandardScaler + Ridge(alpha=1.0)`，在 `(meta_train, y_train)` 上拟合，即学习 `ŷ = b + Σ wᵢ·ŷ_modelᵢ` 的最优线性组合（Ridge 正则化处理基模型预测间共线性）。
3. **最终分数**（`stacking.py:179-182, 220-230`）：`GPSE 分数 = Pearson r(元学习器(meta_test), y_测试)`，与单模型同度量、同尺度、同测试样本，可直接比较。

### 10.2 口径警示

单模型是 50 次重复的 r 均值±SD，而 stacking 当前只在 holdout 上**评估一次**（std 恒 0.0）。单次划分方差大；严谨做法是把 stacking 也纳入重复协议报告均值±SD（逐 repeat 结果在各原始运行目录，尚未汇总——见 `question_1.md` panel c 备注）。

### 10.3 落盘位置（每次运行，`optimization_results_v2/` 下）

| 文件 | 内容 |
|---|---|
| `reports/model_comparison_holdout.csv`（及 `holdout_metrics.csv/.json`） | 最终 holdout 指标表，含 `model == "stacking_ensemble"` 行（**推荐取数处**，`_pipeline.py:65-110`） |
| `model_comparison.csv` | stacking 完成后重写，追加 `stacking_ensemble` 行（`_pipeline.py:466-468`） |
| `ensemble_stacking/stacking_results.pkl` | 完整指标字典（joblib；`test_metrics.pearson` 等） |
| `stacking_selected_models.json` | 本次 GPSE 的基模型名单与选型来源 |
| `ensemble_stacking/stacking_ensemble_model.pkl` | 训练好的 stacking 模型本体 |

**⚠️ 最大的坑：`final_predictions.csv` 不是 GPSE！** 它存的是 TOPSIS rank-1 **单模型**的测试预测（`_pipeline.py:113-119` `_write_topsis_final_predictions`），名字有误导性。找不到 GPSE 分数的常见原因：① 看了 `final_predictions.csv`；② 运行时未开 `use_stacking` 或 stacking 静默失败（`_pipeline.py:470-472` 只 log 不中断，检查有无 `ensemble_stacking/` 目录）；③ std 恒 0.0 被当成中间产物忽略。

---

## 十一、结果目录结构改进提案（待确认后实施）

### 11.1 现状问题

- GPSE 分数散落于至少 4 处（holdout csv / model_comparison.csv 行 / pkl / leaderboard），无单一权威入口；
- `final_predictions.csv` 命名误导（实为 rank-1 单模型预测）；
- TOPSIS 选型明细（各模型 C 值、排名）与 GPSE 结果分属不同文件，无法一眼看到"该表型选了谁、为什么、集成后多少分"；
- 跨 run（多表型/多物种）汇总需人工拼接。

### 11.2 提案：新增单一份"选型-集成汇总"文件

每个 run 增加 `reports/selection_summary.json`（机器可读）+ `reports/selection_summary.csv`（人读），一次说清：

```json
{
  "trait": "...", "task_type": "regression", "top_n": 3,
  "topsis": {
    "criteria": ["CV Pearson", "CV Std"], "weights": [0.8, 0.2],
    "min_transform": "neglog", "selection_source": "train_only_cv",
    "candidates": [
      {"model": "kernelridge", "cv_mean": 0.552, "cv_std": 0.0014,
       "topsis_score": 1.000, "topsis_rank": 1, "selected": true},
      {"model": "xgboost", "topsis_score": 0.87, "topsis_rank": 2, "selected": true}
    ]
  },
  "gpse": {
    "base_models": ["kernelridge", "xgboost", "svr"],
    "meta_learner": "Ridge(alpha=1.0)",
    "pearson": 0.61, "spearman": 0.60, "r2": 0.35, "rmse": 0.04,
    "evaluation": "single_holdout"
  }
}
```

配套改动：

1. `final_predictions.csv` 改名为 `topsis_top1_holdout_predictions.csv`（或至少加 README 说明），消除"final = GPSE"的误读；
2. batch 层（`batch/`）增加聚合脚本：遍历各 run 的 `selection_summary.json`，产出跨表型汇总表（每行：物种/表型、top_n、入选模型、GPSE pearson、最优单模型 pearson、Δr）——直接对接 figure_2 数据包需求；
3. （可选，配合第十节口径警示）将 stacking 纳入重复评估协议后，在汇总中给出 `pearson_mean ± pearson_std`。

该提案不改变任何计算逻辑，只是**重组与汇总已有输出**，对历史结果目录也可离线回填（topsis 明细、selected_models、stacking_results.pkl 均已在盘上）。

---

*核查人：Kimi Code；核查日期：2026-10-22；相关文件：`question_1.md`、`毕业论文答辩_刘晓.pdf`、`docs/methods_code_audit.md`。第八至十一节为同日后续讨论（文献调研、流程口径、GPSE 分数定位、目录结构提案）的补充。*
