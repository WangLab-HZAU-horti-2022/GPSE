# GPSE Results Architecture v3
## 训练结果目录标准化与可追溯性重构实施计划

**版本：** v1.0  
**目标输出根目录：** `optimization_results_v3/`  
**原始输出根目录：** `optimization_results_v2/`  
**改造类型：** Output / Storage / Artifact Organization Refactoring  
**方法学变更：** 无  
**核心原则：** 只修改输出结构，不改变计算结果。

---

## 1. 项目背景

目前，GPSE 训练流程已能够完成单模型训练、重复交叉验证、TOPSIS 模型选择、Stacking 集成与结果评估。

但结果文件随着功能增长不断累积，逐渐出现以下问题：

1. 根目录、模型子目录和 `reports/` 之间缺乏明确职责。
2. TOPSIS 决策矩阵、排名与入选模型分别存储，关联困难。
3. GPSE Stacking 的指标分散在 CSV、JSON 和 PKL 中。
4. `final_predictions.csv` 实际是 TOPSIS Rank-1 单模型预测，名称存在误导。
5. Stacking 运行状态缺少可靠的结构化记录。
6. 单模型、折模型、部署集成模型和 GPSE 模型的身份不够清晰。
7. 结果文件存在覆盖写入和累积追加行为，不易区分运行边界。
8. 多物种、多表型及不同配置的结果缺少统一索引。
9. 历史模型文件与下游读取逻辑对现有路径存在依赖。

本次重构的目标，是建立清晰、稳定、可检索、可扩展的 GPSE V3 结果目录。

**本任务不处理任何机器学习方法学问题。**

## 2. 重构范围和硬性约束

### 2.1 允许修改

- 默认结果输出根目录。
- 结果文件的保存路径。
- 结果文件名称。
- 目录自动创建逻辑。
- 输出路径配置与管理模块。
- 模型和结果文件的读取路径。
- 结果文件索引与 Manifest。
- 已有结果字段的只读汇总和展示。
- 日志输出路径。
- 结果目录 README。
- 跨 Run 的现有结果收集机制。
- 路径兼容逻辑。
- 与输出结构相关的测试和文档。

允许在包含训练代码的模块中修改**纯 I/O 路径表达式或调用输出模块的位置**，但不得修改这些模块的数值计算行为。

### 2.2 严格禁止修改

- 模型算法及超参数。
- 训练流程和模型拟合行为。
- 特征预处理、标准化和特征选择逻辑。
- 交叉验证划分、重复次数和随机种子。
- CV 聚合方法及指标公式。
- 最佳 repeat 选择逻辑。
- Pearson、Spearman、RMSE、R² 等指标计算。
- TOPSIS 权重、正向化、归一化与评分公式。
- TOPSIS 候选模型筛选规则。
- Top-N 数量及默认值。
- Stacking 的 OOF 构造和基学习器训练。
- Ridge 元学习器训练与预测。
- Holdout 数据划分与评价。
- 模型预测和反标准化计算。
- 训练异常后的原有控制流及退出语义。

不得借本次重构修复已有算法缺陷。

发现计算逻辑问题时，只记录到独立的问题清单，不得直接修改。

### 2.3 必须保持的行为

在相同数据、配置、划分和随机种子条件下：

- 训练参数不变。
- 各 CV 指标不变。
- TOPSIS 分数与排名不变。
- 入选 Stacking 的模型不变。
- GPSE 的预测行为不变。
- Holdout 评估指标不变。
- 已训练模型能够正常序列化、加载和预测。

输出路径允许改变，数值和模型语义不得改变。

## 3. V3 目录结构设计

默认目录：

```text
optimization_results_v3/
├── README.md
├── runs_index.csv
├── runs/
│   └── <run_id>/
│       ├── README.md
│       ├── run_manifest.json
│       │
│       ├── data/
│       │   ├── split_manifest.json
│       │   ├── train_ids.txt
│       │   └── test_ids.txt
│       │
│       ├── models/
│       │   └── <model_id>/
│       │       ├── summary_results.json
│       │       ├── deployment_ensemble/
│       │       ├── representative_model/
│       │       └── repeat_<n>/
│       │           ├── repeat_results.json
│       │           ├── all_predictions.json
│       │           ├── cv_train_only.csv
│       │           ├── optuna_best_params.json
│       │           └── fold_<k>_model.pkl
│       │
│       ├── selection/
│       │   ├── model_comparison_cv.csv
│       │   ├── model_comparison_cv_topsis.csv
│       │   ├── model_comparison_cv_topsis_simple.csv
│       │   ├── stacking_selected_models.json
│       │   └── selection_summary.json
│       │
│       ├── ensembles/
│       │   └── gpse/
│       │       ├── stacking_ensemble_model.pkl
│       │       └── stacking_results.pkl
│       │
│       ├── evaluation/
│       │   ├── model_comparison.csv
│       │   ├── model_comparison_holdout.csv
│       │   ├── holdout_metrics.csv
│       │   ├── holdout_metrics.json
│       │   └── predictions/
│       │       └── topsis_top1_holdout_predictions.csv
│       │
│       ├── reports/
│       │   ├── model_leaderboard.csv
│       │   ├── run_summary.json
│       │   ├── training_report.md
│       │   └── training_report.html
│       │
│       └── logs/
│
└── aggregates/
    ├── README.md
    └── runs_summary.csv
```

说明：

1. 上述为逻辑目标结构，实际只创建本次运行需要的文件。
2. `representative_model/`、Optuna 文件等仅在现有逻辑实际产生时保存。
3. 不要求每个 Run 都具有 Stacking 产物。
4. `evaluation/predictions/` 仅存放现有代码已经生成并能够直接持久化的预测结果。
5. GPSE 逐样本预测在现有实现中没有独立落盘，本次不得为补齐该文件而新增预测运算。
6. 如果已存储的结果中不存在 GPSE 逐样本预测，应在 Manifest 中明确标记未提供。
7. 不复制大型模型文件来制造多个目录视图，保证每个正式模型 Artifact 只有一个权威物理位置。
8. 不在本次重构中启用任何新指标或新实验协议。

## 4. Run 级隔离机制

每次独立训练执行都必须具有唯一的 `run_id`。

建议：

`run_<YYYYMMDD_HHMMSS>_<short_uuid>`

每个 Run 具有独立输出空间：

`optimization_results_v3/runs/<run_id>/`

要求：

- 不同 Run 不得覆盖文件。
- 不同表型不得在同一运行目录中混合。
- 不同随机种子或不同实验配置应独立记录。
- 相同配置重新运行仍应产生新的 Run 身份。
- 不允许仅使用时间戳作为唯一性保证。
- 原有单次任务内多模型训练仍属于同一个 Run。
- 原有批量执行中每个独立训练任务使用独立 Run 目录。
- 并发运行时必须使用原子化的目录创建及冲突检测。

如果项目当前的批量执行使用共享结果目录，需要识别真实训练任务边界，不能简单把同一批任务错误归并为一个 Run。

## 5. 统一输出路径管理

新增统一的输出路径管理组件，例如：

`gpse/output/paths.py`

名称和所在位置可根据仓库实际架构调整。

建议提供：

- `ResultPaths`
- `create_run_directory`
- `get_model_dir`
- `get_selection_dir`
- `get_ensemble_dir`
- `get_evaluation_dir`
- `get_report_dir`
- `get_logs_dir`
- `resolve_artifact_path`

所有输出路径统一从 `ResultPaths` 获取。

禁止继续在多个训练模块中自行拼接结果目录字符串。

该模块只负责路径与目录，不得引入 NumPy 模型训练逻辑、指标计算或 TOPSIS 实现。

### 5.1 默认目录调整

将默认结果根目录从：

`optimization_results_v2/`

改为：

`optimization_results_v3/`

需要调查并同步修改所有实际生效的默认路径、配置、CLI 和输出文档。

不能只替换 `workflow.py` 中的一处字符串。

### 5.2 自定义结果目录兼容

保留当前 `--results_dir` 的可用性。

由于 V3 引入 Run 级目录，必须明确区分结果根目录与单次运行目录，不能静默改变已有自动化脚本对 `--results_dir` 的解释。

实施前调查当前参数语义，并设计兼容方案。可新增显式 `--results_layout v3|legacy` 或等价配置。

要求：

- 默认新任务使用 V3。
- 显式兼容模式保留旧式直接写入行为。
- 用户指定自定义根目录时，V3 能创建独立 Run。
- 旧版读取接口能够根据明确的布局标识解析历史路径。
- 不允许隐式猜测路径后把不相关的历史目录视为当前 Run。

## 6. 现有文件映射

### 6.1 数据划分

| 旧路径 | V3 路径 |
|---|---|
| `split_manifest.json` | `data/split_manifest.json` |
| `train_ids.txt` | `data/train_ids.txt` |
| `test_ids.txt` | `data/test_ids.txt` |

只迁移文件路径，不改变划分数据。

### 6.2 单模型训练

原有：

`<model>/summary_results.json`

更改为：

`models/<model>/summary_results.json`

原有：

`<model>/repeat_<n>/...`

更改为：

`models/<model>/repeat_<n>/...`

原有的：

- fold 模型文件
- repeat 指标
- 逐样本预测
- CV 划分记录
- Optuna 参数
- deployment ensemble
- representative model

均保留原有结构和内容，仅增加上级 `models/` 分类。

不强制重命名 repeat、fold 或模型文件，避免破坏模型引用与读取逻辑。

### 6.3 TOPSIS 选型

将以下文件统一放入 `selection/`：

- `model_comparison_cv.csv`
- `model_comparison_cv_topsis.csv`
- `model_comparison_cv_topsis_simple.csv`
- `stacking_selected_models.json`

这些文件的计算与字段保持不变。

新增的 `selection_summary.json` 只读取已有选型与评估文件，整合已有信息。

不得重新计算 TOPSIS 或自行重新确定模型排名。

### 6.4 Stacking

原有：

`ensemble_stacking/`

更改为：

`ensembles/gpse/`

保留：

- `stacking_ensemble_model.pkl`
- `stacking_results.pkl`

同时修改所有依赖旧模型路径的读取位置。

不得重新训练模型，也不得通过重新保存已加载模型的方式执行文件迁移。

### 6.5 最终评估

原有 Holdout 相关文件统一放入：

`evaluation/`

其中：

`final_predictions.csv`

调整为：

`evaluation/predictions/topsis_top1_holdout_predictions.csv`

这个文件仍然只包含现有 TOPSIS Rank-1 单模型的预测，不能改名为 GPSE 预测。

原有 GPSE Holdout 指标仍然取自现有评估结果。

`model_comparison.csv` 按原有生成逻辑保留其最终内容；如需保留覆盖前版本，只能在原有写入边界添加输出快照，不得重新计算前置数据。

### 6.6 报告

现有：

- `model_leaderboard.csv`
- `run_summary.json`
- `training_report.md`
- `training_report.html`

放入 `reports/`。

不得修改报告中数值字段的计算逻辑。

只允许更新内部路径、相对链接、展示名称及文件定位说明。

## 7. Run Manifest

为每个 Run 创建：

`run_manifest.json`

建议 Schema：

```json
{
  "schema_version": "1.0",
  "results_layout": "gpse_v3",
  "run_id": "run_20261022_103000_ab12cd",
  "species": null,
  "trait": null,
  "task_type": null,
  "status": "unknown",
  "execution": {
    "started_at": null,
    "completed_at": null,
    "code_commit": null,
    "config_path": null
  },
  "selection": {
    "status": "unknown",
    "method": "topsis",
    "source": "train_only_cv",
    "artifact": "selection/stacking_selected_models.json"
  },
  "stacking": {
    "status": "unknown",
    "model_artifact": null
  },
  "artifacts": {}
}
```

正式设计时扩展：

- 已知配置与方法版本
- 数据来源和已有数据指纹
- 实际运行的候选模型
- 输出 Artifact 的相对路径
- 文件大小和校验值
- 产物类型与内容语义
- 运行状态及失败信息
- Manifest 生成时间和 Schema 版本

要求：

1. 字段必须来自真实配置、已有产物或可验证的文件元数据。
2. 不得通过目录存在推断全部训练成功。
3. 不得把缺失数据自动填充为 0。
4. 无法确认的状态标记为 `unknown`。
5. 所有 Artifact 路径采用 Run 根目录下的相对路径。
6. 清单采用临时文件加原子替换方式写入，避免留下部分 JSON。
7. 产物哈希只用于验证文件一致性，不参与模型计算。

## 8. 选型与集成汇总

新增：

`selection/selection_summary.json`

它需要把已有结果中的以下信息关联起来：

- 候选模型名单
- TOPSIS 输入准则
- TOPSIS 实际权重
- TOPSIS Score 和 Rank
- 入选 Stacking 的模型
- Top-N 配置
- Stacking 执行状态
- 已有 GPSE 评估指标
- 相关 Artifact 路径

注意：

- 只做字段提取、结构转换与来源关联。
- 不重新计算 CV 均值。
- 不重新计算 TOPSIS 分数。
- 不重新计算 GPSE Pearson。
- 不执行 GPSE 推理。
- 不通过 TOPSIS 排名自行补造缺失的入选名单。
- 已有记录冲突时，必须报告冲突及其来源。

可另外生成面向用户的简洁 CSV 展示，但 JSON 应作为该汇总内容的唯一权威版本。

原始训练与评估文件仍是数值事实来源，`selection_summary.json` 是只读生成的聚合视图，不应成为另一套计算来源。

## 9. Artifact 注册机制

为输出层建立 Artifact Registry。

建议为每个文件记录：

- `artifact_id`
- `artifact_type`
- `relative_path`
- `producer`
- `source_artifact`
- `file_size`
- `checksum`
- `created_at`
- `status`

例如：

- `cv_model_summary`
- `cv_repeat_result`
- `topsis_decision_matrix`
- `topsis_ranking`
- `stacking_selection`
- `gpse_model`
- `gpse_metrics`
- `holdout_metrics`
- `topsis_top1_predictions`

Artifact Registry 只管理文件身份和来源，不接管计算功能。

不引入数据库、对象存储或复杂后台服务，优先使用现有文件系统和 JSON。

## 10. 运行状态与缺失产物

运行级状态与组件级状态分离。

组件至少支持：

- `not_requested`
- `running`
- `succeeded`
- `failed`
- `skipped`
- `unknown`

对 Stacking：

- 明确未启用：`not_requested`
- 有可信成功结果及完成记录：`succeeded`
- 有明确异常证据：`failed`
- 主动跳过且有记录：`skipped`
- 无法核实：`unknown`

不得仅因为缺少 `stacking_results.pkl` 就一律标记失败。

不得为了完善状态记录修改现有异常处理或吞掉异常。

## 11. V2 历史兼容

历史路径：

`optimization_results_v2/`

必须保持原样。

禁止：

- 自动删除旧目录。
- 自动覆盖旧文件。
- 直接移动旧版大型模型。
- 修改旧版 PKL 内容。
- 假定旧版实验符合新版方法学。
- 将历史 Test-based TOPSIS 结果标记为 Train-only CV。

建议实现 `ResultLocator` 或等价的路径解析层，显式支持：

- V3 规范目录。
- V2 旧版目录。

兼容策略：

1. 新运行默认写入 V3。
2. 历史文件按原路径读取。
3. 不要求先迁移 V2 才能查看历史结果。
4. 如确需历史整理，只生成只读索引或独立副本。
5. 无法确认结果语义时保留原始字段，标记来源为 legacy。
6. 历史模型是否可从新路径加载必须经过独立测试。

兼容期内保留旧版模型 ID 与机器字段，不强制修改所有既有消费者。

## 12. 跨 Run 索引

在 V3 根目录生成：

`runs_index.csv`

每行对应一个 Run，包含：

- run_id
- species
- trait
- task_type
- started_at
- status
- stacking_status
- results_path

跨物种、跨表型的已有结果整理写入：

`aggregates/runs_summary.csv`

这个汇总只收集已经存在的指标及路径引用。

不计算新的性能增益、均值、标准差或统计显著性。

如果需要额外计算论文统计，应作为独立分析任务执行。

## 13. 实施阶段

### P0：源码与读写依赖审计

实施内容：

1. 全局搜索 V2 路径和结果文件名。
2. 列出全部生产者、消费者及文件打开模式。
3. 调查模型加载与 PKL 依赖。
4. 调查 batch、CLI、报告和 predict 模块。
5. 确认 `--results_dir` 的真实语义。
6. 记录所有文件的当前读写时机。
7. 建立一份 V2 基准实验输出清单。

交付：

`docs/results_v3_io_inventory.md`

验收：每个关键输出文件有明确生产者与消费者，不遗漏重要路径引用。

### P1：路径管理与 V3 Run 初始化

实施内容：

1. 创建 `ResultPaths`。
2. 设置 V3 默认根目录。
3. 实现 Run ID。
4. 创建 Run 目录。
5. 创建基础 Manifest。
6. 实现原子目录创建及路径校验。
7. 保留明确的 legacy 兼容模式。

验收：不启动训练即可正确创建独立运行目录；重复、并发和自定义路径测试通过。

### P2：单模型与数据输出迁移

实施内容：

1. 迁移 split 与 sample ID 文件。
2. 迁移模型子目录。
3. 保留原有 repeat/fold 内部结构。
4. 更新相关模型加载与引用路径。
5. 更新日志路径。

验收：相同配置的训练能够生成原有全部文件，模型文件可以正常加载。

### P3：TOPSIS、Stacking 与评估输出迁移

实施内容：

1. 将 TOPSIS 相关文件统一到 `selection/`。
2. 将 GPSE 模型和指标统一到 `ensembles/gpse/`。
3. 将 Holdout 指标统一到 `evaluation/`。
4. 修正 `final_predictions.csv` 的对外文件名称。
5. 修改所有结果消费者的路径引用。
6. 保持原有结果数据和模型内容不变。

验收：TOPSIS 排名、选中模型、GPSE 指标与基准一致。

### P4：结果索引与只读汇总

实施内容：

1. 完善 Run Manifest。
2. 增加 Artifact Registry。
3. 生成 Selection Summary。
4. 生成根目录 README。
5. 生成 Runs Index。
6. 更新训练报告内部路径。

验收：无需阅读原始代码，即可定位一次 Run 的模型、选型、指标和报告。

### P5：兼容性与回归验证

实施内容：

1. V2 旧结果读取测试。
2. V3 新结果读取测试。
3. 自定义输出目录测试。
4. 并发 Run 测试。
5. 单模型运行测试。
6. Stacking 开启和关闭测试。
7. Stacking 原有失败路径测试。
8. 已有模型加载与预测测试。
9. batch 多表型结果隔离测试。
10. 全量文件和数值回归测试。

验收：

- 所有必要产物齐全。
- 产物引用有效。
- 模型可加载。
- 原始指标与预测行为保持一致。
- 不出现新的结果覆盖。
- V2 原始目录保持不变。

## 14. 回归验证策略

使用一组已知成功的 V2 基准运行。

对照 V3：

**文件层：**

- 应生成文件数量
- 文件类型
- 文件大小
- 模型 Artifact 完整性
- 文件引用正确性

**数值层：**

- CV 指标
- TOPSIS Score
- TOPSIS Rank
- Stacking 入选模型
- GPSE 评估指标
- Holdout 指标
- 现有逐样本预测值

新输出的路径、时间戳和 Manifest 元数据允许不同。

除明确的路径相关字段外，已有结果中的指标数据必须保持一致。

对 PKL 等二进制产物，优先验证同一产物的字节完整性；跨两次独立训练时，使用可重复性控制下的模型加载和预测一致性验证，不能单纯要求二进制哈希完全相等。

不能因新旧运行本身存在随机性，就未经核查地认定路径重构改变了算法。

## 15. 风险及应对

### 风险 A：已有代码硬编码路径

应对：统一路径管理，逐项更新消费者；禁止无调查的全局字符串替换。

### 风险 B：模型文件包含相对或绝对路径引用

应对：调查序列化对象和加载逻辑；必要时保留兼容解析，不修改模型内部计算。

### 风险 C：结果目录中存在覆盖与追加行为

应对：通过 Run 级隔离解决跨运行覆盖；同一 Run 的原有写入行为只做路径适配，不私自改变数据生成时机。

### 风险 D：新旧指标语义不一致

应对：原始字段原样保留，记录来源和方法版本；不得重新解释历史指标。

### 风险 E：新架构复制大量折模型导致磁盘膨胀

应对：正式模型只保存一次，Manifest 使用引用而不是复制。

### 风险 F：批量运行结果混淆

应对：使用 Run ID 和独立运行目录；聚合层只通过 Manifest 搜索结果。

## 16. 最终验收标准

重构完成必须满足：

- [ ] 新默认输出根目录为 `optimization_results_v3/`。
- [ ] 每次独立训练创建唯一 Run 目录。
- [ ] 目录符合 V3 规范。
- [ ] 模型训练和评估计算代码未被修改。
- [ ] CV 结果不变。
- [ ] TOPSIS 计算结果不变。
- [ ] Stacking 基模型名单不变。
- [ ] GPSE 计算结果不变。
- [ ] 所有原有必需文件成功保存。
- [ ] `final_predictions.csv` 的语义误导得到解决。
- [ ] Run Manifest 与 Artifact 引用完整。
- [ ] 没有从缺失数据中伪造指标或预测结果。
- [ ] V2 历史数据不受影响。
- [ ] 现有模型加载接口可正常使用。
- [ ] 批量运行不发生文件覆盖。
- [ ] 主要回归测试通过。
- [ ] 文档、CLI 帮助与示例路径同步更新。

**完成定义：GPSE 原有计算完全不变，但每次训练的文件能够按照统一 V3 结构自动保存、被准确定位、被正确读取，并与历史 V2 结果明确隔离。**