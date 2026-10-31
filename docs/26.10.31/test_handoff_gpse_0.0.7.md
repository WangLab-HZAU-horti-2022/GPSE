# GPSE 0.0.7 高性能服务器测试交接文档

> 日期：2026-10-31
> 代码基线：commit `a7bea9c`（`chore(release): bump version to 0.0.7`），上一提交 `2d48e20`（HPO 文档与示例配置）
> 构建产物：`dist/gpse-0.0.7-py3-none-any.whl`、`dist/gpse-0.0.7.tar.gz`
> 测试目标：验证 **Optuna HPO 默认开启** 后，在真实数据上相对历史默认参数结果的精度提升；同时确认 0.0.7 全流程产物完整。

---

## 一、本次变更摘要（测试对象）

| 变更 | 内容 | 影响 |
|---|---|---|
| HPO 使用方式澄清 | `gpse train` 不传 `--use_default_params` 即启用 Optuna 调参（CLI 默认行为未变，本次是文档与示例配置纠正） | 训练时间显著增加，精度应提升 |
| `batch/batch_config.example.yaml` | `use_default_params: false`，新增 `trials: 100`、`patience: 20` | batch 模板默认走 HPO |
| 版本号 | `pyproject.toml`、`gpse/config/software.yaml`、`gpse/config/default.yaml` 统一为 0.0.7 | `gpse --version` 应显示 0.0.7 |

关键机制（测试前必读）：

- **冻结复用**：默认只在 repeat 1 做 Optuna 搜索，后续 repeat 冻结复用该参数（`gpse/train/_repeat_training.py:458-459`）。加 `--optuna_per_repeat` 才每个 repeat 重调（时间 ×50，不建议首轮使用）。
- **调参目标**：`mean(CV) − λ·std(CV)`，λ 由 `--cv_stability_penalty`（默认 0.5）控制。
- **例外**：`gblup_reg` 为固定参数实现，不参与 HPO，属正常。
- `tests/validation/` 下的配置仍是 `--use_default_params`（历史复现基线），不要拿它们当 HPO 测试入口。

---

## 二、服务器环境准备

```bash
# 1. 拉取/确认代码
cd /path/to/gpse
git log --oneline -2   # 应看到 a7bea9c / 2d48e20

# 2. 安装 0.0.7（建议独立环境）
python -m pip install dist/gpse-0.0.7-py3-none-any.whl

# 3. 验证
gpse --version                 # Version 必须显示 0.0.7
gpse convert --check-deps      # PLINK 1.9 必须可用；Java 仅 Beagle 插补时需要
python -m pip freeze > run_metadata_pip_freeze.txt   # 归档环境
```

资源前提：

- HPO 成本 ≈ 默认参数的 `trials × n_splits` 倍（冻结机制下仅 repeat 1 承担）。`--trials 100`、5 折时约 500 次 fit/模型。
- `--threads N` 不得超过服务器可用核数（会校验 affinity/cgroup，超限直接报错）。示例配置里的 `threads: 60` 是占位值，务必改成实际核数或 `auto`。
- 大矩阵（20 万+ SNP）下 MLP / KernelRidge / KNN 内存占用高，首轮建议保守线程数观察。

---

## 三、测试矩阵

按 P0 → P1 → P2 顺序执行，每一层通过后再进下一层。

### P0：冒烟测试（验证流程，不看精度）

目的：确认安装、数据加载、CV 解析、HPO 触发、产物结构全部正常。

```bash
gpse train \
  --geno_file <geno.csv> \
  --pheno_file <pheno.csv> \
  --target_trait <TRAIT> \
  --task_type regression \
  --models svr_reg kernelridge_reg \
  --trials 10 \
  --n_repeats 2 \
  --n_splits 5 \
  --test_size 0.3 \
  --random_seed 42 \
  --threads 8 \
  --results_dir test_007/smoke_<TRAIT>
```

通过标准：

- 日志中出现 Optuna 搜索过程（trial 计数），而不是 "使用默认参数/Using default parameters"；
- `test_007/smoke_<TRAIT>/run_manifest.json`、`feature_manifest.json`、`data/split_manifest.json`、`models/<model>/summary_results.json` 均存在；
- 无 Python traceback 退出。

### P1：HPO vs 默认参数对比（核心精度验证）

同一性状、同一 CV 文件、同一种子，跑两遍——唯一变量是 HPO 开关：

```bash
# A 组：HPO 开启（0.0.7 推荐配置）
gpse train \
  --geno_file <geno.csv> --pheno_file <pheno.csv> --target_trait <TRAIT> \
  --models elasticnet_reg gbdt_reg svr_reg mlp_reg knn_reg rf_reg xgboost_reg \
           adaboost_reg lightgbm_reg catboost_reg kernelridge_reg \
           histgradientboost_reg sgd_reg lasso_reg \
  --trials 100 --patience 20 \
  --n_repeats 50 --n_splits 5 --test_size 0.3 --random_seed 42 \
  --cv_file <历史_cv_50x5.csv> --use_same_test_set \
  --use_stacking --top_n_models 3 --cv_folds 5 \
  --threads <实际核数> \
  --results_dir test_007/hpo_<TRAIT>

# B 组：默认参数（历史口径对照）
gpse train ...（同上参数，仅去掉 --trials/--patience，加 --use_default_params）... \
  --results_dir test_007/default_<TRAIT>
```

判定标准：

| 检查项 | 期望 |
|---|---|
| A 组单模型 holdout Pearson 均值 ≥ B 组 | 大多数模型应提升；个别模型持平可接受 |
| A 组 TOPSIS rank-1 模型 holdout Pearson | ≥ B 组对应值 |
| A 组 stacking（`ensembles/gpse/`）holdout Pearson | ≥ B 组 |
| 若 A 组多数模型反而更差 | 异常，回收日志与 `summary_results.json` 反馈 |

> 若与历史 6 物种基线（`docs/6species_result_baseline.tsv`）对比：B 组单模型 Test Pearson 与基线绝对差应 ≤ 0.02，超差先按 `docs/6species_reproduction_validation.md` §12.2 的排查顺序检查（二次标准化、CV 文件用错、test_size/种子不一致、混入 gblup_reg 等）。

### P2：batch 多性状全流程（可选，HPO 生效后跑全量）

复制 `batch/batch_config.example.yaml` 为 `my_batch.yaml`，修改：

```yaml
defaults:
  geno_file: /path/to/geno.csv
  pheno_file: /path/to/pheno.csv
  task_type: regression
  use_default_params: false
  trials: 100
  patience: 20
  n_repeats: 50
  n_splits: 5
  test_size: 0.3
  use_stacking: true
  top_n_models: 3
  threads: <实际核数>
  results_root: test_007/batch_results

traits:
  - name: <TRAIT1>
    cv_file: /path/to/TRAIT1_cv_50x5.csv
  - name: <TRAIT2>
    cv_file: /path/to/TRAIT2_cv_50x5.csv
```

```bash
gpse batch --config my_batch.yaml --dry_run   # 先预览，确认每条命令含 --trials 100 且无 --use_default_params
gpse batch --config my_batch.yaml             # 正式执行
```

通过标准：每个性状目录产物完整（见第四节清单），`<results_root>/merged/` 汇总表生成。

### 可选增强（独立变量，单独跑，不要混进 P1）

- 折内特征选择：`--feature_selection univariate --select_k 5000`（p>>n 数据预期收益最大）；
- 缺失插补：`--genotype_imputation mean`（convert 输出含缺失编码 3 时）；
- 每个 repeat 重调：`--optuna_per_repeat`（时间 ×50，仅小数据验证）。

---

## 四、结果回收清单

每个 run 目录需要回收：

```text
run_manifest.json
reports/model_comparison_holdout.csv        # holdout 指标总表（首选取数处）
selection/model_comparison_cv.csv           # TOPSIS 输入（CV 口径）
selection/model_comparison_cv_topsis_simple.csv
selection/stacking_selected_models.json     # stacking 基模型名单
ensembles/gpse/stacking_results.pkl         # GPSE 集成完整指标
models/<model>/summary_results.json         # 每个模型的 repeat 汇总
logs/                                       # 完整日志
```

跨 run 汇总时建议填表：

| Trait | 组别 | rank-1 model | rank-1 holdout r | stacking r | 最优单模型 r | 备注 |
|---|---|---|---|---|---|---|

环境归档：`git rev-parse HEAD`、`gpse --version`、`pip freeze`、运行命令原文。

---

## 五、常见问题排查

| 症状 | 排查 |
|---|---|
| `--threads must be between 1 and N, or auto` | 线程预算超过检测到的核数；改小或用 `auto` |
| 日志没有 Optuna trial 输出 | 检查命令是否残留 `--use_default_params`；`gblup_reg` 本来就无 HPO |
| CatBoost 大量并发时进程池异常 | 历史已知问题（见 `docs/6species_reproduction_validation.md` §13.1），降低 `--threads` 或单独补跑 |
| LightGBM 报列名不支持字符 | 特征名含 `:|[]{},` 等，`gpse convert` 会自动清洗；自查输入矩阵来源 |
| 内存爆（MLP/KernelRidge/KNN） | 降 `--threads`；大矩阵先跑 `--feature_selection univariate --select_k 5000` |
| 结果想对比历史但数字对不上 | 按 `docs/6species_reproduction_validation.md` §12.2 顺序排查；不要换种子追数 |

---

## 六、测试完成后回传

1. 第四节汇总表（HPO vs 默认参数）；
2. 异常 run 的 `logs/` 与 `summary_results.json`；
3. `pip freeze` 与运行命令归档；
4. 结论：HPO 是否在各性状上稳定优于默认参数、是否有模型调参后反而变差（反馈后可针对性收窄该模型的搜索空间）。

---

*相关文档：`docs/6species_reproduction_validation.md`（历史复现 SOP）、`docs/26.10.22/topsis_investigation_report.md`（口径审计）、README "Legacy scripts/ Results Are Not a Fair Baseline" 章节。*
