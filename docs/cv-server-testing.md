# CV 修复：其他服务器测试说明

这份交接包包含独立的 CV 统计修复补丁，基于提交
`3e6a5357051a990fb21f52f068e5f3fc2ee550b2`。
包中未包含当前工作区并行进行的 V3 结果目录改造，使用原有结果目录结构。
迁移补丁已在原始提交的隔离副本验证；不要将它叠加到已修改相同文件的 V3 分支。

## 应用补丁

将 `gpse-cv-server-handoff.tar.gz` 复制到服务器并解压。进入服务器 GPSE 仓库：

```bash
sha256sum -c SHA256SUMS
cd /path/to/GPSE
git rev-parse HEAD
git apply --check /path/to/gpse-cv-server-handoff/cv-statistics.patch
git apply /path/to/gpse-cv-server-handoff/cv-statistics.patch
```

第一条校验命令在解压后的 `gpse-cv-server-handoff/` 目录运行。
`git apply --check` 必须通过才应用。其他基线版本需要先核对差异。
使用已有 GPSE Python 环境；如果通过安装包调用，应用源码后执行 `python -m pip install -e .`。
这份包没有模型、历史实验数据、账号配置或运行日志。

## 自动化验收

在应用后的仓库根目录执行：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest \
  tests/test_cv_statistics.py tests/test_train_pipeline_optimization.py -q
```

测试包括两个候选模型实际执行 50×5，走正式 TOPSIS 和 Stacking；另一次固定训练数据与种子，
只改变 holdout 标签，验证全部 CV 聚合指标相同。这里使用小型合成数据，不读取真实基因组数据。

全仓测试可以另跑 `python -m pytest tests -q`。原始提交已知的失败项是：
`test_threadpoolctl_declared_in_pyproject`、`test_find_result_file_depth_limit` 和有波动的
`test_multi_trait_genotype_files_are_hardlinked`；不能用这些既有问题冒充 CV 回归。

## 真实数据训练

沿用原实验命令的模型清单、预处理、调参配置和 Stacking Top-N，明确设置：

```text
--n_repeats 50 --train_folds 5 --random_seed 42
--cv_stability_metric cv_std_repeat --results_dir 一个全新的目录
```

不要指向旧实验目录。默认稳定性使用 repeat 均值的样本 SD（ddof=1），不接受单 repeat。
若只是单 repeat 冒烟测试，必须显式选择 `--cv_stability_metric cv_std_within`，
这不替代正式 50×5 实验。

每个成功候选模型应有 250 条折记录和 50 条 repeat 记录；在 `model_comparison_cv.csv`
和 TOPSIS 输出中核对 `selection_source=train_only_cv`、`accuracy_metric=cv_mean`、
`stability_metric=cv_std_repeat`、`weight_method=manual`、`weights=[0.8,0.2]`。
正式准确性不能是 `cv_best_repeat_mean`；后者只用于诊断。

## 历史数据离线重聚合

输入是包含 `{model}/repeat_{i}/repeat_results.json` 的旧结果根目录，
输出必须是历史输入树外一个尚不存在的新目录：

```bash
python -m gpse.tools.reaggregate_cv /path/to/OLD_RUN \
  --output /path/to/NEW_CV_ANALYSIS --expected-repeats 50 --expected-folds 5
```

无需重训折模型。完整指标并且共享划分可验证时，生成：

- `cv_fold_metrics.csv`、`cv_repeat_metrics.csv`
- `model_comparison_cv.csv`、`cv_aggregation_summary.json`
- `model_comparison_cv_topsis.csv`、`model_comparison_cv_topsis_simple.csv`
- `cv_ranking_comparison.csv`：旧最佳 repeat 策略与新全部重复聚合的排名对照
- `reaggregation_audit.json`：恢复范围、缺失/损坏输入、需重拟合及训练来源未知的模型

缺失或损坏数据会产生审计且命令返回非零；不会用剩下的候选模型生成看似完整的排名。
旧结果不会被修改。应先恢复坏文件，确实无法恢复原始指标时再决定补跑 CV。

外部验收后可提供 `cv_ranking_comparison.csv`、`reaggregation_audit.json` 和失败测试日志，
即可核对真实排名变化与逐实验重训清单，不需要传输全部模型文件。
