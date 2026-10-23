# KernelRidge 的 gamma 与数值策略

默认 `kernelridge_reg` 现在搜索 `gamma_scale`，每次拟合时按 `gamma = gamma_scale / p` 计算实际 gamma。这里的 p 是该次拟合经过插补、特征选择/PCA 后实际输入模型的列数，不使用完整数据提前拟合的统计量。相同规则用于 Optuna fold、常规 CV、代表模型、部署 ensemble 和 Stacking。

## 参数范围

默认的非线性核搜索 `gamma_scale ∈ [0.001, 10]`，使用对数采样。以每折最终保留 5000 列为例，实际 gamma 范围是 `[2e-7, 0.002]`；若特征选择后只剩 100 列，则为 `[1e-5, 0.1]`。`gamma_scale=1` 对应 RBF 常用的 `1/p` 基准。[sklearn RBF 定义与默认值](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.rbf_kernel.html)

这是针对已标准化输入的经验范围，不能保证所有数据都不会产生常数或非有限预测。已有的无效 Pearson 检查仍然生效，任何 fold 无效都会淘汰整个 trial。常数预测、常数标签、非有限预测和非有限标签会分别记录原因。

显式数字 `gamma` 继续表示绝对值，便于读取旧参数和复现实验。`gamma` 与 `gamma_scale` 不能同时指定为非空值。拟合后的模型保存 `gamma_` 和 `n_kernel_features_`，预测时复用该次拟合的 gamma，不用待预测样本重新估计它。

## sigmoid 的处理

默认搜索包含 linear、RBF、Laplacian 和 polynomial。sigmoid 需要在用户模型 YAML 中显式启用，因为其参数组合不保证产生正半定核矩阵。

打包的 `kernelridge_reg` 默认 `numerical_policy: raise`。正定求解失败、求解器报告病态矩阵、核矩阵或求解结果非有限时，会抛出 `KernelNumericalError`。该异常在参数搜索中转换为 PRUNED；它不会进入 sklearn 的最小二乘回退路径。[SciPy 正定求解器与数值异常](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve.html)

每条 trial 的 alpha 由搜索参数确定，失败分支不会自行增加 alpha 或修改核矩阵后继续评分。若全部 trial 都无效，则模型不能进入选优流程。常规 CV 和最终模型重拟合也使用同样的求解策略。

直接使用 Python `SafeKernelRidge` 的默认策略仍为 `legacy`，保留旧调用和旧 joblib 模型的行为；复现旧搜索时也可在 YAML 中显式指定 `numerical_policy: legacy`。新训练建议使用打包的严格策略。

显式搜索 sigmoid 的 YAML 示例：

```yaml
models:
  - name: kernelridge_reg
    param_func: null
    default_params:
      gamma: null
      numerical_policy: raise
    search_space:
      - {name: kernel, type: categorical, choices: [rbf, sigmoid]}
      - {name: alpha, type: float, low: 0.01, high: 10.0, log: true}
      - {name: gamma_scale, type: float, low: 0.001, high: 10.0, log: true}
      - {name: coef0, type: fixed, value: 0.0}
```

将该文件通过 `--model_config` 传入；显式 `param_func: null` 用于替换默认 builtin 搜索函数。此例会保留满足数值要求的 sigmoid 候选，剔除无效候选。

## 记录与结果兼容

- `optuna_best_params.json` 的 `best_params` 保存获胜候选的完整模型参数，包括固定 YAML 参数和数值策略。Optuna 原始的 `study.best_params` 不包含固定参数，不能直接作为完整重拟合输入。
- `logs/optuna/kernelridge_reg_repeat_*.jsonl` 增加 `model_params` 和 `prune_reason`，保留采样参数、实际模型参数、状态和剪枝原因。
- 搜索 fold 的事件记录 `kernel_parameters`；常规 CV 的 `repeat_results.json` 在每个 fold 中保存 `kernel_parameters`，含实际维数、gamma_scale、实际 gamma 与数值策略。
- 模型文件内保存已经拟合的 gamma 与预处理器，加载后预测复用它们；旧模型显式 gamma 的含义不变。

预期的数值剪枝在 worker 中记录 DEBUG 细节，每条 trial 在终端汇总一条 WARNING；其他训练异常继续记录 ERROR。完整逐折事件仍保留。

本次修改会改变新的搜索空间，新的训练结果应写入新目录，并与旧结果作为不同实验比较。此前对 [trial 7/8 的排查](26.10.23/kernelridge-numerical-diagnosis.md)保留原始参数和源码哈希，不覆盖为新结果。

## 验证

新增 `tests/test_kernelridge_numerics.py` 的 14 项测试通过，覆盖复制特征后的预测不变性、特征选择后实际维数、旧模型状态及绝对 gamma 兼容、clone/joblib 一致性、严格拒绝 sigmoid 回退、病态/非有限矩阵、固定 YAML 参数持久化，以及真实串行/并行 CLI 的结果一致性和整 trial 剪枝。相关旧模型与注册表的先行检查为 9 项通过。

全量运行结果为 `271 passed, 1 failed in 111.20s`，失败项仍为先前已复现的 `test_find_result_file_depth_limit`，完整日志 `/tmp/gpse-gamma-full-tests.log`。

用原诊断相同的 120 样本、2048 SNP 合成数据复验：旧绝对 gamma 的 RBF 五折均得到常数预测，改为 gamma_scale=1 后五折均有有效 Pearson；旧 sigmoid 五折均回退最小二乘，严格策略五折均抛出数值异常且没有回退。新的默认搜索前 10 个 trial 全部完成五折评估。逐折数据、参数、源码哈希与验证结果见 [复验记录](26.10.23/kernelridge-policy-validation.json)。这些是数值稳定性检查，不是用户真实数据上的准确率或耗时承诺。
