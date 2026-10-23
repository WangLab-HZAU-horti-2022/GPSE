# KernelRidge 搜索数值告警排查

> 本文保留修复前的诊断与复现。后续实现及使用方式见 [gamma 缩放与严格数值策略](../kernelridge-gamma-and-numerics.md)。

排查对象：repeat 1 / trial 7 的 `Undefined validation Pearson`，trial 8 的 `Singular matrix in solving dual problem`，以及随后成功完成的模型输出。

这是对当前源码和本机合成数据的排查。用户日志来自 `/home/jzhang/...` 的另一台环境，本机没有该次运行目录，因此没有直接检查该次参数历史、输入 SNP 数量或最终最优 kernel。下面的 trial 参数对应当前默认配置、Optuna 4.9.0 和随机种子 42；与用户日志中的 trial 编号高度吻合。

## 可复现的原因

默认参数空间在不同输入维度上都搜索绝对 gamma `[1e-6, 1]`，同时包含 sigmoid 核。重放前 10 次建议，在默认 TPE 启动阶段可以得到：

| Trial（显示编号） | kernel | alpha | gamma | coef0 |
| --- | --- | ---: | ---: | ---: |
| 7 | rbf | 0.6218704728 | 0.3398172415 | — |
| 8 | sigmoid | 0.0651699061 | 0.0938480072 | 3.5675332669 |

实验使用 120 个样本、2048 个合成二倍体 SNP，5 折 CV，每折单独拟合标准化，使用项目当前的 `SafeKernelRidge` 和训练预处理 Pipeline。所有实验串行执行，未启用 fold 并行，因此这两类现象本身不需要并行调度即可出现。

**Trial 7：验证预测成为常数。** RBF 为 `exp(-gamma * squared_distance)`，高维标准化输入的距离较大，gamma=0.3398 可使验证样本到训练样本的全部核值在 float64 中下溢为 0。[RBF 官方定义及默认 gamma](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.rbf_kernel.html)

本机复现的五折中，每折 24×96 个验证/训练核值全部为 0，预测范围也全部为 0；真实标签范围非零。这不是原始输入含 NaN 的证据，而是这组超参数下的常数预测。`evaluate_optimization_fold` 将其转换成 `TrialPruned`，整条 trial 被排除，不会把未定义的 Pearson 填成 0 后参与最优选择。

作为机制对照，仅将同一组参数中的 gamma 改成 `1/2048` 后，五折核值均非零、预测均有变化、Pearson 均可计算。这个对照说明维度相关的 gamma 值可以避免本实验的下溢，不表示该 gamma 对用户数据是最优值，也不等于已经修改项目搜索空间。

**Trial 8：正定求解失败后回退。** sklearn 使用 `solve(..., assume_a="pos")` 求解 `K + alpha I`；捕获 `LinAlgError` 后打印统一的 Singular matrix 警告，再执行 `lstsq`。[sklearn 1.7.2 求解器源码](https://github.com/scikit-learn/scikit-learn/blob/1.7.2/sklearn/linear_model/_ridge.py#L225-L283)

本机五折系统矩阵均为 96×96、秩均为 96，但最小特征值约 -14.58 至 -13.94。也就是说，警告文本中的 singular 不能直接证明严格的秩亏；本次复现实际是矩阵非正定。五折都打印了同一条警告并继续完成最小二乘求解。这类 warning 本身不会使 trial 自动变成 PRUNED；只要之后所有 fold 的评分有效，该 trial 仍可参与选优。

## 为什么会出现在面板上方

`ConsoleRenderer.accept` 默认将 WARNING/ERROR 交给同一个 Rich Console，作为持久日志放在动态面板上方。5 个并行 fold 分别记录各自的问题，所以同一 trial 会出现 5 行近似消息。

当前错误信息将几种原因合并为 `constant/non-finite input`，并把随后会剪枝的 fold 记为 ERROR；因此终端视觉上像是整个模型失败。日志中的 `kernelridge_reg 1 3 7` 按顺序表示 model / repeat / fold / trial，即 R1、F3、T7。最后的 `model completed ...` 是模型最终成功完成的摘要。

当前模型选择只允许完整、有效的 CV 结果，因而成功完成与部分搜索 trial 被剪枝可以同时成立。`CV Mean=0.550803` 和 holdout Pearson=0.559437 表明这次模型有有效结果，但单凭这些数字不能确定最终选择了哪个 kernel，也不能证明包含数值告警的 trial 8 没有入选。需要核对该次运行的：

- `logs/optuna/kernelridge_reg_repeat_001.jsonl`：trial 7、8 的实际参数、状态、分数和耗时。
- `models/kernelridge_reg/repeat_1/optuna_best_params.json`：实际最优参数。

告警中的回退求解和无效 trial 都会增加搜索成本；仅凭粘贴的片段不能将 842 秒总耗时归因于其中某一条警告。

## 验证与后续处理

详细逐折数值见 [合成数据复现记录](kernelridge-numerical-reproduction.json)。另外运行了已有的常数 Pearson 剪枝测试及 SafeKernelRidge 测试，6 项通过；没有修改训练算法、搜索空间、评分标准或隐藏告警。

优先处理方向是让 gamma 搜索与预处理后的特征维度或训练样本距离相关，并为 sigmoid 的非正定情况制定明确策略。日志可单独改进为区分常数预测、非有限预测和标签异常，并按 trial 汇总重复信息。修改搜索空间或核策略会改变后续训练实验，应该与现有结果分开记录。
