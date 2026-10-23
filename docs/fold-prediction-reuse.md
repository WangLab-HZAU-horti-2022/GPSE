# 复用 CV fold 预测，消除 ensemble 重复训练

2026-10-23。此改动接在[单服务器调度研究](single-server-threading-study.md)之后；研究中的旧测量保留为优化前快照。

## 已实现的变化

正常 repeat 训练中，每个 CV fold 只拟合一次。该模型已经计算过的 holdout 预测会直接用于 fold ensemble；分类同时复用概率矩阵，没有概率接口时继续做硬投票。回归继续在平均预测后还原表型尺度。

之前，CLI 默认 `save_models=False` 时，ensemble 阶段会把 K 个 fold 全部重新拟合；启用保存时则重新读取模型、预处理 holdout 并预测。现在这两条正常训练路径都复用第一次生成的输出。`--save_models` 只决定 fold 模型是否写盘，已有命令无需增加选项即可生效。

仅缓存 K 个 holdout 预测向量，分类另缓存 K 个 holdout 概率矩阵。**没有保留 K 个大模型或 K 份 SNP 矩阵**。fold 顺序、随机种子、CV 划分、Optuna 搜索、TOPSIS 和 Stacking 规则没有调整。模型部署所需的独立 refit 仍然保留；单独调用内部 ensemble 评估函数时，也保留旧的加载/重建入口。

涉及正常训练的源码：

- `gpse/train/_fold_training.py`：成功的 fold 将 holdout 输出传给 repeat。
- `gpse/train/_repeat_training.py`：按原 fold 顺序收集输出，交给 ensemble。
- `gpse/train/_ensemble.py`：优先聚合既有输出，不再为正常 repeat 重新拟合或重新读盘。

未完整成功的 fold 集合不能当成完整 ensemble；既有的 CV 聚合验证仍会拒绝缺失 fold 的模型。

## 前后对照

| 负载 | CPU 预算 | 优化前秒数 | 优化后秒数 | 耗时下降 |
| --- | ---: | ---: | ---: | ---: |
| tiny | 1 | 3.28 | 3.12 | 4.6% |
| tiny | 4 | 3.33 | 3.23 | 3.0% |
| forest | 1 | 16.29 | 10.08 | 38.1% |
| forest | 4 | 8.79 | 7.01 | 20.2% |
| frozen | 1 | 9.33 | 6.96 | 25.4% |
| frozen | 4 | 8.07 | 7.26 | 10.1% |
| mixed | 1 | 20.52 | 16.63 | 19.0% |
| mixed | 4 | 15.19 | 13.67 | 10.0% |
| svm | 1 | 3.43 | 2.92 | 14.9% |
| svm | 4 | 3.83 | 3.78 | 1.5% |

表中为两次端到端测量的均值，包含启动、读取、训练、保存和报告。小任务差异接近 0.1 秒采样分辨率，不作强性能结论；较重森林和混合负载的差异更明显。


计时不包含测试，训练组之间顺序执行。使用修改前保存的源码目录 `/tmp/gpse-speedup-before-20261023`，通过研究脚本新增的 `--source-root` 选择真实旧源码；两边采用相同模型配置、种子和生成数据，各测两次。设置为单 CPU 预算和四 CPU 预算，后者每任务一个内部线程。完整 argv、源码哈希、原始时间和 RSS 见测量 JSON。

```bash
PYTHONPATH=. /home/zj/.pyenv/versions/gpse/bin/python scripts/profile_single_server.py \
  --source-root /tmp/gpse-speedup-before-20261023 \
  --output /tmp/gpse-fold-reuse-before-20261023 \
  --workloads tiny forest frozen mixed svm --settings serial shared4 --repetitions 2

PYTHONPATH=. /home/zj/.pyenv/versions/gpse/bin/python scripts/profile_single_server.py \
  --output /tmp/gpse-fold-reuse-after-20261023 \
  --workloads tiny forest frozen mixed svm --settings serial shared4 --repetitions 2 \
  --audit-fit-counts
```

目录必须为新目录；旧源码快照需要在改动前保存。没有用减少 repeat、fold、trial 或关闭搜索/Stacking 的方式制造前后差异。本机仍是 8 个物理核、16 个逻辑 CPU，数据是前一报告中定义的合成负载；这些数字不等于 90 核服务器或真实高维数据的性能承诺。

## 验证

前后共 40 次 CLI 训练全部成功退出。逐一核对 20 对运行，输入文件和 holdout ID 完全相同；每个 repeat/fold 的参数、搜索次数、CV 指标、预测、ensemble 指标、TOPSIS 排名和最终 holdout 输出在相对/绝对容差 1e-12 内一致，最大数值绝对差为 0。

独立 cProfile 计数：R=4、K=2、双模型及 Stacking 场景，RF 默认参数的总拟合次数在不保存模型时从原来的 22 次降到 14 次；冻结搜索场景从 25 次降到 17 次。启用保存时也分别为 14 次和 17 次，拟合次数不再取决于持久化开关。仍保留代表模型、部署 ensemble 和 Stacking 的必要 refit。

完整[前后测量、源码哈希与一致性检查](benchmarks/fold-prediction-reuse-20261023.json)已经保存。原始资源采样、事件和产物位于上面两条命令的输出目录。


新增 `tests/test_fold_prediction_reuse.py` 的 **6 项专门测试通过**（1.64 秒）：回归 RF、分类 RF 和无概率接口的 LinearSVC 每个 fold 只拟合一次，保存/不保存的指标相同；多分类按概率平均而不是直接投票；标准化后的回归预测恢复原尺度；不完整 fold 集合被拒绝。覆盖中也启用 fold 内插补和特征选择。

全量测试本次执行为 `2 failed, 246 passed in 75.30s`，完整日志在 `/tmp/gpse-speedup-full-tests.log`。两个失败均位于未修改的模块：

- `test_find_result_file_depth_limit`：优化前保存的源码单独运行也失败。
- `test_multi_trait_genotype_files_are_hardlinked`：当前源码和旧快照单独复跑均通过；现有转换代码会在文件写完前发布可复用路径，存在并发时机相关问题。当前改动不涉及这段代码。

没有将全量测试描述为全部通过。新增训练测试及已有串/并行训练对照的先行检查通过，模型保存开关、分类概率和回归尺度的行为另有专门覆盖。

## 仍然存在的瓶颈

本报告的测量快照中，首次 Optuna 搜索的 trial 和 trial 内 fold 仍串行。后续的 [fold 级调度优化](fold-parallel-scheduling.md)已经将搜索和常规 CV 的独立 fold 接入同一个计算池；trial 仍顺序推进。进程启动、跨进程矩阵复制、部署 refit 和报告阶段仍有成本，不能据此宣称已经实现全程充分使用 90 核。共享矩阵和按内存准入仍需单独处理。
