# Fold 级并行调度

训练默认使用的 CPU 预算池现在同时执行 Optuna 搜索、常规 CV 的独立 fold 和模型 refit。只有一个计算进程池；父进程的线程管理 repeat 和 study。`--repeat_workers 1` 时，当前 repeat 内的 fold 也能同时计算，首次冻结搜索不再只能使用一个计算进程。

## 使用

在现有 `gpse train` 命令后加：

```bash
--threads 90 --n_jobs 1
```

未指定并行参数时仍自动使用可用 CPU 预算，fold 并行默认开启。也可以使用 `--threads auto`，可用核数受 affinity/cgroup 配额约束。`--fold_workers 3` 将每个 repeat/trial 的并发 fold 限为 3；`--fold_workers 1` 恢复先前按 repeat 分配计算进程的方式。Python API 使用 `cpu_budget=90, n_threads=1, fold_workers=None`；未传 `cpu_budget` 的旧 API 保留原调度方式。

显式指定的模型、repeat 和 native thread 限制继续有效。手工 worker 参数且未传 `--threads` 时，CLI 继续用 `n_jobs×max_workers×repeat_workers` 推算预算，fold 任务也受此预算约束；显式 `--fold_workers > 1` 则开启自动 CPU 预算推导。若上层批处理只给当前训练进程分配 8 核，本层也只能使用这 8 核预算。

## 容量与依赖

设总 CPU 预算为 B，单个计算任务的线程数为 T，模型上限为 M，每模型 repeat 上限为 R，每 repeat 的 fold 上限为 F。计算进程数最多为 `min(floor(B/T), M×R×F)`，实际数量也受模型、repeat、fold 总数限制。所有提交中的计算任务共用一个有界信号量。模型 refit 和结构感知 CV 的 PCA/KMeans 也使用同一计算池。

各 study 仍调用 `Optuna.optimize(n_jobs=1)`，同一个 trial 的 fold 可以并行。fold 的完成顺序不影响其汇总顺序；分数、预测、CV 指标均按原 fold 编号归并。trial 失败会取消尚未运行的 fold，等待已经运行的 fold 收尾后才进入下一个 trial。冻结参数仍在首次 repeat 完成后分发给后续 repeat，holdout 划分、种子、特征选择和插补的拟合边界保持不变。

repeat 控制线程复用父进程中的同一组 holdout 数据；计算进程各缓存一份 holdout 分区，后续任务只传 fold 索引、参数和结果路径。仍然需要向每个 spawn 进程传递完整输入，并非跨进程共享内存。高维 SNP 矩阵下需要同时观察内存；当前 CPU 预算不是内存准入限制。

计算池在主线程预先启动，取消时先停止提交、清理计算进程，再回收等待 future 的控制线程。事件日志区分 `queue=compute` 与 `queue=controller`；重建实际训练并发应使用 `fold` 和 `optimization_fold` 的开始/结束事件。

## 本机前后测量

基线是已完成预测复用、尚未实施 fold 调度的源码快照 `/tmp/gpse-before-fold-scheduling-20261023`。比较使用同一份合成数据、模型配置、种子、trial/repeat/fold 数量和 `--threads 4 --n_jobs 1`，保留 Stacking。每种负载前后各测两次，第二轮交换版本顺序；每次仅运行一条训练命令。

| 负载 | 优化前端到端秒数 | 优化后秒数 | 耗时变化 |
| --- | ---: | ---: | ---: |
| 冻结搜索，允许并发 repeat | 6.71 | 5.95 | 下降 11.2% |
| 搜索，每次仅运行一个 repeat | 9.88 | 7.46 | 下降 24.5% |
| 默认参数，每次仅运行一个 repeat | 5.90 | 6.21 | 增加 5.3% |
| 微小双模型任务 | 3.13 | 3.13 | 持平（差异 <0.1%） |

表中为两次测量的均值，包含导入、启动、训练、保存、Stacking 和报告。冻结搜索自身耗时从 1.60 秒降到 0.62 秒；限制一个 repeat 的搜索从 4.30 秒降到 1.23 秒，分别减少约 61% 和 71%。端到端收益小于搜索阶段收益，因为仍有进程启动、refit 和 Stacking。

短的默认参数任务出现约 0.31 秒回退，说明新增进程的启动成本可能超过短 fold 的并行收益；微小任务两版都已有 4 个计算进程，耗时基本相同。需要优先避免额外启动时可使用 `--fold_workers 1` 对照。限制一个 repeat 的搜索场景中，计算进程从 1 个增至 4 个，进程树峰值 RSS 求和约从 409 MiB 增至 994 MiB；该口径含共享页重复计数，不是唯一物理内存占用。

16 次 CLI 运行均成功，8 对结果通过逐项一致性检查。参数、trial 状态/目标、CV 指标、TOPSIS 和最终 holdout 指标在容差 1e-12 内一致，最大数值绝对差为 0；所有 fold 预测 JSON 与训练集 CV 分配文件也逐字节一致。代码版本指纹、耗时和输出路径属于预期不同的元数据，不作为科学输出比较。


复现命令：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/zj/.pyenv/versions/gpse/bin/python -m scripts.profile_fold_scheduler \
  --baseline /tmp/gpse-before-fold-scheduling-20261023 \
  --output /tmp/gpse-fold-scheduling-20261023
```

输入、详细日志、产物、事件区间和进程树资源采样保留在输出目录。完整命令、源码哈希和汇总见 [测量记录](benchmarks/fold-parallel-scheduling-20261023.json)。本机是 8 个物理核、16 个逻辑 CPU，未在 90 核硬件或用户真实 SNP 数据上测量。

## 验证

新增 10 项测试覆盖全局任务上限、有序归并、失败 trial 收尾、等待提交的取消、理论 90 核参数推导，以及只有一个 repeat 控制任务时的真实 fold 并行。回归的自定义 CV、插补、PCA、表型标准化、每 repeat 搜索、结构感知 CV，和三分类冻结搜索均与 `--fold_workers 1` 的参数、trial 历史、全部 fold 指标/预测和最终 holdout 结果一致。集成对照启用模型保存，已有预测复用测试另行覆盖保存开关。

进程启动、首次冻结搜索、常规 CV 三种阶段注入阻塞，并向真实 CLI 发送 SIGTERM，均按 130 退出且无遗留计算进程。已有 SIGINT、SIGTERM、SIGHUP 的共享池与 Stacking 清理测试也通过。

全量运行 `256 passed, 2 failed in 101.35s`，日志 `/tmp/gpse-fold-scheduler-full-tests.log`。两个失败在优化前源码也复现：结果文件搜索深度测试，以及多性状转换硬链接的并发时机问题；后者当前版本单独重跑通过。本次未修改其实现。

## 对 90 核的含义

本次消除了“首次搜索期间每个模型仅有一个活跃进程”的限制。若当前可运行 14 个模型、每个 trial 有 5 折，且每个任务用 1 线程，首次搜索理论最多同时提供 70 个 fold；后续 repeat 可以进一步补充任务。一个模型、5 折、冻结搜索时只有 5 个独立 fold，即使预算是 90，也无法让 90 个单线程任务同时工作。

进程启动、逐 trial 的依赖、模型尾部 refit、Stacking、I/O 和报告仍有串行部分。`--n_jobs` 可以给支持内部并行的模型更多线程，但是否更快取决于算法、数据和内存带宽；此处没有保证全程 90 核满载。
