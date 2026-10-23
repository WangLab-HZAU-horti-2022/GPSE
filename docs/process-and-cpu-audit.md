# GPSE 停止行为与 CPU 调度核查

## 结论和复现证据

Ctrl+C 后仍有后台计算进程的情况确实可能发生。核查时可见环境为 16 个逻辑 CPU，未发现此前遗留的 GPSE 训练进程；这一观察仅覆盖当前可见的进程命名空间。

使用原始 HEAD 的 `graceful_process_pool` 单独复现：worker 忽略 SIGTERM，另起一个 session 的孙进程。向主进程发送 SIGINT 后，CLI 返回 **130**、worker 已退出，但孙进程仍存活。本次核查创建的残留测试进程随后按已记录 PID 清理，没有操作其他训练任务。修复后的同一场景、连续两次 Ctrl+C、外部命令及批处理场景均检查独立 PID 文件，确认计算进程已退出或仅剩不计算的 zombie 等待系统回收。

原始代码仍有以下遗漏：

1. 只终止 Executor 已登记的直接 worker；深层后代、另起会话的子进程无法完整覆盖。
2. 模型 worker 收到 SIGTERM 后依赖 Python 展开内部池清理，主进程只等待 0.5 秒。原生计算阻塞时，Python 信号处理会延迟。
3. `Process.start()` 到 Executor 记录句柄之间存在窗口；此时 Ctrl+C 可能遗漏新 worker。
4. 正常 `shutdown(wait=True)` 原来位于异常捕获之外，退出等待本身被中断时缺少相同清理。
5. 外部 PLINK/Java 命令和自适应批处理有各自停止方式；只等待主命令结束不能证明整个进程树停止。
6. 自适应调度器停止后可能仍将任务记为 RUNNING，并且未恢复其安装的信号处理器；同进程后续调用会受影响。
7. train 内置预处理没有向转换器传递线程预算；独立 convert/pipeline 在转换阶段也需要与 train 一致的信号退出入口。

## 退出实现

`gpse/utils/process_control.py` 统一管理 GPSE 自己创建的进程。Pool worker 和外部命令使用独立 session/process group。Linux `/proc` 快照跟踪已登记 worker 及其后代，并补上启动中尚未登记的新子进程；PID start time 用于防止 PID 重用后的误杀。不会按进程名称执行全局 kill，也不会清理预先存在的无关进程。

Ctrl+C、SIGTERM 和 SIGHUP 进入取消流程：停止提交工作，先 TERM、再对仍存活的目标 KILL，回收直接子进程。清理期间屏蔽重复取消信号。原始 KeyboardInterrupt 继续向上返回，CLI 退出码为 130，Run 记录 interrupted，Manifest 标记 failed。Stacking 也使用相同的可取消池。

外部命令由 `run_subprocess` 创建独立会话，异常离开时清理进程树。自适应批处理调用同一停止机制，停止后持久化 INTERRUPTED/130，并恢复原有信号处理器。SIGKILL、机器断电等无法执行 Python 清理的情况不在这一保证范围内。

## CPU 调度调查与新行为

| 层级 | 原来行为 / 问题 | 当前实现 |
| --- | --- | --- |
| 可用 CPU | 只看 affinity，未考虑容器 CPU quota | 取逻辑 CPU、affinity、cgroup v1/v2 当前及祖先 quota 的限制；分数 CPU quota 向上取整为可调度 worker 数，由内核控制 CPU 时间 |
| CLI 默认 | 三个并行参数均为 1，默认只做一个任务 | 没有显式手动限额时自动使用可用 CPU；`--threads auto`、`--threads -1` 同义，整数为总预算 |
| 模型 / Repeat | 嵌套池，每个模型固定 Repeat 配额；尾部模型不能接管其他模型的空闲槽位 | CLI 统一为一个共享池，独立 Repeat 和模型聚合/refit 任务使用同一总预算；空闲槽位可以重新分配 |
| 冻结 Optuna 参数 | `optuna_per_repeat=False` 时 Repeat 一直串行 | 每个模型先成功完成一次搜索，再将缓存参数传给后续并行 Repeat；失败不构成有效完整 CV，仍由原聚合规则拒绝 |
| Optuna Trial / CV Fold | Trial 与单次搜索内 Fold 顺序执行 | 保持原搜索顺序、随机种子、Fold 定义、稳健目标和剪枝含义；通过跨模型/Repeat 的独立任务并行使用 CPU |
| 默认参数 / GBLUP | 不需要先做参数搜索，但也受到冻结分支串行限制 | 独立 Repeat 可立即并行 |
| 模型聚合 / 保存 | 代表模型与部署 ensemble 的 refit 会消耗计算资源 | 聚合/refit 本身也占共享池槽位，不在满负荷 worker 之外再额外计算一份 |
| Stacking | OOF Fold 与基础模型 refit 逐个执行；没有统一 native 线程限制 | OOF Fold / refit 使用受同一 CPU 预算限制的池，完成后填回原来的行与列位置；元学习器只有 Top-N 个特征，单线程拟合 |
| Native 线程 | CLI 入口预解析与自动推导不一致；spawn 导入数值库后再改环境变量可能太晚 | 创建池时设置 `GPSE_NATIVE_THREADS` 及 6 个原生库变量，root CLI 和 direct module bootstrap 均优先使用该值；实际 fit/predict 阶段再用 threadpoolctl 限制 |
| 自适应批处理 | 固定插入 repeat_workers=1，默认总预算 90，可能与容器 CPU 不符 | 总预算默认 auto、每性状预算受总预算限制；不再强制关闭 Repeat 并行；统一复用 train 参数构建并保留用户限额 |
| 转换 / 预处理 | 独立转换有自己的线程选项，train 内置预处理未传递预算 | train 内置预处理传递本阶段总预算；独立 convert 的 `--threads`、pipeline 的 `--convert-threads` 仍控制转换阶段，pipeline 的 `--threads` 单独转交训练阶段；三个 CLI 入口均安装相同取消处理 |
| 批处理启动参数 | 默认 `python -m gpse train` 在项目缺少 gpse/__main__.py 时不可用；部分训练选项没有传递 | 使用当前 Python 的 `-m gpse.train.cli`；复用标准 batch/train 参数处理 |

默认调度更偏向多个独立任务，每个任务尽量只使用 1 个 native 线程；独立任务不足时可分配更多线程。共享池实际任务数满足：

```text
worker_limit = min(
    floor(cpu_budget / n_jobs),
    model_limit × repeat_limit
)
worker_limit × n_jobs <= cpu_budget
```

`--max_workers` 和 `--repeat_workers` 是高级限额，不是要求同时启动多少进程；显式传入 1 仍受尊重。没有 `--threads` 但显式提供这些旧选项时，先校验其乘积预算，再在共享池中执行。直接 Python API 的传统执行入口保留兼容，`GenomicPredictorV2(cpu_budget=N, ...)` 可以使用共享池。

进度事件中 `scheduler/configured` 记录预算、worker 上限和每任务线程数；`scheduler/submitted` 记录占用槽位；`worker/ready` 记录实际加载的 native thread pools。这些都是容量/任务证据，不能将 Python 日志线程、空闲 BLAS 线程数量简单相加视为 CPU 使用率。

### 当前环境的原生库问题

最小脚本可复现：NumPy/SciPy 的 OpenBLAS 已按 1 线程初始化，再用 threadpoolctl 扩到 4 线程拟合 LogisticRegression，会在 SciPy L-BFGS 中崩溃；如果在 import 之前设置 `OPENBLAS_NUM_THREADS=4` 则通过。因此不能只在 worker initializer 中修改环境变量。本实现把配置前移至 spawn 导入之前，并对仅几个特征的元学习器固定单线程，不改变其模型或超参数。新增测试检查 worker 实际 native pool 的线程数不超过预算。

## 推荐命令

```bash
# 在现有训练命令上使用这一项即可，其他 worker 选项通常不需要：
gpse train <原有数据与训练参数> --threads auto

# 等价的自动预算写法：
gpse train <原有数据与训练参数> --threads -1

# 例如给该任务最多 16 个逻辑 CPU：
gpse train <原有数据与训练参数> --threads 16

# 保留严格串行计算配置：
gpse train <原有数据与训练参数> --threads 1 --n_jobs 1
```

默认冻结参数时，单模型首次 Optuna 搜索仍有顺序依赖，无法一直用满全部核心。数据读取、最后少量 refit、很小的元学习器和报告写入也不是持续满核阶段。多进程会增加内存占用，内存不足时用较小的 `--threads N`，多性状任务也可继续使用带内存限额的自适应调度器。

## 验收范围

`tests/test_cpu_scheduler.py` 额外覆盖：自动/手动限额、cgroup v1/v2 与祖先 quota、冻结搜索只执行一次、共享池总任务数、spawn 的真实 native 线程数、回归与分类的串行/并行指标和预测一致性、独立 OOF Fold、SIGINT/SIGTERM/SIGHUP、连续取消、忽略 TERM 的 worker、独立 session 孙进程、尚未登记的启动窗口、正常关闭等待、Stacking 中断、外部命令和自适应调度器、批处理参数和退出状态/信号恢复。

测试不只检查日志中的 PID，还使用受控子进程独立写出的 PID 文件核对进程状态。训练保持随机种子、划分、搜索与 CV/Stacking/Holdout 算法不变；统计对比在同样的单任务线程设置下执行，provenance hash 随源码和配置变化属于预期。


## 最终验收结果

2026-10-22，在 `/home/zj/.pyenv/versions/gpse/bin/python` 环境执行：

```text
python -m pytest -q tests/
1 failed, 241 passed in 83.72s
```

新增调度/取消验收共 **19 个参数化用例，全部通过**。唯一失败为既有的 `test_find_result_file_depth_limit`，属于预测文件查找深度，在单独提取的原始 HEAD 中同样复现，本次未修改该预测代码。还补全了实际使用的 `threadpoolctl` 依赖声明。

`python -m compileall -q gpse tests/test_cpu_scheduler.py`、`git diff --check` 均通过。未改动用户原有的 `reaggregate_cv.py`、`test_cv_statistics.py` 修改或 PDF 删除。
