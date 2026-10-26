# GPSE 线程治理与弹性 CPU 调度优化设计

版本：设计稿 v1.0  
日期：2026-10-26（Asia/Shanghai）  
配套文件：`GPSE_Threading_Implementation_Prompts_20261026.md`

> 本文依据用户提供的 GPSE 核查结果与本次讨论编写。未直接访问 GPSE 仓库、服务器、实际日志或数据；文中的现有行为属于用户提供的证据，文件行号仅作定位线索。所有新增类、配置、事件和文件名均为建议接口，不代表已实现。本文交付的是实施设计，不是已完成优化或性能验证的报告。

## 1. 目标与决策

目标：让快模型优先完成，同时让慢模型获得合理基础资源；随着其他任务完成，自动把空闲 CPU 预算分配给仍有可运行工作的模型。最后剩下的 XGBoost、CatBoost、LightGBM、RF 或其他模型，都能申请可用预算，无需另起一次训练、手工拼结果。

本次采用以下决策：

1. 一个 GPSE run 使用一个全局 CPU 资源池，模型、repeat、trial 只组织任务，实际计算任务领取资源。
2. 先统一线程设置来源，再做弹性调度；不能在冲突线程设置上叠加新调度器。
3. 第一版固定各模型的单 fit 线程数，优先动态增加活跃 folds；不调整已经开始的 fit。
4. Optuna 保持每个 study 的串行 trial 语义；新开 trial 并行、改变搜索空间或 early stopping 不属于第一版。
5. 单 fit 多线程、fold 并行、repeat 并行共享同一本资源账，不相乘失控。
6. CPU、worker 槽位、内存共同约束启动。80 个 CPU 单位不意味着 80 个 Python 进程。
7. 同一 run 保留原有统计定义、OOF、leaderboard、stacking、报告与恢复语义。
8. 只有实测总耗时受益才推广；CPU 图表更满不是唯一成功标准。

### 1.1 本次不承诺

- 不承诺 CPU 持续 100% 利用，也不承诺 83 分钟必然缩短为某个数值。
- 不把 80 个逻辑 CPU 默认称为 80 个物理核心。
- 不保证一个正在训练的实例能从 8 线程在线扩展到 80 线程。
- 不因模型没有 `n_jobs` 就认定它只能单线程。
- 不修改实验参数、树方法、精度、数据集或 trial 数来制造加速结果。
- 不在既有训练进程中热更新调度器，不覆盖正在运行或已完成的结果目录。

## 2. 已提供的现状与仍需核实的问题

### 2.1 现状基线

| 项目 | 用户提供的信息 | 实施前的核查动作 |
|---|---|---|
| 模型注册表 | 21 个：15 回归、6 分类 | 枚举实际 ID、构造器、任务类型、线程字段 |
| 直接可配置模型并行 | RF、XGBoost、LightGBM、CatBoost，共 8 个 ID | 验证真实训练路径的参数，而不只实例化 |
| 线程注入 | `max(n_threads, min_threads)`；上述 8 个 `min_threads=8` | 确认所有入口、clone、refit 和配置覆盖顺序 |
| 全局预算 | `--threads` 管理共享 CPU 预算 | 确认与 `--n_jobs`、worker 参数的实际语义 |
| Optuna | `optuna_n_jobs=1` | 确认是每个 study 串行，还是存在更高层并发 |
| fold 限制 | 支持 trial/repeat 的 fold 并发上限 | 确认是否有固定 executor 或同步屏障限制扩容 |
| 多处线程限制 | 外层模型有效线程数、fold 的 `self.n_threads`、工具函数的限制 | 画出嵌套关系、进程边界和覆盖时间 |
| XGBoost 时间 | HPO 约 4991.8 秒，模型总计约 5589 秒 | 查计时起止位置、等待时间和实际 CPU 使用量 |

用户提供的版本：scikit-learn 1.7.2、XGBoost 3.2.0、LightGBM 4.7.0、CatBoost 1.2.10、threadpoolctl 3.6.0。Optuna、Python、NumPy、SciPy、joblib 和 BLAS/OpenMP 版本尚需补齐。部署机器上的真实版本优先于本文。

### 2.2 优先检查的代码位置

| 文件线索 | 检查重点 |
|---|---|
| `gpse/train/cli.py:99` | 入口线程预算、参数解析、CPU 自动检测 |
| `gpse/train/_scheduler.py:48,71` | 外层线程池限制、模型成本、executor、资源占用 |
| `gpse/train/_fold_scheduler.py:96,165` | fold 成本、并发上限、阻塞等待、任务提交方式 |
| `gpse/train/_fold_training.py:63` | 训练内部线程限制、fit/predict 生命周期 |
| `gpse/train/workflow.py:225` | repeat/trial/fold 依赖和屏障 |
| `gpse/config/_model_registry.py:258,298` | 线程注入、最小线程、模型实例创建与克隆 |
| `gpse/config/models.yaml` | 21 个模型和线程能力元数据 |
| `gpse/config/constants.py:59` | Optuna 并发默认值 |
| `gpse/utils/genomic_utils.py:506` | GBLUP/矩阵计算中的额外线程限制 |

行号可能来自不同版本。先记录 git commit 和工作区差异，再用符号名和调用关系定位，不按行号盲改。

## 3. 模型能力与运行速度分开建模

### 3.1 并行能力分类

| 已提供模型/ID | 并行机制 | 第一版资源策略 |
|---|---|---|
| `rf_reg`、`rf_clf` | estimator `n_jobs`，内部 joblib 并行 | 显式 job 数；约束内层线程池；动态 fold 并发 |
| `xgboost_reg`、`xgboost_clf` | estimator `n_jobs`，现配置还传 `nthread` | 两字段若保留则一致；动态 fold 并发 |
| `lightgbm_reg`、`lightgbm_clf` | estimator `n_jobs` | 显式正整数；动态 fold 并发 |
| `catboost_reg`、`catboost_clf` | estimator `thread_count` | 显式正整数；动态 fold 并发 |
| `knn_reg` | `n_jobs`，主要涉及查询/预测 | 将 predict/score 纳入任务生命周期，不承诺 fit 加速 |
| `histgradientboost_reg` | 底层 OpenMP | 在隔离进程中设置并核验线程池；先验证后启用 profile |
| ElasticNet、Lasso、SGD、SVR、AdaBoost、GBLUP、GBDT、Kernel Ridge、MLP 回归；SVM、MLP 分类 | 未注册直接线程参数；具体计算可能调用底层库 | 先审计后分别配置；优先受限进程间 fold 并发 |

最后一行的模型 ID 尚未全部提供，实施时从注册表读取，不根据名称杜撰。21 个模型全部需要能力清单；第一批重点验证 8 个直接可配置模型。

### 3.2 每个模型保存两类信息

**能力 profile（相对稳定）**：线程参数名、并行机制、线程池控制方法、是否可用进程隔离、允许的线程数、内存估算方法。

**运行成本 profile（随负载变化）**：相近样本量/特征数/稀疏性/模型参数下的 fit 时间、峰值内存、扩展曲线。

`fast` / `heavy` 是调度提示，不是永久算法标签。RF 在某个数据集很快，不应因为支持多线程就一律推迟。初期采用可覆盖的人工分类，缺少历史信息时用中性优先级。

## 4. 资源模型：单一账本、任务级租约

### 4.1 CPU 预算口径

CPU token 表示允许并行计算的逻辑 CPU 预算单位，不是物理核心绑定，也不是操作系统硬隔离。

启动时检测并记录：用户请求预算、CPU affinity/cpuset、容器 cgroup quota、作业分配范围、逻辑 CPU 数、物理核心/NUMA 信息（能取得时）。有效预算采用这些约束的保守交集，不直接使用 `os.cpu_count()` 作为唯一依据。

若 quota 是分数 CPU，第一版明确采用保守整数策略；不足 1 CPU 的环境必须显式降级并记录，不能静默按整机核数运行。用户请求超过可用范围时打印 requested/effective 及原因。留给系统的资源可配置，不必机械预留固定百分比。

统一不变量：

```text
sum(active_lease.cpu_units) <= effective_cpu_budget
sum(active_lease.memory_reservation) + fixed_process_memory <= memory_budget
active_compute_tasks <= worker_slots
active_fits_per_model <= model.max_active_fits
```

内存 reservation 是保守估算，不是 OOM 的数学保证；需要监控和失败处理。token 是逻辑 admission control；如果需要硬 CPU 限制，应由作业/cgroup 等外部机制提供，不能把 token 当成硬隔离。

### 4.2 建议数据结构

```python
# 接口示意，非可直接运行的实现
TaskKey = (run_id, model_id, repeat_id, trial_id, fold_id, stage)

TaskSpec:
    key
    attempt_id
    seed
    dependency_ids
    resource_request
    input_refs

ResourceProfile:
    parallelism_kind
    min_cpu_per_fit
    preferred_cpu_per_fit
    max_cpu_per_fit
    max_active_fits
    max_cpu_share
    adapter_id
    memory_estimator_id

FitLease:
    lease_id
    task_key
    attempt_id
    cpu_units
    memory_bytes
    worker_id
    worker_generation
    thread_plan
```

`min_cpu_per_fit` 表示调度粒度/经确认的限制；`preferred_cpu_per_fit` 表示起始性能选择。当前 `min_threads=8` 应先核实用途，不能自动当成算法硬要求。

`max_cpu_share` 是模型级借用上限；基础份额是竞争时的调度权益，不是预先从账本扣除的资源。无 runnable task 的模型不保留额度。

### 4.3 单一资源所有者

模型和 repeat 的编排任务不能占着 CPU token 等待子任务。具体 fold、final refit、需要大量计算的预处理、stacking fit 等任务才领取资源。

父层模型份额用于决定优先级，不再次扣 CPU 账。所有入口共享资源管理器；禁止模型池、repeat 池和 fold 池分别认为自己拥有完整 80 核。

调度器在主进程中原子完成：检查 CPU、内存、可用 worker、模型并发上限 → 生成 lease → 扣账 → 分派。提交失败立即回滚。不要提前把成千上万个任务塞入 executor 后让它们阻塞等待资源。

## 5. 执行隔离与线程设置统一

### 5.1 先核实实际并发拓扑

需要查清：模型、repeat、fold 各自使用 ThreadPoolExecutor、ProcessPoolExecutor、joblib 还是自定义 worker；是否形成嵌套池；是否在同一 worker 中同步等待子 future。

推荐第一版拓扑：轻量编排器管理依赖和研究状态；一层受控计算进程池执行 fits；一个 worker 同一时刻只执行一个计算任务。worker 内部模型仍可使用自己的线程。

若现有进程池已满足隔离，可复用，不必重写全部框架。避免在初始化过 OpenMP/BLAS 的多线程父进程后不受控地 fork；使用与平台和依赖兼容的启动方式并记录实际选择。

### 5.2 线程池限制是进程状态

针对用户报告的 threadpoolctl 3.6.0，不能假设在同一进程的不同 Python 线程中交叉进入不同 `threadpool_limits` 上下文就获得独立资源隔离。支持的库、版本和 OpenMP 组合也会影响控制结果。[R2]

因此：

1. 异构线程额度的并发 fits 优先放在独立 worker 进程。
2. 任务开始时导入需要的库，获得当前线程池信息，再施加本次限制；避免 controller 创建后才加载的新库被漏管。
3. 任务结束或异常后恢复进程状态；worker 复用时核实下一任务不会继承上一任务的额度。
4. 不允许多个并发任务在同一进程中动态修改不同的环境变量或全局限制。
5. 环境变量作为启动默认值；不是运行中针对不同线程分别设置额度的方法。
6. 若无法确认安全隔离，先保留明确的固定进程级限制，并记录不支持异构动态限额，不能宣称已实现。

### 5.3 一次分配生成完整 ThreadPlan

唯一来源为 `FitLease`，再由模型适配器派生：

```text
lease.cpu_units
  → estimator 参数
  → joblib 并发设置
  → OpenMP / BLAS 限制
  → 数据构建与预测阶段设置
  → 运行日志与资源记账
```

不是把所有线程池都设为 N。若外层 joblib 已有 N 个并行任务，内层 BLAS/OpenMP 通常需要更保守的限制，具体由真实调用拓扑决定。官方文档区分了 joblib、OpenMP 和 BLAS，`n_jobs` 不统一控制全部并行层。[R1]

建议适配器行为：

| 类型 | 建议 ThreadPlan | 必须验证 |
|---|---|---|
| RF | `n_jobs=N`；内层数值库保守限制 | joblib 后端、是否派生子进程、预测并行 |
| XGBoost | sklearn `n_jobs=N`；保留 `nthread` 时同值；匹配 OpenMP 策略 | DMatrix/QuantileDMatrix 构造、fit/predict、参数是否覆盖 |
| LightGBM | `n_jobs=N`；匹配其 native 线程策略 | fit 后真实参数、OpenMP runtime、预测 |
| CatBoost | `thread_count=N` | 自有线程不能仅由 threadpoolctl 信息推断 |
| HGB | 经核实的 OpenMP 限制 N；内层 BLAS 保守控制 | 内部实现是否调整线程池、运行时是否遵守 |
| BLAS 型模型 | estimator 无额外并发；BLAS 限制 N | GBLUP、Kernel Ridge、MLP 等实际热点与内存 |
| 尚未验证模型 | 保守线程计划 + 任务间并行 | 不能声称已完全限制未知 native 库 |

XGBoost 的模型线程与外层并发需要协调，线程竞争可能抵消收益。[R3] 对所有后端，参数值只是配置证据，实际 CPU 指标是运行证据。

### 5.4 清理冲突配置

新调度路径中，注册表不再自行执行 `max(global_n_jobs, min_threads)`；应接受经过校验的 lease/thread plan。搜索空间不得覆盖资源线程参数。模型实例 clone、最终 refit 和序列化恢复需要重新注入本次执行计划。

`_fold_training.py` 和 `genomic_utils.py` 不应继续从独立的过期 `self.n_threads` 生成另一套限制；改为接收执行上下文，或在安全情况下复用已有上下文。嵌套上下文只能保守收紧，不得静默放宽或覆盖其他任务。

旧入口可保留 legacy resolver；新旧 resolver 不得混用。默认值改变必须记录，不得无提示改变历史运行的资源行为。

## 6. 弹性分配策略

### 6.1 默认：fast_first_elastic

快模型在有可运行任务时获得较高权重；慢模型保留基础调度机会；完成事件驱动资源再分配。不能给 fast 队列绝对永久优先级，避免持续新任务饿死慢模型。

建议采用“模型级公平选择 + 模型内任务选择”：

1. 只考虑依赖已满足、未取消、资源配置合法的 runnable tasks。
2. 按模型竞争份额、实际已用额度和等待时长选候选模型；初期可以使用带 aging 的加权 deficit round-robin。
3. 候选模型内部按稳定任务 ID/轮转选择 ready fit，防止单 repeat 占满。
4. 检查 CPU、内存、worker 槽位；满足则原子授予 lease。
5. 候选任务暂时放不下时允许其他任务回填；大任务持续等待时应临时停止部分小任务回填，积攒其所需额度，避免永久饥饿。
6. 任务完成、失败回收、取消、依赖满足和定时健康检查触发下一轮分配；避免频繁忙轮询。

权重和份额是策略参数，首版采用简单可解释值，不引入复杂预测器。设计公平性测试时给定有限任务时长和可满足资源条件；非抢占运行无法保证任意长任务下的固定秒级响应。

### 6.2 资源回收不等于在线缩放

正在运行的 fit 保持原 lease。竞争份额下降时停止向该模型发新任务，等待已运行任务自然结束；不撤销活跃 lease，也不随意修改运行中实例。

当其他模型结束，只要慢模型存在 runnable fits、可用 worker 和内存，就获得释放资源。已完成 fold 不能为了凑满 CPU 重跑，也不能提前启动依赖尚未满足的 trial。

### 6.3 预算示例

假设其他模型共占 64、32、0 CPU 单位，XGBoost 单 fit 固定 16，且至少有相应数量的 ready folds：

| 阶段 | 其他模型已用 | XGBoost 并发 fits | XGBoost 已用 | 总预算 |
|---|---:|---:|---:|---:|
| 初始 | 64 | 1 | 16 | 80 |
| 部分完成 | 32 | 3 | 48 | 80 |
| 仅余 XGBoost | 0 | 5 | 80 | 80 |

实际日志中不会保证恰好经历这三个状态。这是解释扩容机制的合成例子，不能填写为实测。

多个慢模型竞争时先共享；某模型无 ready 工作时其未使用份额可被借用。无需等待整个快模型批次结束才释放已经空闲的资源。

### 6.4 可选严格两阶段

`fast_then_heavy` 可作为显式选择：快模型终态后再开放 heavy 队列。快模型失败如何处理必须继承运行失败策略，不能永远等“成功”。默认不采用严格两阶段，以免拖长慢模型尾部。

两阶段仍属于同一个 run，不通过重复执行训练命令覆盖同一个目录实现。

## 7. 第一版与后续优化边界

### 7.1 第一版：固定单 fit 额度，动态并发

- 每个模型有自己的 `preferred_cpu_per_fit`；起始值仅作实验候选。
- 单个 study 保持串行 trial；当前 trial 内允许并行 folds。
- repeat 并行只在现有统计与搜索语义允许的前提下使用；首版保留已有 repeat 并发语义，不为占满 CPU 擅自扩大。
- 单模型活跃 fit 上限统计跨全部 repeats/trials 的总量。
- `max_workers` 必须能提供足够 fit 槽位，但不能为每个 CPU 单位启动一个常驻进程。

### 7.2 第二版：新任务边界提高线程数

当只剩少数 runnable tasks，可依据已验证的线程扩展曲线，为尚未开始的 fit 分配更高额度。候选形状例如 `5×16`、`2×40`、`1×80`，应比较完成同一批工作需要的总墙钟时间，不能只比较最快一个 fold。

不同模型保持独立 profile；不预设所有模型的最优形状一致。配置优先级、线程数变化和性能选择证据必须写入 manifest。

### 7.3 明确尾部限制

若 5 个 folds 中 4 个已经结束，只剩一个已开始的 16 线程 fit，即使另 64 CPU 空闲，第一版也不会在线调整它。这是合法的低利用率，记录 `dependency_barrier` 或 `only_running_task` 等原因。

final refit 只有一个 fit 时，第一版可能只用该模型固定额度；第二版才能按经过验证的 refit profile 提高到 32/40/80 等值。文档和 UI 不得把第一版表述成“任何阶段都用满 80 核”。

### 7.4 后续才考虑 trial 并行

并行 trial 会改变自适应搜索的反馈时序；Optuna 文档也讨论了并行情况下的复现限制。[R4] 若后续启用，需要独立实验配置、storage 设计、确定的提交规则和新的复现声明，不能偷偷把 `optuna_n_jobs` 从 1 改成 80。

## 8. 建议配置与兼容策略

以下 YAML 是拟议配置，不能直接作为现有 GPSE CLI 参数使用。8 个模型可共享能力模板，但要分别覆盖性能参数。示例仅列 XGBoost 一个 ID。

```yaml
execution:
  scheduler: elastic_v1
  cpu_budget: 80
  worker_backend: isolated_process
  worker_slots: auto_memory_bounded
  policy: fast_first_elastic
  optuna_jobs_per_study: 1
  resize_running_fit: false
  memory_budget: auto_detect_with_headroom

resource_profiles:
  xgboost_reg:
    adapter: xgboost_cpu
    parallelism_kind: estimator_threads
    scheduling_class: heavy  # 可覆盖的初始提示
    min_cpu_per_fit: 1       # 待源码和兼容性核验，不继承 min_threads=8 的硬约束
    preferred_cpu_per_fit: 16
    max_cpu_per_fit: 80      # v1 固定 preferred；v2 才允许选择更大额度
    max_active_fits: 5
    base_cpu_share: 16
    max_cpu_share: 80
    elastic: true
```

`auto_memory_bounded` 和 `auto_detect_with_headroom` 都是配置语义说明，实际实现应使用项目既有 schema 形式并明确计算方法，不能原样当成已支持的字符串常量。

### 8.1 旧 CLI 参数迁移

| 现有参数线索 | 新设计中的定位 | 兼容要求 |
|---|---|---|
| `--threads` | 全局总预算 | 核查旧语义后保留或显式映射 |
| `--n_jobs` | 旧全局模型线程默认值 | 不再隐式代表总预算；冲突必须诊断 |
| `--max_workers` | 某层执行器并发上限 | 查明旧层级；不要直接认定是 fit worker 数 |
| `--repeat_workers` | repeat 编排并发上限 | 不形成第二个 CPU 池 |
| `--fold_workers` | fold 并发上限 | 与全局模型 cap 取共同约束，不相乘预算 |

建议提供 dry-run/resource-plan 输出，展示每个模型最后解析的 profile、来源、有效预算和瓶颈。用户显式给出的互相矛盾参数应报错或给出明确兼容映射；不要静默选一个。

单 fit 最小请求大于预算属于不可调度配置，应启动前失败。首版 preferred 大于有效预算时，需要明确选择“报错要求调整”或“预先解析为保守较小值并记录”；不得无限排队。

## 9. 任务生命周期、异常与恢复

推荐状态：`PENDING → READY → LEASED → RUNNING → SUCCEEDED / FAILED / CANCELLED`；重试产生新的 attempt，不复用旧 attempt 的资源和临时输出。

关键规则：

- task lease 的分配和释放必须幂等，每个 `lease_id` 只归还一次。
- 正常/异常路径执行清理；提交失败也归还额度。
- 心跳超时只表示可疑，不足以证明原 worker 停止。先确认旧进程及其计算子进程停止，再回收并重试，避免旧新两份任务同时耗费同一份额度。
- 以 worker generation / fencing token 拒绝过期结果；过期 worker 不能提交最终 artifact。
- 取消 pending 任务直接移除；取消 running 任务先请求停止，完成进程清理后再归还。
- 若通用进程池单个 worker 死亡导致整池失效，重建池并重排所有受影响 attempts；不能只修补一个 future 后假定其他任务正常。
- coordinator 重启前确认旧运行仍在/已退出；新 run 不得忽略存活旧 workers 直接分配全部预算。
- 失败重试不无限循环，OOM 重试先调整并发/内存 profile，保留失败记录。

把 input 数据共享作为优化项：只读内存映射或共享不可变数据可以降低复制；但 fold 内预处理与拟合必须留在训练分区，不能为共享缓存而在全数据上拟合 scaler、特征选择或其他有状态变换。

## 10. 统计与结果完整性

这是性能改造，不是实验方法修改。必须保持：

1. 数据身份、训练/验证/holdout 划分、fold ID、repeat 定义、指标、搜索空间和 trial 数不变。
2. seed 由稳定任务身份推导，不能来自 worker 编号、PID、提交/完成顺序或随时间变化的 Python hash。
3. trial 参数在编排侧生成；workers 只执行指定参数，按 fold ID 返回结果。
4. OOF 按样本 ID/索引定位写回，不能按异步返回顺序拼接。
5. trial 指标按稳定 fold 顺序聚合；若有 pruning/intermediate report，必须保留原步序，不能谁先完成就先上报改变决策。
6. 不改此前重复 CV 统计、模型选择、TOPSIS 与 stacking 的训练侧信息边界；具体已实现行为以仓库核查为准。
7. 线程/并发变化可能导致浮点差异，不默认承诺逐位相同。测试需给出事前约定的数值容差与模型差异解释。
8. 不能用 holdout 表现选择线程配置、改变模型超参数或挑选更有利的优化结果。资源 profile 依据时间和资源指标选择。

建议 artifact key 包含 run/model/repeat/trial/fold/stage/attempt。worker 写独立临时文件，协调器校验后原子提交；公共 leaderboard 和报告由单一汇总入口写入。

汇总屏障等待必需任务的终态，区分完整成功、允许的部分失败、取消和未完成。缺失 OOF 时不得填零后继续正常 stacking。重启跳过已成功且 manifest/hash 匹配的任务，不能仅凭文件存在认定完成。

## 11. 日志与性能指标

### 11.1 建议事件

`resource_plan_resolved`、`task_ready`、`lease_granted`、`fit_started`、`fit_completed`、`fit_failed`、`task_cancel_requested`、`lease_released`、`worker_lost`、`allocation_changed`、`aggregation_started`、`run_completed`。

JSONL 建议由协调器单写入，或通过集中日志队列串行落盘，避免多进程交叉写。事件带 schema version 和序号。

### 11.2 关键字段

```json
{
  "schema_version": 1,
  "event": "fit_started",
  "run_id": "example-run",
  "model_id": "xgboost_reg",
  "repeat_id": 0,
  "trial_id": 2,
  "fold_id": 1,
  "stage": "hpo_cv",
  "attempt_id": 0,
  "lease_id": "example-lease",
  "cpu_units": 16,
  "estimator_threads": 16,
  "memory_reserved_bytes": 8589934592,
  "worker_id": "example-worker",
  "pid": 12345,
  "thread_plan_id": "example-plan",
  "timestamp_utc": "2026-10-26T00:00:00Z"
}
```

该 JSON 全部为示例，8 GiB 不是 XGBoost 内存推荐值。真实线程池清单、CPU affinity、版本、有效参数可单独存快照并以 ID 引用，避免每条事件重复大对象。

### 11.3 时间口径

| 指标 | 定义 |
|---|---|
| 依赖等待 | 创建到 READY；与资源排队区分 |
| 资源排队 | READY 到获得 lease |
| 分派/准备 | lease 到 fit 开始；包含反序列化和必要数据准备 |
| fit wall | 单 fit 起止的单调时钟差 |
| score wall | 预测、评分阶段耗时 |
| task wall | 整个带资源计算任务的耗时 |
| HPO wall | HPO 阶段起止墙钟，含必要等待和调度开销 |
| 模型 wall | 模型阶段完整起止，包括 HPO、OOF/refit 等实际阶段 |
| makespan | 同一完整 run 起止墙钟 |

并行 fits 的累计 wall 可以大于阶段 wall，不能相加冒充用户等待时间。对 4991.8 秒必须定位实际计时代码再命名。

### 11.4 CPU 与内存监控

定期采样该 run 的 worker 进程树，包括内部子进程；结合租约时间积分计算：

```text
平均占用逻辑 CPU = 运行进程树 CPU_seconds / elapsed_wall_seconds
全局 CPU 利用率 = CPU_seconds / (effective_budget × elapsed_wall_seconds)
租约计算利用率 = CPU_seconds / integral(active_lease_cpu_units dt)
```

避免父进程累计 children time 与逐个子进程采样重复计数；短生命周期子进程需用累计退出统计或 run 级 cgroup 指标补足。监控线程数量不能代替 CPU 使用量。区分 `top` 的多核百分比与归一化百分比。

记录进程树 RSS 峰值，条件允许时同时记录 PSS 或 cgroup memory peak；共享页导致 RSS 求和可能高估独占内存。记录 I/O wait、page faults、swap、cgroup throttling 和系统竞争作为诊断项，采样频率以约 1–5 秒为起点，测量采样开销。

空闲原因必须可解释：无 ready 工作、trial barrier、内存不足、worker 槽位不足、单 fit 不可扩展、I/O/数据准备、预算粒度碎片、公平性等待、配置限制或外部 throttling。

## 12. 性能验证计划

### 12.1 实验拆分

| 实验 | 目的 | 控制条件 |
|---|---|---|
| A：原实现基线 | 定位原有等待、线程冲突和尾部 | 原配置、独立输出目录、完整监控 |
| B：仅线程治理 | 隔离统一 thread plan 的收益/代价 | 保持任务并发与工作量相同 |
| C：弹性 fold 调度 | 验证释放资源后增加并发 | 同一 workload、固定 per-fit 额度 |
| D：不同线程形状 | 找各模型较优 profile | 对同一组已固定 fit 任务比较 |
| E：完整训练回归 | 验证最终 makespan 和统计产物 | 相同数据、种子、fold/repeat/trial 设置 |

B 与 C 分开，避免无法区分收益来自修复线程限制还是调度策略。

### 12.2 两种基准负载

**固定任务重放**：固定一组 trial 参数与 folds，比较调度吞吐量，排除自适应搜索路径变化。所有方案执行同一组任务到相同完成边界。

**真实 HPO + 全流程**：保持搜索空间、sampler seed、trial 数、pruning、repeat 逻辑一致，检查实际用户体验与结果完整性。即使串行 trial，底层浮点差异也可能影响自适应后续选择，需报告 trial 参数轨迹是否相同。

### 12.3 分级执行

1. 合成调度测试：不加载大数据，验证资源不变量、扩容、饥饿、异常、恢复。
2. 小数据真实 smoke：8 个直接线程模型分别验证 fit/predict 和进程隔离；其余 13 个建立能力清单并运行适用的 smoke。
3. 代表性负载：选择真实数据规模和高维输入，先少量固定任务，控制总体成本。
4. 独占或明确记录其他负载的 80 CPU 环境：交替执行方案，记录冷/热缓存、版本、NUMA、affinity、温度/频率因素（可观测时）。
5. 最终完整 run：验证单 run 汇总与实际总时间。

初筛用小规模；最终选中方案建议至少 3 次成对测量，成本允许再增加。报告中位数和范围，不拿一次最快值宣称普适加速。没有 80 CPU 或内存不足时如实标记未执行，可用缩比测试验证调度正确性，但不能称其已证明 80 CPU 性能。

### 12.4 性能报告表

| 项目 | A 原实现 | B 线程治理 | C 弹性调度 | 备注 |
|---|---|---|---|---|
| effective CPU budget | 待测 | 待测 | 待测 | 固定/明确差异 |
| 全 run wall | 待测 | 待测 | 待测 | 主指标 |
| 快模型完成分位时间 | 待测 | 待测 | 待测 | 验证优先策略 |
| XGBoost / 其他慢模型 HPO wall | 待测 | 待测 | 待测 | 按模型分列 |
| 资源排队 wall | 待测 | 待测 | 待测 | 不能与并行时间混加 |
| 平均 CPU 与有效利用率 | 待测 | 待测 | 待测 | 实际监控 |
| 进程树 RSS / PSS / cgroup peak | 待测 | 待测 | 待测 | 注明口径 |
| 失败、重试、取消 | 待测 | 待测 | 待测 | 不隐藏失败 |
| 数值与产物一致性 | 待测 | 待测 | 待测 | 预设容差 |

推广条件建议：所有正确性测试通过；无死锁/资源泄漏/明显过度订阅；目标负载总时间改善超出测量波动；内存和快模型完成时间没有不可接受退化。可在运行前约定例如 10% 的工程目标，但它只是项目选择阈值，不是本文预测的收益。无显著收益则保留可观测性与正确性修复，不强推新默认策略。

## 13. 必要测试与验收矩阵

| ID | 测试场景 | 通过要求 |
|---|---|---|
| T01 | 预算 1/4/8/16/80；请求超过预算 | 不超账；非法配置启动前诊断；无无限等待 |
| T02 | 8 个直接参数模型 | estimator 参数与 lease/thread plan 一致 |
| T03 | 线程限制冲突 | 不再出现外层 16、内层 8 或 registry 私自放宽 |
| T04 | worker 先运行 16 再运行 4 | 线程池状态正确恢复，不污染后续任务 |
| T05 | 两个不同额度任务并发 | 隔离有效；不依赖同进程交叉限制上下文 |
| T06 | 快任务逐步完成，慢模型有 backlog | 慢模型无需重启获得更多 fit 槽位与 CPU |
| T07 | 多个慢模型竞争及大小任务混排 | 无永久饥饿；借用份额可在任务边界回收 |
| T08 | 父模型/repeat 等待子 fold | 父层不重复扣账；worker 池不会被阻塞父任务占满 |
| T09 | 异常、提交失败、取消、worker crash | 恰好一次释放；无旧 worker 与新 attempt 同时占用同一额度 |
| T10 | 内存紧张和 OOM | 不盲目提高并发；保留失败诊断和受限重试 |
| T11 | fold 返回顺序打乱 | 指标、OOF 映射、稳定聚合不受完成顺序影响 |
| T12 | 有 pruning/intermediate metrics | 保持原有 step 与决策顺序，不随完成时间改变 |
| T13 | final refit、predict、stacking | 计算阶段均受资源约束，汇总触发一次 |
| T14 | resume + 旧 attempt 迟到 | 结果去重，旧结果被 fencing 拒绝，完整性检查有效 |
| T15 | legacy 参数与新配置组合 | 兼容映射或显式报错，无静默歧义 |
| T16 | 不足 80 CPU / quota 限制 | requested/effective 明确，测试结论不冒充 80 CPU |
| T17 | 只剩一个已运行 fit | 允许有解释的空闲，不重跑任务或改统计工作量 |
| T18 | 所有 21 个模型与 run 汇总 | 能力列表完整，模型产物/失败状态可核验，缺失 OOF 不伪造 |

调度核心测试用 fake clock、可控完成事件和 barrier，避免依靠真实 sleep 造成脆弱时序测试。资源约束的测试独立于模型质量测试，真实模型 smoke 用小数据即可。

## 14. 实施工作包

| 阶段 | 产物 | 完成条件 |
|---|---|---|
| P0 源码与运行审计 | `threading_audit.md`、21 模型清单、调用拓扑、原始基线 | 线程来源、CPU 记账、进程边界均有源码证据 |
| P1 线程治理 | ResourceProfile/ThreadPlan、统一 resolver、兼容层 | T01–T05、T15 通过；实际 fit 参数可追溯 |
| P2 弹性调度 | 全局 lease、ready queue、模型公平策略、内存 gate | T06–T10、T16–T17 通过 |
| P3 全流程集成 | final refit、OOF、恢复、单 run 汇总 | T11–T14、T18 通过 |
| P4 性能验收 | 固定负载与完整训练 benchmark、差异分析、推荐配置 | 无虚构性能；有完整 run 证据或明确环境限制 |
| P5 可选后续 | 新 fit 边界线程扩容、历史 profile | 单独验证后推广；不混入首版验收 |

建议新增模块如 `_resource_manager.py`、`_thread_policy.py`、`_model_resource_profiles.py`，但若仓库已有同等职责应扩展复用，避免产生第二套调度器。既有 `_scheduler.py` 负责模型编排和策略，`_fold_scheduler.py` 负责 ready task 接入；具体划分以 P0 的调用图为准。

交付时记录源码 commit、工作区差异、环境 manifest、配置来源、执行命令、测试结果与未完成项。CLI 和 README 必须明确旧参数与新参数含义。新策略通过显式开关启用并保留 legacy 回退；回退只能在新运行/安全恢复边界进行，不热切换已活跃的资源账本。

## 15. 给维护者的最终验收问题

1. 任意 fit 的线程数能否追溯到同一个 lease，而不是三个互相覆盖的配置？
2. 快模型结束后，慢模型是否在下一次可分配事件自动增加实际并发，而不仅是日志中的目标额度增加？
3. 多个慢模型是否公平共享预算，是否存在父任务占着 worker/token 等子任务？
4. CPU、内存、executor 三个约束是否都生效？
5. 取消和崩溃后是否确认旧计算停止才回收资源？
6. OOF、重复 CV 统计、模型选择和 stacking 是否仍遵循原有信息边界？
7. 全 run 是否比基线更快，收益是否超出运行波动？
8. 低 CPU 利用率是否有明确原因，而不是用更多进程掩盖真正瓶颈？

## 16. 参考资料与适用边界

以下为并行控制原则的官方参考，访问日期为 2026-10-26。工程设计、测试矩阵和配置均是针对本次 GPSE 需求的建议，不是这些文档提供的现成 GPSE 实现。

- [R1] scikit-learn 1.7.2 并行与资源管理：<https://scikit-learn.org/1.7/computing/parallelism.html>。用于区分 joblib、OpenMP、BLAS 与嵌套并行。
- [R2] threadpoolctl 官方仓库：<https://github.com/joblib/threadpoolctl>。在线主分支可能已含后续版本功能；本项目必须以已安装 3.6.0 的源码和行为核验，不能直接套用主分支新增 API。
- [R3] XGBoost 3.2 release 系列 Python API：<https://xgboost.readthedocs.io/en/release_3.2.0/python/python_api.html>。页面当前标示 3.2.1，用户环境为 3.2.0，具体参数路径以安装包核验为准。
- [R4] Optuna 官方 FAQ：<https://optuna.readthedocs.io/en/stable/faq.html>。用于并行优化与复现限制；用户实际 Optuna 版本尚未提供。

本文没有宣称已测得实际 80 CPU 利用率、速度提升或修复完成。第一项行动是 P0 审计，第一项工程改造是统一线程执行计划，首个性能目标是用不改变实验工作量的弹性 fit 调度缩短完整运行时间。
