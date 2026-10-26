# GPSE 线程与弹性 CPU 调度：开发 Agent 实施提示词

日期：2026-10-26  
配套设计：`GPSE_Threading_Optimization_Design_20261026.md`

> 使用方法：将两份 Markdown 一起提供给能访问 GPSE 仓库的开发 Agent。首次建议发送下面的“完整实施提示词”。需要控制单次工作量时，可逐个发送后面的 P0–P4 提示词。内容中的模块名是设计建议；Agent 必须按实际代码定位，不能假装已经存在。本文件不包含已执行的源码修改或性能测试结果。

## 1. 完整实施提示词（可整段复制）

```text
你现在负责 GPSE 的线程治理和弹性 CPU 调度改造。请阅读配套《GPSE 线程治理与弹性 CPU 调度优化设计》，在当前真实仓库中完成源码核查、分阶段实施、适当测试和交付。不要只给口头建议，也不要未经核查把所有 n_jobs 改成 80。

一、用户目标

机器计划为一次训练提供最多 80 个 CPU 单位。希望快模型优先完成，慢模型从开始就得到适量资源；其他任务完成后，空闲资源自动由仍有待运行工作的模型接管。支持 XGBoost、RF、LightGBM、CatBoost 的回归和分类模型，并为全部 21 个模型建立统一能力描述。所有阶段仍属于同一个 run，保留现有 leaderboard、OOF、stacking 和报告汇总。

二、已提供背景，必须自行核实

1. 注册表有 21 个模型：15 回归、6 分类。
2. 8 个直接并行模型：rf_reg/rf_clf、xgboost_reg/xgboost_clf、lightgbm_reg/lightgbm_clf、catboost_reg/catboost_clf。
3. RF/LightGBM 使用 n_jobs；XGBoost 当前注入 n_jobs/nthread；CatBoost 使用 thread_count。
4. 8 个模型的 min_threads=8，注册表当前采用 max(n_threads, min_threads)。这只是已有配置，不能当成算法硬要求。
5. knn_reg 的 n_jobs 主要涉及查询/预测；histgradientboost_reg 依赖底层 OpenMP。其他模型可能有 BLAS/OpenMP 并行，需要具体核查。
6. 外层 scheduler、fold training 和 genomic_utils 都可能调用 threadpool_limits，线程值来源不完全一致。
7. 当前 optuna_n_jobs=1，已有模型/repeat/fold 多层调度。
8. 用户报告版本：sklearn 1.7.2，XGBoost 3.2.0，LightGBM 4.7.0，CatBoost 1.2.10，threadpoolctl 3.6.0。部署版本与源码为准。
9. 一次 XGBoost HPO 约 4991.8 秒，模型总计约 5589 秒。尚不知道计时口径与实际 CPU 使用量，不允许把它当作已知纯计算时间。

三、先完成 P0 源码审计

先阅读 AGENTS.md、README 和相关测试；记录 git commit、工作区已有差异、Python/依赖版本、CPU affinity/cgroup quota 和内存限制，不覆盖用户修改。

重点检查：
- gpse/train/cli.py
- gpse/train/_scheduler.py
- gpse/train/_fold_scheduler.py
- gpse/train/_fold_training.py
- gpse/train/workflow.py
- gpse/config/_model_registry.py
- gpse/config/models.yaml
- gpse/config/constants.py
- gpse/utils/genomic_utils.py

搜索所有 n_jobs、nthread、thread_count、min_threads、threadpool_limits、ThreadPoolExecutor、ProcessPoolExecutor、Parallel、OMP_NUM_THREADS、MKL_NUM_THREADS、OPENBLAS_NUM_THREADS、fit、predict、refit 的相关调用。

输出：21 个模型能力表、真实调用拓扑、线程配置优先级、预算扣账位置、执行器容量、所有训练/预测入口。必须区分源码确认、运行确认、推断、未知。没有日志或命令时如实写缺失，不能补造。

四、必须满足的架构约束

1. 一个全局 ResourceManager 管理 CPU、内存与计算 worker 槽位。实际计算任务申请 lease；父模型/repeat/trial 编排不重复占用 CPU token。
2. lease 是本次任务的唯一资源来源，模型适配器从 lease 生成 ThreadPlan，统一 estimator、joblib、OpenMP/BLAS、数据构建与预测阶段的设置。
3. 不把所有内层线程池都设置为 lease 数量；按实际嵌套并行结构约束，避免外层 N × 内层 N。
4. 新路径不得继续由 registry 在分配后通过 max(..., 8) 私自提高线程。保留显式、可测试的旧路径兼容规则。
5. 异构线程额度的并发 fits 优先在独立进程执行；不要假设同进程多个 Python 线程中的 threadpool_limits 上下文相互隔离。核验安装的 threadpoolctl 3.6.0 行为，不套用主分支新增 API。
6. worker 同一时刻只执行一个 fit，可复用；任务前后恢复线程池状态。所有计算子进程纳入资源与生命周期管理。
7. 调度器只向可用 worker 分派已获资源的任务；不要用阻塞等待 token 的任务占满 worker 池。
8. 资源分配/释放原子且幂等。异常、提交失败、取消、进程崩溃都有处理。心跳超时不能直接当作旧 worker 已停止；确认停止后才回收重试，拒绝过期 attempt 结果。
9. 配置不可满足时启动前报错，不能无限等待。有效 CPU 预算必须考虑 affinity、quota 和作业分配，不只看 os.cpu_count()。
10. 调度器的 token 是逻辑预算，不是操作系统硬隔离；日志不能把配置额度当作实际 CPU 使用率。

五、第一版调度策略

实现 fast_first_elastic：快模型较高权重，慢模型基础调度机会；空闲份额可借用；完成/资源释放时重排。多个慢模型共享预算，有 aging 或等价机制避免饥饿。给大任务留出最终可启动的机会，避免小任务持续回填导致它永远得不到额度。

第一版固定每个模型 preferred_cpu_per_fit，动态增加该模型活跃 folds。保留每个 study 的 optuna_n_jobs=1，不增加 trial 并行，不修改正在运行的 fit。

示意：XGBoost 每 fit 16 线程，其他模型占用从 64 降到 32 再到 0 时，如有 ready folds、worker 和内存，XGBoost 并发可从 1 增至 3 再到 5，总使用预算 16→48→80。这个示例不是普适最优配置，也不是要求所有模型复制 16 线程。

单模型 max_active_fits 必须跨全部 repeats/trials 计数。repeat 并发保留原有允许语义，不为凑满 CPU 改研究流程。只有一个已经运行的 fit 时允许剩余 CPU 空闲，并记录原因。

可选 fast_then_heavy 只作为显式策略，不作为默认，不通过两次训练覆盖同目录来实现。

六、结果与统计约束

不得改变数据、train/holdout 边界、CV splits、repeat 统计定义、搜索空间、trial 数、指标、模型选择、TOPSIS、stacking 输入来源。
seed 与稳定任务身份绑定，不能随 worker/PID/完成顺序变化。
OOF 按样本 ID 和 fold ID 定位，不能按异步返回顺序拼接。
trial 聚合按稳定 fold 顺序；已有 pruning/intermediate reports 维持原 step 顺序。
一个 run 统一汇总；final refit、预测评分、stacking 等重计算阶段也必须纳入资源约束。
结果按 task/attempt 独立写入并由协调器原子提交；恢复必须校验 manifest/hash，旧 attempt 结果不得覆盖新结果。缺失 OOF 不得填零伪装完成。

七、日志与验收

记录 task identity、attempt、lease、PID/worker generation、CPU 分配、真实模型参数、线程池快照、ready/lease/start/end 时间、失败与释放原因。
区分依赖等待、资源排队、数据准备、fit、predict/score、HPO wall、model wall、全 run makespan。
采集运行进程树 CPU 时间与内存峰值，避免子进程重复计数，记录监控口径。不能用参数值或线程数量代替实际 CPU 利用率。
说明低利用率原因：无 ready 工作、trial barrier、内存不足、worker 不足、线程粒度、I/O、throttling 等。

按配套文档 T01–T18 设计有意义的测试。先用 fake tasks 和可控事件验证调度，再用小数据执行真实模型 smoke。
性能对照拆分 A 原实现、B 仅统一线程治理、C 弹性调度；再比较不同线程形状。固定任务重放用相同参数列表与同一组 folds；全流程验证使用相同实验设置与独立结果目录。
不要未经资源核验就启动长时间 80 CPU 全量实验。若本会话已有该环境和预算授权则继续；否则完成代码、小规模验证和准确 benchmark 命令，把依赖条件写清楚。没有 80 CPU 实测不能标记 80 CPU 性能验收通过。

八、执行与交付方式

按照 P0 审计、P1 线程治理、P2 弹性调度、P3 全流程集成、P4 性能验收执行。在已授权范围内连续推进，不在每阶段机械询问确认。若事实与设计冲突，采用满足目标的最小改动并说明理由；真正缺少必需信息时继续无依赖工作，明确阻塞点。

保留 legacy 回退开关，先 opt-in 新调度策略。不要修改无关文件，不重构与任务无关的统计/模型逻辑，不删除用户产物，不自动启动第二份训练去占用正在工作的机器。

最终输出：
1. 已确认的问题及源码证据；
2. 修改文件和设计取舍；
3. 测试命令、真实结果、未覆盖项；
4. 实际支持的 CLI/配置示例与兼容说明；
5. 基线与优化性能表，未实测字段写“未执行”；
6. legacy 回退与恢复方式；
7. 仍需用户提供的最小运行信息。

不要把“线程参数注入成功”“CPU 曲线更满”“合成调度测试通过”写成“已证实训练更快”。主要性能结论来自相同工作量的完整墙钟时间。
```

## 2. P0：仅源码与运行行为核查

适用于希望先拿到调查结果、暂不改生产代码的场景。

```text
请对当前 GPSE 做线程与调度专项只读核查。参考配套设计文档，但不要先假定设计里的问题已存在。

当前阶段不修改生产代码、不启动正式训练、不覆盖结果目录。可运行小型参数检查或已有轻量测试，明确区分实例化检查与真实运行检查。

任务：
1. 记录仓库 commit、dirty files、Python 和相关依赖版本。
2. 从真实注册表列出全部 21 个模型，输出 model_id/task_type/estimator/线程字段/min_threads/实际注入逻辑/底层并行可能性。
3. 追踪 cli → 模型调度 → repeat → Optuna trial → fold → fit/predict → refit → 汇总调用链。
4. 标出每一层进程/线程池、worker 数、等待位置、token 扣账与释放位置。检查父任务等待子任务时是否持有资源和计算 worker。
5. 列出所有 threadpool_limits 及环境变量的来源、作用域、嵌套顺序；核实同进程异构任务是否交叉修改全局线程池。
6. 检查 clone、refit、pipeline 内部、DMatrix 或其他数据结构构建是否重新覆盖线程参数。
7. 定位 4991.8 秒对应计时器；如缺少本次日志和命令，写明无法确定，不推测实际核数。
8. 如运行环境允许，读取 affinity/cgroup quota/内存与可用 worker 限制；不把宿主机 CPU 数当作当前分配额度。
9. 输出最小改造方案与可复用模块，避免复制已有调度器。

交付：threading_audit.md、完整模型能力表、调用关系图或表、问题分级和建议修改清单。每项问题附 file:symbol/line 和触发条件；未知项单列。
```

## 3. P1：统一线程执行计划

```text
基于已完成的 P0 审计，只实施线程治理层，暂不改变 trial 数、fold 并发策略或研究统计语义。

实现目标：
- 用 ResourceProfile + ThreadPlan 或项目内等价结构统一 estimator、joblib、OpenMP/BLAS 设置。
- 新路径中模型实例从分配额度获取线程参数，不再被 registry 的 max(n_threads, min_threads) 私自放大。
- 查明并消除外层有效线程数与 fold/genomic_utils 的 self.n_threads 不一致。
- 8 个直接多线程模型提供明确 adapter，其余模型提供经核查的 profile 或显式 unknown/conservative 状态。
- 对并发任务的线程池状态进行进程隔离；复用 worker 时正确设置和恢复限制。
- 旧 CLI 行为通过独立兼容解析保留；冲突参数显式诊断。
- 输出解析后的 resource plan，并记录来源和实际 estimator 参数。

验证：
1. 参数解析与预算 1/4/8/16 的行为，包括 n_jobs 小于原 min_threads 的场景。
2. 同 worker 先执行 16 再执行 4 的线程计划，状态正确恢复。
3. 两个不同资源计划并发执行，不能依赖同进程交错全局限制获得隔离。
4. clone、predict、refit 也取得正确资源计划。
5. 8 个模型使用小数据 smoke；无法安装/加载的依赖如实列为阻塞，不写通过。
6. 比较线程治理前后的数值产物，解释容差和差异，不要求所有 native 实现逐位相同。

交付修改、必要测试与运行结果。此阶段不宣传弹性调度已完成，也不提前声称加速收益。
```

## 4. P2：任务级资源池与弹性 fold 调度

```text
在 P1 的统一 ThreadPlan 基础上，实现共享 CPU/内存/worker admission control 和 fast_first_elastic 策略。

必须实现：
1. 全局资源账本，只在实际计算任务层扣账；模型/repeat 编排不重复占用 token。
2. ready queue 与非阻塞分派；任务只有在 CPU、内存和 worker 都可满足时才获得 lease 并提交。
3. 模型级公平选择，快模型优先但慢模型不饿死；空闲份额允许借用。
4. 第一版每模型固定单 fit 额度，只动态增加活跃 folds，保持每个 study trial 串行。
5. max_active_fits 跨 repeats/trials 统计；不能由每一层独立上限相乘超配。
6. 完成/异常/提交失败释放资源后自动分派下一任务。
7. lease release 幂等；worker generation/attempt fencing 防止迟到结果覆盖。
8. 进程崩溃后确认旧计算停止再回收；考虑整个进程池失效的恢复路径。
9. 大任务等待与小任务回填同时存在时避免永久饥饿。
10. 资源不可满足启动前报错；内存不足时不盲目提高并发。

用可控 fake tasks 验证：
- 缩比预算 10，其他任务释放后慢模型 fit 并发从 1 增至 3 再至 5（每 fit 2）。
- 合成预算 80 也验证不变量，但明确这不是 80 CPU 实机性能测试。
- 多慢模型竞争、公平回填、取消、重复完成事件、提交异常、worker 崩溃、旧 attempt 迟到。
- 只剩一个已运行 fit 时允许空闲，不能修改运行中实例或补跑无用任务。

输出每项不变量的测试证据、事件日志样例、配置说明和剩余限制。用显式开关启用新策略，保留 legacy。
```

## 5. P3：接入完整训练、恢复与汇总

```text
请将已完成的统一线程与弹性调度接入 GPSE 完整训练流程，重点保证统计和产物完整性。

需要检查并实现：
1. HPO folds、必要 OOF 重训、final refit、predict/score、stacking fit 等计算入口都走同一资源约束。
2. 预处理在正确训练分区内拟合，禁止为了共享缓存而引入数据泄漏。
3. TaskKey/seed 与固定模型、repeat、trial、fold、stage 关联，独立于 worker 和完成顺序。
4. fold 结果按稳定 ID 聚合；OOF 按样本索引填入，不按完成顺序拼接。
5. 保留已有 pruning/intermediate reports 的步序与决策语义。
6. 模型完成条件明确区分 HPO、OOF、refit 和后续必需产物完成。
7. 同一个 run 统一生成 leaderboard、stacking 和报告；写入者唯一，缺失 OOF 必须阻止正常 stacking 或明确标记部分失败。
8. 任务独立临时输出、校验、原子提交；resume 校验 run manifest、任务状态和文件 hash。
9. 过期 attempt 结果被拒绝；取消/失败/恢复不产生重复结果或重叠计算。
10. 新线程策略不覆盖旧结果目录，不改历史统计输出定义。

测试时故意打乱 fold 完成顺序、注入一个失败、恢复一次、模拟旧 attempt 迟到。验证指标、OOF 覆盖、产物数量、汇总触发次数和状态准确性。

输出端到端 smoke 的完整命令、产物清单、测试结果与需要长实验验证的项目。没有真实运行就不写“报告已正确生成”。
```

## 6. P4：性能基准与验收

```text
请验证 GPSE 新调度是否缩短相同工作量的总耗时，不以 CPU 配置或利用率曲线替代结论。

先确认运行资源和是否已有训练占用，再生成准确命令；只有当前授权范围允许的负载才执行。先运行小规模固定任务重放，再做成本可控的真实 HPO，全量 80 CPU 实验在资源条件满足后执行。

对照设计：
A. 旧调度、旧线程路径；
B. 统一线程路径，尽量保留原并发；
C. 统一线程路径 + 弹性 fold 调度；
D. 在同一固定任务集上比较不同 per-fit 线程与并发组合。

控制变量：
- 数据 fingerprint、数据规模与 dtype、fold splits、repeat、种子、trial 参数列表或搜索配置、early stopping、预处理、版本、CPU quota/affinity、输出边界。
- 每个方案使用独立输出目录，执行同样的工作到同样阶段。
- 不以更少 trials、更少 folds、更少树、不同 tree_method 或改变算法获得的收益冒充调度加速。
- 固定重放用于吞吐归因；真实 HPO 用于最终体验验证。记录真实 HPO 参数轨迹是否一致。

指标：
- 全 run makespan；各模型 HPO/总 wall；快模型完成时间；队列/准备/fit/score 时间；
- 进程树 CPU_seconds、有效 CPU 利用率、租约使用效率；
- 同时活跃 fits、实际线程参数和线程池快照；
- RSS/PSS/cgroup peak（有哪个写哪个，说明口径）、page faults、swap/throttling；
- 失败/重试/取消、结果完整性和数值容差。

执行顺序交替，记录冷/热缓存和其他负载。最终候选方案建议至少 3 次成对测量（预算不足则注明），报告中位数和范围。

输出 benchmark_report.md：配置与环境、命令、原始日志位置、对照表、加速比、数值一致性、解释低利用率原因、推荐 profile、未完成项和回退建议。

如果没有 80 CPU 环境，完成缩比正确性验证和可执行 benchmark 脚本/命令，明确“80 CPU 实机性能待验收”。如果新策略更慢，分析原因并保留旧默认，不选择性隐藏结果。
```

## 7. P5：可选的新 fit 边界线程扩容

仅在 P1–P4 稳定后使用，避免首版同时引入过多变量。

```text
在已通过验收的 elastic_v1 上，增加“仅在新 fit 启动前选择线程额度”的可选优化。

目标：当前 ready fits 很少或进入 final refit 时，如果该模型的实测扩展曲线表明收益，可以为尚未开始的 fit 分配更多 CPU。正在运行的 fit 不能被在线修改。

要求：
1. 每模型单独保存线程形状与耗时/内存 profile，按相近数据规模匹配；缺少证据时回退固定额度。
2. 决策同时考虑 ready fits 数、剩余 CPU、内存、worker 槽位和其他模型公平份额。
3. profile 选择只依据性能与资源指标，不能用 holdout 准确率挑选。
4. 资源计划先原子授予再实例化模型；显式写入 manifest，便于复现运行配置。
5. 单独比较 5×16、2×40、1×80 等候选在同一任务集合上的总时间；线程数不必只允许这些值。
6. final refit 单任务使用高线程也必须实测，不能保证 80 最快。
7. 保留 v1 回退；不顺便增加 trial 并行或修改搜索参数。

交付调度选择规则、基准证据、适用范围、反例和默认启用建议。
```

## 8. 审查实施结果的提示词

适用于开发 Agent 完成后，再进行一次以证据为中心的代码审查。

```text
请审查 GPSE 线程与弹性调度改造，不以开发者结论代替代码和测试证据。先阅读 diff 与实施报告，执行必要的针对性验证。

重点查：
- 是否只是把 n_jobs 改成 80，却没有统一预算和调度？
- registry/min_threads、fold self.n_threads、环境变量是否仍覆盖 lease？
- 是否在同一进程的并发线程中交叉修改全局线程池？
- 模型/repeat/fold 是否重复扣 token 或形成 worker 等待死锁？
- 外层并发与内部 joblib/BLAS 是否相乘超配？
- 快模型结束后慢模型实际是否扩容，而非只更新目标预算？
- 多慢模型公平性、内存 gate、大任务饥饿是否处理？
- worker 只超时但未停止时是否已归还 token 并重试？
- 取消、崩溃和旧 attempt 迟到是否可能污染结果？
- OOF 索引、fold 聚合、pruning 顺序、repeat 统计是否被改变？
- final refit/预测/stacking 是否绕过资源池？
- benchmarks 是否固定工作量，是否真实测量 CPU/内存与完整 wall？
- 是否把合成 80 token 测试冒充 80 CPU 实机性能？
- legacy 回退、版本 manifest 和 CLI 文档是否准确？

输出按严重程度排序的问题清单，每项包括 file:symbol/line、触发条件、影响、最小修复建议及测试缺口。无问题则说明实际检查范围和仍未覆盖项，不给无条件正确保证。
```

## 9. 接收交付时的快速核对清单

- [ ] 已核查全部 21 个真实模型 ID，没有遗漏或杜撰。
- [ ] 新线程计划是唯一来源，旧 `min_threads` 不再私自放宽 lease。
- [ ] 多进程隔离/其他安全隔离方案有证据，worker 复用不会遗留限制。
- [ ] 快模型完成后，慢模型在有 ready 工作时确实自动扩容。
- [ ] 多模型竞争和 CPU/内存/worker 三类约束均已测试。
- [ ] 失败、取消、崩溃、恢复不会泄漏或重复释放额度。
- [ ] 完整训练的 OOF、leaderboard、stacking 和报告被验证。
- [ ] 第一版未擅自并行 trials、改变超参数或改变统计方法。
- [ ] 有同工作量的耗时对照；无环境的项目明确标“未执行”。
- [ ] 有可用的 legacy 回退路径和准确命令。

推荐实施顺序：P0 → P1 → P2 → P3 → P4。P5 为后续性能优化，不作为第一版必须完成条件。先让资源分配可信，再让空闲资源被有效使用，最后用真实总时间决定是否成为默认策略。
