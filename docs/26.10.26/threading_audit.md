# Threading audit — P0 work in progress

> 接续核查更新：下文保留早期审计时间点，不能作为当前实现状态。当前已接入 ThreadPlan、全局 admission、fold/refit/stacking 任务提交与跨 run checkpoint 恢复，并验证 21 模型默认参数完整 run；命令、测试结果、配置和剩余限制见 [实施记录](threading_implementation_report.md)，逐项范围见 [验收矩阵](threading_acceptance_matrix.md)，实测见 [性能记录](benchmark_report.md)。失败 HPO 轨迹恢复、自动重试/OOM 与 80 CPU 验收仍未完成。

日期：2026-10-26。基准 commit：`b12ccf5f65afbb4b51cb9bdfeda6975e03b7f9ad`。
初始工作区仅 `docs/26.10.26/` 未跟踪；保留两份用户设计文档。本文为部分审计，尚未完成 P0–P4 验收。

## 已确认与设计基线的差异

- `gpse/config/_model_registry.py:ModelRegistry.create_model` 在 defaults 与调用参数合并后用 `self.n_threads` 覆盖线程字段。当前源码无 `min_threads`，不能把历史“至少 8 线程”当作现存 bug。
- `gpse/train/_scheduler.py:run_budgeted_models` 已有 spawn 共享进程池；容量取 CPU budget / n_threads 与模型、repeat、fold 并发乘积的较小值。所有任务使用相同 n_threads，没有模型独立 profile。
- `gpse/train/_fold_scheduler.py:_Dispatcher` 用 semaphore 限制计算槽位。controller 在 submit 中阻塞等待，不占计算 worker。没有 CPU/内存/worker 三维 lease、模型级公平队列或跨 repeat 的 max_active_fits。
- `_Dispatcher.map_folds` 按 fold 顺序等待和返回，保持稳定聚合，但窗口补充也等待队首，可能在后面的 fold 已完成时闲置槽位。需要将完成驱动补充与有序结果聚合分离，并保持 trial failure drain 语义。
- `_scheduler._execute` 在独立进程中设置统一 threadpool_limits；fold 和 optimization 内层再次设置相同 n_threads。异构额度尚不存在，不能宣称异构隔离已验证。
- `gpse/train/stacking.py` 有独立的预算进程池及 meta fit；尚未接入全 run 唯一资源账本。
- `gpse/utils/parallel.py:get_available_cpu_cores` 已检查 affinity 与 cgroup 祖先 quota。仍需验证当前挂载布局和作业分配约束。

## 实际模型能力清单

线程字段来自当前 YAML；无字段不等于 native 库单线程。底层扩展曲线全部未知，后续须以实际 fit/predict smoke 和线程池快照验证。
注入优先级：YAML defaults → 调用 params → registry n_threads 覆盖 thread_strategy 指定字段。

| model_id | task | estimator | 线程字段 | min_threads | 证据 |
|---|---|---|---|---|---|
| elasticnet_reg | regression | sklearn.linear_model.ElasticNet | none | 无 | 源码确认；运行未验证 |
| lasso_reg | regression | sklearn.linear_model.Lasso | none | 无 | 源码确认；运行未验证 |
| sgd_reg | regression | sklearn.linear_model.SGDRegressor | none | 无 | 源码确认；运行未验证 |
| svr_reg | regression | sklearn.svm.LinearSVR | none | 无 | 源码确认；运行未验证 |
| adaboost_reg | regression | sklearn.ensemble.AdaBoostRegressor | none | 无 | 源码确认；运行未验证 |
| knn_reg | regression | sklearn.neighbors.KNeighborsRegressor | n_jobs | 无 | 源码确认；运行未验证 |
| rf_reg | regression | sklearn.ensemble.RandomForestRegressor | n_jobs | 无 | 源码确认；运行未验证 |
| lightgbm_reg | regression | lightgbm.LGBMRegressor | n_jobs | 无 | 源码确认；运行未验证 |
| histgradientboost_reg | regression | sklearn.ensemble.HistGradientBoostingRegressor | none | 无 | 源码确认；运行未验证 |
| gblup_reg | regression | gpse.models.safe_kernel_ridge.SafeKernelRidge | none | 无 | 源码确认；运行未验证 |
| gbdt_reg | regression | sklearn.ensemble.GradientBoostingRegressor | none | 无 | 源码确认；运行未验证 |
| xgboost_reg | regression | xgboost.XGBRegressor | [n_jobs, nthread] | 无 | 源码确认；运行未验证 |
| catboost_reg | regression | catboost.CatBoostRegressor | thread_count | 无 | 源码确认；运行未验证 |
| kernelridge_reg | regression | gpse.models.safe_kernel_ridge.SafeKernelRidge | none | 无 | 源码确认；运行未验证 |
| mlp_reg | regression | sklearn.neural_network.MLPRegressor | none | 无 | 源码确认；运行未验证 |
| rf_clf | classification | sklearn.ensemble.RandomForestClassifier | n_jobs | 无 | 源码确认；运行未验证 |
| svm_clf | classification | sklearn.svm.SVC | none | 无 | 源码确认；运行未验证 |
| xgboost_clf | classification | xgboost.XGBClassifier | [n_jobs, nthread] | 无 | 源码确认；运行未验证 |
| lightgbm_clf | classification | lightgbm.LGBMClassifier | n_jobs | 无 | 源码确认；运行未验证 |
| catboost_clf | classification | catboost.CatBoostClassifier | thread_count | 无 | 源码确认；运行未验证 |
| mlp_clf | classification | sklearn.neural_network.MLPClassifier | none | 无 | 源码确认；运行未验证 |

## 已追踪调用拓扑

`train/cli.py` 解析预算 → `predictor` → `_pipeline.run_all_models` → `_scheduler.run_budgeted_models` → `_RepeatExecutor` controller → 串行 Optuna study / training fold runner → `_Dispatcher` → spawn `_execute` → `execute_fold` → optimization fit 或 `_train_single_fold`。
repeat 结束后 finalize 通过同一 dispatcher 提交 `aggregate_model_repeats`。stacking 在模型汇总之后使用自己的执行路径。未使用 budget 的 API legacy 分支仍需逐入口审计。

线程来源：CLI bootstrap 环境变量；`parallel.native_thread_environment` 子进程启动环境；scheduler initializer；worker `_execute`；optimization/fold/genomic_utils 内层限制；registry estimator 参数；stacking/ensemble 各自限制。完整 clone、refit、predict 路径和第三方内部嵌套尚未全部核查。

## 本机环境证据

默认 `python` 为 3.10.14；affinity 16 个逻辑 CPU。默认解释器未安装 numpy/scipy/scikit-learn/threadpoolctl/optuna/xgboost/lightgbm/catboost/ngboost；pytest 9.1.1。不能据此推断其他环境也缺依赖。
`/proc/self/cgroup` 为 unified user.slice 路径；根 `/sys/fs/cgroup/cpu.max`、memory.max、memory.current 不存在，当前 quota 与内存上限未确认。未启动正式训练。

## 后续实施与验收缺口

1. 完成环境定位、现有测试基线、全部计算入口和恢复产物审计。
2. 在现有共享池上增加 opt-in ThreadPlan 和三维资源 admission，保留 legacy；不要复制第二套统计流程。
3. 实现 ready queue、公平/aging、模型总并发上限、attempt fencing；资源释放必须与计算停止证据绑定。
4. 接入 final refit、预测、stacking；校验 OOF、稳定种子、原子产物与恢复。
5. 完成设计 T01–T18、真实模型 smoke、固定工作量 A/B/C 对照。所有性能结果目前为“未执行”，80 CPU 实机性能待验收。

4991.8 秒的原始命令与日志不在本次已检查证据中，计时口径和 CPU 使用无法确认。

## 2026-10-26 后续进度：资源账本与 ready queue

新增 `gpse/train/_resources.py`：CPU/内存/worker 原子非阻塞 admission、跨 repeat 模型并发计数、幂等租约、attempt/pool generation fencing；释放接口要求调用方确认计算已经停止。新增按模型加权的 ready queue，老任务达到等待阈值后停止回填，为大任务积攒资源。当前尚未接入生产训练，不能宣称整体弹性调度完成。

验证：`python -m pytest tests/test_resources.py tests/test_parallelism.py -q` → **26 passed**。覆盖并发申请、1/4/8/16/80 逻辑预算、内存/worker/模型 gate、重复释放与过期 attempt、1→3→5 fit admission、快慢模型加权服务、大任务 aging、重复入队拒绝。80 token 用例为合成正确性验证。

环境补充：已检查当前 cgroup 及所有可见祖先，cpu.max 均为 `max 100000`，memory.max 均为 `max`；affinity 16 是当前已确认 CPU 上限。Poetry 缓存环境同样缺少 numpy 等依赖，原 fold 测试收集失败（ModuleNotFoundError），并非测试通过。正在 `/tmp/gpse-threading-venv` 构建隔离依赖环境，不修改用户全局 Python。

待继续：将 ThreadPlan 接入 registry/worker 与全部计算入口，ready queue 接入共享池，验证进程终止后回收、重试/原子产物恢复，以及 T01–T18 剩余内容和真实性能对照。
