# 线程优化实施与验证记录

> 最终默认变更：按用户要求，CLI 和 Python 训练入口默认启用 elastic_v1，无 YAML 时在加载数据后自动生成内存受限计划并保存 resource_plan.json；显式 --scheduler legacy 回退。自定义 execution_config 继续生效，resume_from 未指定配置时读取来源 resource_plan.json。下面的 opt-in/legacy 默认描述属于此前实施阶段。用户要求停止继续测试并提交，未进行默认变更后的完整全仓重跑；此前已启动的默认训练/恢复与取消通知针对性测试为 11 passed，信号/stacking 中断测试为 5 passed。跨服务器验证由用户进行，整体设计验收仍未完成。

日期：2026-10-26。基准 commit：`b12ccf5f65afbb4b51cb9bdfeda6975e03b7f9ad`。实现位于未提交工作区，包含接续前已有修改。P0–P4 整体验收尚未完成。

## 本次确认与修复

现有实现已将 ThreadPlan、三维资源账本和 ready queue 接入 opt-in 共享 spawn 计算池；模型/repeat controller 不持有计算 lease。`_elastic_dispatcher.py:map_folds` 按完成事件补充任务、按 fold ID 返回结果，Optuna study 仍串行。stacking fold/refit 和 coordinator meta fit 使用同一 run 账本。

修复 `_elastic_dispatcher.py:_coordinate` 的 broken-submit 挂起：若 `pool.submit()` 抛出 BrokenProcessPool，停止 admission 后立即再次处理尚未提交的 promises，不等待一个可能永不发生的 worker callback。新增 `test_broken_submission_resolves_all_queued_parents_without_worker_callback`，验证所有排队父任务收到异常、未成功提交的 lease 释放。已运行任务仍须在 pool owner 确认进程停止后回收。

README 补充真实的预算语义、新路径配置入口及 legacy 回退，移除旧的固定模型数/并发推导描述。

## 本次测试

使用此前已建立的 `/tmp/gpse-threading-venv/bin/python`；系统 Python 和 Poetry 缓存环境缺少 numpy，不能直接用它们运行完整测试。

```bash
/tmp/gpse-threading-venv/bin/python -m pytest -q \
  tests/test_resources.py tests/test_thread_plan.py tests/test_elastic_dispatcher.py \
  tests/test_execution_config.py tests/test_fold_scheduler.py tests/test_cpu_scheduler.py
/tmp/gpse-threading-venv/bin/python -m pytest -q
```

接续时针对性测试为 **120 passed**；加入 broken-submit 回归后，全仓结果为 **363 passed, 1 failed, 3 warnings**。分派器单独复核 **8 passed**。警告来自小数据 smoke 中 SVR/MLP 的低迭代上限，不代表性能调优或模型收敛验收。

此前唯一失败为 `tests/test_audit_fixes.py::test_find_result_file_depth_limit`：预测模块 `_find_result_file` 找到了测试认为过深的 label_encoder。该模块和测试均未在本次线程改动中修改；另行复现仍失败，未扩展本任务修改它。

恢复接入后的全仓实测为 **373 passed / 2 failed / 7 warnings**：另一个失败是未修改的 `tests/test_convert_perf.py::test_multi_trait_genotype_files_are_hardlinked`，单独重跑 **1 passed**，表现为并发转换时硬链接未复用的非稳定失败。最终提交身份和 CatBoost 日志隔离补强后，执行 `tests/test_task_store.py tests/test_thread_plan.py tests/test_execution_config.py tests/test_elastic_dispatcher.py` 得到 **90 passed / 7 warnings**。该最后一轮为针对性回归，不能当作最后源码的全仓重跑。格式检查与 `git diff --check` 通过。

测试实际覆盖 21 个注册 estimator 的小数据 fit/predict；8 个直接线程模型预算映射；worker 16→4 及异常后恢复；不同进程异构计划；10/80 合成 token；内存/worker/模型 gate；1→3→5 活跃 fit；大小任务 aging；真实 worker exit；打乱返回顺序；legacy/elastic 的默认参数 stacking、回归 HPO、分类 HPO 和 thread_plan 完整 CLI 运行。端到端测试比较 split、指标、预测、TOPSIS、holdout，并检查报告文件、lease 成对释放；不能外推为所有模型的完整生产训练验收。

## 可执行配置

以下为 4 CPU 小数据示例，内存 reservation 需要按实际数据和模型调整；不是生产推荐 profile。

```yaml
scheduler: elastic_v1
cpu_budget: 4
worker_slots: 3
memory_budget_bytes: 4294967296
fixed_memory_bytes: 268435456
worker_memory_bytes: 268435456
aging_seconds: 5
resource_profiles:
  kernelridge_reg:
    preferred_cpu_per_fit: 2
    memory_bytes: 134217728
    max_active_fits: 3
    weight: 1
  elasticnet_reg:
    preferred_cpu_per_fit: 1
    memory_bytes: 134217728
    max_active_fits: 3
    weight: 2
```

```bash
python -m gpse.train.cli --geno_file X.csv --pheno_file y.csv \
  --target_trait trait --task_type regression \
  --models kernelridge_reg elasticnet_reg --n_repeats 2 --train_folds 3 \
  --cv_folds 3 --use_default_params --save_models \
  --execution_config execution.yaml --results_dir results_elastic_new
```

CPU budget 必须符合 affinity/quota/job allocation；配置中的所有已选模型必须有 profile。`--threads` 若提供必须与配置一致；显式 `--n_jobs` 若与任一 profile 冲突会报错，异构配置建议省略。`--max_workers`/`--repeat_workers` 仍约束 controller，`--fold_workers` 约束每 repeat/trial 的 fold 窗口；模型 `max_active_fits` 跨 repeats/trials 计数。worker 常驻内存预留为 worker_slots × worker_memory_bytes，另加 fixed_memory_bytes 和活跃任务 memory_bytes。

`scheduler: thread_plan` 保留旧的顺序 fold 窗口补充方式，用于 B 对照；它仍使用新资源 admission，并不等价于原实现。省略 `--execution_config`，在全新结果目录开始运行，即回退 legacy。不要对正在运行的 run 热切换，也不要复用已有训练结果目录。offline reaggregation 只重新汇总，不能替代训练恢复。

## 后续接续：任务提交与恢复

已新增 `_task_store.py`，接入 training/optimization folds、finalize（代表模型与部署 ensemble）、stacking folds/refit/meta。worker 按 task/lease 写 `.task_store/pending/<task>/<lease>`，协调器检查当前 lease、attempt、generation、任务身份、manifest 和全部产物 SHA-256 后移动至不可变 commit 目录，再原子替换 `current.json` 成功指针。公开模型文件逐文件原子替换，task pointer 是整个任务完成的权威标志；公开多文件集合并不保证同一时刻全部可见。未发布目录不用于恢复。

`--resume_from old_run --results_dir new_run` 只读复用已停止来源 run 的成功任务，新目录重新生成划分、研究编排、repeat 汇总和单 run 报告。输入内容、源码、有效配置、依赖与 Python 版本形成 run 指纹；任务参数、划分和最终汇总的科学内容形成任务签名。任何指纹或已提交产物 hash 不匹配即报错，不能静默换参数重跑。来源目录内容保持不变，模型路径重写到目标目录。旧版无 `.task_store/run.json` 的 run 不支持此恢复。

```bash
python -m gpse.train.cli --geno_file X.csv --pheno_file y.csv \
  --target_trait trait --task_type regression \
  --models kernelridge_reg elasticnet_reg --n_repeats 2 --train_folds 3 \
  --cv_folds 3 --use_default_params --save_models \
  --execution_config execution.yaml --resume_from results_elastic_old \
  --results_dir results_elastic_recovered
```

来源 run 的 owner/live 文件锁用于拒绝重叠恢复；spawn worker 持有 live 共享锁直至退出，因此协调器 SIGKILL 后尚存的 worker 也会阻止恢复。此实现依赖 POSIX flock 语义（当前在 Linux 验证），不是跨机器作业管理器；不支持在不可靠网络文件锁上承诺同样保证。API 和 CLI 均拒绝重叠目录，CLI 在写 manifest 前拒绝已有结果目录。

恢复测试包含部分 fold 失败后补跑、完整回归与分类 HPO 重放、thread_plan 与 elastic_v1、stacking 所有 fit 复用、旧来源移走后的模型加载预测、来源文件 hash 不变、旧 attempt 迟到拒绝、manifest/产物损坏拒绝，以及真实协调器被终止后孤儿 worker 锁保持。HPO 保持原 trial 顺序，成功 fold 从 checkpoint 读取；失败/未提交任务重新执行。

CatBoost 额外审计发现 run 级 `catboost_info` 原本由并发 fold 共享。新路径现在于 adapter fit 边界把辅助日志写入 lease 暂存目录，每次 fit 使用独立 `native_logs/fit_*`；日志纳入任务 hash 清单。真实回归/分类 fit 测试确认共享目录无写入。stacking 签名会规范化运行时 train_dir，避免仅因输出位置不同错失缓存。legacy 保持原行为。

全模型小数据运行位于 `/tmp/gpse-all-models-threading-20261026`：180 样本、8 特征、原注册表默认参数、2 repeats × 3 folds，4 CPU；15 个回归和 6 个分类模型均成功，并分别生成 GPSE stacking、TOPSIS 与报告。回归 90 个、分类 36 个 fold 模型；每 repeat 的 144 个训练样本 OOF 恰好覆盖一次。157/76 个 lease 成对释放。校验摘要见 `all_models_smoke_checks.json`。这不是生产性能结果。

## 尚未完成的要求

- pool crash 采用终止与失败传播，支持停止后跨 run checkpoint 恢复，没有透明自动重建/受限重试；OOM 注入和受限重试仍待补齐。
- HPO 恢复目前重建 study 并复用成功 folds，没有持久化恢复原失败/剪枝 trial 轨迹。若临时错误曾改变后续建议路径，不一致签名会报错，不能声称该类 study 的无损恢复已完成。
- 来源恢复以任务 manifest 为准；没有对整个 run 断电后的 fsync 持久性作保证，也未验证非 POSIX 平台或网络文件系统。汇总/报告按新 run 重新生成，未将每个轻量汇总步骤做成独立 checkpoint。
- T01–T18 的逐项证据见 `threading_acceptance_matrix.md`；全模型运行已验证默认参数路径，21 模型真实 HPO 全覆盖未执行。
- 任务日志已有 lease、attempt、generation、PID、thread plan、CPU time 和 peak RSS；尚未完整拆分所有数据准备/fit/score 时间，也未覆盖所有平台的 cgroup memory v1 观测。
- 性能报告见 `benchmark_report.md`。80 CPU、生产数据、不同 per-fit 线程形状和完整生产 HPO 仍待验收；P5 是可选后续。

继续实施顺序：T01–T18 逐项证据审计 → 当前持久化版本的性能回归与线程形状对照 → OOM/受限重试等剩余故障验收。生产 80 CPU 验收最终还需要实际运行命令、数据路径、可用资源和独立输出位置；原 4991.8 秒日志仍缺失。
