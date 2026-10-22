# CLI 日志调查与实施方案

## P0 调查（2026-10-22，修改前代码）

- 调用链：`train/cli.py:main → predictor.py:GenomicPredictorV2 → _pipeline.py:run_all_models → _run_model_task / run_model_multiple_repeats → _run_repeat_task / train_and_evaluate_model_for_repeat → _optimization.py / _fold_training.py`。Stacking 在模型选择完成后由 `_pipeline.py` 调用 `stacking.py:fit`。
- Loguru 全局 Logger 被 CLI 的懒初始化和 Predictor 初始化各重配一次。`workflow.py:_configure_logging` 使用标准库的 `setLevel`，对 Loguru 实际无效；Predictor 硬编码 INFO，再次覆盖用户日志级别。`logger_init` 已 remove 旧 sink，因此没有发现同一进程持续累加 Handler 的证据。
- `setup_subprocess_logging` 给每个 Repeat 子进程增加终端 sink；模型进程继承父进程 sink。`collect_subprocess_logs` 将所有 worker 日志重新作为 INFO 打印，嵌套多模型时还会扫描其他模型的日志，这是明确的重复输出源。
- Optuna 默认 Handler 与根日志拦截器独立；`study.optimize(show_progress_bar=True)` 另起 tqdm 进度条。CLI 配置 Panel、RichHandler、worker 输出并无共享 Console/Live 管理。
- `parallel.py:derive_parallelism_from_threads` 使用 `value == 1` 判断是否可自动推导，无法区分显式 `--repeat_workers 1` 与默认值，因而会将其改成 2 等值。train 的参数来自 argparse；项目 YAML 当前没有合入训练参数（`load_default_config` 默认 `auto_project_config=False`）。模型 YAML 负责模型配置，不能覆盖训练 worker 参数。修复优先级为：显式 CLI > threads 推导 > argparse 默认；不新增 YAML 训练配置入口。
- `_repeat_training.py` 仅当 `repeat_workers > 1 and optuna_per_repeat` 才启用 Repeat 进程池；默认冻结参数时顺序执行。因此 `n_jobs * max_workers * repeat_workers` 是配置容量上限，不是真实执行并发。模型内部并行效率、BLAS 和当前阶段还会进一步影响实际 CPU 使用，不能用容量冒充测量值。
- 调度器每个列表元素提交一次，无额外递归训练证据；但 `--models a a` 会提交重复模型，需在入口稳定去重并警告。Fold-ensemble 重训与训练分区 refit 属于现有算法，不是重复触发错误，不修改。
- Holdout 在重复 CV 前固定划分；Fold 的 Train / Validation 属于训练分区内部。现有 `Best CV Pearson` 实际为 `mean(CV)-λ*std(CV)`，应改为 Robust Objective；现有 Test 指标实为 Holdout。仅修正文案，不更改统计数据和文件格式。

## 分阶段实施

1. P1：增加单一进程内 Console Renderer；父进程消费跨进程结构化事件，Rich Live 只在交互 TTY 启用（CI、TERM=dumb、重定向降级）。Worker 保留详细文件并通过队列发送日志，取消终端 sink 与重复回放。渲染状态仅为事件的显示投影，不调度训练。
2. P2：增加 NORMAL / VERBOSE / DEBUG；普通模式只显示初始化、模型/大阶段、警告错误、结束指标。所有原始 DEBUG 仍写文件。Optuna 原生日志统一拦截，禁止其 tqdm。
3. P3：Run / Model / Repeat / Fold / Trial 统一关联，JSONL 记录开始/完成/失败/中断、耗时、指标、参数与日志。完整 Trial 历史独立写入 logs，不修改现有结果结构；在 Manifest artifact_index 登记日志关联。
4. P4：参数显式值回归、真实单模型/并行模型/并行 Repeat/Optuna/Stacking/失败/中断/重定向验证，并比较未监控与监控训练输出。测试日志写入或渲染失效不改变训练异常及返回结果。

所有修改限于可观测性、参数解析和文字，不改变训练算法、随机种子、数据划分、模型选择与 Holdout 计算。

## 实现与使用

默认 `gpse train ...` 使用 NORMAL。交互终端（包括分配 PTY 的 SSH）显示一个 Rich Live 表格，包含总体阶段、完成模型数、耗时、每个模型的 Repeat/Fold 计数及正在执行的阶段。刷新上限约 4 Hz；警告、错误、完成指标由同一个 Console 输出并保留。

```bash
# 日常使用：简洁进度与关键异常
gpse train ... --verbosity NORMAL
# 查看每个 Repeat、Fold、Optuna Trial / 搜索 Fold、Stacking OOF Fold
gpse train ... --verbosity VERBOSE
# 参数、原始 DEBUG 日志和阶段详情
gpse train ... --verbosity DEBUG
# 关闭动态刷新，或重定向为纯文本
gpse train ... --progress off
gpse train ... > train.txt 2>&1
```

`-l DEBUG` 兼容旧命令。`--log-level WARNING/ERROR/CRITICAL` 继续限制普通日志的严重级别。非 TTY、`CI` 非空或 `TERM=dumb` 自动关闭动态显示及颜色；无需手动关闭进度。

并行参数的显式值（包括 `1`、`--repeat_workers=1` 和 argparse 缩写）优先于 `--threads` 推导。展示的 configured capacity、training task ceiling、thread capacity ceiling 都是理论上限，不能解释为实测 CPU 使用率。冻结参数模式保持原有顺序 Repeat 逻辑；不为“用满线程”改变算法。

### 日志产物

所有路径相对于当前 Run 目录：

- `logs/gpse_<timestamp>.log`：父进程 DEBUG 文本，保留完整配置和上下文。
- `logs/workers/<model>.log`、`<model>_repeat_NNN.log`：各计算进程的 DEBUG 文本，不再回放到终端。
- `logs/events.jsonl`：父进程统一汇总的结构化事件及原始日志。
- `logs/events/<run_execution_id>/producer_<pid>.jsonl`：worker 的独立事件生产日志，也是中断后的恢复审计来源。
- `logs/optuna/<model>_repeat_NNN.jsonl`：完整 Trial 历史，包括 COMPLETE/PRUNED/FAIL、参数、CV Mean、CV Std、Robust Objective、耗时。`trial` 从 1 开始，`optuna_trial_number` 保留 Optuna 原生从 0 开始的编号。

事件统一含 `run_id`（同 Manifest）、`run_execution_id`、`event_id`、UTC 时间戳、PID、Model/Repeat/Fold/Trial 上下文。阶段事件另含 `execution_id`、status、耗时、指标或异常；父进程与 worker 分别写自己的流，消费者只读取完整 JSON 行，终端不依赖 worker 的存活。

最终在已有 Manifest 的 `artifact_index` 中登记日志；不改变统计结果、预测文件或已有 JSON/CSV 数据含义。CV Std 的具体统计口径随 `cv_std_metric` 标注；优化目标 `mean(CV)-λ·std(CV)` 与 Holdout 指标分别命名。

### 中断调查补充

真实 SIGINT 暴露 `parallel.py:graceful_process_pool` 先 shutdown、后读取 `_processes` 的既有清理错误。修复为提前保存句柄，异常时取消并终止，正常结束等待 worker 完成并刷新事件。模型 worker 收到终止时会展开内部 Repeat 池的清理。此修改只涉及退出，不改变成功执行的调度与计算。

初版验证也证实：管理器 socket 在受限环境被禁止；原生 multiprocessing.Queue 在强杀 writer 时可能损坏共享管道/锁，吞掉最后中断事件。最终采用父进程本地队列 + worker 独立追加 JSONL 的事件通道，无管理器服务、无共享写锁，跨 fork/spawn 工作。渲染、队列、日志文件失效均不替换训练返回值或原始异常。Python stdout/stderr 也进入统一日志路由；原生库仍沿用现有模型静默配置。

## P4 验收结果（2026-10-22）

执行环境：本机 `/home/zj/.pyenv/versions/gpse/bin/python`。没有增加依赖。

`python -m pytest -q tests/`：**220 passed，3 failed（33.63 秒）**。新增日志验收的 **21 个用例全部通过**；失败项均属于既有基线问题，已用 `git archive HEAD` 提取原始代码在 `/tmp/gpse-log-baseline` 复现：

1. `test_threadpoolctl_declared_in_pyproject`：原始 pyproject 缺少显式依赖声明。
2. `test_find_result_file_depth_limit`：既有预测结果搜索深度行为与测试预期不同。
3. `test_multi_trait_genotype_files_are_hardlinked`：既有多线程转换缓存竞态；原始代码单独重复执行同样失败，亦有偶发通过。不属于本次日志修改范围。

| 验收项 | 当前证据 |
| --- | --- |
| 无进度条覆盖 | 真实 PTY 双模型 + Stacking 训练只有一张 Live 表，退出恢复光标；`test_live_and_warning_share_console` 验证警告保留且仅一次，CI/dumb 降级、显式 off 生效 |
| 事件不重复 | 单模型三级 verbosity、并行 Optuna/Repeat 的阶段数量与 execution/event ID 检查；重复模型 CLI 输入仅执行一次 |
| CLI 显式值一致 | `--threads 4 --repeat_workers 1 --n_jobs 1` 完整 CLI 训练的 Run config 与 cli_config 均保留 1；另验证等号语法与项目 YAML 不覆盖训练参数 |
| 各阶段准确 | 检查 Model / Repeat / Fold / Trial、优化 Fold、Stacking OOF Fold 的实际数量、上下文、状态和耗时；PRUNED 独立记录，不冒充 COMPLETE |
| 科研指标不变 | 默认参数和 Optuna 两种模式比较开启/绕过监控的真实 CV、Holdout、Stacking；另与原始 HEAD 执行同样 2 模型 × 2 Repeat × 2 Fold × 2 Trial：CV 除 provenance 两列外逐值相同，Holdout JSON 和最终预测 CSV 逐字节相同，4 份 Optuna 结果 JSON 完全相同。来源/配置 hash 随代码及显示配置改变是预期行为 |
| 日志可检索 | DEBUG 文本、worker 事件、汇总 JSONL、4 份完整 Trial 历史；检查 Manifest 日志登记和最终文件大小 |
| 重定向无控制字符 | NORMAL/VERBOSE/DEBUG 的真实 subprocess 输出均无 ANSI 与回车进度残留；额外验证 FORCE_COLOR、CI、TERM=dumb |
| 日志故障不改变计算 | 损坏队列、不可写日志目标、事件文件为目录、渲染器异常；真实双模型并行 + Stacking 在日志故障下仍保存 Holdout 和模型 |
| 异常及中断 | 无效模型失败事件可见；真实 SIGINT 的单模型 Repeat 池和多模型嵌套 Repeat 池均返回 130，run 最后事件为 interrupted，Manifest 为 failed，已启动的 Repeat worker 不再计算 |

`python -m compileall -q gpse tests/test_training_monitor.py` 与 `git diff --check` 通过。未更改用户已有的 `reaggregate_cv.py` / `test_cv_statistics.py` 修改及 PDF 删除。
