# T01–T18 验收证据矩阵

> 默认策略已按用户要求改为 elastic_v1，legacy 使用 --scheduler legacy 显式选择。本矩阵保留此前验收范围；默认变更没有进行完整全仓重跑，跨服务器验收待用户执行。

2026-10-26，未提交工作区。此表保持原设计范围；部分通过不计作完整验收。

| 项目 | 当前证据 | 结论/缺口 |
|---|---|---|
| T01 预算与非法请求 | test_resources 的 1/4/8/16/80 原子申请、test_execution_config 的不可调度配置拒绝 | 合成预算正确性通过；不是 80 CPU 实机 |
| T02 八个直接线程模型 | test_thread_plan 预算映射、clone、真实 fit/predict；全模型默认参数 run | 小数据通过 |
| T03 限制冲突 | estimator_thread_count、training_thread_limits 与 registry 参数检查 | 新路径通过；legacy resolver 保留 |
| T04 worker 16→4 | 独立进程复用及异常恢复 threadpool 快照测试 | 通过 |
| T05 异构隔离 | 两个 spawn 进程的不同计划同时运行；拒绝 controller 线程修改 native limits | 通过 |
| T06 空闲资源扩容 | ControlledPool 真实 dispatcher 1→3→5；80 token 账本测试 | 合成事件验证通过；生产吞吐另验 |
| T07 公平与大任务 | 加权服务、aging 停止回填、模型 cap 不阻塞其他模型 | 有限可满足负载通过 |
| T08 父层不占计算额度 | 共享 repeat controller/compute pool 测试、完整 CLI 和取消测试 | 所测路径通过 |
| T09 异常/提交/取消/crash | dispatcher 测试、真实 worker os._exit、停止后回收、broken-submit、孤儿 worker 锁 | 释放与拒绝重叠通过；整池透明重建与自动重排未实现 |
| T10 内存/OOM | 内存 reservation、常驻数据 floor、worker/model gate | admission 通过；真实 OOM 注入、诊断与受限重试未验收 |
| T11 完成顺序与 OOF | 完成驱动补充、有序返回、legacy/elastic 预测比较、全模型 OOF 每行一次 | 小数据通过 |
| T12 pruning/报告步序 | 当前源码没有逐 fold trial.report/should_prune；Optuna 串行聚合，回归/分类 HPO 历史对照 | 当前成功轨迹通过；原失败/剪枝轨迹持久化恢复未实现 |
| T13 refit/predict/stacking | 同一账本资源日志，finalize、stacking fold/refit/meta；模型移走来源后预测 | 所测路径通过；所有轻量阶段细分计时尚不完整 |
| T14 resume/迟到 attempt | TaskStore manifest/hash、故障 fold 补跑、输入/seed/损坏拒绝、旧 attempt 不覆盖新指针、孤儿 worker 阻止恢复 | fit checkpoint 路径通过；失败 HPO 轨迹恢复、断电持久性、非 POSIX 未验收 |
| T15 参数兼容 | CLI 冲突测试、legacy 默认、新路径配置、已有结果目录不改写 | 所测组合通过 |
| T16 quota/分配 | affinity/cgroup v1/v2/作业分配 resolver、小预算测量 | 缩比通过；80 CPU 实机未执行 |
| T17 仅一任务 | 固定 lease 不在线扩展，dispatcher only_running_task/no_ready_work 事件 | 行为与设计一致，未补跑无用任务 |
| T18 全模型与汇总 | 15 回归+6 分类默认参数完整 run，126 fold 模型、OOF、TOPSIS、两个 GPSE、报告、租约全释放；all_models_smoke_checks.json | 默认参数小数据通过；21 模型完整 HPO/恢复组合未全覆盖 |

最后针对性验证：90 passed；全仓上一轮：373 passed、2 个未修改模块的失败，其中转换硬链接用例单独重跑通过。相关原始命令和路径见实施记录。

P4 当前源码固定工作量与真实 HPO 各 3 次 ABC，以及每 fit 1/2/4 CPU 形状比较已执行，数值对照通过；未证明加速，继续 opt-in。80 CPU、生产数据、完整阶段监控与真实 OOM 尚不具备验收证据。P5 非首版必要项。
