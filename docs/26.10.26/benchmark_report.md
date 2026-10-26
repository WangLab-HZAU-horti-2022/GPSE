# 线程策略缩比性能记录

日期：2026-10-26。结论：**本次小工作量未证实弹性调度缩短完整墙钟时间。测量后按用户要求改为默认 elastic_v1，legacy 可显式回退；80 CPU 实机性能未执行。**

本报告原始摘要对应默认切换前的源码，不作为最终默认自动计划版本的性能验收；原始 hash 保留供复核。基准脚本的 A 路径现在显式传入 --scheduler legacy。

## 环境与控制变量

Python 3.10.18，NumPy 2.2.6，SciPy 1.15.3，pandas 2.3.3，scikit-learn 1.7.2，Optuna 4.9.0，threadpoolctl 3.6.0，joblib 1.6.0；XGBoost 3.2.0、LightGBM 4.7.0、CatBoost 1.2.10 已用于模型 smoke，本性能工作量仅 RF/Lasso。机器 affinity 为 16 个逻辑 CPU，本实验逻辑预算 4、每 fit 2、2 worker；并非独占 4 核硬绑定。

固定生成的 160 样本 × 24 特征，rf_reg/lasso_reg，2 repeats × 3 folds，无 stacking；固定默认参数重放与真实 HPO（2 trials）分开执行。每模式同一数据、seed、划分、模型参数配置、相同产物边界，独立结果目录。A legacy；B 新 ThreadPlan/admission + 原顺序 fold 窗口；C 新 ThreadPlan/admission + 完成驱动窗口和加权策略。B/C 差异包含公平策略，不能把两者差值仅归因于窗口补充。

每项 3 次，顺序 ABC / CBA / ABC，未强制清空缓存。正式记录运行时没有并发本任务 pytest；宿主机其他负载不可控。最早与 pytest 重叠的 `/tmp/gpse-threading-verified-20261026` 保留但不用于下表。更早 `/tmp/gpse-threading-abc-20261026` 缺少与当前源码一致的 hash 证据，也不用于本结论。

## 完整进程墙钟（秒）

| 工作量 | 模式 | 中位数 | 最小–最大 | CPU seconds 中位数 | 进程树 RSS 和峰值中位 MiB |
|---|---|---:|---:|---:|---:|
| 固定默认参数 | A | 2.884 | 2.884–3.012 | 4.760 | 596.32 |
| 固定默认参数 | B | 3.008 | 3.005–3.149 | 4.525 | 601.03 |
| 固定默认参数 | C | 3.238 | 3.012–3.390 | 4.755 | 604.13 |
| 真实 HPO | A | 7.190 | 7.184–7.194 | 11.208 | 602.82 |
| 真实 HPO | B | 7.317 | 7.297–7.325 | 10.994 | 602.26 |
| 真实 HPO | C | 7.306 | 7.291–7.307 | 11.023 | 606.34 |

C 相对 A 的中位加速比：固定重放约 0.891×，真实 HPO 约 0.984×，均未显示加速。当前 B/C 已包含 checkpoint 写入与 hash 校验开销；B 不能称为仅改变线程配置的纯对照。整个任务只有数秒，启动、导入、调度、持久化和汇总可能占明显比例，但本测量没有分离各因素的因果贡献。固定重放归一化 CPU 利用率中位约 A 40.75%、B 37.18%、C 36.71%，真实 HPO 约 39.00%、37.56%、37.72%；这些是 CPU seconds/(预算×wall)，不是配置线程数。

CPU seconds 使用 wait/rusage（含已回收子进程）；RSS 是采样进程树求和，会重复计共享页，不能当 PSS/cgroup peak。swap、throttling、page fault 的完整 benchmark 汇总和详细阶段分解未执行。所有方案各 3 次退出码为 0；每类 6 组输出对照通过，容差 1e-9；固定重放最大绝对差 2.66e-15，HPO 为 2.22e-16。HPO 参数/状态/指标历史也由 compare_outputs 对照。每次运行期间源码 hash 保持一致。

## 重现与证据

```bash
python -m scripts.benchmark_threading --output /tmp/gpse-threading-fixed-new \
  --cpu 4 --per-fit 2 --repetitions 3
python -m scripts.benchmark_threading --output /tmp/gpse-threading-hpo-new \
  --cpu 4 --per-fit 2 --repetitions 3 --hpo --trials 2
```

当前版本原始 console、进程采样、events、结果位于 `/tmp/gpse-threading-final-fixed-20261026` 和 `/tmp/gpse-threading-final-hpo-20261026`。摘要 `benchmark_fixed_summary.json`、`benchmark_hpo_summary.json` 含命令、环境、输入/源码 hash、逐次测量和比较结果；已核验与最终源码一致且测量期间未改源码。此前未接入 checkpoint 的摘要保留为 `benchmark_fixed_pre_checkpoint_summary.json`、`benchmark_hpo_pre_checkpoint_summary.json`，仅作历史证据。临时目录不是长期存储保证。

## 固定工作量的线程形状

| 每 fit CPU | worker 上限 | A 中位秒 | B 中位秒 | C 中位秒 | C RSS 中位 MiB |
|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 2.636 | 2.751 | 2.649 | 985.30 |
| 2 | 2 | 2.884 | 3.008 | 3.238 | 604.13 |
| 4 | 1 | 3.357 | 3.459 | 3.460 | 408.77 |

形状对照使用相同生成规则、种子、默认参数和输出边界，各形状内 6 组 A/B/C 数值比较通过。1 与 4 CPU/fit 的原始目录为 `/tmp/gpse-threading-shape1-20261026`、`/tmp/gpse-threading-shape4-20261026`；摘要保存为 `benchmark_shape1_summary.json`、`benchmark_shape4_summary.json`。所有四类配置共 36 个 run 退出码为 0、24 组比较通过。形状之间未交替执行，存在不同时段负载干扰；多 worker 对本任务更快但常驻内存增加，不能直接推广为生产 profile。重现时修改 `--per-fit` 并使用新的输出目录。

当前无推荐生产 profile。需继续执行更大且受控的固定 trial 参数列表重放、不同 per-fit/worker 形状及真实生产 HPO，再决定推广。不得把合成 80 token 或本次 4 CPU 测量写成 80 CPU 性能验收。
