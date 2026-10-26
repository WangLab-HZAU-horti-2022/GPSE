# TODO: GPSE Web 界面（GPSE Studio）实施文档

> 创建日期：2026-10-21
> 目标：为 GPSE 提供图形化 Web 界面，把"命令行 + 长参数列表"的使用方式变成
> "上传数据 → 表单配置 → 一键运行 → 在线看结果"，降低基因组选择工具的使用门槛。
> 参考实现：作者已有的前端项目
> [Treasure-Seeking_Mouse](https://github.com/xsx123123/Treasure-Seeking_Mouse)
> （React/Vite SPA + Node 服务 + SSE + Docker），技术栈和部署经验可直接迁移。

---

## 一、可行性结论：可以做，且基础比想象中好

GPSE 现有的三个事实决定了 Web 化成本低：

1. **CLI 边界清晰**：全部能力收敛在 5 个子命令
   `convert / train / predict / batch / tools`（`gpse/cli.py`），
   Web 后端只需做"表单 → CLI 参数 → 子进程"的映射。
2. **任务编排代码已存在**：`gpse/mcp/server.py` 已实现完整的长任务管理模型——
   后台子进程、任务目录（`$GPSE_MCP_JOBS_DIR/<job_id>/`）、日志落盘、
   状态探测（`_probe_pid`）、停止（`_job_stop`）、列表（`_job_list`）。
   Web 后端可以**直接复用这套 job 模式**（抽成公共模块，MCP 与 HTTP 两个前端共用）。
3. **输出结构标准化**：每个训练任务产出
   `reports/model_leaderboard.csv`、`model_comparison_cv_topsis.csv`、
   `stacking_selected_models.json`、`reports/final_predictions.csv`、日志等，
   前端结果页只需解析这些固定文件，无需侵入训练逻辑。

主要风险与对策见第七节。

---

## 二、总体架构

```text
浏览器
  └── React/Vite 前端（GPSE Studio SPA）
        └── HTTP/WebSocket(SSE) 
              └── FastAPI 后端（Python，新增 gpse/web/ 包）
                    ├── 复用 gpse/mcp/server.py 的 job 管理（子进程模型）
                    ├── gpse convert/train/predict/batch CLI
                    └── 结果目录解析（leaderboard / topsis / 日志）
```

**选型理由**：
- 后端用 **FastAPI（Python）** 而不是 Node：GPSE 是 Python 包，job 管理代码已是
  Python，同进程内可以直接 import 校验逻辑（如表型列名校验），减少跨语言胶水。
  Treasure-Seeking_Mouse 的 Node 服务里与 LLM 相关的部分这里用不上。
- 前端沿用 **React + Vite + pnpm**（与 Treasure-Seeking_Mouse 一致，
  组件/主题/部署配置可大量借用）。
- 日志与进度用 **SSE**（与寻宝鼠一致的流式方案；训练日志是单向流，SSE 足够，
  不需要 WebSocket）。
- 部署沿用 **Docker + nginx** 双容器方案（前端静态 + 后端 API）。

---

## 三、页面设计（5 个页面）

### P1. 数据上传与校验（对应 `gpse convert` / `gpse tools`）
- 拖拽上传基因型文件（VCF / PLINK / CSV）与表型文件（CSV）
- 上传后自动做**轻量预检**（同步、秒级）：行列数、样本 ID 交集、表型列名列表、
  缺失率；预检逻辑直接调用 `gpse/convert/validators.py` 与
  `gpse/tools/analyze_phenotypes.py` 的函数（不走子进程）
- 大文件（VCF 可能 GB 级）：分块上传 + 服务端落盘到
  `$GPSE_WEB_DATA_DIR/<dataset_id>/`；预检只读头部和前 N 行
- 表型列名列表缓存下来，供 P2 的"目标性状"下拉框使用

### P2. 任务配置（对应 `gpse train`，向导式表单）
按 `gpse/train/workflow.py` 的参数分组组织为分步表单：
1. **基本**：数据集（P1 已上传）、目标性状（下拉）、任务类型（回归/分类，
   分类时需 `n_classes`）
2. **CV 设置**：重复次数（默认 50）、folds（默认 5）、随机种子（默认 42）
3. **模型与集成**：模型池勾选（默认全 11 个）、是否启用 stacking、
   top_n_models（默认 3）、TOPSIS 配置（默认双准则 0.8:0.2，高级选项）
4. **特征选择**（高维数据）：GWAS-P / LD / 无，阈值参数（高级选项折叠）
5. **资源**：n_jobs / threads 滑块（显示服务器可用核数）
- 右侧实时预览**将要执行的完整 CLI 命令**（透明性 + 方便老用户核对）
- "保存为模板"：配置 JSON 可复用

### P3. 任务监控（对应 job 管理）
- 任务列表：状态（排队/运行/完成/失败）、进度、开始时间、耗时
- 详情页：SSE 实时滚动日志（复用 `_job_log` 的 tail 语义）、
  停止按钮（复用 `_job_stop`）、CLI 命令回显
- 进度估计：从日志中解析 `repeat i/50` 模式渲染进度条
  （训练日志已有此格式；解析失败则退化为"运行中"无百分比）

### P4. 结果可视化（核心卖点页）
解析任务输出目录的固定文件渲染：
- **模型排行榜表格** ← `reports/model_leaderboard.csv`
  （CV Pearson ± SD、holdout、是否入选 stacking）
- **TOPSIS 排名图** ← `model_comparison_cv_topsis.csv`
  （横轴 TOPSIS score，双准则散点：精度 vs 稳定性）
- **GPSE vs 单模型对比条形图** ← leaderboard 中 stacking_ensemble 行 vs 各模型
- **预测-观测散点图** ← `reports/final_predictions.csv`
- 一键下载：全部结果打包 zip、单图导出 PNG/SVG
- 图表库建议 ECharts（中文文档好、大数据量散点性能好）

### P5. 在线预测（对应 `gpse predict`）
- 选择已完成任务的已保存模型（`--save_models` 产物 / deployment_ensemble）
- 上传新基因型 → 返回预测值 CSV 下载

---

## 四、后端 API 设计（FastAPI）

```text
POST   /api/datasets                 # 分块上传初始化
PUT    /api/datasets/{id}/chunks     # 分块上传
POST   /api/datasets/{id}/validate   # 预检（同步调用 validators）
GET    /api/datasets/{id}            # 数据集元信息（含表型列名）

GET    /api/config/schema            # 任务配置表单 schema（从 argparse 定义生成，避免前后端重复维护）
POST   /api/jobs                     # 提交任务（body = 表单 JSON → 构建 CLI argv）
GET    /api/jobs                     # 任务列表
GET    /api/jobs/{id}                # 任务状态
GET    /api/jobs/{id}/logs           # SSE 日志流（tail -f 语义）
POST   /api/jobs/{id}/stop           # 停止任务
GET    /api/jobs/{id}/results        # 解析后的结果 JSON（leaderboard/topsis/predictions）
GET    /api/jobs/{id}/download       # 结果打包下载

POST   /api/predict                  # 在线预测任务
```

**表单 schema 自动生成**：`GET /api/config/schema` 从
`gpse/train/workflow.py` 的 argparse 定义反射生成（参数名/类型/默认值/帮助文本/
分组），前端据此渲染表单——这样 CLI 加参数时前端表单自动跟进，单一事实源。

**CLI 参数构建**：复用并泛化 `gpse/mcp/server.py` 的
`_build_train_args` / `_build_convert_args` / `_build_predict_args`
（建议抽到 `gpse/utils/arg_builders.py`，MCP 与 Web 共用）。

---

## 五、目录结构（新增部分）

```text
GPSE/
├── gpse/
│   ├── web/                     # 新增：FastAPI 后端
│   │   ├── __init__.py
│   │   ├── app.py               # FastAPI app 与路由
│   │   ├── jobs.py              # 从 mcp/server.py 抽出的 job 管理（通用化）
│   │   ├── uploads.py           # 分块上传与数据集管理
│   │   ├── results.py           # 结果目录解析器
│   │   └── schema.py            # argparse → JSON schema 反射
│   ├── mcp/server.py            # 改为 import gpse.web.jobs（去重）
│   └── utils/arg_builders.py    # 新增：表单 JSON → CLI argv（MCP/Web 共用）
├── webui/                       # 新增：React/Vite 前端（参照寻宝鼠结构）
│   ├── src/
│   │   ├── pages/               # Upload / Configure / Jobs / Results / Predict
│   │   ├── components/          # 表单渲染器、日志流组件、图表面板
│   │   └── services/api.ts      # REST + SSE 客户端
│   ├── package.json / vite.config.ts
│   └── ...
├── deploy/
│   ├── Dockerfile.webui         # 双阶段：node build → nginx
│   └── docker-compose.webui.yml # webui + gpse-api 两容器
└── todo.md                      # 本文档
```

---

## 六、分期实施计划

### Phase 0：地基（1–2 天）
- [ ] 抽公共模块：`gpse/web/jobs.py`（自 mcp/server.py 泛化）、
      `gpse/utils/arg_builders.py`；MCP server 改为复用，回归测试 MCP 工具不受影响
- [ ] FastAPI 骨架 + `/api/config/schema` 反射 + `POST /api/jobs` 全链路打通
      （命令行能通，先不要前端）

### Phase 1：MVP（3–5 天）—— 能演示的最小闭环
- [ ] P2 简化版表单（仅：数据集路径、性状、任务类型、repeats/folds、stacking 开关）
- [ ] P3 任务列表 + SSE 日志
- [ ] P4 排行榜表格 + TOPSIS 图（只读 `reports/` 下 CSV，ECharts）
- [ ] 先用"服务器本地路径"代替上传（内部用户场景够用）

### Phase 2：数据闭环（3–4 天）
- [ ] P1 分块上传 + 预检（validators 直接 import）
- [ ] `gpse convert` 向导（VCF→CSV 转换作为后台 job）
- [ ] P4 完整化：预测-观测散点、结果打包下载

### Phase 3：打磨（按需）
- [ ] P5 在线预测
- [ ] batch 模式（多性状批量提交，对应 `gpse batch`）
- [ ] 中英 i18n、明暗主题（寻宝鼠组件可迁移）
- [ ] Docker 一键部署 + 文档
- [ ] （可选）账号/多用户：现阶段定位**单用户/课题组内部署**，先不做鉴权；
      若将来公网部署再加（寻宝鼠的 Supabase/guest 方案可参考）

---

## 七、风险与对策

| 风险 | 说明 | 对策 |
|---|---|---|
| 长任务（数小时~数天） | 50×5 CV × 11 模型的 manuscript 协议单个性状可跑数天 | job 模型天然支持；HTTP 请求不持有任务，只持句柄；服务器重启后按 jobs 目录恢复状态 |
| 大文件上传 | VCF 可达 GB 级 | 分块上传 + 流式落盘；nginx `client_max_body_size` 调高；磁盘配额检查 |
| 路径安全 | 表单接收文件路径，防目录穿越 | 一切路径限制在 `$GPSE_WEB_DATA_DIR` / `$GPSE_WEB_JOBS_DIR` 内；复用 MCP 的参数白名单校验（`_validate_extra_args`），禁止自由 extra_args |
| 资源争抢 | 多任务并发打满 CPU | 任务队列串行/限制并发数；n_jobs 上限校验；显示服务器负载 |
| 进程树清理 | 停止任务需杀整棵进程树 | `_job_stop` 已有 grace 逻辑，泛化时保留；子进程用独立进程组启动 |
| 结果文件版本漂移 | 前端解析依赖 reports/ 文件格式 | 结果解析器 `results.py` 对各 CSV 做 schema 校验 + 缺列降级；给解析器配单元测试（用 `tests/manuscript_runs/` 下的真实结果目录做 fixture） |

---

## 八、与寻宝鼠仓库的复用清单

可直接迁移/参照的部分：
- Vite + React + pnpm 工程骨架、明暗主题、响应式布局
- SSE 客户端封装（`src/services/` 的流式解析模式）
- Docker 双容器 + nginx gzip + 健康检查（`deploy/`）
- Makefile 开发命令组织方式
- 不需要迁移：LLM 工具调用循环、Supabase 账号体系、桌宠（除非想要 🙂）

## 九、验收标准（MVP）

1. 内部用户零命令行完成一次完整训练：上传（或选路径）→ 配置 → 运行 → 看到排行榜
2. 任务中断/刷新页面后日志与状态不丢
3. 同一配置在 Web 与 CLI 下产出的结果目录逐文件一致（用 cucumber 小数据集对比）
4. `pytest tests/` 原有测试全绿（确保 MCP 重构无回归）

---

## 十、GPSE 性能排查与优化

### 优先级 1：核对并验证 CPU 并行参数
- [ ] 检查启动命令中 `--threads` 与旧式 `--n_jobs`、`--max_workers`、
      `--repeat_workers`、`--fold_workers` 的组合，确认任务进入共享 CPU 调度器，
      而不是意外走串行/旧式调度路径。
- [ ] 用小规模、可重复的训练任务对照不同 worker/thread 配置，记录实际 worker 数、
      每 worker 线程数、总耗时及 CPU 利用率；不要仅凭参数值推断并行已生效。

### 优先级 2：控制 CV 与超参数搜索成本
- [ ] 对高耗时模型先做小规模试跑，评估降低 `--trials`、`--n_repeats` 或
      `--train_folds` 的影响；必要时只运行当前分析需要的模型。
- [ ] 从训练事件日志（`logs/events.jsonl`）汇总 data、optimization/trial、fold、
      repeat、stacking 等阶段的耗时，比较不同模型和参数配置，定位主要计算时间。

### 优先级 3：降低大矩阵 IO 与内存压力
- [ ] 大型基因型矩阵优先用 Parquet 或 Feather，避免不必要的 CSV 读写；记录文件大小、
      加载时间和转换时间，确认实际收益。
- [ ] 在高维、多 worker 任务中记录主进程及 worker 的峰值 RSS，并结合矩阵维度与
      worker 数评估内存放大；据此限制并发 worker，或进一步评估减少矩阵复制的方案。
- [ ] 同时检查缓存命中与未命中时的耗时和 RSS：缓存可减少重复读取/对齐，但需确认
      DataFrame 副本没有抵消收益或造成峰值内存问题。
