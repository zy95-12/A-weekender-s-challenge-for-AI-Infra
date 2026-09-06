# 真实可运行的 Split-Infer 研究界面

所有操作调用实际后端；已归档实验明确标为历史实测，不模拟进度、输出文本或性能数值。

## 启动

在仓库根目录运行：

```bash
python3 demo/server.py --host 127.0.0.1 --port 8088
```

访问 http://127.0.0.1:8088。远程机器可通过 SSH 转发 8088 端口。
页面“启动 Demo”实际执行固定命令 `./poc up --wan`，包含首次环境/模型准备。
完成健康检查后显示“服务已启动”，对话转发至真实 `/v1/chat/completions` SSE。
该命令启动 baseline；如果之前本目录运行 optimized，会按既有 manage.py 行为切回 baseline。
外部 checkout 的 GPU 服务应先从其自身目录停止，本项目共用四卡与 network namespace。
启动环境要求见 [serving 文档](../docs/integrated-serving.md)：Linux/root、四张 A10、模型与 vLLM 依赖。
UI/仿真需要 Python 3.11+；GPU serving 的既有环境为 Python 3.12。

也可以先 `./poc setup` 准备环境，然后运行攻击。攻击独立使用 CPU，模型版本与 serving 相同。
UI 后端串行执行启动、攻击、仿真和对话，避免内部任务竞争；外部 CLI 压测应独占服务。
UI 进程退出不会停止 serving；停止服务使用 `./poc down`。

## 模块与真实数据来源

- **方案分析**：`Q1/demo_attack.py` 调用 PR #15 原版 `retrieval.recover`，把输入编码为 FP16 embedding，然后真实执行全词表 FP32 余弦最近邻恢复。CPU 实现，不是密码学加解密，也不声称反演了 4 层后的 serving 激活。每次运行可下载 NPY 和逐位置 JSON，真实耗时包含在结果中。原始 IDs 只用于检索之后评分。
- **功能实现**：固定 `./poc up --wan`；轮询实际健康状态。回复来自模型，TTFT/TPOT 来自浏览器 token 事件计时，包含网络、代理和浏览器开销；少于两个输出 token 时 TPOT 不定义。
- **精度**：只展示 Split 4/27/5 对原生完整单卡 TP1/full-prefill 的 8 道 GSM8K 真实归档结果。逐绝对位置对齐，decode 使用同一 teacher-forcing 序列；不比较自由回答。`accuracy-samples.json` 展示这 8 组不同输入。原生与 Split TP2+2 已于2026-09-07重新采集，现采用 logits cosine、top1 一致率和 top-5/10/20 overlap，不沿用绝对误差门槛；原始 logits、模型版本和门槛见 [精度报告](../docs/gsm8k-native-tp1.md)。
- **优化曲线**：05 章节主要展示 baseline 与优化系统的真实开环对比：同一目标到达率使用相同泊松请求计划，客户端不限制并发，服务端统一 max_active=96 / kv_blocks=32768。横轴为实测完成 QPS，均值/P99 按计划到达 cohort 计算；TTFT 包括发包延误，错误计失败，到达和完成 cohort 均须至少99%请求同时满足两项SLO。显示每点实际到达率、在途并发和测量时长；页面不展示闭环吞吐曲线；保留 C16 Baseline 的实测耗时表，并在表格下方展示通信、流水和调度分析结论。见 [开环报告](evidence/open-loop-report.md)。
- **仿真**：`demo/simulate.py` 对接当前主线 Q4，只开放 Qwen2.5-3B/A10/4K/79。optimized 使用 vendored real-serving scheduler、C40 empirical command 和 host profile，seed17；baseline 使用 behavioral scheduler、operator/host cost，沿用 baseline 精度回归的活动上限（C1 为8，其余为16）。每点真实执行、不缓存结果；不同并发仍是预测，参见 [校准与局限](../Q4/docs/CALIBRATION_AND_LIMITS.md)。

## 复现开环对比

独占四卡服务后运行：

```bash
.venv/bin/python scripts/open_loop_sweep.py --out results/my-open-loop
.venv/bin/python scripts/package_open_loop.py --source results/my-open-loop
python3 scripts/plot_open_loop.py
```

绘图需 matplotlib；测量脚本使用现有 serving venv。运行器先测试 baseline，再测试 optimized，结束恢复默认 baseline WAN 服务。保存逐请求计划、实际时间、health、trace、配置、源文件和 hash。打包器从逐请求数据重新计算 QPS、均值、P99、联合 SLO 与平均在途并发，检查相同负载的到达计划一致。

每点预热30秒、粗扫测量60秒（目标到达率≥4时120秒）、继续到达30秒后排空。首次失败后补一个中间点；每点只有 seed17，不是精确边界或长期生产容量验证。原始数据可下载，UI 不会点击后伪造新的性能测试。

## API

- `POST /api/service/start {}`：异步执行固定启动命令。
- `GET /api/health`：本工作目录服务的真实健康状态。
- `POST /api/chat {"messages":[{"role":"user","content":"你好"}]}`：真实 SSE。
- `POST /api/security/encode {"text":"…"}` / `POST /api/security/recover {"encode_id":"…"}`：真实生成与恢复。
- `POST /api/simulate {"variant":"optimized","concurrency":40}`：当前 Q4 仿真。
- `GET /api/jobs/{id}`：实际状态、日志、结果；失败明确报告。
- `GET /api/artifacts/{id}/{filename}`：下载本次产物。
- `GET /api/evidence`：归档实测 JSON。

任务产物在 `results/demo/<uuid>/`，日志与命令可追溯。任务索引在内存中，重启 UI 后旧产物仍保留在磁盘，但不支持恢复轮询；不实现断点续跑。没有任意 shell 输入；默认 loopback、POST 检查同源。保持为本地研究工具，不提供公网多用户部署。

## 验证

```bash
python3 -m unittest demo/test_demo.py -v
node --check demo/app.js
.venv/bin/python -m unittest tests/test_open_loop_benchmark.py
python3 scripts/check_demo_evidence.py
```

端到端验收须在真实四卡环境运行启动、对话；CPU 攻击仍需要固定模型权重。

可选浏览器端到端脚本：`demo/browser_smoke.cjs`（需要单独安装 Playwright/Chromium）。先 `./poc down`，启动 UI 后运行 `node demo/browser_smoke.cjs`；它会实际启动四卡服务并运行对话、攻击和两个仿真点。可通过 `DEMO_URL`、`DEMO_EVIDENCE_DIR`、`PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH` 指定地址、产物目录及浏览器。前端运行本身不依赖 Playwright。图表另可通过安装 matplotlib 后运行 `scripts/plot_demo_sweep.py` 导出 PNG/SVG。

本次完整验收与新增数据见 [验证报告](../docs/live-demo-validation.md)。

05 章节可用 `node demo/browser_open_loop.cjs` 做只读浏览器验收（需要 Playwright/Chromium）：核对实际证据与表格/曲线一致，检查 CSV、图片、原始 ZIP 下载及桌面/移动视图，不启动模型或发送推理请求。
