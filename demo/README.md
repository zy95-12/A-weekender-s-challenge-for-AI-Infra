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
- **C16 拆解**：来自历史 baseline refinement C16 完成 cohort 的 96 个请求和 7488 个 decode 位置。`baseline-c16-raw.zip` 保留原始请求、trace 和环境，区间为 wall-clock/RPC 区间，不伪装成 kernel 时间。`python3 scripts/check_demo_evidence.py` 可独立复算。
- **优化曲线**：本次新增四卡最优 PD、4K/79 closed-loop 测量。每点至少 60 秒、每 worker 至少 3 轮；边界另做更长确认。SLO 为至少 99% 请求同时 TTFT≤3s、请求平均 TPOT≤100ms，完成和到达 cohort 都检查。CSV 和 ZIP 直接从真实请求生成。
- **仿真**：`demo/simulate.py` 对接当前主线 Q4，只开放 Qwen2.5-3B/A10/4K/79。optimized 使用 vendored real-serving scheduler、C40 empirical command 和 host profile，seed17；baseline 使用 behavioral scheduler、operator/host cost，沿用 baseline 精度回归的活动上限（C1 为8，其余为16）。每点真实执行、不缓存结果；不同并发仍是预测，参见 [校准与局限](../Q4/docs/CALIBRATION_AND_LIMITS.md)。

## 复现实测曲线

确保其他任务已停止：

```bash
./poc up --preset optimized --wan
.venv/bin/python scripts/demo_sweep.py --output results/my-demo-sweep
python3 scripts/package_demo_evidence.py --sweep results/my-demo-sweep \
  --serving-results "$(cat run/current_results)" --output results/my-demo-evidence
```

扫描默认从 C1 增至 C96，遇到第一个不满足联合 SLO 的点停止；没有找到失效点时不会宣称找到最大容量。
边界可使用 `scripts/pd_benchmark.py --seconds 180 --cycles 6` 另测确认点；归档器在同一并发存在多次测量时选最长窗口绘图，保留全部原始运行。

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
python3 scripts/check_demo_evidence.py
```

端到端验收须在真实四卡环境运行启动、对话；CPU 攻击仍需要固定模型权重。

可选浏览器端到端脚本：`demo/browser_smoke.cjs`（需要单独安装 Playwright/Chromium）。先 `./poc down`，启动 UI 后运行 `node demo/browser_smoke.cjs`；它会实际启动四卡服务并运行对话、攻击和两个仿真点。可通过 `DEMO_URL`、`DEMO_EVIDENCE_DIR`、`PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH` 指定地址、产物目录及浏览器。前端运行本身不依赖 Playwright。图表另可通过安装 matplotlib 后运行 `scripts/plot_demo_sweep.py` 导出 PNG/SVG。

本次完整验收与新增数据见 [验证报告](../docs/live-demo-validation.md)。
