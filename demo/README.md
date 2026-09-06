# Split-Infer 优化/建模 Demo 演示

页面以 **SplitShield** 作为暂定展示名。这是一个零前端依赖的单页 Demo，汇总行业洞察、Split Inference 安全实验、真实系统 baseline 和 Q4 性能建模能力。

Demo 的目标不是把所有功能包装成已经完成，而是提供一个统一、可演示的研究入口，并明确标注能力边界：

- 安全攻击区重放 PR #15 已验证的实验结果；
- baseline 启动和对话区使用假数据展示预期产品形态；
- 默认性能配置会调用 Q4 仿真器真实计算，不使用前端伪造曲线；
- 尚无模型或硬件 cost model 的组合只保留接口，并返回明确的 `not_implemented`。

## 环境要求

- Python 3.10 或更高版本；
- 仓库中已存在 `Q4/split_serving_sim`；
- 不需要 Node.js、前端构建工具、CDN 或额外 Python 包。

## 启动

在仓库根目录运行：

```bash
python demo/server.py --host 127.0.0.1 --port 8088
```

浏览器访问 `http://127.0.0.1:8088`。远端服务器可以通过 SSH 端口转发查看：

```bash
ssh -L 8088:127.0.0.1:8088 <user>@<server-ip>
```

服务默认设置 768 MiB 地址空间上限，并将仿真串行化，给 4 GB 服务器的 SSH 和系统进程留出空间。可用 `--memory-limit-mib` 调整；不建议在共享服务器上关闭限制。

只查看静态排版可以直接打开 `demo/index.html`，但性能建模需要通过 `server.py` 访问，否则 `/api/simulate` 不可用。

## 页面内容与数据边界

1. **背景**：说明云端大模型运行时隐私问题。
2. **业界洞察**：只提炼主仓 [完整洞察报告](https://github.com/zy95-12/A-weekender-s-challenge-for-AI-Infra/blob/main/docs/01_tech_insight.md) 的核心结论。
3. **安全分析**：用户可以输入文本或自动生成 token，通过假矩阵交互回放 embedding 与最近邻反演过程；同时展示不同企业侧层数下的固定预算攻击结果。按钮不会在浏览器中运行 GPU 攻击。详见 [安全报告](https://github.com/zy95-12/A-weekender-s-challenge-for-AI-Infra/blob/feat/issue3-hidden-state-security/docs/hidden-state-security.md)。
4. **功能实现**：介绍 PR #8 的 Qwen2.5-3B、4/27/5、TP2+2 baseline，提供一行启动命令和产品交互，并展示三种切分与原生完整模型的真实精度对比。启动按钮和对话结果是假数据；精度表来自 [POC 实测报告](https://github.com/zy95-12/A-weekender-s-challenge-for-AI-Infra/blob/feat/issue-4-real-split-vllm-poc/docs/poc-validation.md)。
5. **性能优化**：只保留后续扩展位置。
6. **性能建模**：前端依次调用 `POST /api/simulate`，每完成一个 closed-loop 并发点就更新 QPS—TTFT/TPOT 曲线。

当前真实可运行组合为 **Qwen3-32B / A10 / 9 卡 / 云侧 TP4**。A10 使用解析 Roofline 参数预设，尚未针对 Qwen3-32B 实测校准，适合演示 Roofline + DAG + event-driven 的系统行为，不应用作生产容量承诺。DeepSeek-V3/V4、H20、L20、Ascend 910B/950 以及其他卡数已保留表单和 API 契约，但后端会返回 `422 not_implemented`，不会生成伪曲线。

## 性能建模链路

```text
HTML 配置表单
    │  concurrency = 1 → 2 → 4 → 8（逐点请求）
    ▼
POST /api/simulate
    │  参数白名单 / 范围校验 / 串行锁 / 结果缓存
    ▼
Q4 Simulator
    │  Roofline cost + DAG + event-driven closed loop
    ▼
observed QPS + mean/P99 TTFT + mean/P99 TPOT
    │
    └── 前端原生 SVG 增量绘图
```

### 支持矩阵

| 模型 | 硬件 | 卡数 / 并行 | 当前行为 |
| --- | --- | --- | --- |
| Qwen3-32B | A10 | 9 卡，云侧 TP4 | 调用 Q4 真实解析仿真 |
| DeepSeek-V3 / V4 | 任意预留硬件 | 任意 | 接口预留，返回 HTTP 422 |
| Qwen3-32B | H20 / L20 / Ascend 910B / 950 | 任意 | 接口预留，返回 HTTP 422 |

当前 Qwen3-32B 拓扑沿用 Q4 示例：企业侧 front/tail 使用 1 卡，云侧 middle 使用 TP4 × PP2 共 8 卡。每个并发点采用 5 秒 closed-loop 测量窗口，先排除 warmup，再统计完成请求。

API 示例：

```bash
curl -s http://127.0.0.1:8088/api/simulate \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-32B","hardware":"A10","cards":9,"tp_degree":4,"input_tokens":256,"output_tokens":12,"concurrency":1}'
```

成功响应：

```json
{
  "status": "ok",
  "engine": "Q4",
  "point": {
    "concurrency": 1,
    "qps": 0.8,
    "ttft_mean_ms": 122.36,
    "ttft_p99_ms": 122.36,
    "tpot_mean_ms": 81.26,
    "tpot_p99_ms": 81.26,
    "measured_requests": 4
  }
}
```

这些数值是默认配置的示例输出，会随输入/输出 token 数及后续 cost model 更新而变化。

## 文件

- `index.html`：六部分页面结构与研究内容
- `styles.css`：响应式视觉与组件样式
- `refinements.css`：标题层级与首屏展示微调
- `app.js`：攻击回放、假对话、逐点仿真和 SVG 曲线
- `server.py`：静态文件服务、参数校验、Q4 仿真适配、内存限制与缓存
- `test_demo.py`：HTML 结构、证据链接与 API 参数契约测试

全部前端资源随仓库提供，不依赖 CDN、Node.js 或第三方 Python 包。

行业洞察卡片中的“安全性”和“性能影响”均为定性评级，用于横向理解技术取舍，不代表统一 benchmark。安全攻击交互里的四维 embedding preview 也是前端生成的假数据；真实实验仍以 PR #15 的脚本、模型和结果文件为准。

## 验证

在仓库根目录运行 Demo 测试：

```bash
python -m unittest demo/test_demo.py -v
```

验证底层 Q4 仿真器：

```bash
PYTHONPATH=Q4 python -m unittest discover -s Q4/tests -v
```

验证 JavaScript 语法（可选，需要本机有 Node.js）：

```bash
node --check demo/app.js
```

## 证据来源

- [大模型安全推理技术洞察](https://github.com/zy95-12/A-weekender-s-challenge-for-AI-Infra/blob/main/docs/01_tech_insight.md)
- [PR #15：hidden-state security](https://github.com/zy95-12/A-weekender-s-challenge-for-AI-Infra/pull/15)
- [PR #8：real split-vLLM baseline](https://github.com/zy95-12/A-weekender-s-challenge-for-AI-Infra/pull/8)
- [PR #7：Q4 split-serving simulator](https://github.com/zy95-12/A-weekender-s-challenge-for-AI-Infra/pull/7)
