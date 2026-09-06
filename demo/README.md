# SplitShield 安全推理研究 Demo

这是一个零前端依赖的单页演示，汇总行业洞察、Split Inference 安全实验、真实系统 baseline 和 Q4 性能建模能力。页面中的能力边界是显式的：安全攻击与对话区重放已有证据/假数据；性能曲线的默认组合会调用仓库中的 Q4 仿真器逐点计算。

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

## 页面内容与数据边界

1. **背景**：说明云端大模型运行时隐私问题。
2. **业界洞察**：只提炼主仓 [完整洞察报告](https://github.com/zy95-12/A-weekender-s-challenge-for-AI-Infra/blob/main/docs/01_tech_insight.md) 的核心结论。
3. **安全分析**：重放 PR #15 的 embedding-only 100% token 恢复证据，并展示不同企业侧层数下的固定预算攻击结果。按钮不会在浏览器中运行 GPU 攻击。详见 [安全报告](https://github.com/zy95-12/A-weekender-s-challenge-for-AI-Infra/blob/feat/issue3-hidden-state-security/docs/hidden-state-security.md)。
4. **系统实现**：介绍 PR #8 的 Qwen2.5-3B、4/27/5、TP2+2 baseline。启动按钮和对话结果是假数据，只表达产品交互；真实部署见 [PR #8](https://github.com/zy95-12/A-weekender-s-challenge-for-AI-Infra/pull/8)。
5. **性能优化**：只保留后续扩展位置。
6. **性能建模**：前端依次调用 `POST /api/simulate`，每完成一个 closed-loop 并发点就更新 QPS—TTFT/TPOT 曲线。

当前真实可运行组合为 **Qwen3-32B / A10 / 9 卡 / 云侧 TP4**。A10 使用解析 Roofline 参数预设，尚未针对 Qwen3-32B 实测校准，适合演示 Roofline + DAG + event-driven 的系统行为，不应用作生产容量承诺。DeepSeek-V3/V4、H20、L20、Ascend 910B/950 以及其他卡数已保留表单和 API 契约，但后端会返回 `422 not_implemented`，不会生成伪曲线。

API 示例：

```bash
curl -s http://127.0.0.1:8088/api/simulate \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-32B","hardware":"A10","cards":9,"tp_degree":4,"input_tokens":256,"output_tokens":12,"concurrency":1}'
```

## 文件

- `index.html`：六部分页面结构与研究内容
- `styles.css`：响应式视觉与组件样式
- `app.js`：攻击回放、假对话、逐点仿真和 SVG 曲线
- `server.py`：静态文件服务、参数校验、Q4 仿真适配、内存限制与缓存

全部前端资源随仓库提供，不依赖 CDN、Node.js 或第三方 Python 包。
