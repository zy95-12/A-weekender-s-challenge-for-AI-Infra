const $ = (s) => document.querySelector(s);
const esc = (v) => {
  const d = document.createElement("div");
  d.textContent = String(v);
  return d.innerHTML;
};
const toast = (m) => {
  const e = $("#toast");
  e.textContent = m;
  e.classList.add("show");
  setTimeout(() => e.classList.remove("show"), 3000);
};
async function api(path, body) {
  const r = await fetch(
    path,
    body === undefined
      ? {}
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  const p = await r.json();
  if (!r.ok) throw Error(p.error || p.message || r.statusText);
  return p;
}
async function job(path, payload, log) {
  let j = await api(path, payload);
  while (j.status === "running") {
    if (log) log.textContent = j.log || "后台实际执行中…";
    await new Promise((r) => setTimeout(r, 1000));
    j = await api("/api/jobs/" + j.id);
  }
  if (log) log.textContent = j.log || j.error || "";
  if (j.status !== "completed") throw Error(j.error || "任务失败");
  return j;
}
document.querySelectorAll("[data-copy]").forEach((b) =>
  b.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(b.dataset.copy);
      toast("命令已复制");
    } catch (e) {
      toast(e.message);
    }
  }),
);
let encoded = null;
$("#attack-text").addEventListener("input", () => {
  encoded = null;
  $("#attack-run").disabled = true;
});
$("#encode-run").addEventListener("click", async () => {
  const b = $("#encode-run");
  b.disabled = true;
  $("#attack-run").disabled = true;
  encoded = null;
  $("#attack-text").disabled = true;
  $("#recovered-output").textContent = "等待恢复…";
  $("#attack-download").textContent = "";
  try {
    const j = await job(
      "/api/security/encode",
      { text: $("#attack-text").value },
      $("#attack-output"),
    );
    encoded = j.id;
    $("#attack-output").textContent = JSON.stringify(j.result, null, 2);
    $("#attack-download").innerHTML =
      `<a href="/api/artifacts/${j.id}/hidden.npy">下载本次激活 NPY</a>`;
    $("#attack-run").disabled = false;
  } catch (e) {
    toast(e.message);
    $("#attack-output").textContent = e.message;
  } finally {
    b.disabled = false;
    $("#attack-text").disabled = false;
  }
});
$("#attack-run").addEventListener("click", async () => {
  const b = $("#attack-run");
  b.disabled = true;
  try {
    const j = await job(
      "/api/security/recover",
      { encode_id: encoded },
      $("#attack-output"),
    );
    const r = j.result;
    $("#recovered-output").textContent =
      `${r.recovered_text}\n\n逐 token：${r.correct}/${r.positions.length} (${(r.token_accuracy * 100).toFixed(2)}%)\n整段完全匹配：${r.exact_sequence_match}\n真实检索耗时：${r.seconds.toFixed(3)}s`;
    $("#attack-output").textContent = JSON.stringify(
      r.positions.slice(0, 32),
      null,
      2,
    );
    $("#attack-download").innerHTML +=
      ` · <a href="/api/artifacts/${j.id}/recover.json">下载全部逐位置结果</a>`;
  } catch (e) {
    toast(e.message);
  } finally {
    b.disabled = false;
  }
});
let ready = false,
  starting = false,
  chatting = false;
async function refreshHealth() {
  try {
    const h = await api("/api/health");
    ready = h.status === "ready";
    $("#service-state").textContent = ready
      ? "服务已启动"
      : starting
        ? "服务启动中"
        : "服务未启动";
    $("#service-dot").classList.toggle("ready", ready);
    if (ready)
      $("#service-state").textContent +=
        " · " + (h.pd ? "优化 PD" : "Baseline TP2+2");
  } catch (e) {
    ready = false;
    $("#service-state").textContent = "健康检查失败";
    $("#service-dot").classList.remove("ready");
  }
}
$("#service-toggle").addEventListener("click", async () => {
  const b = $("#service-toggle");
  b.disabled = true;
  starting = true;
  ready = false;
  $("#service-state").textContent = "服务启动中";
  try {
    await job("/api/service/start", {}, $("#service-log"));
    await refreshHealth();
    toast("真实服务已启动");
  } catch (e) {
    $("#service-log").textContent += "\n" + e.message;
    toast(e.message);
  } finally {
    starting = false;
    b.disabled = false;
    await refreshHealth();
  }
});
const history = [];
$("#chat-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (chatting) return;
  if (!ready) {
    toast("请先启动服务");
    return;
  }
  const input = $("#chat-text"),
    text = input.value.trim();
  if (!text) return;
  chatting = true;
  const b = $("#chat-form button");
  b.disabled = true;
  const box = $("#messages");
  box.insertAdjacentHTML(
    "beforeend",
    `<div class="message user"><span>U</span><p>${esc(text)}</p></div><div class="message assistant"><span>S</span><p></p></div>`,
  );
  const output = box.lastElementChild.querySelector("p");
  input.value = "";
  $("#chat-ttft").textContent = "—";
  $("#chat-tpot").textContent = "—";
  let answer = "",
    first = null,
    last = null,
    count = 0,
    usage = null,
    done = false;
  const start = performance.now();
  try {
    const r = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: [...history, { role: "user", content: text }],
      }),
    });
    if (!r.ok) {
      const x = await r.json();
      throw Error(x.error || "推理失败");
    }
    const reader = r.body.getReader(),
      decoder = new TextDecoder();
    let pending = "";
    while (true) {
      const chunk = await reader.read();
      if (chunk.done) break;
      pending += decoder.decode(chunk.value, { stream: true });
      let pos;
      while ((pos = pending.indexOf("\n")) >= 0) {
        const line = pending.slice(0, pos).trim();
        pending = pending.slice(pos + 1);
        if (!line.startsWith("data:")) continue;
        const raw = line.slice(5).trim();
        if (raw === "[DONE]") {
          done = true;
          continue;
        }
        const event = JSON.parse(raw);
        if (event.error) throw Error(JSON.stringify(event.error));
        if (event.usage) usage = event.usage;
        const choice = event.choices?.[0];
        if (choice && typeof choice.delta?.content === "string") {
          const now = performance.now();
          if (first === null) {
            first = now;
            $("#chat-ttft").textContent = (first - start).toFixed(1) + " ms";
          }
          last = now;
          count++;
          answer += choice.delta.content;
          output.textContent = answer;
          box.scrollTop = box.scrollHeight;
        }
      }
    }
    if (!done || !answer) throw Error("流提前结束或输出为空");
    if (usage && usage.completion_tokens !== count)
      throw Error("token 事件数量与 usage 不一致，无法报告 TPOT");
    $("#chat-tpot").textContent =
      count > 1
        ? ((last - first) / (count - 1)).toFixed(2) + " ms"
        : "单 token，无 TPOT";
    history.push(
      { role: "user", content: text },
      { role: "assistant", content: answer },
    );
    while (history.length > 20) history.splice(0, 2);
  } catch (e) {
    output.textContent = answer + "\n[失败] " + e.message;
    toast(e.message);
  } finally {
    chatting = false;
    b.disabled = false;
  }
});
function table(headers, rows) {
  return (
    '<div class="data-table"><table><thead><tr>' +
    headers.map((x) => "<th>" + esc(x) + "</th>").join("") +
    "</tr></thead><tbody>" +
    rows
      .map(
        (row) =>
          "<tr>" + row.map((v) => "<td>" + esc(v) + "</td>").join("") + "</tr>",
      )
      .join("") +
    "</tbody></table></div>"
  );
}
function chart(selector, points, key, limit) {
  if (!points.length) return;
  const w = 760,
    h = 340,
    p = { l: 65, r: 20, t: 35, b: 45 },
    maxX = Math.max(1, ...points.map((x) => x.qps)) * 1.12,
    maxY =
      Math.max(
        limit,
        ...points.map((x) => x[key]),
        ...points.map((x) => x[key.replace("mean", "p99")] || 0),
      ) * 1.12;
  const x = (v) => p.l + (v / maxX) * (w - p.l - p.r),
    y = (v) => h - p.b - (v / maxY) * (h - p.t - p.b);
  const path = (k) =>
    points.map((v, i) => `${i ? "L" : "M"}${x(v.qps)},${y(v[k])}`).join(" ");
  let svg = "";
  for (let i = 0; i <= 4; i++) {
    const v = (maxY * i) / 4;
    svg += `<line x1="${p.l}" x2="${w - p.r}" y1="${y(v)}" y2="${y(v)}" stroke="#29404c"/><text x="${p.l - 8}" y="${y(v) + 4}" fill="#a9c5ce" text-anchor="end" font-size="11">${v.toFixed(0)}</text>`;
  }
  svg += `<line x1="${p.l}" x2="${w - p.r}" y1="${y(limit)}" y2="${y(limit)}" stroke="#e6ac69" stroke-dasharray="5 4"/><text x="${p.l + 5}" y="${y(limit) - 5}" fill="#e6ac69" font-size="11">SLO ${limit}ms</text>`;
  for (const [k, color, label] of [
    [key, "#43e3cf", "mean"],
    [key.replace("mean", "p99"), "#a78bfa", "P99"],
  ]) {
    if (points.some((v) => v[k] === undefined)) continue;
    svg += `<path d="${path(k)}" fill="none" stroke="${color}" stroke-width="2"/>`;
    points.forEach((v) => {
      svg += `<circle cx="${x(v.qps)}" cy="${y(v[k])}" r="4" fill="${color}"><title>C${v.concurrency}: ${v[k].toFixed(2)}ms, ${v.qps.toFixed(3)} QPS ${label}</title></circle>`;
    });
  }
  points.forEach((v, i) => {
    svg += `<text x="${x(v.qps) + 3}" y="${y(v[key]) - 10 - (i % 2) * 12}" fill="#43e3cf" font-size="10">C${v.concurrency}</text>`;
  });
  for (let i = 0; i <= 5; i++) {
    const tick = (maxX * i) / 5;
    svg += `<text x="${x(tick)}" y="${h - p.b + 18}" fill="#a9c5ce" text-anchor="middle" font-size="10">${tick.toFixed(1)}</text>`;
  }
  $(selector).innerHTML =
    `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${key} 对 QPS"><text x="10" y="18" fill="#a9c5ce">${key.startsWith("ttft") ? "TTFT" : "TPOT"} ms</text><text x="110" y="18" fill="#43e3cf">mean</text><text x="170" y="18" fill="#a78bfa">P99</text>${svg}<text x="${w - 110}" y="${h - 5}" fill="#a9c5ce">QPS</text></svg>`;
}
function openLoopChart(selector, points, metric, limit) {
  if (!points.length) return;
  const w = 820,
    h = 360,
    p = { l: 70, r: 25, t: 55, b: 45 };
  const maxX = Math.max(1, ...points.map((r) => r.completed_qps)) * 1.1;
  const maxY =
    Math.max(
      limit,
      ...points.flatMap((r) => [
        r[`mean_${metric}_ms`] || 0,
        r[`p99_${metric}_ms`] || 0,
      ]),
    ) * 1.12;
  const x = (v) => p.l + (v / maxX) * (w - p.l - p.r);
  const y = (v) => h - p.b - (v / maxY) * (h - p.t - p.b);
  let svg = "";
  for (let i = 0; i <= 5; i++) {
    const a = (maxX * i) / 5,
      b = (maxY * i) / 5;
    svg += `<line x1="${p.l}" x2="${w - p.r}" y1="${y(b)}" y2="${y(b)}" stroke="#29404c"/><text x="${p.l - 8}" y="${y(b) + 4}" fill="#a9c5ce" text-anchor="end" font-size="11">${b.toFixed(0)}</text><text x="${x(a)}" y="${h - p.b + 20}" fill="#a9c5ce" text-anchor="middle" font-size="11">${a.toFixed(1)}</text>`;
  }
  svg += `<line x1="${p.l}" x2="${w - p.r}" y1="${y(limit)}" y2="${y(limit)}" stroke="#e6ac69" stroke-dasharray="5 4"/><text x="${p.l + 4}" y="${y(limit) - 5}" fill="#e6ac69" font-size="11">SLO ${limit} ms</text>`;
  for (const [system, color, name] of [
    ["baseline", "#e6ac69", "Baseline"],
    ["optimized", "#43e3cf", "优化系统"],
  ]) {
    const rows = points
      .filter((r) => r.system === system)
      .sort((a, b) => a.target_arrival_rate - b.target_arrival_rate);
    for (const stat of ["mean", "p99"]) {
      const key = `${stat}_${metric}_ms`,
        valid = rows.filter((r) => r[key] != null);
      const path = valid
        .map((r, i) => `${i ? "L" : "M"}${x(r.completed_qps)},${y(r[key])}`)
        .join(" ");
      svg += `<path d="${path}" fill="none" stroke="${color}" stroke-width="2" ${stat === "p99" ? 'stroke-dasharray="6 4"' : ""}/>`;
      valid.forEach((r) => {
        svg += `<circle cx="${x(r.completed_qps)}" cy="${y(r[key])}" r="4" fill="${r.slo_pass ? color : "#132833"}" stroke="${color}" stroke-width="2"><title>${name} · λ=${r.target_arrival_rate.toFixed(3)} · 实际到达=${r.actual_arrival_rate.toFixed(3)}/s · 完成=${r.completed_qps.toFixed(3)} QPS · ${stat} ${r[key].toFixed(2)}ms · 联合SLO ${r.slo_pass ? "通过" : "未通过"}</title></circle>`;
      });
    }
  }
  $(selector).innerHTML =
    `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="开环 ${metric.toUpperCase()} 对完成 QPS，Baseline 与优化系统"><text x="10" y="18" fill="#a9c5ce">${metric.toUpperCase()} ms</text><text x="150" y="18" fill="#e6ac69">Baseline</text><text x="260" y="18" fill="#43e3cf">优化系统</text><text x="150" y="38" fill="#a9c5ce" font-size="12">实线：均值　虚线：P99　空心点：联合 SLO 未通过</text>${svg}<text x="${w - 160}" y="${h - 4}" fill="#a9c5ce">实测完成 QPS</text></svg>`;
}

async function evidence() {
  try {
    const d = await api("/api/evidence");
    if (d["security-depth"])
      $("#security-depth").innerHTML = table(
        ["前层深度", "4K 样本恢复率", "独立公开测试恢复率"],
        d["security-depth"].map((r) => [
          r.depth,
          (r.current_prompt_mlp_token_accuracy_mean * 100).toFixed(2) + "%",
          (r.public_test_mlp_token_accuracy_mean * 100).toFixed(2) + "%",
        ]),
      );
    const v = d.accuracy.variants.baseline;
    $("#accuracy-results").innerHTML =
      table(
        [
          "配置",
          "阶段",
          "位置数",
          "Logits cosine 平均 / 最低",
          "Top1 一致",
          "Top5 overlap 平均 / 最低",
          "Top10 overlap 平均 / 最低",
          "Top20 overlap 平均 / 最低",
        ],
        ["prefill", "decode"].map((p) => [
          "Split · 4 / 27 / 5",
          p,
          v[p].rows,
          v[p].mean_cosine_similarity.toFixed(8) +
            " / " +
            v[p].min_cosine_similarity.toFixed(8),
          `${v[p].top1_matches}/${v[p].rows}`,
          ...[5, 10, 20].map(
            (k) =>
              (v[p].top_k_overlap[k].mean * 100).toFixed(2) +
              "% / " +
              (v[p].top_k_overlap[k].min * 100).toFixed(2) +
              "%",
          ),
        ]),
      ) +
      `<p>数据来源：2026-09-07 重新进行真实 GPU 采集；原生完整模型单卡 TP1，split 为 TP2+2，均 full prefill。此对照包含 TP 差异；逐请求采集，不代表高并发数值一致性。固定 test indices：${d.accuracy_manifest.indices.join(", ")}；prefill 只比较最后一个输入位置，decode 每题 7 个位置。</p><p>Cosine 使用全词表原始 logits 向量；top-k overlap = 交集大小 / k，逐位置计算后统计平均与最低值。不比较集合内部顺序；同分按 token ID 从小到大排序。本次报告描述性指标，不沿用绝对误差门槛判定。</p>`;
    if (d["accuracy-samples"])
      $("#accuracy-results").innerHTML += table(
        ["GSM8K index", "问题", "输入 token 数"],
        d["accuracy-samples"].map((r) => [r.index, r.question, r.tokens]),
      );
    const b = d["baseline-c16"];
    $("#breakdown").innerHTML = ["ttft", "tpot"]
      .map(
        (k) =>
          `<h4>${k.toUpperCase()}：${b.breakdown[k].total_ms.toFixed(2)}ms</h4>` +
          table(
            ["实测区间", "平均耗时 ms"],
            Object.entries(b.breakdown[k].parts_ms).map(([x, y]) => [
              x,
              y.toFixed(3),
            ]),
          ),
      )
      .join("");
    const openLoop = d["open-loop-sweep"];
    if (openLoop) {
      openLoopChart("#open-loop-ttft", openLoop.points, "ttft", 3000);
      openLoopChart("#open-loop-tpot", openLoop.points, "tpot", 100);
      $("#open-loop-status").textContent = openLoop.conclusion;
      $("#open-loop-table").innerHTML = table(
        [
          "系统",
          "目标 / 实际到达率",
          "完成 / 达标 QPS",
          "TTFT mean/P99 ms",
          "TPOT mean/P99 ms",
          "到达 / 完成 SLO %",
          "平均 / 峰值在途",
          "到达请求数",
          "测量 s",
        ],
        openLoop.points.map((r) => [
          r.system === "baseline" ? "Baseline" : "优化系统",
          r.target_arrival_rate.toFixed(3) +
            " / " +
            r.actual_arrival_rate.toFixed(3),
          r.completed_qps.toFixed(3) + " / " + r.goodput_qps.toFixed(3),
          r.mean_ttft_ms.toFixed(1) + " / " + r.p99_ttft_ms.toFixed(1),
          r.mean_tpot_ms.toFixed(2) + " / " + r.p99_tpot_ms.toFixed(2),
          (r.arrival_slo * 100).toFixed(2) +
            " / " +
            (r.completion_slo * 100).toFixed(2),
          r.mean_inflight.toFixed(1) + " / " + r.peak_inflight,
          r.requests,
          r.duration_s,
        ]),
      );
    } else
      $("#open-loop-status").textContent =
        "开环实验进行中，尚未发布完整对比结果";
    const sweep = d["optimized-sweep"];
    if (sweep) {
      const points = sweep.points.map((r) => ({
        concurrency: r.concurrency,
        qps: r.completed_qps,
        ttft_mean_ms: r.mean_ttft_ms,
        tpot_mean_ms: r.mean_tpot_ms,
        ttft_p99_ms: r.p99_ttft_ms,
        tpot_p99_ms: r.p99_tpot_ms,
      }));
      chart("#sweep-ttft", points, "ttft_mean_ms", 3000);
      chart("#sweep-tpot", points, "tpot_mean_ms", 100);
      $("#sweep-status").textContent = sweep.conclusion;
      $("#sweep-table").innerHTML = table(
        [
          "C",
          "QPS",
          "TTFT mean/P99 ms",
          "TPOT mean/P99 ms",
          "完成/到达 SLO %",
          "窗口 s",
        ],
        sweep.points.map((r) => [
          r.concurrency,
          r.completed_qps.toFixed(3),
          r.mean_ttft_ms.toFixed(1) + " / " + r.p99_ttft_ms.toFixed(1),
          r.mean_tpot_ms.toFixed(2) + " / " + r.p99_tpot_ms.toFixed(2),
          (r.slo_attainment * 100).toFixed(2) +
            " / " +
            (r.start_cohort_slo_attainment * 100).toFixed(2),
          r.duration_s.toFixed(1),
        ]),
      );
    } else $("#sweep-status").textContent = "新增扫描尚未归档，不展示占位数字";
  } catch (e) {
    toast("证据加载失败：" + e.message);
  }
}
$("#sim-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const levels = $("#sim-levels")
    .value.split(",")
    .map((x) => Number(x.trim()));
  if (
    !levels.length ||
    levels.length > 16 ||
    levels.some((c) => !Number.isInteger(c) || c < 1 || c > 96)
  ) {
    toast("输入 1–16 个并发点，每点为 1–96 整数");
    return;
  }
  const b = $("#sim-run");
  b.disabled = true;
  const points = [],
    variant = $("#variant").value;
  $("#chart").innerHTML =
    '<div id="sim-ttft"></div><div id="sim-tpot"></div><div id="sim-downloads"></div>';
  try {
    for (const [i, c] of levels.entries()) {
      $("#sim-status").textContent = `实际仿真中：${variant} C${c}`;
      const j = await job("/api/simulate", { variant, concurrency: c });
      points.push(j.result.point);
      chart("#sim-ttft", points, "ttft_mean_ms", 3000);
      chart("#sim-tpot", points, "tpot_mean_ms", 100);
      $("#sim-downloads").innerHTML +=
        `<a href="/api/artifacts/${j.id}/result.json">C${c} JSON</a> `;
      $("#metric-qps").textContent = j.result.point.qps.toFixed(3);
      $("#metric-ttft").textContent =
        j.result.point.ttft_mean_ms.toFixed(1) + " ms";
      $("#metric-tpot").textContent =
        j.result.point.tpot_mean_ms.toFixed(2) + " ms";
      $("#sim-progress").textContent = `${i + 1} / ${levels.length}`;
      $("#progress-bar").style.width = ((i + 1) / levels.length) * 100 + "%";
    }
    $("#sim-status").textContent = "仿真完成 · 预测值";
  } catch (e) {
    $("#sim-status").textContent = e.message;
    toast(e.message);
  } finally {
    b.disabled = false;
  }
});
evidence();
refreshHealth();
setInterval(refreshHealth, 10000);
