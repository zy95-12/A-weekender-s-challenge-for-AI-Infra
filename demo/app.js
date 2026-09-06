const $ = (selector) => document.querySelector(selector);
const toast = (message) => { const el = $('#toast'); el.textContent = message; el.classList.add('show'); setTimeout(() => el.classList.remove('show'), 1800); };

document.querySelectorAll('[data-copy]').forEach((button) => button.addEventListener('click', async () => {
  await navigator.clipboard.writeText(button.dataset.copy); toast('命令已复制');
}));

const generatedSamples = [
  'secure cloud inference protects private input',
  'enterprise data stays inside trusted boundary',
  'split inference connects edge and cloud',
];

$('#token-generate').addEventListener('click', () => {
  const next = generatedSamples[Math.floor(Math.random() * generatedSamples.length)];
  $('#attack-input').value = next;
  toast('已生成 token 序列');
});

function demoTokens(value) {
  return value.match(/[\p{Script=Han}]|[\p{L}\p{N}_-]+|[^\s]/gu) || [];
}

function demoVector(token, index) {
  let hash = 2166136261;
  for (const char of token) hash = Math.imul(hash ^ char.codePointAt(0), 16777619) >>> 0;
  return Array.from({length: 4}, (_, offset) => {
    const value = Math.sin((hash % 10007) * (offset + 1) + index) * .92;
    return Number(value.toFixed(4));
  });
}

$('#attack-run').addEventListener('click', async (event) => {
  const button = event.currentTarget, output = $('#attack-output');
  const source = $('#attack-input').value.trim();
  if (!source) { toast('请先输入文本或生成 token'); return; }
  const tokens = demoTokens(source), vectors = tokens.map(demoVector);
  button.disabled = true; button.textContent = '正在执行…';
  output.textContent = `$ python embedding_attack.py --demo\ninput_tokens = ${JSON.stringify(tokens)}\n\n[1/3] embedding hidden-state matrix (preview, dim=4):\n${JSON.stringify(vectors, null, 2)}\n`;
  await new Promise((resolve) => setTimeout(resolve, 650));
  output.textContent += '\n[2/3] cosine nearest-neighbor over public vocabulary…\n';
  await new Promise((resolve) => setTimeout(resolve, 550));
  output.textContent += `[3/3] recovered_tokens = ${JSON.stringify(tokens)}\n\n✓ comparison: ${tokens.length} / ${tokens.length} tokens consistent (cosine ≈ 0.9998)`;
  button.textContent = '加密 / 解密'; button.disabled = false;
  output.scrollTop = output.scrollHeight;
} );

let serviceReady = false;
$('#service-toggle').addEventListener('click', async (event) => {
  const button = event.currentTarget;
  if (serviceReady) { toast('Demo 服务已处于就绪状态'); return; }
  button.disabled = true; button.textContent = '启动中…'; $('#service-state').textContent = '加载 Front / Middle / Back';
  await new Promise((resolve) => setTimeout(resolve, 900));
  serviceReady = true; button.textContent = 'Demo Ready'; $('#service-state').textContent = 'Qwen2.5-3B · 4/27/5'; $('#service-dot').classList.add('ready'); toast('Baseline 界面已就绪（假数据）');
});

$('#chat-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!serviceReady) { toast('请先启动 Demo'); return; }
  const input = $('#chat-text'), text = input.value.trim(); if (!text) return;
  const messages = $('#messages');
  messages.insertAdjacentHTML('beforeend', `<div class="message user"><span>U</span><p>${escapeHtml(text)}</p></div>`); input.value = '';
  await new Promise((resolve) => setTimeout(resolve, 600));
  messages.insertAdjacentHTML('beforeend', '<div class="message assistant"><span>S</span><p>因为推理输入可能包含业务机密与个人数据。Split Inference 缩小了直接暴露面，但中间激活仍需结合威胁模型验证，不能视为天然加密。</p></div>');
  $('#chat-ttft').textContent = '186 ms'; $('#chat-tpot').textContent = '31 ms'; messages.scrollTop = messages.scrollHeight;
});

function escapeHtml(value) { const div = document.createElement('div'); div.textContent = value; return div.innerHTML; }

const supported = () => $('#model').value === 'Qwen3-32B' && $('#hardware').value === 'A10' && $('#cards').value === '9';
function updateCapability() {
  const box = $('#capability');
  if (supported()) { box.classList.remove('unsupported'); box.querySelector('b').textContent = '当前可运行'; box.querySelector('span').textContent = 'Qwen3-32B · A10 · 9 卡；Q4 解析仿真。'; }
  else { box.classList.add('unsupported'); box.querySelector('b').textContent = '接口已预留'; box.querySelector('span').textContent = '该模型 / 硬件 / 卡数组合尚无 cost model，暂不生成伪结果。'; }
}
['#model','#hardware','#cards'].forEach((id) => $(id).addEventListener('change', updateCapability));

$('#sim-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!supported()) { toast('该组合当前仅预留接口，请选择 Qwen3-32B / A10 / 9 卡'); return; }
  const levels = [...document.querySelectorAll('input[name="concurrency"]:checked')].map((el) => Number(el.value));
  if (!levels.length) { toast('请至少选择一个并发量'); return; }
  const button = $('#sim-run'), points = []; button.disabled = true;
  $('#sim-progress').textContent = `0 / ${levels.length}`; $('#progress-bar').style.width = '0%';
  try {
    for (let index = 0; index < levels.length; index += 1) {
      const concurrency = levels[index]; $('#sim-status').textContent = `正在计算 concurrency = ${concurrency}…`;
      const response = await fetch('/api/simulate', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:$('#model').value,hardware:$('#hardware').value,cards:Number($('#cards').value),tp_degree:Number($('#tp').value),input_tokens:Number($('#input-tokens').value),output_tokens:Number($('#output-tokens').value),concurrency})});
      const payload = await response.json(); if (!response.ok) throw new Error(payload.message || '仿真失败');
      points.push(payload.point); renderChart(points);
      $('#metric-qps').textContent = payload.point.qps.toFixed(2); $('#metric-ttft').textContent = `${payload.point.ttft_mean_ms.toFixed(1)} ms`; $('#metric-tpot').textContent = `${payload.point.tpot_mean_ms.toFixed(1)} ms`;
      $('#sim-progress').textContent = `${index + 1} / ${levels.length}`; $('#progress-bar').style.width = `${(index + 1) / levels.length * 100}%`;
    }
    $('#sim-status').textContent = `完成 · ${points.length} 个并发点`; toast('仿真曲线已生成');
  } catch (error) { $('#sim-status').textContent = `失败：${error.message}`; toast(error.message); }
  finally { button.disabled = false; }
});

function renderChart(points) {
  const width=760,height=340,pad={l:58,r:62,t:25,b:46};
  const maxX=Math.max(...points.map(p=>p.qps),1)*1.15, maxY=Math.max(...points.flatMap(p=>[p.ttft_mean_ms,p.tpot_mean_ms]),10)*1.18;
  const x=v=>pad.l+v/maxX*(width-pad.l-pad.r), y=v=>height-pad.b-v/maxY*(height-pad.t-pad.b);
  const grid=[0,.25,.5,.75,1].map(f=>`<line x1="${pad.l}" y1="${y(maxY*f)}" x2="${width-pad.r}" y2="${y(maxY*f)}" stroke="#193040"/><text x="${pad.l-10}" y="${y(maxY*f)+4}" text-anchor="end" fill="#70858f" font-size="10">${(maxY*f).toFixed(0)}</text>`).join('');
  const path=key=>points.map((p,i)=>`${i?'L':'M'} ${x(p.qps)} ${y(p[key])}`).join(' ');
  const dots=(key,color)=>points.map(p=>`<circle cx="${x(p.qps)}" cy="${y(p[key])}" r="4" fill="${color}"/><text x="${x(p.qps)}" y="${y(p[key])-9}" text-anchor="middle" fill="${color}" font-size="9">C${p.concurrency}</text>`).join('');
  $('#chart').innerHTML=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="QPS 与 TTFT TPOT 曲线"><g>${grid}</g><line x1="${pad.l}" y1="${height-pad.b}" x2="${width-pad.r}" y2="${height-pad.b}" stroke="#47606c"/><line x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${height-pad.b}" stroke="#47606c"/><text x="18" y="20" fill="#91a3ae" font-size="10">ms</text><text x="${width-pad.r}" y="${height-14}" text-anchor="end" fill="#91a3ae" font-size="10">observed QPS</text><path d="${path('ttft_mean_ms')}" fill="none" stroke="#43e3cf" stroke-width="2.5"/>${dots('ttft_mean_ms','#43e3cf')}<path d="${path('tpot_mean_ms')}" fill="none" stroke="#a78bfa" stroke-width="2.5"/>${dots('tpot_mean_ms','#a78bfa')}${points.map(p=>`<text x="${x(p.qps)}" y="${height-pad.b+18}" text-anchor="middle" fill="#70858f" font-size="9">${p.qps.toFixed(2)}</text>`).join('')}</svg>`;
}
