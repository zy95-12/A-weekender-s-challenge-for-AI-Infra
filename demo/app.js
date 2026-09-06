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

const optimizationEvidence = {
  baseline: [
    {qps:.2996,ttftMean:806.5,ttftP99:834.5,tpotMean:32.43,tpotP99:34.00},
    {qps:.4761,ttftMean:1218.6,ttftP99:1648.3,tpotMean:38.23,tpotP99:43.53},
    {qps:.6657,ttftMean:1937.4,ttftP99:3160.9,tpotMean:48.48,tpotP99:64.03},
    {qps:.9301,ttftMean:3297.1,ttftP99:6141.5,tpotMean:67.89,tpotP99:103.62},
    {qps:1.0325,ttftMean:4776.5,ttftP99:8940.9,tpotMean:87.56,tpotP99:140.68},
    {qps:1.0915,ttftMean:6331.4,ttftP99:11828.1,tpotMean:106.98,tpotP99:179.18},
  ],
  optimized: [
    {qps:.3318,ttftMean:447.4,ttftP99:465.0,tpotMean:32.87,tpotP99:33.11},
    {qps:.5794,ttftMean:552.0,ttftP99:565.8,tpotMean:37.23,tpotP99:37.50},
    {qps:.9333,ttftMean:544.7,ttftP99:605.5,tpotMean:45.93,tpotP99:47.13},
    {qps:1.4264,ttftMean:563.9,ttftP99:636.3,tpotMean:64.28,tpotP99:70.04},
    {qps:1.7469,ttftMean:556.1,ttftP99:603.3,tpotMean:80.56,tpotP99:82.68},
    {qps:1.9652,ttftMean:550.9,ttftP99:568.2,tpotMean:96.42,tpotP99:97.73},
  ],
};

function renderOptimizationCurves() {
  const target = $('#optimization-curves');
  if (!target) return;
  const width=1040,height=370,panelWidth=460,plotHeight=255,top=50,bottom=55;
  const panels=[{left:52,title:'QPS — TTFT',mean:'ttftMean',p99:'ttftP99',maxY:12000,tick:3000},{left:552,title:'QPS — TPOT',mean:'tpotMean',p99:'tpotP99',maxY:200,tick:50}];
  const colors={baseline:'#ff6f73',optimized:'#43e3cf'},maxX=2.1;
  const parts=['<svg viewBox="0 0 1040 370" role="img" aria-label="QPS 与 TTFT TPOT 实测对比曲线">'];
  panels.forEach((panel) => {
    const x=(value)=>panel.left+48+value/maxX*(panelWidth-66), y=(value)=>top+plotHeight-value/panel.maxY*plotHeight;
    parts.push(`<text x="${panel.left}" y="22" fill="#eff6f6" font-size="13" font-weight="650">${panel.title}</text>`);
    for(let value=0;value<=panel.maxY;value+=panel.tick){parts.push(`<line x1="${panel.left+48}" y1="${y(value)}" x2="${panel.left+panelWidth-18}" y2="${y(value)}" stroke="#193040"/><text x="${panel.left+40}" y="${y(value)+4}" text-anchor="end" fill="#70858f" font-size="9">${value}</text>`);}
    [0,.5,1,1.5,2].forEach((value)=>parts.push(`<text x="${x(value)}" y="${height-bottom+22}" text-anchor="middle" fill="#70858f" font-size="9">${value.toFixed(1)}</text>`));
    parts.push(`<line x1="${panel.left+48}" y1="${top}" x2="${panel.left+48}" y2="${height-bottom}" stroke="#49616c"/><line x1="${panel.left+48}" y1="${height-bottom}" x2="${panel.left+panelWidth-18}" y2="${height-bottom}" stroke="#49616c"/><text x="${panel.left+panelWidth-18}" y="${height-12}" text-anchor="end" fill="#70858f" font-size="9">observed QPS</text><text x="${panel.left+4}" y="${top-9}" fill="#70858f" font-size="9">ms</text>`);
    Object.entries(optimizationEvidence).forEach(([name,points])=>{
      [panel.mean,panel.p99].forEach((key,index)=>{const path=points.map((point,i)=>`${i?'L':'M'} ${x(point.qps).toFixed(1)} ${y(point[key]).toFixed(1)}`).join(' ');parts.push(`<path d="${path}" fill="none" stroke="${colors[name]}" stroke-width="${index?1.5:2.5}" ${index?'stroke-dasharray="6 5"':''}/>`);});
      points.forEach((point)=>parts.push(`<circle cx="${x(point.qps)}" cy="${y(point[panel.mean])}" r="3.5" fill="${colors[name]}"/>`));
    });
  });
  parts.push('</svg>'); target.innerHTML=parts.join('');
}

renderOptimizationCurves();
