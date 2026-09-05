from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def render_gantt_html(traces: list[dict[str, Any]], destination: Path) -> None:
    """Render a standalone hierarchical, zoomable pipeline Gantt page."""
    payload = json.dumps(traces, ensure_ascii=False).replace("<", "\\u003c")
    prefix = """<!doctype html>
<html><head><meta charset="utf-8"><title>Split-serving pipeline Gantt</title>
<style>
body{font-family:ui-sans-serif,system-ui;margin:20px;color:#172033;background:#f8fafc}
h1{margin:0 0 8px}.toolbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:12px 0}
button{border:1px solid #cbd5e1;background:white;border-radius:6px;padding:6px 11px;cursor:pointer}
button:hover{background:#eef2ff}.viewport{overflow:hidden;border:1px solid #cbd5e1;border-radius:10px;background:white}
svg{display:block;user-select:none;cursor:grab}.lane-main{fill:#f1f5f9}.lane-child{fill:#fff}
.grid{stroke:#dce3ed;stroke-width:1}.bar{stroke:#172033;stroke-width:.6;cursor:pointer}
.decode{stroke-width:2}.label{font-size:13px;font-weight:600}.child-label{font-size:11px;fill:#475569}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:13px}.swatch{display:inline-block;width:11px;height:11px;margin-right:4px;border-radius:2px}
#details{margin-top:12px;padding:12px;background:white;border:1px solid #cbd5e1;border-radius:8px;white-space:pre-wrap;font:12px ui-monospace,monospace;min-height:40px}
.hint{color:#475569;font-size:13px}code{background:#e2e8f0;padding:2px 5px;border-radius:4px}
</style></head><body>
<h1>Split-serving pipeline</h1>
<div class="hint">滚轮缩放；按住拖动平移；点击资源前的 ▶ 展开算子/通信子流；点击 batch 查看完整详情。</div>
<div class="toolbar">
  <button id="zoomIn">放大 +</button><button id="zoomOut">缩小 −</button>
  <button id="panLeft">←</button><button id="panRight">→</button><button id="reset">适应全图</button>
  <span id="windowText"></span>
</div>
<div class="legend">
 <span><i class="swatch" style="background:#2563eb"></i>edge_front</span>
 <span><i class="swatch" style="background:#06b6d4"></i>wan_up</span>
 <span><i class="swatch" style="background:#7c3aed"></i>cloud_middle</span>
 <span><i class="swatch" style="background:#14b8a6"></i>wan_down</span>
 <span><i class="swatch" style="background:#f59e0b"></i>edge_tail</span>
 <span><i class="swatch" style="background:#93c5fd"></i>operator</span>
 <span><i class="swatch" style="background:#fb7185"></i>node communication</span>
</div>
<div class="viewport"><svg id="chart"></svg></div>
<div id="details">点击任意主流 batch 或子流色块查看详情。</div>
<script>const traces="""
    suffix = r""";
const stageColors={edge_front:'#2563eb',wan_up:'#06b6d4',cloud_middle:'#7c3aed',wan_down:'#14b8a6',edge_tail:'#f59e0b'};
const categoryColors={compute:'#93c5fd',node_communication:'#fb7185',communication:'#22d3ee',overhead:'#94a3b8'};
const preferred=['edge_gpu','wan_up','cloud_gpu','wan_down'];
const found=[...new Set(traces.map(x=>x.resource))];
const resources=[...preferred.filter(x=>found.includes(x)),...found.filter(x=>!preferred.includes(x)).sort()];
const globalStart=Math.min(...traces.map(x=>x.start_time_ms));
const globalEnd=Math.max(...traces.map(x=>x.end_time_ms));
let viewStart=globalStart,viewEnd=globalEnd; const expanded=new Set();
const svg=document.getElementById('chart'),details=document.getElementById('details');
const NS='http://www.w3.org/2000/svg',labelWidth=210,plotWidth=1350,mainH=54,childH=36,top=58;
function node(name,attrs,text){const n=document.createElementNS(NS,name);for(const [k,v] of Object.entries(attrs||{}))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;return n;}
function subNames(resource){const names=[];for(const row of traces.filter(x=>x.resource===resource))for(const op of row.sub_operations||[])if(!names.includes(op.name))names.push(op.name);return names;}
function rows(){const out=[];for(const resource of resources){out.push({resource,main:true,key:resource});if(expanded.has(resource))for(const name of subNames(resource))out.push({resource,main:false,name,key:resource+'::'+name});}return out;}
function xAt(t){return labelWidth+(t-viewStart)/(viewEnd-viewStart)*plotWidth;}
function visible(a,b){return b>=viewStart&&a<=viewEnd;}
function tooltip(target){const title=node('title',{});title.textContent=target;return title;}
function select(value){details.textContent=JSON.stringify(value,null,2);}
function render(){
 const rowList=rows(),height=top+rowList.reduce((n,r)=>n+(r.main?mainH:childH),0)+40,width=labelWidth+plotWidth+20;
 svg.replaceChildren();svg.setAttribute('width',width);svg.setAttribute('height',height);svg.setAttribute('viewBox',`0 0 ${width} ${height}`);
 let y=top;const positions={};
 for(const row of rowList){const h=row.main?mainH:childH;positions[row.key]={y,h};svg.append(node('rect',{x:0,y,width,height:h,class:row.main?'lane-main':'lane-child'}));
  if(row.main){const toggle=node('text',{x:12,y:y+30,class:'label',style:'cursor:pointer'},expanded.has(row.resource)?'▼':'▶');toggle.addEventListener('click',()=>{expanded.has(row.resource)?expanded.delete(row.resource):expanded.add(row.resource);render();});svg.append(toggle);svg.append(node('text',{x:34,y:y+30,class:'label'},row.resource));}
  else svg.append(node('text',{x:34,y:y+23,class:'child-label'},'↳ '+row.name)); y+=h;
 }
 for(let i=0;i<=10;i++){const ratio=i/10,x=labelWidth+ratio*plotWidth,t=viewStart+ratio*(viewEnd-viewStart);svg.append(node('line',{x1:x,y1:42,x2:x,y2:height-35,class:'grid'}));svg.append(node('text',{x,y:30,'text-anchor':'middle','font-size':11},t.toFixed(1)+' ms'));}
 for(const batch of traces){if(!visible(batch.start_time_ms,batch.end_time_ms))continue;const pos=positions[batch.resource];const x=Math.max(labelWidth,xAt(batch.start_time_ms)),right=Math.min(labelWidth+plotWidth,xAt(batch.end_time_ms)),w=Math.max(right-x,1);const yb=pos.y+10,h=pos.h-20;
  const rect=node('rect',{x,y:yb,width:w,height:h,rx:2,fill:stageColors[batch.stage]||'#64748b',class:'bar '+(batch.phases.includes('decode')?'decode':'')});rect.append(tooltip(`B${batch.batch_id} | ${batch.stage} | ${batch.phases.join('/')} | requests=${JSON.stringify(batch.request_ids)} | ${batch.duration_ms.toFixed(3)} ms | ${batch.input_shape||''}`));rect.addEventListener('click',e=>{e.stopPropagation();select(batch);});svg.append(rect);if(w>30)svg.append(node('text',{x:x+4,y:yb+h/2+4,fill:'white','font-size':10,'pointer-events':'none'},'B'+batch.batch_id));
  if(expanded.has(batch.resource))for(const op of batch.sub_operations||[]){if(!visible(op.start_time_ms,op.end_time_ms))continue;const child=positions[batch.resource+'::'+op.name];if(!child)continue;const ox=Math.max(labelWidth,xAt(op.start_time_ms)),oright=Math.min(labelWidth+plotWidth,xAt(op.end_time_ms)),ow=Math.max(oright-ox,1),oy=child.y+7,oh=child.h-14;const sub=node('rect',{x:ox,y:oy,width:ow,height:oh,rx:2,fill:categoryColors[op.category]||'#a5b4fc',class:'bar'});sub.append(tooltip(`${op.name} | ${op.duration_ms.toFixed(3)} ms | ${op.input_shape}`));sub.addEventListener('click',e=>{e.stopPropagation();select({batch_id:batch.batch_id,resource:batch.resource,stage:batch.stage,...op});});svg.append(sub);if(ow>72)svg.append(node('text',{x:ox+3,y:oy+oh/2+4,'font-size':9,'pointer-events':'none'},op.input_shape));}
 }
 document.getElementById('windowText').textContent=`窗口 ${viewStart.toFixed(2)} – ${viewEnd.toFixed(2)} ms（跨度 ${(viewEnd-viewStart).toFixed(2)} ms）`;
}
function setWindow(start,end){const full=globalEnd-globalStart,minSpan=Math.max(full/10000,.001),span=Math.min(Math.max(end-start,minSpan),full);let s=start;if(s<globalStart)s=globalStart;if(s+span>globalEnd)s=globalEnd-span;viewStart=s;viewEnd=s+span;render();}
function zoom(factor,center=(viewStart+viewEnd)/2){const span=(viewEnd-viewStart)*factor,ratio=(center-viewStart)/(viewEnd-viewStart);setWindow(center-ratio*span,center+(1-ratio)*span);}
function pan(ratio){const delta=(viewEnd-viewStart)*ratio;setWindow(viewStart+delta,viewEnd+delta);}
document.getElementById('zoomIn').onclick=()=>zoom(.5);document.getElementById('zoomOut').onclick=()=>zoom(2);document.getElementById('panLeft').onclick=()=>pan(-.25);document.getElementById('panRight').onclick=()=>pan(.25);document.getElementById('reset').onclick=()=>{viewStart=globalStart;viewEnd=globalEnd;render();};
svg.addEventListener('wheel',e=>{e.preventDefault();const rect=svg.getBoundingClientRect(),ratio=Math.max(0,Math.min(1,(e.clientX-rect.left-labelWidth)/(plotWidth)));zoom(e.deltaY<0?.75:1.33,viewStart+ratio*(viewEnd-viewStart));},{passive:false});
let drag=null;svg.addEventListener('mousedown',e=>{if(e.clientX<svg.getBoundingClientRect().left+labelWidth)return;drag={x:e.clientX,start:viewStart,end:viewEnd};svg.style.cursor='grabbing';});window.addEventListener('mousemove',e=>{if(!drag)return;const delta=-(e.clientX-drag.x)/plotWidth*(drag.end-drag.start);setWindow(drag.start+delta,drag.end+delta);});window.addEventListener('mouseup',()=>{drag=null;svg.style.cursor='grab';});
render();</script></body></html>
"""
    destination.write_text(prefix + payload + suffix, encoding="utf-8")
