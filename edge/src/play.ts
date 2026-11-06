// A generic "play" page: lists the spec's actions as buttons, shows the state the network returns, its energy,
// and the runner-up futures. Rendering is derived from the export's field/action metadata, so it works for every spec.
export function playPage(spec: string): string {
  return `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>nerve · ${spec}</title>
<style>
:root{--bg:#0b0d12;--fg:#e8eaf0;--mut:#8a90a0;--card:#141824;--acc:#6ea8fe;--ok:#3ddc97;--bad:#ff6b6b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 -apple-system,Inter,Helvetica,sans-serif}
main{max-width:860px;margin:0 auto;padding:24px 16px}h1{font-size:20px;margin:0 0 4px}.sub{color:var(--mut);margin:0 0 18px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:640px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border-radius:14px;padding:16px}.card h2{font-size:13px;letter-spacing:.06em;text-transform:uppercase;color:var(--mut);margin:0 0 10px}
.state{display:grid;grid-template-columns:auto 1fr;gap:6px 14px;font-family:ui-monospace,Menlo,monospace;font-size:15px}
.state b{color:var(--mut);font-weight:500}.val{color:var(--fg)}.changed{color:var(--ok)}
.actions{display:flex;flex-wrap:wrap;gap:8px}button{background:#1d2333;color:var(--fg);border:1px solid #2a3147;border-radius:10px;padding:10px 14px;font:inherit;cursor:pointer}
button:hover{border-color:var(--acc)}select{background:#1d2333;color:var(--fg);border:1px solid #2a3147;border-radius:8px;padding:8px}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.energy{font-family:ui-monospace,Menlo,monospace}
.alt{color:var(--mut);font-family:ui-monospace,Menlo,monospace;font-size:13px}.log{max-height:260px;overflow:auto;font-family:ui-monospace,Menlo,monospace;font-size:12px;color:var(--mut)}
.log div{padding:2px 0;border-bottom:1px solid #1d2333}.pill{display:inline-block;padding:2px 8px;border-radius:999px;background:#1d2333;color:var(--mut);font-size:12px}
.viol{color:var(--bad)}a{color:var(--acc)}
</style></head><body><main>
<h1>${spec}</h1><p class="sub">The state below is produced by a neural network, not by code: each click scores every candidate future and takes the lowest energy. <span id="who" class="pill"></span></p>
<div class="grid">
<div class="card"><h2>State</h2><div id="state" class="state"></div><p class="energy" id="energy"></p><div id="alts"></div><div class="row" style="margin-top:10px"><button id="reset">reset session</button><span class="pill" id="sid"></span></div></div>
<div class="card"><h2>Actions</h2><div id="actions" class="actions"></div></div>
</div>
<div class="card" style="margin-top:16px"><h2>Trace</h2><div id="log" class="log"></div></div>
<p class="sub" style="margin-top:14px">Weights: <a href="/${spec}/weights">edge.json</a> · health: <a href="/${spec}/health">digest + patches</a> · <a href="https://github.com/alexwelcing/end-to-end-negotiator/tree/nerve">source</a></p>
</main>
<script>
const spec=${JSON.stringify(spec)};
const sid=(location.hash.slice(1)||Math.random().toString(36).slice(2,8));location.hash=sid;document.getElementById('sid').textContent='session '+sid;
let meta=null,prev=null;
const el=id=>document.getElementById(id);
function fmt(v){return typeof v==='number'?(Number.isInteger(v)?v:v.toFixed(2)):String(v)}
function render(r){const st=el('state');st.innerHTML='';for(const k of Object.keys(meta.fields)){const b=document.createElement('b');b.textContent=k;const s=document.createElement('span');s.textContent=fmt(r.state[k]);s.className='val'+(prev&&prev[k]!==r.state[k]?' changed':'');st.append(b,s)}
 el('energy').textContent=r.energy==null?'':'energy '+r.energy.toFixed(3)+(r.violations&&r.violations.length?'  ⚠ '+r.violations.join('; '):'');el('energy').className='energy'+(r.violations&&r.violations.length?' viol':'');
 el('alts').innerHTML=(r.alternatives||[]).map(a=>'<div class="alt">runner-up '+JSON.stringify(a.state)+' · '+a.energy.toFixed(2)+'</div>').join('');prev=r.state}
function logline(a,r){const d=document.createElement('div');d.textContent=new Date().toLocaleTimeString()+'  '+a.name+(Object.keys(a.params||{}).length?' '+JSON.stringify(a.params):'')+'  →  '+JSON.stringify(r.state)+'  E='+(r.energy??0).toFixed(2);el('log').prepend(d)}
async function act(a){const r=await fetch('/'+spec+'/session/'+sid,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(a)}).then(x=>x.json());if(r.error){alert(r.error);return}render(r);logline(a,r)}
async function init(){meta=await fetch('/'+spec+'/meta').then(x=>x.json());el('who').textContent=meta.states+' states · '+meta.actions.length+' action instances · '+meta.params+' learned params';
 const byName={};for(const a of meta.actions){(byName[a.name]=byName[a.name]||[]).push(a.params)}
 const box=el('actions');for(const [name,plist] of Object.entries(byName)){const row=document.createElement('div');row.className='row';const b=document.createElement('button');b.textContent=name;row.append(b);
  const keys=Object.keys(plist[0]);const sels={};for(const k of keys){const s=document.createElement('select');for(const p of plist){const v=p[k];if(![...s.options].some(o=>o.value===String(v))){const o=document.createElement('option');o.value=String(v);o.textContent=k+'='+v;o.dataset.json=JSON.stringify(v);s.append(o)}}sels[k]=s;row.append(s)}
  b.onclick=()=>act({action:name,params:Object.fromEntries(keys.map(k=>[k,JSON.parse(sels[k].selectedOptions[0].dataset.json)]))});box.append(row)}
 const r=await fetch('/'+spec+'/session/'+sid).then(x=>x.json());render(r)}
el('reset').onclick=async()=>{const r=await fetch('/'+spec+'/session/'+sid,{method:'DELETE'}).then(x=>x.json());prev=null;render(r);el('log').innerHTML=''};
init();
</script></body></html>`;
}
