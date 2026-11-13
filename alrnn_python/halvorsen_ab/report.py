"""Offline Plotly report with parameter-plane slices and held-out free orbits."""
from __future__ import annotations

import html
import json

import numpy as np
from plotly.offline import get_plotlyjs


def thin(values, maximum):
    values = np.asarray(values)
    if len(values) > maximum:
        values = values[np.linspace(0, len(values) - 1, maximum, dtype=int)]
    return values


def report_payload(arrays, metadata, max_peaks):
    from halvorsen_ab.evaluate import finite_json, unpack_ragged
    truth = unpack_ragged(arrays["truth_peaks"], arrays["truth_offsets"])
    model = unpack_ragged(arrays["model_peaks"], arrays["model_offsets"])
    groups = []
    for i, (a, b) in enumerate(arrays["pairs"]):
        groups.append(dict(a=float(a), b=float(b), truth=thin(truth[i], max_peaks),
                           model=thin(model[i], max_peaks),
                           truthEquilibrium=arrays["truth_equilibria"][i],
                           modelEquilibrium=arrays["model_equilibria"][i],
                           modelFailedAt=int(arrays["model_failed_at"][i])))
    orbits = []
    for i, (a, b) in enumerate(arrays["orbit_pairs"]):
        orbits.append(dict(a=float(a), b=float(b), truth=thin(arrays["orbit_truth"][i], 5000),
                           model=thin(arrays["orbit_model"][i], 5000), metrics=metadata["metrics"][i]))
    cloud_z = [float(z) for group in groups for z in [*group["truth"], *group["model"],
                group["truthEquilibrium"], group["modelEquilibrium"]] if np.isfinite(z)]
    low, high = (min(cloud_z), max(cloud_z)) if cloud_z else (-1.0, 1.0)
    margin = max((high - low) * .03, .1)
    settings = metadata["ground_truth"]["settings"]
    return finite_json(dict(
        label=metadata["label"], groundTruthOnly=metadata["ground_truth_only"],
        aValues=np.unique(arrays["pairs"][:, 0]), bValues=np.unique(arrays["pairs"][:, 1]),
        trainingPairs=arrays["orbit_pairs"], groups=groups, orbits=orbits,
        zRange=[low - margin, high + margin], dt=metadata["data_config"]["dt"],
        burnIn=metadata["data_config"]["burn_in_steps"], warmup=settings["warmup_steps"],
        observations=settings["observation_steps"], initialStates=settings["initial_states"],
        maxPlotPeaks=max_peaks, truthFailures=len(metadata["ground_truth"]["failures"]),
        modelFailures=metadata["model_failures"], modelHash=metadata["model_sha256"]))


def write_report(arrays, metadata, path, max_peaks=128):
    payload = report_payload(arrays, metadata, max_peaks)
    # Escaping '<' also prevents a user-provided label from closing the script.
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    document = TEMPLATE.replace("__TITLE__", html.escape(metadata["label"]))
    document = document.replace("__PLOTLY_JS__", get_plotlyjs()).replace("__PAYLOAD__", data)
    path.write_text(document, encoding="utf-8")


TEMPLATE = r"""<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{font:15px/1.5 system-ui,sans-serif;color:#172b40;background:#f2f5f9;color-scheme:light}
*{box-sizing:border-box}body{margin:0}main{max-width:1600px;margin:auto;padding:28px}
h1{font-size:28px;letter-spacing:-.6px;margin:0 0 8px}h2{font-size:19px;margin:0 0 12px}
p{margin:8px 0}.muted{color:#536779}.card{background:white;border:1px solid #dce4ed;border-radius:12px;padding:20px;margin:18px 0}
.row{display:flex;gap:22px;align-items:center;flex-wrap:wrap}.plots{display:grid;grid-template-columns:1.15fr 1fr;gap:10px}
label{display:flex;gap:9px;align-items:center}select,input{font:inherit;padding:7px;border:1px solid #a8b8c9;border-radius:6px;background:white;color:#172b40}
input[type=range]{width:min(310px,65vw);padding:0}input[type=number]{width:120px}.plot{width:100%;height:560px;min-width:0}
#orbit{height:590px}#slice-label{font-variant-numeric:tabular-nums;font-weight:600}.badge{display:inline-block;background:#e8f0fb;color:#17488a;border-radius:5px;padding:2px 8px}
.notice{padding:10px 14px;border-left:3px solid #dc812a;background:#fff5e9}button{font:inherit;padding:7px 12px;border:1px solid #a8b8c9;border-radius:6px;cursor:pointer;background:#f5f8fc}
details{margin-top:12px}code{font-size:13px}#orbit-metrics{white-space:pre-wrap}
@media(max-width:1000px){main{padding:14px}.plots{grid-template-columns:1fr}.plot{height:460px}.card{padding:12px}}
</style>
<script>__PLOTLY_JS__</script>
</head>
<body><main>
<h1>__TITLE__</h1>
<p class="muted">Halvorsen mit c = 4 · Parameterraum (a,b) · physikalische Zustandskoordinaten</p>
<p id="mode" class="badge"></p>
<p id="experiment-info" class="muted"></p>
<p id="failures" class="notice" hidden></p>
<section class="card">
<h2>Bifurkationen und Parameterschnitte</h2>
<div class="row">
<label>Fixieren <select id="fixed-axis"><option value="a">a — b variieren</option><option value="b">b — a variieren</option></select></label>
<label>Rasterwert <input id="slice-index" type="range" min="0" step="1" aria-label="Fixierten Rasterwert wählen"></label>
<label>Wert <input id="fixed-value" type="number" step="any" aria-label="Fixierten Parameterwert eingeben"></label>
<span id="slice-label" aria-live="polite"></span>
<button id="download-slice">Schnitt als CSV</button>
</div>
<p class="muted">Die Ebene markiert den gewählten Schnitt. Eingaben werden auf den nächsten berechneten Rasterwert gesetzt.
Ein Klick auf einen Punkt der 3D-Wolke wählt ebenfalls einen Schnitt. Die Legenden schalten Datensätze ein und aus.</p>
<div class="plots"><div id="cloud" class="plot"></div><div id="slice" class="plot"></div></div>
<p class="muted">Kreise: lokale z-Maxima nach der Einschwingphase. Rauten: numerisch stationäre Zustände, keine Maxima.
Grüne Kreuze am Boden markieren die Trainingsparameter. Jeder Schnitt enthält alle berechneten Anfangszustände.</p>
</section>
<section class="card">
<h2>Freie 3D-Trajektorien gegenüber Ground Truth</h2>
<div class="row"><label>Parameterpaar <select id="orbit-pair" aria-label="Trajektorienpaar auswählen"></select></label><span id="orbit-metrics"></span></div>
<p class="muted">Je Paar: erste Testtrajektorie aus dem zeitlich zurückgehaltenen Ende einer langen Simulation.
Beide starten am selben beobachteten Zustand; das ALRNN läuft danach frei. Die Einschwingphase wird ausgeblendet.
Bei chaotischer Dynamik sind langfristig unterschiedliche Phasen zu erwarten.</p>
<div id="orbit" class="plot"></div>
</section>
<details class="card"><summary>Berechnung und Darstellung</summary>
<p id="details"></p><p>Die vollständigen Maxima und Trajektorien stehen in analysis.npz; die Tabellen in metrics.json.
Die HTML enthält Plotly und alle dargestellten Daten und funktioniert ohne Internet oder Server.</p>
</details>
</main>
<script id="halvorsen-data" type="application/json">__PAYLOAD__</script>
<script>
"use strict";
const D=JSON.parse(document.getElementById('halvorsen-data').textContent);
const COLORS={truth:'#2563eb',model:'#e87926'}, FORMAT=v=>Number(v.toPrecision(7)).toString();
const AXIS=document.getElementById('fixed-axis'), SLIDER=document.getElementById('slice-index');
const NUMBER=document.getElementById('fixed-value'), LABEL=document.getElementById('slice-label');
const PCONFIG={responsive:true,displaylogo:false,toImageButtonOptions:{format:'png',scale:2}};
const BASE={font:{family:'system-ui,sans-serif',size:13,color:'#243b53'},paper_bgcolor:'white',plot_bgcolor:'white',
  margin:{l:55,r:20,t:65,b:55},legend:{orientation:'h',y:-.13},uirevision:'halvorsen-ab'};
const SCENE={xaxis:{title:{text:'a'}},yaxis:{title:{text:'b'}},zaxis:{title:{text:'z (Maxima)'}},aspectmode:'cube'};
document.getElementById('mode').textContent=D.groundTruthOnly?'Nur Ground Truth':'Ground Truth + ALRNN';
document.getElementById('experiment-info').textContent=`Training: ${D.trainingPairs.length} Parameterpaare · Bifurkationen: ${D.aValues.length} × ${D.bValues.length} Rasterwerte × ${D.initialStates} Anfangszustände · dt=${D.dt}`;
document.getElementById('details').textContent=`ODE-Burn-in: ${D.burnIn} Schritte; danach gemeinsamer Startzustand, ${D.warmup} weitere Einschwing-Schritte und ${D.observations} ausgewertete Zustände. Maximal ${D.maxPlotPeaks} zeitlich gleichmäßig ausgewählte Maxima pro System, Parameterpaar und Anfangszustand in der HTML; vollständige Maxima im Cache. Modell-SHA256: ${D.modelHash||'kein Modell'}.`;
if(D.truthFailures||D.modelFailures){const node=document.getElementById('failures');node.hidden=false;
  node.textContent=`Fehlgeschlagene Simulationen: Ground Truth ${D.truthFailures}, Modell ${D.modelFailures}. Für fehlgeschlagene vollständige Modellrollouts werden keine Maxima eingezeichnet. Details in metrics.json.`;}

function cloudTrace(key,equilibrium){
  const x=[],y=[],z=[];
  for(const g of D.groups){const values=equilibrium?(g[key+'Equilibrium']===null?[]:[g[key+'Equilibrium']]):g[key];
    for(const value of values){x.push(g.a);y.push(g.b);z.push(value);}}
  return {type:'scatter3d',mode:'markers',x,y,z,name:(key==='truth'?'Ground Truth':'ALRNN')+(equilibrium?' stationär':''),
    legendgroup:key,visible:key==='model'&&D.groundTruthOnly?false:true,
    marker:{size:equilibrium?3.5:2,color:COLORS[key],opacity:.7,symbol:equilibrium?'diamond':'circle'},
    hovertemplate:'a=%{x:.6g}<br>b=%{y:.6g}<br>z=%{z:.6g}<extra>%{fullData.name}</extra>'};
}
function values(){return AXIS.value==='a'?D.aValues:D.bValues;}
function plane(value){
  const xmin=D.aValues[0],xmax=D.aValues.at(-1),ymin=D.bValues[0],ymax=D.bValues.at(-1);
  return {type:'surface',x:AXIS.value==='a'?[[value,value],[value,value]]:[[xmin,xmax],[xmin,xmax]],
    y:AXIS.value==='b'?[[value,value],[value,value]]:[[ymin,ymax],[ymin,ymax]],
    z:[[D.zRange[0],D.zRange[0]],[D.zRange[1],D.zRange[1]]],
    surfacecolor:[[0,0],[0,0]],colorscale:[[0,'#506882'],[1,'#506882']],opacity:.19,
    showscale:false,hoverinfo:'skip',name:'Schnittebene',showlegend:false};
}
function sliceGroups(){const value=values()[Number(SLIDER.value)];return D.groups.filter(g=>g[AXIS.value]===value);}
function sliceTrace(key,equilibrium){
  const free=AXIS.value==='a'?'b':'a',x=[],y=[];
  for(const g of sliceGroups()){const zs=equilibrium?(g[key+'Equilibrium']===null?[]:[g[key+'Equilibrium']]):g[key];
    for(const z of zs){x.push(g[free]);y.push(z);}}
  return {type:'scattergl',mode:'markers',x,y,name:(key==='truth'?'Ground Truth':'ALRNN')+(equilibrium?' stationär':''),
    legendgroup:key,visible:key==='model'&&D.groundTruthOnly?false:true,
    marker:{size:equilibrium?5:3,color:COLORS[key],opacity:.7,symbol:equilibrium?'diamond':'circle'},
    hovertemplate:free+'=%{x:.6g}<br>z=%{y:.6g}<extra>%{fullData.name}</extra>'};
}
function updateSlice(){
  const value=values()[Number(SLIDER.value)],free=AXIS.value==='a'?'b':'a';
  NUMBER.value=FORMAT(value);LABEL.textContent=`${AXIS.value} = ${FORMAT(value)}; ${free} variieren`;
  const p=plane(value);
  Plotly.restyle('cloud',{x:[p.x],y:[p.y],z:[p.z]},[4]);
  const freeValues=AXIS.value==='a'?D.bValues:D.aValues;
  return Plotly.react('slice',[sliceTrace('truth',false),sliceTrace('model',false),sliceTrace('truth',true),sliceTrace('model',true)],
    {...BASE,title:{text:`${AXIS.value} = ${FORMAT(value)}`},uirevision:'slice-'+AXIS.value,
      xaxis:{title:{text:free},range:[freeValues[0],freeValues.at(-1)]},
      yaxis:{title:{text:'lokale z-Maxima / stationäres z'}},datarevision:`${AXIS.value}-${SLIDER.value}`},PCONFIG);
}
function configureAxis(){SLIDER.max=values().length-1;SLIDER.value=Math.floor(values().length/2);
  NUMBER.min=values()[0];NUMBER.max=values().at(-1);return updateSlice();}
function setNearest(value){if(!Number.isFinite(value))return;let best=0;
  values().forEach((v,i)=>{if(Math.abs(v-value)<Math.abs(values()[best]-value))best=i;});SLIDER.value=best;return updateSlice();}

function orbitTrace(points,name,color){return {type:'scatter3d',mode:'lines',
  x:points.map(p=>p[0]),y:points.map(p=>p[1]),z:points.map(p=>p[2]),name,
  line:{width:2,color},hovertemplate:'x=%{x:.5g}<br>y=%{y:.5g}<br>z=%{z:.5g}<extra>%{fullData.name}</extra>'};}
function updateOrbit(){
  const item=D.orbits[Number(document.getElementById('orbit-pair').value)],m=item.metrics;
  const f=v=>v===null?'n/a':Number(v).toPrecision(4);
  document.getElementById('orbit-metrics').textContent=D.groundTruthOnly?'Ground Truth':
    `Status: ${m.status} · Dstsp=${f(m.dstsp)} · DH=${f(m.dh)} · RMSE=${f(m.rmse)}`;
  const traces=[orbitTrace(item.truth,'Ground Truth',COLORS.truth)];
  if(!D.groundTruthOnly)traces.push(orbitTrace(item.model,'ALRNN',COLORS.model));
  return Plotly.react('orbit',traces,{...BASE,title:{text:`a=${FORMAT(item.a)}, b=${FORMAT(item.b)}`},
    scene:{xaxis:{title:{text:'x'}},yaxis:{title:{text:'y'}},zaxis:{title:{text:'z'}},aspectmode:'data'},
    datarevision:String(document.getElementById('orbit-pair').value)},PCONFIG);
}
async function initialize(){
  const first=D.aValues[Math.floor(D.aValues.length/2)];
  await Plotly.newPlot('cloud',[cloudTrace('truth',false),cloudTrace('model',false),cloudTrace('truth',true),cloudTrace('model',true),plane(first),
    {type:'scatter3d',mode:'markers',x:D.trainingPairs.map(p=>p[0]),y:D.trainingPairs.map(p=>p[1]),z:D.trainingPairs.map(()=>D.zRange[0]),
      name:'Trainingsparameter',marker:{color:'#13835d',size:4,symbol:'cross'},hovertemplate:'Training a=%{x:.6g}, b=%{y:.6g}<extra></extra>'}],
    {...BASE,title:{text:'Zwei-Parameter-Bifurkationsdiagramm'},scene:SCENE},PCONFIG);
  await configureAxis();
  const orbitSelect=document.getElementById('orbit-pair');
  D.orbits.forEach((g,i)=>{const option=document.createElement('option');option.value=i;option.textContent=`a=${FORMAT(g.a)}, b=${FORMAT(g.b)}`;orbitSelect.appendChild(option);});
  orbitSelect.value=Math.floor(D.orbits.length/2);await updateOrbit();
  AXIS.addEventListener('change',configureAxis);SLIDER.addEventListener('input',updateSlice);
  NUMBER.addEventListener('change',()=>setNearest(NUMBER.valueAsNumber));orbitSelect.addEventListener('change',updateOrbit);
  document.getElementById('cloud').on('plotly_click',event=>{const p=event.points[0];if(p&&p.curveNumber!==4)setNearest(AXIS.value==='a'?p.x:p.y);});
  document.getElementById('download-slice').addEventListener('click',()=>{
    const rows=['a,b,system,kind,z'];for(const g of sliceGroups()){for(const key of ['truth','model']){
      for(const z of g[key])rows.push([g.a,g.b,key,'maximum',z].join(','));
      if(g[key+'Equilibrium']!==null)rows.push([g.a,g.b,key,'equilibrium',g[key+'Equilibrium']].join(','));}}
    const url=URL.createObjectURL(new Blob([rows.join('\n')],{type:'text/csv;charset=utf-8'}));
    const anchor=document.createElement('a');anchor.href=url;anchor.download=`slice_${AXIS.value}_${FORMAT(values()[Number(SLIDER.value)])}.csv`;anchor.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
  window.halvorsenReady=true;
}
initialize().catch(error=>{document.getElementById('failures').hidden=false;document.getElementById('failures').textContent=error.stack;console.error(error);});
</script></body></html>"""
