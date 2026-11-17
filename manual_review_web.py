#!/usr/bin/env python3
"""Local browser interface for reviewing MyoFuse-Hybrid nucleus calls.

The server binds only to 127.0.0.1. Automatic calls are preserved, and every
manual change is stored separately when the reviewer exports the project.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import tempfile
import threading
import uuid
import webbrowser
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, send_file
import pandas as pd
from PIL import Image
from werkzeug.utils import secure_filename

from manual_review_app import EXCLUDE, IN, OUT, ReviewProject, STATE_NAMES

ACTIN_STRUCTURE_PROBABILITY_GUARD = 0.024


HTML = r"""<!doctype html>
<html lang="en" class="notranslate">
<head>
<meta charset="utf-8"><meta name="google" content="notranslate"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MyoFuse Hybrid Actin-structure Review</title>
<style>
:root{color-scheme:light;--bg:#eef2f6;--panel:#fff;--ink:#18212b;--muted:#66717e;--line:#dce3ea;--blue:#1778f2}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;overflow:hidden}
.app{display:grid;grid-template-columns:360px 1fr;height:100vh}.side{background:var(--panel);border-right:1px solid var(--line);padding:22px;overflow:auto}
h1{font-size:21px;margin:0 0 4px}h2{font-size:14px;margin:22px 0 10px}.sub,.hint{color:var(--muted);font-size:12px}.file{overflow-wrap:anywhere;margin-top:12px}
.metrics{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:18px 0}.metric{border:1px solid var(--line);border-radius:10px;padding:11px;background:#f8fafc}.metric b{display:block;font-size:25px;margin-top:5px}.metric small{color:#15803d}
.row{display:flex;gap:8px}.row>*{flex:1}button,input{font:inherit}button{border:1px solid #cfd8e3;border-radius:7px;background:#fff;padding:8px 10px;cursor:pointer}button:hover{background:#f1f5f9}button.primary{background:var(--blue);border-color:var(--blue);color:#fff}button.danger{color:#b42318}button.adding{background:#7a3ff2;border-color:#7a3ff2;color:#fff}#stage.adding canvas{cursor:crosshair}.addbox{margin-top:14px;border:1px solid #d9cdfa;border-radius:9px;padding:10px;background:#faf8ff}
label.check{display:block;padding:5px 0}input[type=checkbox]{margin-right:7px}.field{width:100%;border:1px solid #ccd6e0;border-radius:6px;padding:8px;margin:4px 0 9px}.hidden{display:none}
details{margin-top:14px;border:1px solid var(--line);border-radius:8px;padding:9px;background:#f8fafc}details summary{cursor:pointer;font-weight:600}details p{margin:8px 0 0;line-height:1.4;color:#53606d;font-size:12px}
.drop{margin:10px 0;border:2px dashed #9eb6cf;border-radius:10px;padding:18px 12px;text-align:center;background:#f7fbff;transition:.15s}.drop.over{border-color:var(--blue);background:#eaf4ff}.drop input{display:none}.drop label{display:inline-block;background:var(--blue);color:#fff;border-radius:7px;padding:8px 13px;cursor:pointer}.picked{margin-top:9px;color:#53606d;font-size:12px;overflow-wrap:anywhere;line-height:1.45}
.loader-intro{margin:20px 0 18px}.loader-intro h2{font-size:18px;margin:0 0 5px}.step{margin:18px 0}.step-title{display:flex;align-items:center;gap:9px;font-size:14px;font-weight:700;margin-bottom:8px}.step-number{display:inline-grid;place-items:center;width:23px;height:23px;border-radius:50%;background:#eaf4ff;color:#1256a0;font-size:12px}.method-grid{display:grid;gap:8px}.method-card{display:flex;gap:9px;border:1px solid var(--line);border-radius:9px;padding:11px;background:#fff;cursor:pointer}.method-card:hover{border-color:#99b9dc;background:#f8fbff}.method-card:has(input:checked){border-color:var(--blue);box-shadow:0 0 0 1px var(--blue);background:#f2f8ff}.method-card input{flex:0 0 auto;margin:3px 0 0}.method-title{display:block;font-weight:700}.method-copy{display:block;color:var(--muted);font-size:12px;line-height:1.4;margin-top:3px}.badge{display:inline-block;background:#dff3e5;color:#16743a;border-radius:99px;padding:2px 6px;margin-left:4px;font-size:10px;font-style:normal}.run-button{width:100%;padding:11px;margin-top:4px;font-weight:700}.compact-details{margin-top:12px}.channel-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:7px}.drop-title{font-size:15px;font-weight:700;margin-bottom:4px}
.analysis-status{display:none;margin:10px 0;padding:10px;border-radius:8px;line-height:1.4;overflow-wrap:anywhere}.analysis-status.on{display:block}.analysis-status.busy{background:#eaf4ff;color:#1256a0}.analysis-status.error{background:#fff0f0;color:#a31818;border:1px solid #f4caca}
.global-drop{display:none;position:fixed;inset:16px;z-index:20;border:3px dashed #1778f2;border-radius:16px;background:#eaf4fff2;color:#1256a0;align-items:center;justify-content:center;font-size:24px;font-weight:700;pointer-events:none}body.dragging .global-drop{display:flex}
.main{display:flex;flex-direction:column;min-width:0}.top{height:54px;background:#fff;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;padding:0 18px}.legend{display:flex;gap:16px;color:#586371}.sw{display:inline-block;width:18px;height:4px;border-radius:2px;margin-right:5px;vertical-align:middle}
#stage{position:relative;flex:1;min-height:0;background:#11151a;overflow:hidden;touch-action:none}canvas{position:absolute;inset:0;width:100%;height:100%}.status{height:31px;background:#fff;border-top:1px solid var(--line);padding:7px 14px;color:#66717e;font-size:12px}
#selectBox{position:absolute;border:1px solid #ffe600;background:#ffe60022;pointer-events:none}.toast{position:fixed;right:18px;top:68px;background:#17212b;color:#fff;padding:10px 14px;border-radius:8px;opacity:0;transition:.2s;z-index:3}.toast.on{opacity:1}
@media (max-width:800px){.app{grid-template-columns:minmax(160px,42vw) minmax(0,1fr)}.side{padding:10px}.top{height:auto;min-height:48px;flex-wrap:wrap;gap:5px;padding:7px 10px}.legend{gap:7px;flex-wrap:wrap;font-size:11px}.status{height:auto;min-height:31px;padding:5px 8px}.main{min-width:0}.row{gap:4px}button{padding:7px 5px}.metrics{grid-template-columns:1fr;gap:5px}}
</style>
</head>
<body><div class="app">
<aside class="side">
  <h1>🔬 MyoFuse Hybrid</h1><div class="sub">Actin-structure manuscript method · local manual quality control</div>
  <section id="loader" class="hidden">
    <div class="loader-intro"><h2>New analysis</h2><div class="hint">CZI image → automatic FI → manual review</div></div>
    <div class="step">
      <div class="step-title"><span class="step-number">1</span><span>Choose an image</span></div>
      <div id="dropZone" class="drop">
        <div class="drop-title">Drop one CZI file here</div><div class="hint" style="margin-bottom:10px">or choose it from this Mac</div>
        <label for="filePicker">Choose file</label><input id="filePicker" type="file" multiple accept=".xlsx,.tif,.tiff,.czi">
        <div id="pickedFiles" class="picked">No file selected</div>
      </div>
    </div>
    <div class="step">
      <div class="step-title"><span class="step-number">2</span><span>Choose the analysis method</span></div>
      <div class="method-grid">
        <div class="method-card"><span><span class="method-title">Actin-structure <em class="badge">Manuscript method</em></span><span class="method-copy">MyoFuse nucleus calls are refined using direction-compatible, signal-continuous Actin and MF20 paths between nearby nuclei. This fixed workflow is the method used for the manuscript release.</span></span></div>
      </div>
    </div>
    <div class="step">
      <div class="step-title"><span class="step-number">3</span><span>Choose the fields</span></div>
      <select id="uploadScene" class="field"><option value="ALL" selected>All separate fields (P1–P9) — recommended</option><option value="AUTO">Stitched overview only (TR1)</option></select>
      <div class="hint">Separate fields are analyzed individually, then pooled as total In nuclei ÷ total nuclei.</div>
    </div>
    <details class="compact-details"><summary>Channel overrides</summary>
      <p>Leave these blank to read DAPI, AF488 Actin, and AF647 MF20 from the CZI metadata.</p>
      <div class="channel-grid"><div><label>Nuclei</label><input id="uploadNuclei" class="field" type="number" placeholder="Auto" min="0"></div><div><label>Actin</label><input id="uploadActin" class="field" type="number" placeholder="Auto" min="0"></div><div><label>MF20</label><input id="uploadMf20" class="field" type="number" placeholder="Auto" min="0"></div></div>
      <p>Channel numbers are zero-based.</p>
    </details>
    <label style="margin-top:10px">…or the path of a CZI file on this Mac</label><input id="cziPath" class="field" placeholder="/Users/…/image.czi">
    <button id="loadFilesButton" class="primary run-button" onclick="uploadProject(this)">Run analysis</button>
    <div id="analysisStatus" class="analysis-status"></div>
    <details><summary>Open a previous analysis</summary>
    <p>For an analysis that is already on this Mac.</p>
    <label>Hybrid results workbook</label><input id="results" class="field">
    <label>MyoFuse label mask</label><input id="labels" class="field">
    <label>Hybrid myotube mask (optional)</label><input id="myotubeMask" class="field">
    <label>Original CZI (optional)</label><input id="czi" class="field">
    <label>Scene</label><input id="scene" class="field" value="Image:0">
    <button class="primary" onclick="loadProject()">Load analysis</button>
    </details>
    <p class="hint" style="margin-top:12px">Analysis can take several minutes. Results stay on this Mac in Documents/MyoFuse-Hybrid-Analyses.</p>
  </section>
  <section id="review" class="hidden">
    <div id="filename" class="file sub"></div><div id="sceneName" class="sub"></div>
    <div id="multiScene" class="hidden"><h2>Fields in this CZI</h2><select id="scenePicker" class="field" onchange="switchScene(this.value)"></select><div id="aggregateSummary" class="sub"></div></div>
    <div id="reviewNotice" class="analysis-status"></div>
    <button style="width:100%;margin-top:10px" onclick="closeProject()">Import / analyze another image</button>
    <div class="metrics"><div class="metric">Automatic FI<b id="autoFi">–</b><small id="autoCount"></small></div><div class="metric">Corrected FI<b id="correctFi">–</b><small id="correctCount"></small></div></div>
    <div id="counts" class="sub"></div><div id="correctionSummary" class="sub" style="margin-top:5px"></div>
    <details><summary>What do In, Out, and Exclude mean?</summary>
      <p><b>In</b>: the nucleus is counted inside a retained multinucleated myotube.</p>
      <p><b>Out</b>: the nucleus is valid, but is not counted inside a retained myotube.</p>
      <p><b>Exclude</b>: manual QC only, for a false or unusable nucleus segmentation. It is removed from the corrected FI denominator.</p>
    </details>
    <h2>Background channels</h2>
    <div class="hint" style="margin-bottom:5px">Select one or more channels. Merged colors: DAPI blue, Actin green, MF20 red.</div>
    <label class="check"><input id="dapi" type="checkbox">DAPI <span class="hint">(blue)</span></label>
    <label class="check"><input id="actin" type="checkbox">Actin <span class="hint">(green)</span></label>
    <label class="check"><input id="mf20" type="checkbox" checked>MF20 <span class="hint">(red)</span></label>
    <div class="row" style="margin-top:6px"><button onclick="setChannels('merge')">Merge all</button><button onclick="setChannels('mf20')">MF20 only</button></div>
    <div class="row" style="margin-top:6px"><button onclick="setChannels('actin')">Actin only</button><button onclick="setChannels('dapi')">DAPI only</button></div>
    <label style="display:block;margin-top:10px">Single-channel style</label>
    <select id="channelStyle" class="field"><option value="color">Pseudocolor</option><option value="gray">Grayscale</option></select>
    <label style="display:block;margin-top:10px">Display brightness <span id="brightnessValue" class="hint">1.0×</span></label>
    <input id="brightness" style="width:100%" type="range" min="0.5" max="3" step="0.1" value="1">
    <button style="width:100%;margin-top:6px" onclick="fitImage();draw()">Fit image to window</button>
    <h2 style="margin-top:14px">Overlays</h2>
    <div class="hint" style="margin-bottom:5px">Nuclei outlines: In green, Out red, automatic review candidates yellow. Yellow nuclei are excluded from FI until set to In.</div>
    <label class="check"><input id="nuclei" type="checkbox" checked>Nuclei</label>
    <div style="margin:5px 0 5px 14px;border-left:2px solid #e3e9ef;padding-left:10px">
      <label class="check"><input id="classIn" type="checkbox" checked>Show In <span class="hint">(green)</span></label>
      <label class="check"><input id="classOut" type="checkbox" checked>Show Out <span class="hint">(red)</span></label>
      <label class="check"><input id="classReview" type="checkbox" checked>Show Review candidates <span class="hint">(yellow)</span></label>
    </div>
    <div class="row" style="margin-top:5px"><button onclick="setNucleusClasses('all')">All</button><button onclick="setNucleusClasses('in')">In only</button></div>
    <div class="row" style="margin-top:5px"><button onclick="setNucleusClasses('out')">Out only</button><button onclick="setNucleusClasses('review')">Review only</button></div>
    <div class="row" style="margin-top:5px"><button onclick="goReview(-1)">← Previous review</button><button class="primary" onclick="goReview(1)">Next review →</button></div>
    <label class="check"><input id="myotube" type="checkbox" checked>Myotube boundary</label>
    <label class="check"><input id="ids" type="checkbox">Label IDs</label>
    <h2>Selected nuclei</h2><div id="selected" class="sub">None</div>
    <div class="row" style="margin-top:10px"><button onclick="act('toggle')">Toggle</button><button onclick="act('reset')">Reset</button></div>
    <div class="row" style="margin-top:8px"><button class="primary" onclick="act('in')">Set In</button><button onclick="act('out')">Set Out</button><button class="danger" onclick="act('exclude')">Exclude</button></div>
    <div class="row" style="margin-top:8px"><button onclick="historyAction('undo')">↶ Undo</button><button onclick="historyAction('redo')">↷ Redo</button></div>
    <div class="addbox"><h2 style="margin:0 0 6px">Add missed nuclei</h2>
      <div class="hint" style="margin-bottom:7px">For nuclei that segmentation did not find. Turn on, then click the empty spot on the nucleus. Right-click an added nucleus to remove it. Added nuclei show a white centre dot and count in FI. Shortcut: A (Esc to stop).</div>
      <button id="addModeButton" style="width:100%" onclick="setAddMode(!addMode)">＋ Add nucleus mode: off</button>
      <div class="row" style="margin-top:7px;align-items:center"><label class="hint" style="flex:0 0 auto">Class</label><select id="addClass" class="field" style="margin:0"><option value="in">In</option><option value="out">Out</option></select><label class="hint" style="flex:0 0 auto">Diameter px</label><input id="addDiameter" class="field" style="margin:0;width:60px" type="number" min="4" max="120" step="1"></div>
      <div id="addedCount" class="hint" style="margin-top:6px">Added: 0</div>
    </div>
    <h2>Export</h2><div class="row"><input id="exportDir" class="field" style="margin:4px 0 9px" readonly><button style="flex:0 0 auto;margin:4px 0 9px" onclick="chooseExportFolder()">Choose folder</button></div><button onclick="exportReview()">Save review and export</button>
    <button style="width:100%;margin-top:8px" onclick="closeProject()">Open another analysis</button>
  </section>
</aside>
<main class="main"><div class="top"><strong id="title">Open a MyoFuse-Hybrid result</strong><div class="legend"><span><i class="sw" style="background:#26de5f"></i>In</span><span><i class="sw" style="background:#ff4d4f"></i>Out</span><span><i class="sw" style="background:#ffdc00"></i>Review</span><span><i class="sw" style="background:#b4b4b4"></i>Exclude</span><span><i class="sw" style="background:#00d2ff"></i>Myotube</span></div></div>
<div id="stage"><canvas id="canvas"></canvas><div id="selectBox" class="hidden"></div></div><div class="status" id="statusBar">Wheel: zoom · middle drag: pan · click: select · drag: box select · right click: exclude</div></main>
</div><div id="toast" class="toast"></div><div id="globalDropHint" class="global-drop">Drop files to open or analyze</div>
<script>
const canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d'),stage=document.getElementById('stage');
let meta=null,img=new Image(),selected=new Set(),zoom=1,base=1,ox=0,oy=0,drag=null,panning=false,imageRequest=0,reviewIndex=-1;
let chosenFiles=[],addMode=false;
const displayKeys=['dapi','actin','mf20','nuclei','classIn','classOut','classReview','myotube','ids'];
function notify(t){let e=document.getElementById('toast');e.textContent=t;e.classList.add('on');setTimeout(()=>e.classList.remove('on'),1800)}
async function json(url,options){let r=await fetch(url,options);let d=await r.json();if(!r.ok)throw Error(d.error||r.statusText);return d}
async function refreshState(){meta=await json('/api/state');document.getElementById('loader').classList.toggle('hidden',meta.loaded);document.getElementById('review').classList.toggle('hidden',!meta.loaded);if(!meta.loaded)return;
 document.getElementById('filename').textContent=meta.filename;document.getElementById('sceneName').textContent='Scene: '+meta.scene;document.getElementById('title').textContent=meta.title;
 let multi=document.getElementById('multiScene'),scenePicker=document.getElementById('scenePicker');multi.classList.toggle('hidden',meta.scene_options.length<2);if(meta.scene_options.length>1){let signature=meta.scene_options.map(x=>x.key).join('|');if(scenePicker.dataset.signature!==signature){scenePicker.innerHTML='';meta.scene_options.forEach(x=>{let o=document.createElement('option');o.value=x.key;o.textContent=x.scene;scenePicker.appendChild(o)});scenePicker.dataset.signature=signature}scenePicker.value=meta.active_scene_key;document.getElementById('aggregateSummary').textContent=`All fields: automatic FI ${meta.aggregate_automatic_fi.toFixed(2)}% (${meta.aggregate_automatic_in}/${meta.aggregate_automatic_total}) · corrected FI ${meta.aggregate_corrected_fi.toFixed(2)}% (${meta.aggregate_corrected_in}/${meta.aggregate_valid_total})`;}
 document.getElementById('autoFi').textContent=meta.automatic_fi.toFixed(2)+'%';document.getElementById('autoCount').textContent=meta.automatic_in+'/'+meta.automatic_total;
 document.getElementById('correctFi').textContent=meta.corrected_fi.toFixed(2)+'%';document.getElementById('correctCount').textContent=meta.corrected_in+'/'+meta.valid_total;
 document.getElementById('counts').textContent=`In ${meta.in_count} · Out ${meta.out_count} · Review ${meta.review_candidate_count} · Excluded ${meta.excluded_count} · Changed ${meta.changed_count}`;
 document.getElementById('correctionSummary').textContent=`FP corrected ${meta.false_positive_corrections} · FN corrected ${meta.false_negative_corrections}`;
 document.getElementById('addedCount').textContent=`Added: ${meta.added_count} (In ${meta.added_in})`;let dia=document.getElementById('addDiameter');if(!dia.value)dia.value=Math.round(meta.typical_diameter);
 document.getElementById('exportDir').value ||= meta.default_export;updateSelected();await refreshImage(false)}
async function switchScene(key){if(!key||key===meta.active_scene_key)return;meta=await json('/api/switch-scene',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({key})});selected.clear();reviewIndex=-1;base=0;await refreshState();fitImage();draw();notify('Opened '+meta.scene)}
function query(){return displayKeys.map(k=>k+'='+(document.getElementById(k).checked?1:0)).join('&')+'&style='+document.getElementById('channelStyle').value+'&brightness='+document.getElementById('brightness').value+'&selected='+(selected.size?[...selected].at(-1):'')+'&t='+Date.now()}
function setChannels(mode){document.getElementById('dapi').checked=mode==='dapi'||mode==='merge';document.getElementById('actin').checked=mode==='actin'||mode==='merge';document.getElementById('mf20').checked=mode==='mf20'||mode==='merge';if(mode!=='merge'){document.getElementById('nuclei').checked=false;document.getElementById('myotube').checked=false;document.getElementById('ids').checked=false}refreshImage(false)}
function setNucleusClasses(mode){document.getElementById('nuclei').checked=true;document.getElementById('classIn').checked=mode==='all'||mode==='in';document.getElementById('classOut').checked=mode==='all'||mode==='out';document.getElementById('classReview').checked=mode==='all'||mode==='review';refreshImage(false)}
async function nucleusAt(ix,iy){let d=await json('/api/nucleus-at?x='+encodeURIComponent(ix)+'&y='+encodeURIComponent(iy));return d.id||null}
function focusNucleus(c){zoom=Math.max(zoom,3);let s=base*zoom;ox=stage.clientWidth/2-c.x*s;oy=stage.clientHeight/2-c.y*s;draw()}
async function goReview(step){if(!meta)return;let candidates=meta.centres.filter(c=>c.review);if(!candidates.length){notify('No remaining review candidates');return}let current=selected.size?[...selected].at(-1):null,index=candidates.findIndex(c=>c.id===current);if(index<0)index=reviewIndex;reviewIndex=(index+step+candidates.length)%candidates.length;let c=candidates[reviewIndex];selected=new Set([c.id]);updateSelected();focusNucleus(c);await refreshImage(false);notify(`Review candidate ${reviewIndex+1}/${candidates.length}`)}
async function refreshImage(fit=false){if(!meta)return;let sequence=++imageRequest,n=new Image();n.onload=()=>{if(sequence!==imageRequest)return;img=n;if(fit||!base)fitImage();draw()};n.onerror=()=>{if(sequence===imageRequest)notify('Could not update the image')};n.src='/api/overlay?'+query()}
function fitImage(){base=Math.min(stage.clientWidth/meta.width,stage.clientHeight/meta.height);zoom=1;ox=(stage.clientWidth-meta.width*base)/2;oy=(stage.clientHeight-meta.height*base)/2}
function draw(){canvas.width=stage.clientWidth*devicePixelRatio;canvas.height=stage.clientHeight*devicePixelRatio;ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);ctx.clearRect(0,0,stage.clientWidth,stage.clientHeight);let s=base*zoom;ctx.imageSmoothingEnabled=true;ctx.drawImage(img,ox,oy,meta.width*s,meta.height*s)}
function toImage(x,y){let s=base*zoom;return[(x-ox)/s,(y-oy)/s]}
function nearest(ix,iy){let best=null,bd=18/(base*zoom);for(let c of meta.centres){let d=Math.hypot(c.x-ix,c.y-iy);if(d<bd){bd=d;best=c.id}}return best}
function updateSelected(){document.getElementById('selected').textContent=selected.size?[...selected].join(', '):'None'}
canvas.addEventListener('wheel',e=>{e.preventDefault();let r=stage.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top,[ix,iy]=toImage(x,y),old=base*zoom;zoom=Math.max(.35,Math.min(12,zoom*(e.deltaY<0?1.15:.87)));let s=base*zoom;ox=x-ix*s;oy=y-iy*s;draw()},{passive:false});
canvas.addEventListener('mousedown',e=>{let r=stage.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;if(e.button===1){panning=true;drag={x,y,ox,oy};e.preventDefault();return}if(e.button===0)drag={x,y,select:true};});
window.addEventListener('mousemove',e=>{if(!drag)return;let r=stage.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;if(panning){ox=drag.ox+x-drag.x;oy=drag.oy+y-drag.y;draw();return}if(drag.select&&Math.hypot(x-drag.x,y-drag.y)>4){let b=document.getElementById('selectBox');b.classList.remove('hidden');b.style.left=Math.min(x,drag.x)+'px';b.style.top=Math.min(y,drag.y)+'px';b.style.width=Math.abs(x-drag.x)+'px';b.style.height=Math.abs(y-drag.y)+'px'}});
window.addEventListener('mouseup',async e=>{if(!drag)return;let r=stage.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;if(panning){panning=false;drag=null;return}if(e.button===0&&drag.select){selected.clear();if(Math.hypot(x-drag.x,y-drag.y)>4){let a=toImage(drag.x,drag.y),b=toImage(x,y),l=Math.min(a[0],b[0]),rr=Math.max(a[0],b[0]),t=Math.min(a[1],b[1]),bb=Math.max(a[1],b[1]);meta.centres.forEach(c=>{if(c.x>=l&&c.x<=rr&&c.y>=t&&c.y<=bb)selected.add(c.id)})}else if(addMode){let p=toImage(x,y);document.getElementById('selectBox').classList.add('hidden');drag=null;await addNucleusAt(p[0],p[1]);return}else{let p=toImage(x,y),id=await nucleusAt(p[0],p[1]);if(id==null)id=nearest(p[0],p[1]);if(id!=null)selected.add(id)}updateSelected();await refreshImage(false)}document.getElementById('selectBox').classList.add('hidden');drag=null});
canvas.addEventListener('contextmenu',async e=>{e.preventDefault();let r=stage.getBoundingClientRect(),p=toImage(e.clientX-r.left,e.clientY-r.top),id=await nucleusAt(p[0],p[1]);if(id==null)id=nearest(p[0],p[1]);if(id!=null){let c=meta.centres.find(c=>c.id===id);selected=new Set([id]);updateSelected();if(c&&c.added){await act('remove');notify('Added nucleus removed');return}if(addMode){notify('Only manually added nuclei can be removed; use Exclude for segmented ones');return}await act('exclude')}});
function setAddMode(on){addMode=on;let b=document.getElementById('addModeButton');b.classList.toggle('adding',on);b.textContent='＋ Add nucleus mode: '+(on?'ON':'off');stage.classList.toggle('adding',on);document.getElementById('statusBar').textContent=on?'ADD MODE · click an empty spot to add a nucleus · right click an added nucleus: remove · Esc: stop':'Wheel: zoom · middle drag: pan · click: select · drag: box select · right click: exclude'}
async function addNucleusAt(ix,iy){try{let d=await json('/api/add-nucleus',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({x:ix,y:iy,state:document.getElementById('addClass').value,diameter:Number(document.getElementById('addDiameter').value)||null})});selected=new Set(d.selected||[]);await refreshState();notify('Nucleus added ('+document.getElementById('addClass').value+')')}catch(e){notify(e.message)}}
document.addEventListener('keydown',e=>{if(!meta||!meta.loaded)return;let t=e.target.tagName;if(t==='INPUT'||t==='SELECT'||t==='TEXTAREA')return;if(e.key==='a'||e.key==='A'){setAddMode(!addMode);e.preventDefault()}else if(e.key==='Escape'&&addMode){setAddMode(false)}});
async function act(action){if(!selected.size){notify('Select one or more nuclei first');return}meta=await json('/api/action',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action,ids:[...selected]})});await refreshState();notify('Review updated')}
async function historyAction(action){meta=await json('/api/'+action,{method:'POST'});selected=new Set(meta.selected||[]);await refreshState()}
async function chooseExportFolder(){try{let d=await json('/api/select-export-folder',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({initial_dir:document.getElementById('exportDir').value})});if(!d.cancelled&&d.folder){document.getElementById('exportDir').value=d.folder;notify('Export folder selected')}}catch(e){notify(e.message)}}
async function exportReview(){let d=await json('/api/export',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({output_dir:document.getElementById('exportDir').value})});notify('Saved '+d.workbook)}
async function loadProject(){try{await json('/api/load',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({results:results.value,labels:labels.value,myotube_mask:myotubeMask.value,czi:czi.value,scene:scene.value})});await refreshState()}catch(e){notify(e.message)}}
function showChosen(){let box=document.getElementById('pickedFiles');box.textContent=chosenFiles.length?chosenFiles.map(f=>f.name).join(' · '):'No files selected'}
function setAnalysisStatus(text,kind='busy'){let box=document.getElementById('analysisStatus');box.textContent=text;box.className='analysis-status on '+kind}
async function waitForAnalysis(jobId){while(true){let d=await json('/api/analysis-status/'+jobId);setAnalysisStatus(d.message||'Analysis running…');if(d.status==='error')throw Error(d.message||'Automatic analysis failed');if(d.status==='complete'){await refreshState();let reviewNotice=document.getElementById('reviewNotice');if(d.analysis_notice){reviewNotice.textContent=d.analysis_notice;reviewNotice.className='analysis-status on busy'}else{reviewNotice.className='analysis-status'}document.getElementById('analysisStatus').className='analysis-status';notify('Analysis loaded');return}await new Promise(resolve=>setTimeout(resolve,800))}}
async function uploadProject(button=document.getElementById('loadFilesButton')){let cziPathValue=(document.getElementById('cziPath').value||'').trim();if(!chosenFiles.length&&!cziPathValue){notify('Choose or drag the analysis files first');return}let old=button.textContent;button.disabled=true;let raw=(chosenFiles.length===1&&chosenFiles[0].name.toLowerCase().endsWith('.czi'))||(!chosenFiles.length&&!!cziPathValue);button.textContent=raw?'Running automatic analysis…':'Loading files…';setAnalysisStatus(raw?'File received. Running MyoFuse and Actin-structure analysis; this can take several minutes.':'Loading the selected result files.');let form=new FormData();chosenFiles.forEach(file=>form.append('files',file));if(!chosenFiles.length&&cziPathValue)form.append('czi_path',cziPathValue);form.append('scene',document.getElementById('uploadScene').value||'ALL');form.append('analysis_mode','actin_structure');form.append('nuclei_channel',document.getElementById('uploadNuclei').value);form.append('actin_channel',document.getElementById('uploadActin').value);form.append('mf20_channel',document.getElementById('uploadMf20').value);try{let response=await fetch('/api/upload',{method:'POST',body:form});let data=await response.json();if(!response.ok)throw Error(data.error||response.statusText);if(data.job_id){await waitForAnalysis(data.job_id);chosenFiles=[];showChosen();return}chosenFiles=[];showChosen();await refreshState();let reviewNotice=document.getElementById('reviewNotice');if(data.analysis_notice){reviewNotice.textContent=data.analysis_notice;reviewNotice.className='analysis-status on busy'}else{reviewNotice.className='analysis-status'}notify('Analysis loaded');document.getElementById('analysisStatus').className='analysis-status'}catch(e){setAnalysisStatus('Analysis error: '+e.message,'error');if(meta){let reviewNotice=document.getElementById('reviewNotice');reviewNotice.textContent='Analysis error: '+e.message;reviewNotice.className='analysis-status on error'}notify('Analysis error — see the message in the left panel')}finally{button.disabled=false;button.textContent=old}}
async function closeProject(){if(meta&&meta.total_changed_count&&!confirm('Unsaved manual changes will be lost. Open another analysis?'))return;await json('/api/close',{method:'POST'});meta=null;selected.clear();document.getElementById('title').textContent='Open a MyoFuse-Hybrid result';document.getElementById('loader').classList.remove('hidden');document.getElementById('review').classList.add('hidden');ctx.clearRect(0,0,canvas.width,canvas.height)}
const picker=document.getElementById('filePicker'),dropZone=document.getElementById('dropZone');picker.addEventListener('change',()=>{chosenFiles=[...picker.files];showChosen();document.getElementById('analysisStatus').className='analysis-status'});['dragenter','dragover'].forEach(name=>dropZone.addEventListener(name,e=>{e.preventDefault();e.stopPropagation();dropZone.classList.add('over')}));dropZone.addEventListener('dragleave',e=>{e.preventDefault();e.stopPropagation();dropZone.classList.remove('over')});dropZone.addEventListener('drop',async e=>{e.preventDefault();e.stopPropagation();dropZone.classList.remove('over');chosenFiles=[...e.dataTransfer.files];showChosen();await uploadProject()});
let dragDepth=0;document.addEventListener('dragenter',e=>{e.preventDefault();dragDepth++;document.body.classList.add('dragging')});document.addEventListener('dragover',e=>e.preventDefault());document.addEventListener('dragleave',e=>{e.preventDefault();dragDepth=Math.max(0,dragDepth-1);if(!dragDepth)document.body.classList.remove('dragging')});document.addEventListener('drop',async e=>{e.preventDefault();dragDepth=0;document.body.classList.remove('dragging');let files=[...e.dataTransfer.files];if(!files.length)return;if(meta&&meta.changed_count&&!confirm('Unsaved manual changes will be lost. Analyze the dropped files?'))return;chosenFiles=files;showChosen();await uploadProject()});
displayKeys.forEach(k=>document.getElementById(k).addEventListener('change',()=>refreshImage(false)));document.getElementById('channelStyle').addEventListener('change',()=>refreshImage(false));document.getElementById('brightness').addEventListener('input',e=>document.getElementById('brightnessValue').textContent=Number(e.target.value).toFixed(1)+'×');document.getElementById('brightness').addEventListener('change',()=>refreshImage(false));window.addEventListener('resize',()=>{if(meta){fitImage();draw()}});refreshState().then(()=>{if(meta)refreshImage(true)}).catch(e=>notify(e.message));
</script></body></html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--project", type=Path, help="Saved manual review project JSON")
    parser.add_argument("--label-mask", type=Path)
    parser.add_argument("--czi", type=Path)
    parser.add_argument("--scene", default="Image:0")
    parser.add_argument("--channel-cache", type=Path)
    parser.add_argument("--myotube-mask", type=Path)
    parser.add_argument("--analysis-python", type=Path)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open-browser", action="store_true")
    return parser.parse_args()


ARGS = parse_args()
if ARGS.analysis_python:
    # Preserve the virtual-environment path. Resolving its Python symlink would
    # bypass the environment and lose the CZI reader installed inside it.
    os.environ["MYOFUSE_ANALYSIS_PYTHON"] = str(ARGS.analysis_python.expanduser().absolute())

app = Flask(__name__)
project: ReviewProject | None = None
project_collection: dict[str, ReviewProject] = {}
active_project_key: str | None = None
analysis_jobs: dict[str, dict] = {}
analysis_jobs_lock = threading.Lock()
upload_root = Path(tempfile.gettempdir()) / "myofuse-hybrid-review-uploads"
upload_root.mkdir(parents=True, exist_ok=True)


def inspect_czi_metadata(
    czi: Path, analysis_python: str
) -> tuple[list[str], list[str], list[str]]:
    """Read CZI channel and scene names with the analysis environment."""
    inspector = Path(__file__).resolve().parent / "list_czi_scenes.py"
    completed = subprocess.run(
        [analysis_python, str(inspector), "--json", str(czi)],
        capture_output=True, text=True, timeout=120,
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout or "Cannot read CZI metadata").strip()
        raise RuntimeError(detail[-1600:])
    line = next(
        (line for line in completed.stdout.splitlines() if line.startswith("METADATA_JSON=")),
        None,
    )
    if line is None:
        raise RuntimeError("Cannot identify the CZI channels and scenes")
    metadata = json.loads(line.removeprefix("METADATA_JSON="))
    return (
        list(metadata["channels"]),
        list(metadata["scenes"]),
        list(metadata.get("overview_scenes", [])),
    )


def matching_channels(names: list[str], expected: str, role: str) -> list[int]:
    """Return channels whose metadata names are plausible for one role."""
    aliases = {
        "Nuclei": ("dapi", "hoechst", "405"),
        "Actin": ("af488", "actin", "phalloidin", "fitc"),
        "MF20": ("af647", "mf20", "myhc", "myosin"),
    }.get(role, (expected.casefold(),))
    normalized = [str(name).casefold().replace(" ", "") for name in names]
    exact = [i for i, name in enumerate(normalized) if name == expected.casefold()]
    if exact:
        return exact
    return [i for i, name in enumerate(normalized) if any(alias in name for alias in aliases)]


def select_channel(requested: str | None, names: list[str], expected: str,
                   fallback: int, role: str) -> tuple[int, str | None]:
    """Select by metadata when blank, while allowing an explicit valid index."""
    if not names:
        raise ValueError("No channels were found in the CZI file")
    requested_text = "" if requested is None else requested.strip()
    if not requested_text:
        matches = matching_channels(names, expected, role)
        recovered = matches[0] if len(matches) == 1 else min(fallback, len(names) - 1)
        return recovered, f"Auto-selected {role} C{recovered} ({names[recovered]})"
    try:
        index = int(requested_text)
    except ValueError as error:
        raise ValueError(f"{role} channel must be a zero-based number") from error
    if 0 <= index < len(names):
        return index, None
    matches = matching_channels(names, expected, role)
    recovered = matches[0] if len(matches) == 1 else min(fallback, len(names) - 1)
    note = f"{role} channel {index} was unavailable; using C{recovered} ({names[recovered]})"
    return recovered, note


def select_analysis_channels(nuclei_requested: str | None,
                             actin_requested: str | None,
                             mf20_requested: str | None,
                             names: list[str]) -> tuple[int, int, int, list[str]]:
    """Resolve the three biological channels and prevent role collisions."""
    nuclei_channel, nuclei_note = select_channel(
        nuclei_requested, names, "DAPI", 0, "Nuclei"
    )
    actin_channel, actin_note = select_channel(
        actin_requested, names, "AF488", 1, "Actin"
    )
    mf20_channel, mf20_note = select_channel(
        mf20_requested, names, "AF647", len(names) - 1, "MF20"
    )
    if nuclei_channel == mf20_channel:
        dapi_matches = matching_channels(names, "DAPI", "Nuclei")
        alternatives = [index for index in dapi_matches if index != mf20_channel]
        if len(alternatives) == 1:
            previous = nuclei_channel
            nuclei_channel = alternatives[0]
            nuclei_note = (
                f"Nuclei C{previous} matched the MF20 channel; automatically corrected "
                f"to C{nuclei_channel} ({names[nuclei_channel]})"
            )
        else:
            raise ValueError(
                "Nuclei and MF20 cannot use the same channel. Leave the channel fields "
                "blank for automatic selection or enter different zero-based indices."
            )
    if len({nuclei_channel, actin_channel, mf20_channel}) != 3:
        raise ValueError(
            "Nuclei, Actin, and MF20 must use three different channels. Leave the "
            "channel fields blank for automatic selection or check the CZI metadata."
        )
    notes = [note for note in (nuclei_note, actin_note, mf20_note) if note]
    return nuclei_channel, actin_channel, mf20_channel, notes


def select_scene(requested: str | None, scenes: list[str]) -> tuple[str, str | None]:
    """Use the requested scene, or a predictable field when it is unavailable."""
    if not scenes:
        raise ValueError("No scenes were found in the CZI file")
    requested = requested or "Image:0"
    if requested in scenes:
        return requested, None
    recovered = "P1" if "P1" in scenes else scenes[0]
    note = (
        f"Scene {requested!r} was unavailable; using {recovered!r}. "
        f"Available scenes: {', '.join(scenes)}"
    )
    return recovered, note


def update_analysis_job(job_id: str, **values) -> None:
    with analysis_jobs_lock:
        if job_id in analysis_jobs:
            analysis_jobs[job_id].update(values)


def run_analysis_job(job_id: str) -> None:
    """Run one CZI analysis in the background while exposing stage updates."""
    with analysis_jobs_lock:
        job = dict(analysis_jobs[job_id])
    log_tail: list[str] = []
    try:
        update_analysis_job(
            job_id,
            status="running",
            message="Step 1/3: Segmenting nuclei and running MyoFuse classification",
        )
        process = subprocess.Popen(
            job["command"],
            env=job["environment"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip()
            if line:
                log_tail.append(line)
                log_tail = log_tail[-80:]
            if "Step 2/2:" in line:
                update_analysis_job(
                    job_id,
                    message=(
                        "Step 2/3: "
                        + line.split("Step 2/2:", 1)[-1].strip().capitalize()
                    ),
                )
        return_code = process.wait()
        if return_code:
            raise RuntimeError("\n".join(log_tail[-20:]) or "Automatic analysis failed")
        output = job["output"]
        workbooks = sorted(
            (output / "hybrid").glob("*__hybrid_nucleus_aware_results.xlsx")
        )
        update_analysis_job(job_id, message="Step 3/3: Loading the analysis results")
        entries = []
        for workbook in workbooks:
            stem = workbook.name.removesuffix(
                "__hybrid_nucleus_aware_results.xlsx"
            )
            label = output / "myofuse" / f"{stem}_mask.tif"
            myotube = output / "hybrid" / f"{stem}__hybrid_nucleus_aware_mask.tif"
            if not label.is_file():
                continue
            summary = pd.read_excel(workbook, sheet_name="Summary")
            scene = str(summary.iloc[0]["Scene"])
            entries.append({
                "scene": scene,
                "results": workbook,
                "labels": label,
                "myotube_mask": myotube if myotube.is_file() else None,
            })
        if not entries:
            raise RuntimeError("Automatic analysis completed but its result files could not be identified")
        update_analysis_job(
            job_id,
            status="ready",
            message="Step 3/3: Analysis complete; opening the review",
            entries=entries,
        )
    except Exception as error:
        update_analysis_job(job_id, status="error", message=str(error)[-2000:])


def open_project(
    results: Path,
    labels: Path,
    czi: Path | None,
    scene: str,
    cache: Path | None = None,
    myotube_mask: Path | None = None,
) -> None:
    global project, project_collection, active_project_key
    project = ReviewProject(
        results, labels, czi, scene,
        channel_cache=cache,
        myotube_mask_path=myotube_mask,
    )
    active_project_key = scene
    project_collection = {scene: project}


def open_project_collection(entries: list[dict], czi: Path) -> None:
    """Load every analyzed CZI scene and select the first one for review."""
    global project, project_collection, active_project_key
    loaded: dict[str, ReviewProject] = {}
    for entry in entries:
        scene = str(entry["scene"])
        loaded[scene] = ReviewProject(
            Path(entry["results"]),
            Path(entry["labels"]),
            czi,
            scene,
            myotube_mask_path=(
                Path(entry["myotube_mask"]) if entry.get("myotube_mask") else None
            ),
        )
    if not loaded:
        raise ValueError("No analyzed scenes were available to open")
    project_collection = loaded
    active_project_key = next(iter(loaded))
    project = loaded[active_project_key]


if ARGS.project:
    project = ReviewProject.from_saved_project(ARGS.project.expanduser().resolve())
    active_project_key = project.scene
    project_collection = {project.scene: project}
elif ARGS.results and ARGS.label_mask:
    open_project(
        ARGS.results, ARGS.label_mask, ARGS.czi, ARGS.scene,
        ARGS.channel_cache, ARGS.myotube_mask,
    )


def state_payload(selected: list[int] | None = None) -> dict:
    if project is None:
        return {"loaded": False}
    states = project.current.values()
    false_positive_corrections = sum(
        project.automatic.get(label_id) == IN and state == OUT
        for label_id, state in project.current.items()
    )
    false_negative_corrections = sum(
        project.automatic.get(label_id) == OUT and state == IN
        for label_id, state in project.current.items()
    )
    all_projects = list(project_collection.values()) or [project]
    aggregate_automatic_in = sum(item.automatic_in for item in all_projects)
    aggregate_automatic_total = sum(item.total_automatic for item in all_projects)
    aggregate_corrected_in = sum(item.corrected_in for item in all_projects)
    aggregate_valid_total = sum(item.valid_total for item in all_projects)
    return {
        "loaded": True,
        "filename": project.results_path.name,
        "title": project.results_path.stem.replace("__hybrid_nucleus_aware_results", ""),
        "scene": project.scene,
        "width": int(project.labels.shape[1]),
        "height": int(project.labels.shape[0]),
        "automatic_fi": project.automatic_fi,
        "corrected_fi": project.corrected_fi,
        "automatic_in": project.automatic_in,
        "automatic_total": project.total_automatic,
        "corrected_in": project.corrected_in,
        "valid_total": project.valid_total,
        "in_count": sum(v == IN for v in states),
        "out_count": sum(v == OUT for v in project.current.values()),
        "excluded_count": sum(v == EXCLUDE for v in project.current.values()),
        "changed_count": project.changed_count,
        "total_changed_count": sum(item.changed_count for item in all_projects),
        "false_positive_corrections": false_positive_corrections,
        "false_negative_corrections": false_negative_corrections,
        "review_candidate_count": sum(
            project.current.get(label_id) == OUT for label_id in project.review_candidates
        ),
        "centres": [
            {
                "id": i,
                "x": xy[0],
                "y": xy[1],
                "state": project.current[i],
                "review": i in project.review_candidates and project.current[i] == OUT,
                "added": i in project.added,
            }
            for i, xy in project.centres.items()
        ],
        "added_count": len(project.active_added),
        "added_in": sum(project.current[i] == IN for i in project.active_added),
        "typical_diameter": 2 * project.typical_radius,
        "default_export": str(project.results_path.parent / "manual_review"),
        "active_scene_key": active_project_key,
        "scene_options": [
            {
                "key": key,
                "scene": item.scene,
                "filename": item.results_path.name,
            }
            for key, item in project_collection.items()
        ],
        "aggregate_automatic_in": aggregate_automatic_in,
        "aggregate_automatic_total": aggregate_automatic_total,
        "aggregate_automatic_fi": (
            100 * aggregate_automatic_in / aggregate_automatic_total
            if aggregate_automatic_total else 0
        ),
        "aggregate_corrected_in": aggregate_corrected_in,
        "aggregate_valid_total": aggregate_valid_total,
        "aggregate_corrected_fi": (
            100 * aggregate_corrected_in / aggregate_valid_total
            if aggregate_valid_total else 0
        ),
        "selected": selected or [],
    }


@app.get("/")
def index():
    return HTML


@app.get("/api/state")
def api_state():
    return jsonify(state_payload())


@app.post("/api/switch-scene")
def api_switch_scene():
    global project, active_project_key
    key = str((request.get_json(silent=True) or {}).get("key", ""))
    if key not in project_collection:
        return jsonify(error=f"Unknown scene: {key}"), 404
    active_project_key = key
    project = project_collection[key]
    return jsonify(state_payload())


@app.get("/api/overlay")
def api_overlay():
    if project is None:
        return jsonify(error="No analysis loaded"), 400
    flag = lambda name, default="0": request.args.get(name, default) == "1"
    selected_text = request.args.get("selected", "")
    selected_id = int(selected_text) if selected_text.isdigit() else None
    image = project.render_overlay(
        show_dapi=flag("dapi"), show_actin=flag("actin"), show_mf20=flag("mf20", "1"),
        show_nuclei=flag("nuclei", "1"), show_myotube=flag("myotube", "1"),
        show_in=flag("classIn", "1"), show_out=flag("classOut", "1"),
        show_review=flag("classReview", "1"),
        show_ids=flag("ids"), selected_id=selected_id,
        display_gain=max(0.5, min(3.0, request.args.get("brightness", type=float) or 1.0)),
        pseudocolor=request.args.get("style", "color") != "gray",
    )
    output = io.BytesIO()
    Image.fromarray(image).save(output, format="PNG")
    output.seek(0)
    return send_file(output, mimetype="image/png", max_age=0)


@app.get("/api/nucleus-at")
def api_nucleus_at():
    """Return the segmentation label under an image-space coordinate."""
    if project is None:
        return jsonify(error="No analysis loaded"), 400
    x = request.args.get("x", type=float)
    y = request.args.get("y", type=float)
    if x is None or y is None:
        return jsonify(error="x and y are required"), 400
    column, row = int(round(x)), int(round(y))
    if row < 0 or row >= project.labels.shape[0] or column < 0 or column >= project.labels.shape[1]:
        return jsonify(id=0)
    label_id = int(project.labels[row, column])
    return jsonify(id=label_id if label_id in project.current else 0)


@app.post("/api/action")
def api_action():
    if project is None:
        return jsonify(error="No analysis loaded"), 400
    data = request.get_json(force=True)
    action = data.get("action")
    ids = [int(value) for value in data.get("ids", []) if int(value) in project.current]
    for label_id in ids:
        if action == "toggle":
            project.toggle(label_id)
        elif action == "reset":
            project.reset(label_id)
        elif action == "remove":
            project.remove_added(label_id)
        elif action in {"in", "out", "exclude"}:
            target = {"in": IN, "out": OUT, "exclude": EXCLUDE}[action]
            project.set_state(label_id, target, f"set {STATE_NAMES[target]}")
        else:
            return jsonify(error=f"Unknown action: {action}"), 400
    return jsonify(state_payload([i for i in ids if i in project.current]))


@app.post("/api/add-nucleus")
def api_add_nucleus():
    """Add a nucleus that automatic segmentation missed."""
    if project is None:
        return jsonify(error="No analysis loaded"), 400
    data = request.get_json(force=True)
    try:
        x, y = float(data["x"]), float(data["y"])
    except (KeyError, TypeError, ValueError):
        return jsonify(error="x and y are required"), 400
    state = {"in": IN, "out": OUT}.get(str(data.get("state", "in")).lower(), IN)
    diameter = data.get("diameter")
    radius = float(diameter) / 2 if diameter else None
    label_id = project.add_nucleus(x, y, state, radius)
    if label_id is None:
        return jsonify(error="There is already a nucleus here (or the click is outside the image)"), 400
    return jsonify(state_payload([label_id]))


@app.post("/api/undo")
def api_undo():
    if project is None:
        return jsonify(error="No analysis loaded"), 400
    label_id = project.undo()
    return jsonify(state_payload([label_id] if label_id in project.current else []))


@app.post("/api/redo")
def api_redo():
    if project is None:
        return jsonify(error="No analysis loaded"), 400
    label_id = project.redo()
    return jsonify(state_payload([label_id] if label_id in project.current else []))


@app.post("/api/export")
def api_export():
    if project is None:
        return jsonify(error="No analysis loaded"), 400
    output_dir = Path(request.get_json(force=True).get("output_dir", "")).expanduser()
    if not str(output_dir):
        return jsonify(error="Choose an output folder"), 400
    exports: dict[str, dict[str, Path]] = {}
    summary_rows = []
    for key, item in (project_collection or {project.scene: project}).items():
        exports[key] = item.export(output_dir)
        summary_rows.append({
            "Scene": item.scene,
            "Automatic nuclei In": item.automatic_in,
            "Automatic total nuclei": item.total_automatic,
            "Automatic FI (%)": item.automatic_fi,
            "Corrected nuclei In": item.corrected_in,
            "Corrected valid nuclei": item.valid_total,
            "Corrected FI (%)": item.corrected_fi,
            "Changed nuclei": item.changed_count,
            "Manually added nuclei": len(item.active_added),
        })
    response = {
        "scenes": {
            key: {name: str(path) for name, path in paths.items()}
            for key, paths in exports.items()
        }
    }
    if len(summary_rows) > 1:
        summary = pd.DataFrame(summary_rows)
        totals = {
            "Scene": "ALL (pooled nuclei)",
            "Automatic nuclei In": int(summary["Automatic nuclei In"].sum()),
            "Automatic total nuclei": int(summary["Automatic total nuclei"].sum()),
            "Corrected nuclei In": int(summary["Corrected nuclei In"].sum()),
            "Corrected valid nuclei": int(summary["Corrected valid nuclei"].sum()),
            "Changed nuclei": int(summary["Changed nuclei"].sum()),
            "Manually added nuclei": int(summary["Manually added nuclei"].sum()),
        }
        totals["Automatic FI (%)"] = (
            100 * totals["Automatic nuclei In"] / totals["Automatic total nuclei"]
            if totals["Automatic total nuclei"] else 0
        )
        totals["Corrected FI (%)"] = (
            100 * totals["Corrected nuclei In"] / totals["Corrected valid nuclei"]
            if totals["Corrected valid nuclei"] else 0
        )
        summary = pd.concat([summary, pd.DataFrame([totals])], ignore_index=True)
        summary_path = output_dir / "all_scenes_manual_review_summary.xlsx"
        summary.to_excel(summary_path, index=False)
        response["workbook"] = str(summary_path)
    else:
        response.update({name: str(path) for name, path in next(iter(exports.values())).items()})
    return jsonify(response)


@app.post("/api/select-export-folder")
def api_select_export_folder():
    """Open the native macOS folder chooser for this local-only application."""
    if project is None:
        return jsonify(error="No analysis loaded"), 400
    data = request.get_json(silent=True) or {}
    requested = str(data.get("initial_dir", "")).strip()
    initial = Path(requested).expanduser() if requested else project.results_path.parent
    while not initial.exists() and initial != initial.parent:
        initial = initial.parent
    script = [
        "on run argv",
        'set chosenFolder to choose folder with prompt "Choose export folder" '
        "default location (POSIX file (item 1 of argv))",
        "return POSIX path of chosenFolder",
        "end run",
    ]
    command = ["/usr/bin/osascript"]
    for line in script:
        command.extend(["-e", line])
    command.append(str(initial.resolve()))
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        detail = (completed.stderr or "").strip()
        if "User canceled" in detail or "(-128)" in detail:
            return jsonify(cancelled=True)
        return jsonify(error=detail or "Could not open the folder chooser"), 500
    folder = completed.stdout.strip()
    if not folder:
        return jsonify(cancelled=True)
    return jsonify(cancelled=False, folder=str(Path(folder).resolve()))


@app.post("/api/load")
def api_load():
    try:
        data = request.get_json(force=True)
        czi = Path(data["czi"]).expanduser() if data.get("czi", "").strip() else None
        myotube_mask = Path(data["myotube_mask"]).expanduser() if data.get("myotube_mask", "").strip() else None
        open_project(
            Path(data["results"]).expanduser(), Path(data["labels"]).expanduser(),
            czi, data.get("scene") or "Image:0", myotube_mask=myotube_mask,
        )
        return jsonify(state_payload())
    except Exception as error:
        return jsonify(error=str(error)), 400


@app.post("/api/upload")
def api_upload():
    """Receive files from the local browser and open one processed field."""
    try:
        uploaded = [item for item in request.files.getlist("files") if item.filename]
        czi_path = (request.form.get("czi_path") or "").strip()
        saved: list[Path] = []
        if not uploaded and czi_path:
            local = Path(czi_path).expanduser()
            if local.suffix.lower() != ".czi" or not local.is_file():
                raise ValueError(f"CZI file not found: {local}")
            saved.append(local.resolve())
        elif not uploaded:
            raise ValueError("No files were selected")
        folder = Path(tempfile.mkdtemp(prefix="review-", dir=upload_root))
        for item in uploaded:
            name = secure_filename(Path(item.filename).name)
            if not name:
                raise ValueError(f"Unsupported filename: {item.filename}")
            destination = folder / name
            item.save(destination)
            saved.append(destination)

        workbooks = [path for path in saved if path.suffix.lower() == ".xlsx"]
        czis = [path for path in saved if path.suffix.lower() == ".czi"]
        tiffs = [path for path in saved if path.suffix.lower() in {".tif", ".tiff"}]
        hybrid_masks = [path for path in tiffs if "hybrid_nucleus_aware_mask" in path.stem.lower()]
        nucleus_masks = [path for path in tiffs if "mask" in path.stem.lower() and path not in hybrid_masks]

        if len(saved) == 1 and len(czis) == 1:
            analysis_python = os.environ.get("MYOFUSE_ANALYSIS_PYTHON")
            if not analysis_python or not Path(analysis_python).is_file():
                raise ValueError("The CZI analysis environment is unavailable; run setup_mac.sh first")
            channel_names, scene_names, overview_scenes = inspect_czi_metadata(
                czis[0], analysis_python
            )
            requested_scene = (request.form.get("scene") or "ALL").strip()
            if requested_scene.casefold() in {"auto", "automatic"}:
                if len(overview_scenes) == 1:
                    scene = overview_scenes[0]
                    scene_note = (
                        f"Auto-selected stitched overview {scene}; separate fields "
                        "are not pooled, preventing duplicate counting"
                    )
                else:
                    scene = "ALL"
                    scene_note = (
                        f"No unique stitched overview was found; analyzing all "
                        f"{len(scene_names) - len(overview_scenes)} separate fields"
                    )
            elif requested_scene.casefold() in {"all", "all scenes", "*"}:
                scene = "ALL"
                fields = [name for name in scene_names if name not in overview_scenes]
                scene_note = (
                    f"Analyzing {len(fields)} separate fields: {', '.join(fields)}. "
                    f"Excluded duplicate stitched overview(s): "
                    f"{', '.join(overview_scenes) or 'none'}"
                )
            else:
                scene, scene_note = select_scene(requested_scene, scene_names)
            nuclei_channel, actin_channel, mf20_channel, channel_notes = (
                select_analysis_channels(
                    request.form.get("nuclei_channel"),
                    request.form.get("actin_channel"),
                    request.form.get("mf20_channel"),
                    channel_names,
                )
            )
            output_root = Path.home() / "Documents" / "MyoFuse-Hybrid-Analyses"
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output = output_root / f"{czis[0].stem}-{timestamp}"
            output.mkdir(parents=True, exist_ok=False)
            analysis_mode = (request.form.get("analysis_mode") or "actin_structure").strip()
            if analysis_mode != "actin_structure":
                raise ValueError("This manuscript release supports Actin-structure analysis only.")
            command = [
                str(Path(__file__).resolve().parent / "run_hybrid_pipeline.sh"),
                str(czis[0]), str(output), scene,
                str(nuclei_channel), str(actin_channel), str(mf20_channel),
                (
                    str(ACTIN_STRUCTURE_PROBABILITY_GUARD)
                    if analysis_mode == "actin_structure" else "0"
                ),
                "0", "0", "0",
                ("1" if analysis_mode == "actin_structure" else "0"),
                "0",
            ]
            notes = [note for note in [scene_note, *channel_notes] if note]
            notes.append(
                "Actin-structure manuscript method: aligned continuous Actin and MF20 "
                "connect nearby nuclei. Hybrid-only nuclei below MyoFuse probability "
                f"{ACTIN_STRUCTURE_PROBABILITY_GUARD:.3f} are treated as Out. "
                "Low-MyoFuse images use stricter anchors and send low-support nuclei "
                "to yellow review"
            )
            environment = os.environ.copy()
            environment["PYTHON_BIN"] = analysis_python
            job_id = uuid.uuid4().hex
            with analysis_jobs_lock:
                analysis_jobs[job_id] = {
                    "status": "queued",
                    "message": "Preparing the analysis",
                    "command": command,
                    "environment": environment,
                    "output": output,
                    "czi": czis[0],
                    "scene": scene,
                    "notice": "; ".join(notes),
                    "activated": False,
                    "analysis_python": analysis_python,
                }
            threading.Thread(target=run_analysis_job, args=(job_id,), daemon=True).start()
            return jsonify(job_id=job_id, status="queued", message="Preparing the analysis"), 202

        if len(workbooks) != 1:
            raise ValueError("Select exactly one hybrid results .xlsx workbook")
        if len(nucleus_masks) == 1:
            label_mask = nucleus_masks[0]
        elif len(tiffs) == 1 and not hybrid_masks:
            label_mask = tiffs[0]
        else:
            raise ValueError("Select exactly one MyoFuse nucleus mask .tif file")
        if len(hybrid_masks) > 1:
            raise ValueError("Select no more than one hybrid myotube mask .tif file")
        if len(czis) > 1:
            raise ValueError("Select no more than one original .czi file")
        open_project(
            workbooks[0], label_mask, czis[0] if czis else None,
            request.form.get("scene") or "Image:0",
            myotube_mask=hybrid_masks[0] if hybrid_masks else None,
        )
        return jsonify(state_payload())
    except Exception as error:
        return jsonify(error=str(error)), 400


@app.get("/api/analysis-status/<job_id>")
def api_analysis_status(job_id: str):
    with analysis_jobs_lock:
        stored = analysis_jobs.get(job_id)
        if stored is None:
            return jsonify(error="Analysis job not found"), 404
        job = dict(stored)
    if job["status"] == "ready" and not job.get("activated"):
        try:
            open_project_collection(job["entries"], job["czi"])
        except Exception as error:
            update_analysis_job(job_id, status="error", message=str(error))
            return jsonify(status="error", message=str(error))
        update_analysis_job(job_id, status="complete", activated=True)
        job["status"] = "complete"
    return jsonify(
        status=job["status"],
        message=job["message"],
        analysis_notice=job.get("notice", ""),
    )


@app.post("/api/close")
def api_close():
    global project, project_collection, active_project_key
    project = None
    project_collection = {}
    active_project_key = None
    return jsonify({"loaded": False})


def main() -> None:
    if ARGS.open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{ARGS.port}")).start()
    app.run(host="127.0.0.1", port=ARGS.port, debug=False, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
