const MODELS = [
  { key:'qwen35', name:'Qwen3.5-9B', color:'#c8f04b' },
  { key:'qwen3vl', name:'Qwen3-VL-8B', color:'#5b7cfa' },
  { key:'claude', name:'Claude Opus 5', color:'#ff5b2e' },
  { key:'gpt', name:'GPT-5.2', color:'#a66cff' }
];

const DATA = {
  chess: {
    kicker:'Sequence reconstruction · n = 140', title:'Can a model remember the board?',
    description:'Models reconstruct every chess move shown in clips of increasing duration.', conditions:['5','10','30','60','120','300','600'], xValues:[5,10,30,60,120,300,600], xUnit:'target clip duration (seconds)',
    conditionMeta:[5,10,30,60,120,300,600].map(seconds=>({mode:'video',targetSeconds:seconds,n:20})),
    metrics:{
      strict:{label:'Strict accuracy',direction:'higher',overall:[.0428,.0124,.5377,.3896],series:[[.1500,.1100,.0225,.0133,.0021,.0014,.0001],[.075,.01,0,.0011,.0005,.0002,0],[1,1,1,.7533,.01,.0004,0],[.95,.94,.485,.2256,.1126,.0106,.0034]]},
      loose:{label:'Loose accuracy',direction:'higher',overall:[.1566,.0556,.5505,.5890]},
      hybrid:{label:'Hybrid accuracy',direction:'higher',overall:[.0564,.0127,.5496,.5316]},
      moves:{label:'Moves matched',direction:'higher',overall:[4.3357,2.5929,9.7429,26.95]}
    }, insight:'Accuracy collapses with duration. <strong>Claude is perfect through 30 seconds</strong>, while GPT retains more signal across the longest clips.'
  },
  asl: {
    kicker:'Continuous translation · n = 130',title:'Can a model read motion as language?',description:'Models translate ASL clips grouped by source duration.',conditions:['3','5','10','15','20','25','30'],xValues:[3,5,10,15,20,25,30],xUnit:'duration bucket upper bound (seconds)',
    conditionMeta:[{range:'0–3 s',n:20},{range:'3–5 s',n:20},{range:'5–10 s',n:20},{range:'10–15 s',n:20},{range:'15–20 s',n:20},{range:'20–25 s',n:20},{range:'25–30 s',n:10}],
    metrics:{
      f1:{label:'Token F1',direction:'higher',overall:[.0377,.0648,.1020,.1198],series:[[.0332,.0425,.0391,.0326,.0346,.037,.0492],[.0276,.0749,.0813,.0748,.0706,.0558,.0698],[.0522,.0808,.1091,.1108,.112,.1301,.1345],[.0806,.088,.1153,.125,.1511,.1407,.1535]]},
      bleu:{label:'BLEU',direction:'higher',overall:[.0109,.0218,.0239,.032]},
      similarity:{label:'Char similarity',direction:'higher',overall:[.111,.197,.2143,.226]},
      wer:{label:'Word error rate',direction:'lower',overall:[19.0615,6.3803,3.6017,2.6549]}
    }, insight:'All systems remain challenged. <strong>GPT leads on every aggregate translation metric</strong>, but even its token F1 is only 0.12.'
  },
  wbw: {
    kicker:'Temporal presentation · n = 140',title:'What if words do not wait?',description:'Models answer multiple-choice questions as words accumulate or vanish at three speeds.',conditions:['Static','0.5','2','5','0.5','2','5'],xUnit:'presentation mode · words/second · observed average length',
    tickDetails:['image','81.7 s','20.4 s','8.2 s','87.1 s','25.9 s','13.6 s'],
    conditionMeta:[
      {mode:'static image',n:20,words:'18–93 (40.85 avg)'},
      {mode:'cumulative',wps:.5,avgSeconds:81.7,range:'36–186 s',n:20,words:'18–93 (40.85 avg)'},
      {mode:'cumulative',wps:2,avgSeconds:20.425,range:'9–46.5 s',n:20,words:'18–93 (40.85 avg)'},
      {mode:'cumulative',wps:5,avgSeconds:8.17,range:'3.6–18.6 s',n:20,words:'18–93 (40.85 avg)'},
      {mode:'vanishing',wps:.5,avgSeconds:87.1466,range:'38.4–198.4 s',n:20,words:'18–93 (40.85 avg)',gap:'4 frames/word'},
      {mode:'vanishing',wps:2,avgSeconds:25.8716,range:'11.4–58.9 s',n:20,words:'18–93 (40.85 avg)',gap:'4 frames/word'},
      {mode:'vanishing',wps:5,avgSeconds:13.6166,range:'6–31 s',n:20,words:'18–93 (40.85 avg)',gap:'4 frames/word'}
    ],
    metrics:{
      correct:{label:'Accuracy',direction:'higher',overall:[.5714,.5143,1,.5429],series:[[.8,.55,.65,.75,.6,.35,.3],[.65,.6,.6,.6,.4,.45,.3],[1,1,1,1,1,1,1],[.85,.55,.65,.65,.25,.45,.4]]},
      answered:{label:'Answered',direction:'higher',overall:[1,1,1,1]},
      refusal:{label:'Refusal rate',direction:'lower',overall:[0,0,0,0]}
    }, insight:'Vanishing text hurts most open models. <strong>Claude remains at 100% in every presentation condition</strong> in this evaluation.'
  }
};

let activeBenchmark='chess', activeMetric='strict';
const $=s=>document.querySelector(s);
const fmt=(v,metric)=> metric==='moves' ? v.toFixed(2) : v.toFixed(4);

function render(){
  const task=DATA[activeBenchmark], metric=task.metrics[activeMetric];
  $('#task-kicker').textContent=task.kicker; $('#task-title').textContent=task.title; $('#task-description').textContent=task.description;
  $('#sample-count').textContent=task.kicker.split('·')[1]?.trim()||''; $('#direction-label').textContent=metric.direction==='lower'?'↓ lower is better':'↑ higher is better';
  $('#chart-title').textContent=`${metric.label} by condition`; $('#insight').innerHTML=task.insight;
  $('#metric-select').innerHTML=Object.entries(task.metrics).map(([k,m])=>`<option value="${k}" ${k===activeMetric?'selected':''}>${m.label}</option>`).join('');
  $('#graph-note').textContent=activeBenchmark==='wbw'
    ? 'Durations are observed across all 20 matched questions (18–93 words). Vanishing clips include a 4-frame blank gap after each word.'
    : activeBenchmark==='asl'
      ? 'X values are the numeric upper bounds of half-open source-duration buckets; the final bucket contains 10 matched clips.'
      : 'X positions use the configured target duration in seconds; each condition contains 20 matched games.';
  renderChart(task,metric); renderRanking(metric); renderMetricStrip(task);
}

function renderChart(task,metric){
  const svg=$('#line-chart'), W=850,H=390,p={l:58,r:25,t:24,b:52};
  let series=metric.series;
  if(!series){ series=MODELS.map((_,i)=>task.conditions.map(()=>metric.overall[i])); }
  const vals=series.flat(), rawMax=Math.max(...vals), rawMin=Math.min(...vals);
  const min=0, max=rawMax<=1?Math.max(1,rawMax):rawMax*1.08;
  const xMin=task.xValues?.[0], xMax=task.xValues?.at(-1);
  const x=i=>task.xValues
    ? p.l+(task.xValues[i]-xMin)/(xMax-xMin)*(W-p.l-p.r)
    : p.l+i*(W-p.l-p.r)/(task.conditions.length-1);
  const y=v=>p.t+(max-v)/(max-min)*(H-p.t-p.b);
  let html='';
  for(let i=0;i<5;i++){const v=min+(max-min)*(4-i)/4,yy=y(v);html+=`<line class="grid-line" x1="${p.l}" x2="${W-p.r}" y1="${yy}" y2="${yy}"/><text class="axis-label" x="${p.l-10}" y="${yy+4}" text-anchor="end">${max<=1?v.toFixed(2):v.toFixed(1)}</text>`}
  task.conditions.forEach((c,i)=>{
    html+=`<text class="axis-label" x="${x(i)}" y="${H-35}" text-anchor="middle">${c}`;
    if(task.tickDetails) html+=`<tspan class="axis-detail" x="${x(i)}" dy="12">${task.tickDetails[i]}</tspan>`;
    html+='</text>';
  });
  html+=`<text class="axis-title" x="${(p.l+W-p.r)/2}" y="${H-8}" text-anchor="middle">${task.xUnit}</text>`;
  series.forEach((arr,mi)=>{const path=arr.map((v,i)=>`${i?'L':'M'}${x(i)},${y(v)}`).join(' ');html+=`<path class="line" stroke="${MODELS[mi].color}" d="${path}"/>`;arr.forEach((v,i)=>html+=`<circle class="point" data-index="${i}" data-model="${MODELS[mi].name}" data-condition="${task.conditions[i]}" data-value="${v}" fill="${MODELS[mi].color}" cx="${x(i)}" cy="${y(v)}" r="5"/>`)});
  svg.innerHTML=html;
  $('#legend').innerHTML=MODELS.map(m=>`<span><i style="background:${m.color}"></i>${m.name}</span>`).join('');
  svg.querySelectorAll('.point').forEach(pt=>{pt.addEventListener('mouseenter',e=>showTip(e,pt,metric));pt.addEventListener('mouseleave',()=>$('#tooltip').hidden=true)});
}
function showTip(e,pt,metric){
  const tip=$('#tooltip'),rect=$('.chart-wrap').getBoundingClientRect(),meta=DATA[activeBenchmark].conditionMeta?.[+pt.dataset.index]||{};
  const details=[meta.mode,meta.wps!=null?`${meta.wps} words/s`:null,meta.avgSeconds!=null?`${meta.avgSeconds.toFixed(1)} s average`:null,meta.range?`${meta.range} range`:null,meta.targetSeconds!=null?`${meta.targetSeconds} s target`:null,meta.n?`n = ${meta.n}`:null,meta.words?`${meta.words} words`:null,meta.gap].filter(Boolean);
  tip.innerHTML=`<b>${pt.dataset.model}</b><br>${metric.label}: ${fmt(+pt.dataset.value,activeMetric)}${details.length?`<hr>${details.join('<br>')}`:''}`;tip.hidden=false;tip.style.left=`${e.clientX-rect.left+10}px`;tip.style.top=`${e.clientY-rect.top-70}px`;
}
function renderRanking(metric){
  const ranked=MODELS.map((m,i)=>({...m,value:metric.overall[i]})).sort((a,b)=>metric.direction==='lower'?a.value-b.value:b.value-a.value), max=Math.max(...metric.overall)||1;
  $('#ranking').innerHTML=ranked.map((m,i)=>`<div class="rank-row"><div class="rank-top"><span class="rank-index">0${i+1}</span><span class="rank-name">${m.name}</span><span class="rank-score">${fmt(m.value,activeMetric)}</span></div><div class="rank-track"><div class="rank-fill" style="width:${m.value/max*100}%;background:${m.color}"></div></div></div>`).join('');
}
function renderMetricStrip(task){
  $('#metric-strip').innerHTML=Object.entries(task.metrics).map(([k,m])=>{const best=m.direction==='lower'?Math.min(...m.overall):Math.max(...m.overall);return `<div class="metric-tile ${k===activeMetric?'active':''}" data-metric="${k}"><span>${m.label}</span><strong>${fmt(best,k)}</strong></div>`}).join('');
  document.querySelectorAll('.metric-tile').forEach(el=>el.onclick=()=>{activeMetric=el.dataset.metric;render()});
}

document.querySelectorAll('.tab').forEach(tab=>tab.onclick=()=>{document.querySelectorAll('.tab').forEach(t=>{t.classList.remove('active');t.setAttribute('aria-selected','false')});tab.classList.add('active');tab.setAttribute('aria-selected','true');activeBenchmark=tab.dataset.benchmark;activeMetric=Object.keys(DATA[activeBenchmark].metrics)[0];render()});
$('#metric-select').addEventListener('change',e=>{activeMetric=e.target.value;render()});
render();
