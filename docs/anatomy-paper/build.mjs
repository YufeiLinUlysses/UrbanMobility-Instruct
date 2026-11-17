import {writeFileSync} from 'node:fs';
import * as k from './kit/iso-kit.mjs';

// Concept: a document inspection bench. Five source files feed a scanner;
// a sliding comparator measures verification accuracy on a full 0–100 scale.
// Parts: 4 feet, base + inset, 5 trays × (floor, 2 walls, back, 3 sheets,
// tab), scanner bed, 2 rails, 2 stops, 2 pillars, lintel, 8 vent slots,
// 4 mount pads, ruler strip, 21 ticks and sliding comparator shoe.
const P=k.fitProjection(k.boxCorners({x:-170,y:-80,w:340,d:180,r:4},-16,90),900,480,{pad:42,azimuth:38});
const slab=(x,y,w,d,z,h,tone='mid',lit=false)=>k.solidSvg(k.slabOf({x,y,w,d,r:1.4},z,h,P,4,.5),{tone,lit});
const line=(plan,z)=>k.lineSvg(k.planOutline(plan,z,P),{tone:'faint'});
let b='';
for(const x of [-148,138])for(const y of [-63,78])b+=slab(x,y,13,13,-16,7,'lo');
b+=`<path class="iso-halo" d="${k.haloOf({x:-170,y:-80,w:340,d:180,r:7},-16,13,P)}"/>`;
b+=slab(-170,-80,340,180,-9,9)+line({x:-164,y:-74,w:328,d:168,r:4},.1);
// Archive rack: the five sources are physical files, not free-floating layers.
for(let i=0;i<5;i++){
 const x=-150+i*60;
 b+=slab(x,-66,48,47,0,3,'lo');
 b+=slab(x,-66,48,3,3,24)+slab(x,-63,3,44,3,24);
 for(let j=0;j<3;j++){
  b+=slab(x+5,-60+j*2,36,33,4+j*5,2,'hi');
  b+=line({x:x+9,y:-55+j*2,w:27,d:2,r:.2},6.1+j*5);
 }
 b+=slab(x+34,-59,7,9,16,2,'hi')+slab(x+45,-63,3,44,3,24);
}
// Inspection platen and guide rails, with a fixed evidence sheet under the head.
b+=slab(-145,-8,286,57,0,4,'lo');
for(const y of [-5,41])b+=slab(-137,y,270,4,4,3,'hi');
for(const x of [-142,130])b+=slab(x,-9,7,60,4,8);
b+=slab(-61,3,94,32,4,1.2,'hi');
for(let i=0;i<5;i++)b+=line({x:-52,y:7+i*5,w:65-i*5,d:1,r:.2},5.3);
// Gantry pads and square uprights avoid artificial decorative piping.
for(const y of [-16,47]){
 b+=slab(48,y,23,15,0,4,'lo')+slab(54,y+3,11,9,4,67);
 b+=line({x:56,y:y+5,w:7,d:5,r:.5},71.1);
}
b+=slab(48,-13,23,72,71,13,'hi');
for(let i=0;i<8;i++)b+=slab(52,-7+i*7,15,2,84,.3,'lo');
b+=slab(48,6,23,24,66,5,'lo')+slab(50,8,19,20,65,1,'hi',true);
// Full-scale comparator: the shoe position is the selected verdict accuracy.
b+=slab(-146,71,280,13,0,2,'lo');
for(let i=0;i<=20;i++) b+=line({x:-140+i*13.4,y:71,w:.45,d:i%5===0?9:4,r:.1},2.2);
b+=`<g id="shoe">${slab(-144,69,8,17,2.4,5,'hi',true)}</g>`;
const svg=k.figureSvg({width:900,height:480,label:'Isometric document inspection bench: five source trays, an evidence scanner, and a sliding verification comparator.',body:b});
const data={
 names:['T5-base','Qwen2.5-7B','Qwen2.5-14B','Llama 3.1-8B','Llama 3.1-70B'],
 regimes:['Held-out concepts','France & Japan','Ten unseen datasets'],
 // Per model: base [normalized MC %, verification %], tuned [same].
 scores:[
 [[[1.20,0],[63.42,54.80]],[[87.71,98.79],[100,98.50]],[[87.41,99.04],[100,99.04]],[[100,89.19],[100,99.96]],[[100,98.16],[100,98.54]]],
 [[[5.31,0],[73.68,54.97]],[[83.53,97.11],[99.18,99.08]],[[74.59,98.69],[99.67,99.74]],[[95.88,92.35],[99.84,98.36]],[[94.48,99.08],[99.92,99.38]]],
 [[[11.33,0],[74,54]],[[90,90.86],[94,91.43]],[[95.33,92.43],[96.67,90.29]],[[88.67,67.43],[90.67,89]],[[97.33,92],[98,92.71]]]
 ],
 notes:[
 'Related examples stay together: 106 concept groups are partitioned into 85 training, 11 validation and 10 test groups. The in-domain test therefore has limited independent concept coverage.',
 'France and Japan are geographically held out, but retain the familiar task interface and controlled augmentation. High scores here do not establish unrestricted geographic generalization.',
 'Ten independent datasets contribute 350 authored questions and 700 balanced verification instances, without paraphrase descendants. Broader source variation reveals transfer limits.'
 ]
};
const origin=k.iso([0,0,0],P), end=k.iso([268,0,0],P);
const live=`const D=${JSON.stringify(data)};const axis=${JSON.stringify([end[0]-origin[0],end[1]-origin[1]])};
const $=s=>document.querySelector(s), controls=['#model','#regime','#stage'];
let target=99.96,current=target,visible=true,frame=0,last=0,tour=false,timer;
const reduced=matchMedia('(prefers-reduced-motion: reduce)');
function paint(){ $('#shoe').setAttribute('transform','translate('+axis[0]*current/100+' '+axis[1]*current/100+')');$('#readout').textContent='Comparator · '+current.toFixed(2)+'%';}
function animate(now){frame=0;const dt=Math.min((now-last)/1000||.016,.04);last=now;current=reduced.matches?target:current+(target-current)*(1-Math.exp(-dt/.23));if(Math.abs(current-target)<.005)current=target;paint();if(visible&&current!==target)frame=requestAnimationFrame(animate);}
function run(){if(visible&&!frame){last=0;frame=requestAnimationFrame(animate)}}
function update(){const m=+$('#model').value,r=+$('#regime').value,s=+$('#stage').value;const [mc,v]=D.scores[r][m][s];target=v;$('#mc').textContent=mc.toFixed(2)+'%';$('#verdict').textContent=v.toFixed(2)+'%';$('#context').textContent=D.notes[r];$('#selection').textContent=D.names[m]+' · '+(s?'Fine-tuned':'Base')+' · '+D.regimes[r];$('#delta').textContent=(D.scores[r][m][1][1]-D.scores[r][m][0][1]).toFixed(2)+' percentage points';run();}
function stop(){tour=false;clearInterval(timer);$('#tour').textContent='Play comparison';$('#tour').setAttribute('aria-pressed','false')}
controls.forEach(s=>$(s).addEventListener('change',()=>{stop();update()}));
$('#tour').onclick=()=>{if(tour){stop();return}tour=true;$('#tour').textContent='Pause comparison';$('#tour').setAttribute('aria-pressed','true');timer=setInterval(()=>{if(!visible||document.hidden)return;$('#regime').value=(+$('#regime').value+1)%3;update()},3500)};
$('#theme').onclick=()=>{document.body.dataset.theme=document.body.dataset.theme==='light'?'dark':'light';$('#theme').textContent=document.body.dataset.theme==='light'?'Dark theme':'Light theme'};
new IntersectionObserver(es=>{visible=es[es.length-1].isIntersecting;if(visible)run()}).observe($('.iso-plate'));
reduced.addEventListener('change',()=>{if(reduced.matches)stop();run()});
update();paint();`;
const css=`${k.ISO_CSS}
*{box-sizing:border-box}body{margin:0;background:#101413;color:#e6ebe7;font:16px/1.6 system-ui,sans-serif}body[data-theme=light]{background:#f5f3ed;color:#23342f}main{max-width:1140px;margin:auto;padding:48px 32px 64px}.eyebrow{font:11px/1.4 monospace;letter-spacing:2px;text-transform:uppercase;color:#6ca18a}header{display:flex;gap:30px;justify-content:space-between;align-items:start}h1{font-size:clamp(32px,5vw,56px);letter-spacing:-2px;line-height:1.08;max-width:800px;margin:17px 0}h2{font-size:20px;line-height:1.3;margin:0 0 12px}p{margin:8px 0 18px}.intro{max-width:770px;opacity:.8;font-size:18px}.iso-plate{border:1px solid #6e81754d;border-radius:16px;margin-top:28px;overflow:hidden;background:#8a9a7d06}.corners{display:flex;justify-content:space-between;gap:16px;padding:16px 22px;font:11px/1.5 monospace;text-transform:uppercase;letter-spacing:1px}.iso-svg{display:block;width:100%;height:auto}.controls{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin:22px 0}label{font-size:12px;display:block}select,button{font:inherit;color:inherit;background:transparent;border:1px solid #7e938970;border-radius:7px;padding:9px 12px}select{display:block;width:100%;margin-top:6px}option{color:#17251e;background:#f5f3ed}button{cursor:pointer;white-space:nowrap;font-size:12px}button:hover{background:#739d8020}:focus-visible{outline:2px solid #4caf87;outline-offset:3px}.results{display:grid;grid-template-columns:1fr 1fr 1.5fr;gap:24px;padding:24px 0;border-top:1px solid #7e938940;border-bottom:1px solid #7e938940}.number{font-size:40px;letter-spacing:-1px;line-height:1.4;color:#6ca18a}.label{font:11px monospace;text-transform:uppercase;letter-spacing:1px}.small{font-size:13px;opacity:.72}.story{display:grid;grid-template-columns:repeat(3,1fr);gap:32px;margin:38px 0}.story strong{color:#6ca18a}.note{padding:22px;border-left:2px solid #6ca18a;background:#83987a0a}.footer{font-size:12px;opacity:.65;margin-top:30px}a{color:inherit}#selection{font-size:13px;margin:22px 0 10px}#context{font-size:14px} @media(max-width:600px){main{padding:26px 18px}header{display:block}h1{letter-spacing:-1px}.controls{grid-template-columns:1fr}.results{grid-template-columns:1fr 1fr;gap:16px}.results>div:last-child{grid-column:1/-1}.story{grid-template-columns:1fr;gap:16px}.corners{padding:12px;font-size:9px;letter-spacing:0}.number{font-size:33px}.intro{font-size:16px}}`;
for(const theme of ['light','dark']){
 const html=`<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>UrbanMobility-Instruct — an interactive explanation</title><style>${css} .iso,.iso[data-theme="light"]{--anatomy-lit-top:#92bea6;--anatomy-lit-shade:#537c67;--anatomy-lit:#36654d}.iso-plate{position:relative;padding:0;box-shadow:none}.iso-svg{aspect-ratio:900/480}</style><body class="iso" data-theme="${theme}"><main>
 <header><div><div class="eyebrow">Paper anatomy / UrbanMobility-Instruct</div><h1>Can a model read<br>a dataset correctly?</h1></div><button id="theme">${theme==='light'?'Dark':'Light'} theme</button></header>
 <p class="intro">Transportation data comes with a rulebook: where it was collected, what each variable means, and which conclusions it supports. Your paper turns that rulebook into training tasks—and tests whether models learn to interpret it.</p>
 <div class="iso-plate"><div class="corners"><span>Fig 01 / Evidence inspection bench</span><span>Source → task → evaluation</span></div>${svg}<div class="corners"><span>Choose a model and test setting below</span><span id="readout" data-readout></span></div></div>
 <p class="small">Five file trays = highD, NGSIM, Road Networks, NYC TLC and the urban-flow literature source. The scanner represents the shared task interface. The green comparator moves on a full 0–100% verification scale; this bench is an explanatory metaphor.</p>
 <div class="controls"><label>Model<select id="model">${data.names.map((n,i)=>`<option value="${i}" ${i===3?'selected':''}>${n}</option>`).join('')}</select></label><label>Evaluation setting<select id="regime">${data.regimes.map((n,i)=>`<option value="${i}">${n}</option>`).join('')}</select></label><label>Checkpoint<select id="stage"><option value="0">Base</option><option value="1" selected>Fine-tuned</option></select></label></div>
 <button id="tour" aria-pressed="false">Play comparison</button><p id="selection" aria-live="polite"></p>
 <div class="results"><div><div class="label">Answer selection</div><div class="number" id="mc"></div><div class="small">Normalized multiple-choice accuracy</div></div><div><div class="label">Candidate verification</div><div class="number" id="verdict"></div><div class="small">Verdict accuracy</div></div><div><div class="label">Effect of tuning on verification</div><p id="delta"></p><p class="small">Fine-tuned minus base, within the selected model and setting. A negative value means accuracy fell.</p></div></div><p id="context"></p>
 <div class="story"><section><div class="eyebrow">01 / Construct</div><h2>Turn documentation into supervision.</h2><p>Four empirical sources and one literature-derived source supply the evidence. Questions progress from <strong>direct comprehension</strong>, through <strong>multi-attribute reasoning</strong>, to <strong>cross-dimensional interpretation</strong>.</p><p class="small">The focus is interpreting supplied descriptions, rather than computing directly over raw trajectories.</p></section><section><div class="eyebrow">02 / Teach</div><h2>Ask, challenge, rephrase.</h2><p><strong>Generate</strong> an answer. <strong>Verify</strong> whether a candidate is supported. <strong>Paraphrase</strong> while retaining required structure and answers.</p><p class="small">Incorrect candidates belong in verification inputs, not generation targets. Related concepts and their descendants stay in the same split.</p></section><section><div class="eyebrow">03 / Test</div><h2>Change the source, check the claim.</h2><p>Five backbones × two checkpoints × three settings produce <strong>30 evaluation combinations</strong>. The broader external benchmark reveals weaknesses that the regional test can conceal.</p><p class="small">Try fine-tuned Llama 70B: verification falls from 99.38% on France/Japan to 92.71% on the ten-dataset benchmark.</p></section></div>
 <section class="note"><h2>The finding to remember</h2><p>Fine-tuning improves structured output compliance, short-answer reference overlap and automated paraphrase preservation. Correctness gains vary by task and model. The 8B Llama leads in-domain verification; the 70B leads the external multiple-choice and verification aggregates.</p><p><strong>Fluent answers, correct answers and reference-like answers are different things.</strong> Both Llama models already have 100% normalized in-domain multiple-choice accuracy before tuning. Low similarity can also reflect valid wording or mismatched scoring targets.</p><p class="small">Limits: only ten held-out in-domain concept groups; one training run per model; uneven source coverage by reasoning level; a different prequantized 70B base; and a small, provisional qualitative audit. These experiments do not establish general planning competence or a causal model-size advantage. T5 base verification scores of 0% reflect unparseable verdicts.</p></section>
 <div class="footer">Source: <a href="../../submission/main.tex">submission/main.tex</a>, construction and evaluation design, matched tuning, regional transfer and external transfer tables, discussion and limitations. Values are reported manuscript results, not new experiments. Built with Anatomy; standalone page, no network requests.</div>
 </main><script>${live}</script></body></html>`;
 writeFileSync(new URL(theme==='light'?'index.html':'index-dark.html',import.meta.url),html);
}
console.log('Built light and dark paper explainers.');
