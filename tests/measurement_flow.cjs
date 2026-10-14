const fs=require('fs'),assert=require('assert'),cp=require('child_process');
const {Counter}=require('../miniprogram/utils/counter');
(async()=>{
 const {instance}=await WebAssembly.instantiate(fs.readFileSync('miniprogram/wasm/fringe.wasm'),{env:{emscripten_notify_memory_growth(){}}});instance.exports._initialize();
 const file='experiments/results/real-track/padded-screen.frc',b=fs.readFileSync(file);
 const frame=id=>{const p=28+id*(12+96*96*4);return {t:b.readDoubleLE(p+4),pixels:new Float32Array(b.buffer,b.byteOffset+p+12,96*96)};};
 const run='experiments/runs/measurement-flow';fs.mkdirSync(run,{recursive:true});
 cp.execFileSync('c++',['-std=c++17','-O3','tests/measurement_replay.cpp','-o',run+'/native']);
 const reports=[];
 function scenario(name,append){
  const c=new Counter(instance.exports),ops=[],states=[];c.reset(16,58*96/345);
  function step(op,id=-1,t=0){if(id>=0)c.input().set(frame(id).pixels);switch(op){case 0:c.collect(t,id);break;case 1:assert(c.start());assert(c.arm());break;case 2:c.monitor(t,id);break;case 3:assert(c.beginMeasurement());break;case 4:c.push(t,id);break;case 5:c.finish();break;case 6:c.interrupt();break;case 7:assert(c.stopMeasurement());break;}
   ops.push([op,id,t]);states.push(Array.from(new Float64Array(c.e.memory.buffer,c.resultPointer,64)));return c.snapshot();}
  for(let id=231;id<=278;id++)step(0,id,frame(id).t);step(1);
  append(step,c);
  fs.writeFileSync(run+'/'+name+'.ops',ops.map(r=>r.join(' ')).join('\n')+'\n');cp.execFileSync(run+'/native',[file,run+'/'+name+'.ops',run+'/'+name+'.native']);
  const native=fs.readFileSync(run+'/'+name+'.native','utf8').trim().split('\n').map(r=>r.split(',').map(Number));assert.strictEqual(native.length,states.length);
  let max=0;for(let i=0;i<states.length;i++)for(let k=0;k<64;k++){const d=Math.abs(states[i][k]-native[i][k]);max=Math.max(max,d);if([0,1,3,4,5,6,7,8,9,36,37,38].includes(k))assert.strictEqual(d,0);else assert(d<1e-6,`${name} op=${i} state=${k} native=${native[i][k]} wasm=${states[i][k]} difference=${d}`);}
  const final=c.snapshot();reports.push({name,operations:ops.length,maxNativeWasmDifference:max,finalCount:final.count,finalCycles:final.cycles,complete:final.complete});c.destroy();
 }
 scenario('whole-video-relative-start',(s,c)=>{
  assert.strictEqual(c.snapshot().state,6);assert(!c.beginMeasurement());s(2,279,frame(279).t);s(3);
  assert.strictEqual(c.snapshot().count,0);assert.strictEqual(c.snapshot().cycles,0);
  for(let id=280;id<=1436;id++){const r=s(4,id,frame(id).t);assert(r.complete);if(Math.abs(r.cycles)<.98)assert.strictEqual(r.count,0);}
  for(let i=1;i<=4;i++)s(4,1436,frame(1436).t+i/24);const r=s(5);assert(r.complete);assert.strictEqual(r.count,Math.floor(r.cycles));
 });
 scenario('wait-stop-and-reuse',(s,c)=>{
  for(let id=279;id<=330;id++){const r=s(2,id,frame(id).t);assert(r.complete);assert.strictEqual(r.count,0);assert.strictEqual(r.cycles,0);assert.strictEqual(c.rows().length,0);}
  s(3);for(let id=331;id<=600;id++)assert(s(4,id,frame(id).t).complete);
  for(let i=1;i<=4;i++)s(4,600,frame(600).t+i/24);const stopped=s(5);assert(stopped.complete);const count=stopped.count,cycles=stopped.cycles;
  for(let id=601;id<=700;id++){const r=s(2,id,frame(id).t+.2);assert(r.complete);assert.strictEqual(r.count,count);assert.strictEqual(r.cycles,cycles);}
  s(3);assert.strictEqual(c.snapshot().count,0);assert.strictEqual(c.snapshot().cycles,0);assert.strictEqual(c.snapshot().calibrationFrames,48);
  for(let id=701;id<=1436;id++)assert(s(4,id,frame(id).t+.2).complete);
  for(let i=1;i<=4;i++)s(4,1436,frame(1436).t+.2+i/24);assert(s(5).complete);
 });
 scenario('real-frames-forward-and-back',(s,c)=>{
  s(2,279,frame(279).t);s(3);let t=frame(279).t;
  for(let id=280;id<=340;id++){t+=1/24;assert(s(4,id,t).complete);}
  assert(c.snapshot().count>0);
  for(let id=339;id>=279;id--){t+=1/24;assert(s(4,id,t).complete);}
  for(let i=0;i<4;i++){t+=1/24;s(4,279,t);}
  const r=s(5);assert(r.complete);assert.strictEqual(r.count,0);assert(Math.abs(r.cycles)<1e-5);
 });
 scenario('stop-cutoff-with-four-confirmation-frames',(s,c)=>{
  s(2,279,frame(279).t);s(3);for(let id=280;id<=600;id++)assert(s(4,id,frame(id).t).complete);
  assert(s(7).stopping);for(let id=601;id<=604;id++){const r=s(4,id,frame(id).t);assert(r.complete);if(id<604)assert(r.stopping);}
  const stopped=c.snapshot();assert.strictEqual(stopped.state,4);assert(!stopped.stopping);assert.strictEqual(stopped.stopTime,frame(600).t);
  assert(c.rows().every(r=>r[0]<=600));const count=stopped.count,cycles=stopped.cycles;
  for(let id=605;id<=650;id++){const r=s(2,id,frame(id).t);assert.strictEqual(r.count,count);assert.strictEqual(r.cycles,cycles);}
 });
 scenario('initial-calibration-compute-gap',(s,c)=>{
  // No measurement exists yet. A fresh observation can anchor after a slow fit.
  const r=s(2,279,frame(279).t+.76);assert(r.complete);assert.strictEqual(r.state,6);s(3);
  for(let id=280;id<=1436;id++)assert(s(4,id,frame(id).t+.76).complete);
  s(7);for(let i=1;i<=4;i++)s(4,1436,frame(1436).t+.76+i/24);
  assert.strictEqual(c.snapshot().state,4);assert.strictEqual(c.snapshot().count,83);
 });
 scenario('measurement-gap-still-rejected',(s,c)=>{
  s(2,279,frame(279).t+.76);s(3);const r=s(4,280,frame(280).t+1.76);assert(!r.complete);assert.strictEqual(r.reason,5);
 });
 scenario('ready-gap-rejects-start',(s,c)=>{s(2,279,frame(279).t);const r=s(2,280,frame(280).t+1);assert(!r.complete);assert.strictEqual(r.reason,5);assert(!c.beginMeasurement());});
 fs.writeFileSync('experiments/results/wechat-core/measurement-flow.json',JSON.stringify({passed:true,scope:'Real-video start/wait/stop/reuse flow and Native/WASM parity; recorded final frame repeated only to represent stopping the knob and clear four-frame confirmation buffer.',checks:['no-count-while-waiting','partial-first-period-not-counted','relative-cycle-start-and-end','real-frame-forward-back-net-zero','stop-cutoff-with-four-confirmation-frames','stop-result-frozen','reuse-without-recalibration','initial-calibration-compute-gap','measurement-gap-still-rejected','ready-gap-rejects-start'],scenarios:reports},null,2)+'\n');console.log(JSON.stringify(reports));
})().catch(e=>{console.error(e);process.exitCode=1;});
