// Controlled renderer -> actual RGBA area preprocessing -> production WASM core.
const fs=require('fs'),assert=require('assert');const {Counter}=require('../miniprogram/utils/counter');const {prepare}=require('../miniprogram/utils/camera');
(async()=>{
 const {instance}=await WebAssembly.instantiate(fs.readFileSync('miniprogram/wasm/fringe.wasm'),{env:{emscripten_notify_memory_growth(){}}});instance.exports._initialize();const c=new Counter(instance.exports);c.reset(16,16);
 const N=256,rgba=new Uint8Array(N*N*4),spatial=new Float64Array(N*N),mask=new Float64Array(N*N);
 for(let y=0;y<N;y++)for(let x=0;x<N;x++){const i=y*N+x,r=Math.hypot((x+.5-N/2)/(N/2),(y+.5-N/2)/(N/2));spatial[i]=28*(r/.9)**2;mask[i]=Math.max(0,Math.min(1,(.92-r)/.02));rgba[i*4+3]=255;}
 const map={x:-64,y:-64,width:384,height:384,frameWidth:N,frameHeight:N};const cycles=t=>t<6?0:t<12?t-6:t<18?6:6+Math.min(10.25,t-18);let calibrated=false,begun=false,stopped=false,rows=0;
 for(let id=180;id<1020;id++){
  const t=id/30,phase=2*Math.PI*cycles(t);
  for(let i=0;i<N*N;i++){const v=Math.round(255*(.008+(.55+.33*Math.cos(spatial[i]+phase)-.008)*mask[i]));rgba[i*4]=rgba[i*4+1]=rgba[i*4+2]=v;}
  prepare({width:N,height:N,data:rgba.buffer},map,c.input());
  if(t<=8.1)c.collect(t,id);else{
   if(!calibrated){assert(c.start(),c.calibrationError());assert(c.arm());calibrated=true;}
   let r=c.snapshot();if(!begun||stopped)r=c.monitor(t,id);else r=c.push(t,id);
   assert(r.complete,JSON.stringify({frame:id,result:r}));rows++;
   if(t===15){assert(c.beginMeasurement());begun=true;}
   if(t===31){assert(c.stopMeasurement());stopped=true;}
   // Stop confirmation still needs four new frames; do not switch to monitor early.
   if(stopped&&c.snapshot().state===2)stopped=false;
   if(c.snapshot().state===4)stopped=true;
  }
 }
 const r=c.snapshot();assert.strictEqual(r.state,4);assert.strictEqual(r.count,10);assert(r.complete);
 fs.writeFileSync('experiments/results/theory-camera-target/direct-pipeline.json',JSON.stringify({passed:true,calibration:[6,8.1],start:15,stop:31,truthNetCycles:10.25,expectedWholeCycles:10,result:r,processedFrames:rows,scope:'Analytic uncompressed target through production RGBA prepare and WASM. Does not cover MP4 compression, monitor display, physical handset capture or its event scheduling.'},null,2)+'\n');c.destroy();console.log(JSON.stringify({passed:true,count:r.count,cycles:r.cycles}));
})().catch(e=>{console.error(e);process.exitCode=1;});
