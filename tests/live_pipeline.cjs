const fs=require('fs'),assert=require('assert');const {Counter}=require('../miniprogram/utils/counter');
const {mapping,prepare,previewPoint,LatestFrame}=require('../miniprogram/utils/camera');
(async()=>{
 const {instance}=await WebAssembly.instantiate(fs.readFileSync('miniprogram/wasm/fringe.wasm'),{env:{emscripten_notify_memory_growth(){}}});instance.exports._initialize();
 const fixture=fs.readFileSync('experiments/results/real-track/padded-screen.frc');
 function frame(id){const p=28+id*(12+96*96*4);return {t:fixture.readDoubleLE(p+4),pixels:new Float32Array(fixture.buffer,fixture.byteOffset+p+12,96*96)};}
 function initialized(){const c=new Counter(instance.exports);c.reset(16,58*96/345);for(let id=231;id<=278;id++){const f=frame(id);c.input().set(f.pixels);c.collect(f.t,id);}assert(c.start());return c;}
 let c=initialized();for(let id=279;id<300;id++){const f=frame(id);c.input().set(f.pixels);c.push(f.t,id);}
 const count=c.snapshot().count;const f=frame(300);c.input().set(f.pixels);let result=c.push(f.t+1,300);assert(!result.complete&&result.reason===5&&result.count===count);c.push(f.t+2,301);assert.strictEqual(c.snapshot().count,count);c.destroy();
 c=initialized();c.input().fill(0);result=c.push(frame(279).t,279);assert(!result.complete&&result.count===0);c.destroy();
 c=initialized();c.input().set(frame(279).pixels);c.push(frame(279).t,279);result=c.push(frame(279).t,280);assert(!result.complete&&result.reason===4);c.destroy();
 c=new Counter(instance.exports);c.reset();c.input().fill(.5);for(let i=0;i<24;i++)c.collect(i/30,i);assert(!c.start());assert(c.snapshot().state===1);c.endCalibration();assert(c.snapshot().state===5);c.destroy();
 // Find a real truncated tail event; pending must not be silently committed.
 let pendingFound=false;
 for(const end of [345,489,684,823,1240]){c=initialized();for(let id=279;id<=end;id++){const f=frame(id);c.input().set(f.pixels);c.push(f.t,id);}result=c.finish();if(!result.complete&&result.reason===6){pendingFound=true;assert(c.rows().some(r=>r[10]===1));}c.destroy();}
 assert(pendingFound,'real prefix must exercise unconfirmed EOF event');
 const map=mapping({x:.2,y:.25,width:.5,height:.5},{width:300,height:300},{width:480,height:640});
 const point=previewPoint({x:map.offsetX-.5,y:map.offsetY-.5},map);assert(Math.abs(point.x-60)<1e-9&&Math.abs(point.y-75)<1e-9);
 const rgba=new Uint8Array(480*640*4);for(let i=0;i<rgba.length;i+=4)rgba[i]=128;
 const out=new Float32Array(96*96);prepare({width:480,height:640,data:rgba.buffer},map,out);assert(out.every(v=>Math.abs(v-128/255)<1e-6));
 assert.throws(()=>prepare({width:481,height:640,data:rgba.buffer},map,out));
 const outside={...map,x:-map.width};prepare({width:480,height:640,data:rgba.buffer},outside,out);assert(out.every(v=>v===0));
 let observed=[];const queue=new LatestFrame(x=>observed.push(x.id),e=>{throw e;});queue.offer({},1,1);queue.offer({},2,2);assert.strictEqual(queue.dropped,1);await new Promise(r=>setTimeout(r,10));assert.deepStrictEqual(observed,[2]);queue.stop();queue.offer({},3,3);assert.strictEqual(queue.pending,null);
 console.log(JSON.stringify({passed:true,checks:['real-frame-gap-freeze','tracking-failure','repeated-timestamp','flat-calibration-rejection','real-prefix-pending-tail','aspect-fill-map','area-normalization','zero-padding','frame-size-change','single-pending-frame']}));
})().catch(e=>{console.error(e);process.exitCode=1;});
