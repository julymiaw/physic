const assert=require('assert');const {StreamDiagnostics}=require('../miniprogram/utils/stream-diagnostics');
let time=0,saved,output=[];const diag=new StreamDiagnostics({build:'test'},{now:()=>time,write:r=>output.push(r),save:r=>saved=r});
const frame={width:480,height:640,data:new ArrayBuffer(4)};
diag.arrival(frame,1);time=40;diag.arrival(frame,1.04);diag.processedFrame({prepareMs:3,coreMs:2,totalMs:5,dropped:1});time=1000;diag.summary('calibrating');assert.strictEqual(output.at(-1).callbackFps,2);assert.strictEqual(output.at(-1).dropped,1);assert.strictEqual(output.at(-1).arrivalAgeMs,960);
time=1100;diag.event('calibration-fit-start');time=1800;diag.event('calibration-fit-end',{fitMs:700,success:false});diag.processedFrame({prepareMs:3,coreMs:700,totalMs:703});assert(output.some(r=>r.type==='slow-frame'));diag.persist();assert.strictEqual(saved.received,2);assert(!JSON.stringify(saved).includes('ArrayBuffer'));
for(let i=0;i<110;i++)diag.event('test');assert.strictEqual(diag.rows.length,96);
// Reproduce the existing watchdog's treatment of synchronous processing time.
let definition;global.Page=p=>definition=p;global.wx={};require('../miniprogram/pages/measure/index');
const watchdog=new StreamDiagnostics({},{now:()=>1800,write:()=>{}});let interrupted=false;
const p={...definition,data:{phase:'calibrating'},_clock:()=>1.8,_lastReceive:1.1,_streamDiag:watchdog,_pipeline:{pending:null},_lastResult:{state:1,reason:8,calibrationFrames:50},setData(x){Object.assign(this.data,x);},interrupt(){interrupted=true;}};
p.checkFrameWatchdog();assert(interrupted);assert(watchdog.rows.some(r=>r.type==='watchdog-timeout'&&Math.abs(r.processedTimestampAgeMs-700)<1e-6));
console.log(JSON.stringify({passed:true,checks:['metadata-only','summary-throttled','callback-vs-processing-ages','calibration-fit-timing','bounded-history','persist-on-failure','existing-watchdog-processing-time-reproduction'],scope:'Diagnostics tested; watchdog policy deliberately unchanged pending physical-device logs.'}));
