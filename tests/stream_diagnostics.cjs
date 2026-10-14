const assert=require('assert');const {StreamDiagnostics}=require('../miniprogram/utils/stream-diagnostics');
let time=0,saved,output=[];const diag=new StreamDiagnostics({build:'test'},{now:()=>time,write:r=>output.push(r),save:r=>saved=r});
const frame={width:480,height:640,data:new ArrayBuffer(4)};
diag.arrival(frame,1);time=40;diag.arrival(frame,1.04);diag.processedFrame({prepareMs:3,coreMs:2,totalMs:5,dropped:1});time=1000;diag.summary('calibrating');assert.strictEqual(output.at(-1).callbackFps,2);assert.strictEqual(output.at(-1).dropped,1);assert.strictEqual(output.at(-1).arrivalAgeMs,960);
time=1100;diag.event('calibration-fit-start');time=1800;diag.event('calibration-fit-end',{fitMs:700,success:false});diag.processedFrame({prepareMs:3,coreMs:700,totalMs:703});assert(output.some(r=>r.type==='slow-frame'));diag.persist();assert.strictEqual(saved.received,2);assert(!JSON.stringify(saved).includes('ArrayBuffer'));
for(let i=0;i<110;i++)diag.event('test');assert.strictEqual(diag.rows.length,96);
// Regress the observed ordering: a 760ms fit completes, a callback arrives,
// then watchdog runs before the pending frame is processed.
let definition;global.Page=p=>definition=p;global.wx={};require('../miniprogram/pages/measure/index');
const watchdog=new StreamDiagnostics({},{now:()=>1800,write:()=>{}});let interrupted=false;
const p={...definition,data:{phase:'calibrating'},_clock:()=>1.8,_lastReceive:1.1,_lastCallbackTime:1.799,_streamDiag:watchdog,_pipeline:{pending:null},_lastResult:{state:1,reason:8,calibrationFrames:50},setData(x){Object.assign(this.data,x);},interrupt(){interrupted=true;}};
p.checkFrameWatchdog();assert(!interrupted);
// Without a fresh callback, allow only 150ms after calibration returns.
p._lastCallbackTime=1.1;p._fitGraceUntil=1.95;p.checkFrameWatchdog();assert(!interrupted);
p._clock=()=>2.1;p.checkFrameWatchdog();assert(interrupted);assert(watchdog.rows.some(r=>r.type==='watchdog-timeout'&&Math.abs(r.callbackAgeMs-1000)<1e-6));
interrupted=false;p.data.phase='measuring';p._fitGraceUntil=100;p._lastCallbackTime=1.1;p.checkFrameWatchdog();assert(interrupted);
console.log(JSON.stringify({passed:true,checks:['metadata-only','summary-throttled','callback-vs-processing-ages','calibration-fit-timing','bounded-history','persist-on-failure','fresh-callback-prevents-false-timeout','bounded-calibration-return-grace','genuine-callback-stall-still-rejected'],scope:'Regression follows supplied handset event ordering; phone confirmation after fix remains required.'}));
