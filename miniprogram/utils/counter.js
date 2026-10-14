// Shared adapter: Node tests and WXWebAssembly read the same C ABI.
class Counter {
  constructor(exports) {
    this.e = exports;
    this.handle = exports.fc_create();
    if (!this.handle) throw new Error('计数器初始化失败');
    this.inputPointer = exports.fc_input(this.handle);
    this.resultPointer = exports.fc_result(this.handle);
  }
  input() { return new Float32Array(this.e.memory.buffer, this.inputPointer, 96 * 96); }
  snapshot() {
    const r = new Float64Array(this.e.memory.buffer, this.resultPointer, 64);
    return { canBegin:!!r[36], stopping:!!r[37], stopTime:r[38], state:r[0], count:r[1], cycles:r[2], complete:!!r[3], reason:r[4], pending:r[5], calibrationFrames:r[6], nodes:r[7], timestamp:r[8], gapLimit:r[11], rows:r[9], signals:[r[10], ...Array.from(r.slice(22,26))], phases:Array.from(r.slice(14,18)), radii:Array.from(r.slice(18,22)), points:Array.from({length:r[7]+1},(_,k)=>({x:r[26+k*2],y:r[27+k*2]})) };
  }
  rows() {
    const count = this.snapshot().rows;
    return Array.from({length:count},(_,i)=>Array.from(new Float64Array(this.e.memory.buffer,this.e.fc_row(this.handle,i),22)));
  }
  reset(x=16,y=16) { if(!this.e.fc_reset(this.handle,x,y)) throw new Error('光屏区域无效'); return this.snapshot(); }
  collect(t,id) { this.e.fc_calibrate(this.handle,t,id); return this.snapshot(); }
  start() { return !!this.e.fc_start(this.handle); }
  arm() { return !!this.e.fc_arm(this.handle); }
  monitor(t,id) { this.e.fc_monitor(this.handle,t,id); return this.snapshot(); }
  beginMeasurement() { return !!this.e.fc_begin_measurement(this.handle); }
  stopMeasurement() { return !!this.e.fc_stop_measurement(this.handle); }
  push(t,id) { this.e.fc_push(this.handle,t,id); return this.snapshot(); }
  finish() { this.e.fc_finish(this.handle); return this.snapshot(); }
  interrupt() { this.e.fc_interrupt(this.handle); return this.snapshot(); }
  endCalibration() { this.e.fc_calibration_end(this.handle); return this.snapshot(); }
  destroy() { if(this.handle){this.e.fc_destroy(this.handle);this.handle=0;} }
}
let modulePromise;
function loadCounter() {
  if(typeof WXWebAssembly==='undefined') return Promise.reject(new Error('当前微信不支持计数模块，请更新微信。'));
  if(!modulePromise) modulePromise=WXWebAssembly.instantiate('wasm/fringe.wasm',{env:{emscripten_notify_memory_growth(){}}}).then(instance=>{
    const e=(instance.instance||instance).exports;e._initialize();return e;
  }).catch(error=>{modulePromise=null;throw error;});
  return modulePromise.then(e=>new Counter(e));
}
module.exports={Counter,loadCounter};
