const SIZE=96;
// Map preview coordinates through a centered aspect-fill camera preview.
function mapping(region, preview, frame) {
  const scale=Math.max(preview.width/frame.width,preview.height/frame.height);
  const cropX=(frame.width*scale-preview.width)/2,cropY=(frame.height*scale-preview.height)/2;
  // Snap only sub-micropixel arithmetic noise, then align the source origin.
  // Offsets carry the fractional remainder: never shift the selected count region.
  const snap=v=>Math.abs(v-Math.round(v))<1e-7?Math.round(v):v;
  const x=snap((region.x*preview.width+cropX)/scale),y=snap((region.y*preview.height+cropY)/scale);
  const width=snap(region.width*preview.width/scale),height=snap(region.height*preview.height/scale);
  const left=Math.floor(snap(x-width/4)),top=Math.floor(snap(y-height/4));
  return {x:left,y:top,width:width*1.5,height:height*1.5,offsetX:(x-left)*SIZE/(width*1.5),offsetY:(y-top)*SIZE/(height*1.5),scale,cropX,cropY,frameWidth:frame.width,frameHeight:frame.height,previewWidth:preview.width,previewHeight:preview.height};
}
// Exact area integration, including zero padding beyond the original frame.
// Writes red-channel floats directly into preallocated WASM memory.
function prepare(frame, map, output) {
  if(frame.width!==map.frameWidth||frame.height!==map.frameHeight) throw new Error('相机画面尺寸改变，请重新标定。');
  const data=new Uint8Array(frame.data);
  if(data.length!==frame.width*frame.height*4) throw new Error('相机帧数据格式不支持');
  for(let oy=0;oy<SIZE;oy++)for(let ox=0;ox<SIZE;ox++){
    const left=map.x+ox*map.width/SIZE,right=map.x+(ox+1)*map.width/SIZE;
    const top=map.y+oy*map.height/SIZE,bottom=map.y+(oy+1)*map.height/SIZE;
    let sum=0;
    for(let y=Math.max(0,Math.floor(top));y<Math.min(frame.height,Math.ceil(bottom));y++)
      for(let x=Math.max(0,Math.floor(left));x<Math.min(frame.width,Math.ceil(right));x++){
        const weight=(Math.min(right,x+1)-Math.max(left,x))*(Math.min(bottom,y+1)-Math.max(top,y));
        sum+=data[(y*frame.width+x)*4]*weight;
      }
    output[oy*SIZE+ox]=sum/((right-left)*(bottom-top)*255);
  }
}
function previewPoint(p,map){return {x:((map.x+(p.x+.5)*map.width/SIZE)*map.scale-map.cropX),y:((map.y+(p.y+.5)*map.height/SIZE)*map.scale-map.cropY)};}
// Only one scheduled/retained frame. New arrivals replace it, with their actual receive time.
class LatestFrame {
  constructor(process,onError){this.process=process;this.onError=onError;this.pending=null;this.timer=null;this.dropped=0;this.active=true;}
  offer(frame,t,id){if(!this.active)return;if(this.pending)this.dropped++;this.pending={frame,t,id};if(this.timer===null)this.timer=setTimeout(()=>this.drain(),0);}
  drain(){this.timer=null;const item=this.pending;this.pending=null;if(!this.active||!item)return;try{this.process(item);}catch(error){this.onError(error);}}
  stop(){this.active=false;this.pending=null;if(this.timer!==null)clearTimeout(this.timer);this.timer=null;}
}
// Remote debugging may expose a shim clock with different units or no progress.
// Probe available high-resolution clocks; otherwise use explicit guarded wall time.
function guardedClock(read,units,source){
  const origin=read();if(!Number.isFinite(origin))throw new Error('接收时钟不可用');let last=origin;
  const clock=()=>{const value=read();if(!Number.isFinite(value)||value<last)throw new Error('接收时间倒退，请重新开始测量。');last=value;return (value-origin)/units;};
  clock.source=source;clock.unitsPerSecond=units;return clock;
}
async function receiveClock(options={}) {
  const wall=options.wall||(()=>Date.now()),wait=options.wait||(ms=>new Promise(resolve=>setTimeout(resolve,ms)));
  let candidates=options.candidates;
  if(!candidates){
    candidates=[];
    // wx.getPerformance().now is not guaranteed by the official mini-program API.
    try{const native=wx.getPerformance();if(native&&typeof native.now==='function')candidates.push({object:native,source:'wx-performance'});}catch(_){}
    if(typeof performance!=='undefined'&&typeof performance.now==='function')candidates.push({object:performance,source:'global-performance'});
  }
  for(const {object,source} of candidates){
    try{
      const read=()=>object.now(),a=read(),start=wall();await wait(100);
      const delta=read()-a,elapsed=wall()-start,ratio=delta/elapsed;
      if(!Number.isFinite(ratio)||delta<=0||elapsed<=0)continue;
      const scale=[.001,1,1000,1000000].find(v=>ratio/v>.75&&ratio/v<1.25);
      if(scale)return guardedClock(read,scale*1000,source);
    }catch(_){}
  }
  // No fabricated ticks or clamping: backwards values throw; large forward gaps
  // remain visible to the core continuity gate. This is wall time, not exposure time.
  return guardedClock(wall,1000,'wall-receive');
}
module.exports={mapping,prepare,previewPoint,LatestFrame,receiveClock};
