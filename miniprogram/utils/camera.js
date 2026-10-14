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
// Probe the units of the available monotonic clock once before enabling measurement.
async function receiveClock() {
  const candidate=typeof performance!=='undefined'&&typeof performance.now==='function'?performance:wx.getPerformance();
  if(!candidate||typeof candidate.now!=='function') throw new Error('当前微信不支持可靠计时，请更新微信。');
  const a=candidate.now(),wall=Date.now();await new Promise(resolve=>setTimeout(resolve,100));
  const delta=candidate.now()-a,elapsed=Date.now()-wall,ratio=delta/elapsed;
  const units=ratio>.25&&ratio<4?1000:(ratio>250&&ratio<4000?1000000:0);
  if(!units)throw new Error('无法确认相机接收时钟，请重试。');
  const origin=candidate.now();return ()=> (candidate.now()-origin)/units;
}
module.exports={mapping,prepare,previewPoint,LatestFrame,receiveClock};
