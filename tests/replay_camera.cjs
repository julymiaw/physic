// Real cached red frames are embedded in RGBA callbacks and pass the production adapter.
const fs=require('fs');const {Counter}=require('../miniprogram/utils/counter');const {mapping,prepare}=require('../miniprogram/utils/camera');
async function main(){
 const [name,rawPath,metaPath,fixture,output]=process.argv.slice(2),meta=JSON.parse(fs.readFileSync(metaPath));
 const {instance}=await WebAssembly.instantiate(fs.readFileSync('miniprogram/wasm/fringe.wasm'),{env:{emscripten_notify_memory_growth(){}}});instance.exports._initialize();
 const c=new Counter(instance.exports),preview={width:366,height:506},frame={width:meta.width,height:meta.height};
 const rect=meta.roi,scale=Math.max(preview.width/frame.width,preview.height/frame.height),cropX=(frame.width*scale-preview.width)/2,cropY=(frame.height*scale-preview.height)/2;
 const region={x:(rect[0]*scale-cropX)/preview.width,y:(rect[1]*scale-cropY)/preview.height,width:rect[2]*scale/preview.width,height:rect[3]*scale/preview.height};
 const map=mapping(region,preview,frame);c.reset(map.offsetX,map.offsetY);
 const fd=fs.openSync(rawPath,'r'),red=new Uint8Array(frame.width*frame.height),rgba=new Uint8Array(red.length*4);frame.data=rgba.buffer;
 const fixed=fs.readFileSync(fixture);let maxPixelDifference=0,pixelMismatch=0,started=false;const observed=[],states=[];
 function rows(){observed.push(...c.rows());}
 for(let id=0;id<meta.timestamps.length;id++){
  const t=meta.timestamps[id];if(t<meta.start||t>meta.end)continue;
  const bytes=fs.readSync(fd,red,0,red.length,id*red.length);if(bytes!==red.length)throw new Error('Truncated raw plane');
  for(let i=0;i<red.length;i++){rgba[i*4]=rgba[i*4+1]=rgba[i*4+2]=red[i];rgba[i*4+3]=255;}prepare(frame,map,c.input());
  const expected=new Float32Array(fixed.buffer,fixed.byteOffset+28+id*(12+96*96*4)+12,96*96),input=c.input();
  for(let j=0;j<input.length;j++){const difference=Math.abs(input[j]-expected[j]);maxPixelDifference=Math.max(maxPixelDifference,difference);pixelMismatch+=difference!==0;}
  if(t<=meta.cal)c.collect(t,id);else{if(!started){if(!c.start())throw new Error('Model initialization failed');started=true;}states.push({source_frame:id,...c.push(t,id)});rows();}
 }
 const final=c.finish();rows();fs.closeSync(fd);c.destroy();
 const header='source_frame,timestamp,count,event_direction,confirmation_frame,confirmed_at,valid_pairs,repairs,ambiguous,complete,tail_pending,primary,tracking_valid,tracking_iou,pair0,phase0,pair1,phase1,pair2,phase2,pair3,phase3';
 fs.writeFileSync(output,[header,...observed.map(r=>r.join(','))].join('\n')+'\n');fs.writeFileSync(output+'.state.json',JSON.stringify({final,measurement:states}));
 console.log(JSON.stringify({name,frames:observed.length,final:final.count,complete:final.complete,maxPixelDifference,pixelMismatch,mapping:map}));
 if(!final.complete)process.exitCode=3;
}
main().catch(e=>{console.error(e);process.exitCode=1;});
