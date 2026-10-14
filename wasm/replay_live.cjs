const fs=require('fs');const {Counter}=require('../miniprogram/utils/counter');
async function main(){
 const [wasm,fixture,startText,calText,endText,xText,yText,out]=process.argv.slice(2);
 const {instance}=await WebAssembly.instantiate(fs.readFileSync(wasm),{env:{emscripten_notify_memory_growth(){}}});instance.exports._initialize();
 const c=new Counter(instance.exports);c.reset(+xText,+yText);
 const b=fs.readFileSync(fixture),count=b.readUInt32LE(16);const start=+startText,cal=+calText,end=+endText;
 const fields='source_frame,timestamp,count,event_direction,confirmation_frame,confirmed_at,valid_pairs,repairs,ambiguous,complete,tail_pending,primary,tracking_valid,tracking_iou,pair0,phase0,pair1,phase1,pair2,phase2,pair3,phase3';
 let rows=[fields],started=false;const raw=['source_frame,'+Array.from({length:64},(_,i)=>'state'+i).join(',')];const stateRows=[];let p=28;
 for(let i=0;i<count;i++){
  const id=i,t=b.readDoubleLE(p+4);p+=12;const pixels=new Float32Array(b.buffer,b.byteOffset+p,96*96);p+=96*96*4;
  if(t<start||t>end)continue;c.input().set(pixels);
  if(t<=cal)c.collect(t,id);else{if(!started){if(!c.start())throw new Error('calibration failed');started=true;}const result=c.push(t,id);rows.push(...c.rows().map(r=>r.join(',')));stateRows.push({source_frame:id,...result});raw.push([id,...new Float64Array(c.e.memory.buffer,c.resultPointer,64)].join(','));}
 }
 const result=c.finish();raw.push([-1,...new Float64Array(c.e.memory.buffer,c.resultPointer,64)].join(','));fs.writeFileSync(out+'.state.csv',raw.join('\n')+'\n');rows.push(...c.rows().map(r=>r.join(',')));fs.writeFileSync(out,rows.join('\n')+'\n');
 fs.writeFileSync(out+'.state.json',JSON.stringify({final:result,measurement:stateRows}));c.destroy();console.log(JSON.stringify(result));
 if(!result.complete)process.exitCode=3;
}
main().catch(e=>{console.error(e);process.exitCode=1;});
