// Development only. JSON fixtures are generated locally and excluded from release.
const {loadCounter}=require('./counter');
async function verify(name){
  if(name==='measurement-flow')return verifyMeasurementFlow();
  if(name!=='original'&&name!=='bilibili')throw new Error('Unknown replay');
  getApp()._coreVerification={name,status:'loading'};
  const fs=wx.getFileSystemManager();
  const read=filePath=>new Promise((resolve,reject)=>fs.readFile({filePath,encoding:'utf8',success:r=>resolve(JSON.parse(r.data)),fail:reject}));
  const config=(await read('verification/manifest.json'))[name];
  const expected=await read(`verification/${config.expected}`);
  const c=await loadCounter();c.reset(config.x,config.y);let started=false,rowIndex=0,maxDifference=0,events=0;
  function check(){for(const row of c.rows()){
    if(!expected[rowIndex])throw new Error('Extra output');
    for(let k=0;k<22;k++){const diff=Math.abs(row[k]-expected[rowIndex][k]);maxDifference=Math.max(maxDifference,diff);if(diff>1e-6)throw new Error(`${name} mismatch row ${rowIndex}, field ${k}`);}
    events+=!!row[3];rowIndex++;
  }}
  try{
    for(const file of config.files){
      const packet=await read(`verification/${file}`);
      for(const [id,t,base64] of packet){
        c.input().set(new Float32Array(wx.base64ToArrayBuffer(base64)));
        if(t<=config.cal)c.collect(t,id);else{if(!started){if(!c.start())throw new Error('Calibration failed');started=true;}c.push(t,id);check();}
        getApp()._coreVerification={name,status:'running',sourceFrame:id};
      }
      await new Promise(resolve=>setTimeout(resolve,0));
    }
    const result=c.finish();check();if(rowIndex!==expected.length||result.complete!==expected.every(r=>r[9]===1))throw new Error('Unexpected completeness');
    const report={name,frames:rowIndex,count:result.count,events,maxDifference,complete:result.complete};getApp()._coreVerification={status:'passed',...report};return report;
  }finally{c.destroy();}
}
module.exports={verify};

async function verifyMeasurementFlow(){
 const fs=wx.getFileSystemManager(),read=filePath=>new Promise((resolve,reject)=>fs.readFile({filePath,encoding:'utf8',success:r=>resolve(JSON.parse(r.data)),fail:reject}));
 const config=await read('verification/measurement-manifest.json'),expected=await read('verification/measurement-expected.json');
 const c=await loadCounter();c.reset(config.x,config.y);let index=0,maxDifference=0;
 try{
  for(const file of config.files){for(const [op,id,t,pixels] of await read('verification/'+file)){
   if(pixels)c.input().set(new Float32Array(wx.base64ToArrayBuffer(pixels)));
   switch(op){case 0:c.collect(t,id);break;case 1:if(!c.start()||!c.arm())throw new Error('Calibration failed');break;case 2:c.monitor(t,id);break;case 3:if(!c.beginMeasurement())throw new Error('Start failed');break;case 4:c.push(t,id);break;case 5:c.finish();break;case 6:c.interrupt();break;case 7:if(!c.stopMeasurement())throw new Error('Stop failed');break;}
   const a=new Float64Array(c.e.memory.buffer,c.resultPointer,64);
   for(let k=0;k<64;k++){const diff=Math.abs(a[k]-expected[index][k]);maxDifference=Math.max(maxDifference,diff);if(diff>1e-6)throw new Error(`Flow mismatch ${index}/${k}`);}
   index++;
  }await new Promise(resolve=>setTimeout(resolve,0));}
  if(index!==expected.length)throw new Error('Missing flow rows');
  const r=c.snapshot(),report={status:'passed',name:'measurement-flow',operations:index,count:r.count,complete:r.complete,maxDifference};getApp()._coreVerification=report;return report;
 }finally{c.destroy();}
}
