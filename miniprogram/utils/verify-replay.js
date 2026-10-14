// Development only. JSON fixtures are generated locally and excluded from release.
const {loadCounter}=require('./counter');
async function verify(name){
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
