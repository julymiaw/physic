const assert=require('assert');const {receiveClock}=require('../miniprogram/utils/camera');
(async()=>{
 for(const units of [1,1000,1000000,1000000000]){
  let wall=1000;const object={now:()=>wall*units/1000};const clock=await receiveClock({wall:()=>wall,wait:async ms=>{wall+=ms;},candidates:[{object,source:'test-performance'}]});
  wall+=40;assert(Math.abs(clock()-.04)<1e-9);wall-=1;assert.throws(clock,/倒退/);
 }
 let wall=1000;const clock=await receiveClock({wall:()=>wall,wait:async ms=>{wall+=ms;},candidates:[{object:{now:()=>0},source:'stuck-shim'}]});assert.strictEqual(clock.source,'wall-receive');wall+=40;assert.strictEqual(clock(),.04);wall+=1000;assert.strictEqual(clock(),1.04);wall-=1;assert.throws(clock,/倒退/);
 const noNow=await receiveClock({wall:()=>wall,candidates:[]});assert.strictEqual(noNow.source,'wall-receive');
 let definition;global.Page=p=>definition=p;global.wx={};require('../miniprogram/pages/measure/index');
 function page(){return {...definition,data:{...definition.data,cameraEnabled:true,cameraReady:true},_previewBounds:{left:12,top:10,width:300,height:400},setData(x){Object.assign(this.data,x);},interrupt(){}};}
 const ev=(touch,end=false)=>end?{changedTouches:[touch]}:{touches:[touch]};
 let p=page();p.onSelectionStart(ev({clientX:42,clientY:50}));assert(p.data.selectionStyle.includes('width:8px'));p.onSelectionMove(ev({clientX:222,clientY:250}));p.onSelectionEnd(ev({clientX:222,clientY:250},true));assert(p.data.roiSelected);assert.deepStrictEqual(p.data.previewRegion,{x:.1,y:.1,width:.6,height:.5});
 p=page();p.onSelectionStart(ev({pageX:42,pageY:50}));p.onSelectionEnd(ev({pageX:42,pageY:50},true));assert(!p.data.roiSelected);assert(p._corner);assert.strictEqual(p.data.status,'点另一个对角');p.onSelectionStart(ev({pageX:222,pageY:250}));p.onSelectionEnd(ev({pageX:222,pageY:250},true));assert(p.data.roiSelected);assert.deepStrictEqual(p.data.previewRegion,{x:.1,y:.1,width:.6,height:.5});
 p=page();p.onSelectionStart(ev({x:210,y:240}));p.onSelectionMove(ev({x:30,y:40}));p.onSelectionEnd({changedTouches:[]});assert(p.data.roiSelected);assert.deepStrictEqual(p.data.previewRegion,{x:.1,y:.1,width:.6,height:.5});p.onSelectionCancel();assert(!p.data.roiSelected);assert(!p._corner);
 p=page();p.data.phase='measuring';p.onSelectionStart(ev({clientX:42,clientY:50}));assert(!p._dragStart);
 p=page();p.onSelectionStart(ev({}));assert(!p._dragStart);
 console.log(JSON.stringify({passed:true,checks:['seconds-ms-us-ns-clocks','stuck-shim-wall-fallback','missing-now-fallback','clock-backwards-rejection','forward-gap-preserved','client-coordinates-drag','page-coordinates-two-tap','local-coordinates-reverse-drag','touchend-empty-fallback','touch-cancel','measuring-selection-blocked']}));
})().catch(e=>{console.error(e);process.exitCode=1;});
