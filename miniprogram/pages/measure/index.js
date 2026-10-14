const {loadCounter}=require('../../utils/counter');
const {mapping,prepare,previewPoint,LatestFrame,receiveClock}=require('../../utils/camera');
const COLORS=['#91a69b','#55aa92','#688bb8','#b5985e','#9982ab'];
const REASONS={1:'标定画面不足',2:'光屏跟踪失效',3:'相位信号不可靠',4:'接收时间异常',5:'有效帧间隔过大',6:'片尾仍有未确认事件',7:'帧数据无效',8:'标定未获得稳定的单向周期',9:'测量已中断'};
Page({
  data:{cameraEnabled:false,cameraReady:false,coreReady:false,roiSelected:false,selectionStyle:'',previewRegion:null,points:[],showSignals:true,helpVisible:false,helpTitle:'使用方法',helpText:'开启相机，拖框选中整个光屏。\n\n点击标定，缓慢、连续地单向移动，使中心暗纹至少出现两次；标定最长10秒。标定方向定义为正方向。\n\n开始计数后，正向增加、反向减少。丢失跟踪或中断会冻结计数并标为不完整。\n\n曲线颜色对应画面采样点，相位图显示各路共同暗心零点的相位。点击“信号”可收起。\n\n画面仅在本地处理。',phase:'idle',status:'加载中',countText:'—',complete:true,mainLabel:'标定',viewportHeight:600,safeBottom:0},
  onLoad(){
    this._active=true;this._source=0;this._history=[];this._lastUI=0;this._events=[];
    const info=wx.getWindowInfo?wx.getWindowInfo():wx.getSystemInfoSync();
    this.setData({viewportHeight:info.windowHeight,safeBottom:Math.max(0,info.screenHeight-(info.safeArea?info.safeArea.bottom:info.screenHeight)),helpVisible:!wx.getStorageSync('fringe-help-v1')});
    Promise.all([loadCounter(),receiveClock()]).then(([core,clock])=>{
      if(!this._active){core.destroy();return;}this._core=core;this._clock=clock;
      this.setData({coreReady:true,status:this.data.roiSelected?'待标定':'待圈选'});
    }).catch(error=>{if(this._active){this.setData({status:'初始化失败'});this.showMessage('无法启动',error.message);}});
  },
  onReady(){this.initCanvas();},
  // Invoked only by the development verification entry; never connected to product UI.
  verifyReplay(name){return require('../../utils/verify-replay').verify(name);},
  noop(){},
  closeHelp(){wx.setStorageSync('fringe-help-v1',true);this.setData({helpVisible:false});},
  showHelp(){if(this.isRunning())return;if(this.data.cameraEnabled)this.stopCamera();this.showMessage('使用方法',this.data.helpText);},
  showMessage(title,text){if(this.data.cameraEnabled)this.stopCamera();this.setData({helpTitle:title,helpBody:text,helpVisible:true});},
  showStatus(){
    if(this.isRunning())return;
    const last=wx.getStorageSync('fringe-last-session'),lost=wx.getStorageSync('fringe-last-incomplete');
    const history=lost?'\n\n此前有不完整测量段（净'+lost.count+'圈），其丢失信息没有被重新标定补回。当前计数只代表当前段。':'';
    this.showMessage('测量状态',(this._lastResult&&!this._hasMeasured?(REASONS[this._lastResult.reason]||this.data.status)+'。尚未开始有效测量。':this._lastResult?`${REASONS[this._lastResult.reason]||this.data.status}。当前净计数：${this._lastResult.count}圈。${this._lastResult.complete?'':'这一段存在丢失信息，重新标定不会补回。'}`:(last?`上一段净计数：${last.count}圈；${last.complete?'完整':'不完整'}。`:'圈选光屏后开始标定。'))+history);
  },
  isRunning(){return this.data.phase==='calibrating'||this.data.phase==='measuring';},
  toggleCamera(){if(this.data.cameraEnabled)this.stopCamera();else this.setData({cameraEnabled:true,cameraReady:false,points:[]});},
  onCameraReady(){
    if(!this.data.cameraEnabled)return;this.setData({cameraReady:true});this.measurePreview();
    this._listener=wx.createCameraContext().onCameraFrame(frame=>{
      this._frameShape={width:frame.width,height:frame.height};
      if(this._pipeline)this._pipeline.offer(frame,this._clock(),this._source++);
    });
    this._listener.start({fail:error=>{this.stopCamera();this.showMessage('相机帧不可用',error.errMsg||'请在手机微信中重试。');}});
  },
  measurePreview(){this._previewBounds=null;wx.createSelectorQuery().in(this).select('.preview').boundingClientRect(rect=>{if(this.data.cameraEnabled&&rect)this._previewBounds=rect;}).exec();},
  stopFeed(){if(this._pipeline)this._pipeline.stop();this._pipeline=null;if(this._watchdog)clearInterval(this._watchdog);this._watchdog=null;},
  remember(){if(this._lastResult&&this._hasMeasured){const saved={count:this._lastResult.count,complete:this._lastResult.complete,reason:this._lastResult.reason,events:this._events.slice(-512),eventsTruncated:!!this._eventsTruncated};wx.setStorageSync('fringe-last-session',saved);if(!saved.complete)wx.setStorageSync('fringe-last-incomplete',saved);}},
  interrupt(){
    this.stopFeed();if(this._core&&this.isRunning()){const result=this._core.interrupt();this.display(result,true);this.remember();}
  },
  stopCamera(){
    this.interrupt();if(this._listener)this._listener.stop();this._listener=null;this._dragStart=null;this._previewBounds=null;this._frameShape=null;
    this.setData({cameraEnabled:false,cameraReady:false,roiSelected:false,selectionStyle:'',previewRegion:null,points:[]});
  },
  onCameraError(event){this.stopCamera();this.showMessage('相机无法开启',(event.detail&&event.detail.errMsg)||'请检查相机权限后重试。');},
  onCameraStop(){if(this.data.cameraEnabled){this.stopCamera();this.setData({status:'相机已中断'});}},
  clearSelection(){
    this.interrupt();this._dragStart=null;this.setData({roiSelected:false,selectionStyle:'',previewRegion:null,points:[]});
    if(!this._lastResult||this._lastResult.state===1)this.setData({status:'待圈选'});
  },
  previewPoint(touch){const rect=this._previewBounds;if(!rect||!touch||!Number.isFinite(touch.clientX)||!Number.isFinite(touch.clientY))return null;return {x:Math.max(0,Math.min(rect.width,touch.clientX-rect.left)),y:Math.max(0,Math.min(rect.height,touch.clientY-rect.top))};},
  onSelectionStart(event){if(!this.data.cameraReady||this.isRunning())return;this._dragStart=this.previewPoint(event.touches[0]);if(this._dragStart)this.setData({roiSelected:false,previewRegion:null,selectionStyle:'',points:[]});},
  regionFromTouch(touch){const p=this.previewPoint(touch);if(!p||!this._dragStart)return null;return {x:Math.min(p.x,this._dragStart.x),y:Math.min(p.y,this._dragStart.y),width:Math.abs(p.x-this._dragStart.x),height:Math.abs(p.y-this._dragStart.y)};},
  drawRegion(region){this.setData({selectionStyle:`left:${region.x}px;top:${region.y}px;width:${region.width}px;height:${region.height}px;`});},
  onSelectionMove(event){if(this._dragStart){const r=this.regionFromTouch(event.touches[0]);if(r)this.drawRegion(r);}},
  onSelectionEnd(event){
    const r=this.regionFromTouch(event.changedTouches[0]);this._dragStart=null;if(!r)return;
    if(r.width<32||r.height<32){this.clearSelection();return;}
    const b=this._previewBounds;this.drawRegion(r);this.setData({roiSelected:true,previewRegion:{x:r.x/b.width,y:r.y/b.height,width:r.width/b.width,height:r.height/b.height},status:'待标定'});
  },
  onSelectionCancel(){this.clearSelection();},
  mainAction(){
    if(this.isRunning()){
      // A retained frame was never submitted to the core: do not flush across it.
      const droppedTail=!!(this._pipeline&&this._pipeline.pending);this.stopFeed();
      const result=droppedTail?this._core.interrupt():this._core.finish();this.display(result,true);this.remember();return;
    }
    if(!this.data.coreReady||!this.data.roiSelected||!this._previewBounds||!this._frameShape)return;
    this.remember();this._map=mapping(this.data.previewRegion,this._previewBounds,this._frameShape);
    this._core.reset(this._map.offsetX,this._map.offsetY);this._hasMeasured=false;this._events=[];this._eventsTruncated=false;this._history=[];this._startedAt=null;this._lastAttempt=-1;
    this._lastReceive=this._clock();this.setData({phase:'calibrating',status:'标定中',countText:'—',complete:true,mainLabel:'停止',points:[]});
    this._pipeline=new LatestFrame(item=>this.processFrame(item),error=>{this.interrupt();this.showMessage('测量已中断',error.message);});
    this._watchdog=setInterval(()=>{if(this.isRunning()&&this._clock()-this._lastReceive>.35){this.interrupt();this.setData({status:'画面已中断 · 不完整'});}},100);
  },
  processFrame({frame,t,id}){
    if(!this.isRunning())return;this._lastReceive=t;prepare(frame,this._map,this._core.input());let result;
    if(this.data.phase==='calibrating'){
      if(this._startedAt===null)this._startedAt=t;result=this._core.collect(t,id);const elapsed=t-this._startedAt;
      if(elapsed>=2&&result.state===1&&elapsed-this._lastAttempt>=1){this._lastAttempt=elapsed;if(this._core.start()){result=this._core.snapshot();this._hasMeasured=true;this.setData({phase:'measuring',status:'计数中',countText:'0'});}}
      if(result.state===1&&(elapsed>=9.8||result.calibrationFrames>=256)){if(!this._core.start())result=this._core.endCalibration();else {result=this._core.snapshot();this._hasMeasured=true;}}
    }else result=this._core.push(t,id);
    for(const row of this._core.rows())if(row[3])this._events.push({frame:row[0],time:row[1],direction:row[3],confirmedFrame:row[4],confirmedAt:row[5]});
    if(this._events.length>512){this._events=this._events.slice(-512);this._eventsTruncated=true;}
    if(result.nodes&&result.state===2){this._history.push({t,signals:result.signals.slice()});while(this._history.length>150||(this._history.length>1&&this._history[0].t<t-2))this._history.shift();}
    this._lastResult=result;this.display(result,false);
    if(result.state>=3){this.stopFeed();this.remember();}
  },
  display(result,force){
    this._lastResult=result;const now=this._clock?this._clock():0;if(!force&&now-this._lastUI<.1&&result.state<=2)return;this._lastUI=now;
    const names={1:'标定中',2:'计数中',3:'不完整',4:'已停止',5:'标定失败'};
    let status=names[result.state];if(result.state===1)status=`标定中 ${Math.min(10,Math.round((this._lastReceive-(this._startedAt===null?this._lastReceive:this._startedAt))*10)/10)}s`;
    const points=this._map&&result.nodes?result.points.map((p,k)=>{const q=previewPoint(p,this._map);return {id:k,color:COLORS[k],style:`left:${q.x-4}px;top:${q.y-4}px;background:${COLORS[k]};`,visible:q.x>=0&&q.y>=0&&q.x<=this._map.previewWidth&&q.y<=this._map.previewHeight};}):[];
    this.setData({phase:result.state===1?'calibrating':result.state===2?'measuring':'stopped',status,countText:!this._hasMeasured||result.state===1||result.state===5?'—':String(result.count),complete:result.complete,mainLabel:result.state<=2?'停止':'新测量',points});
    this.drawSignals(result);
  },
  toggleSignals(){this.setData({showSignals:!this.data.showSignals},()=>{this.initCanvas();this.measurePreview();});if(this.isRunning())this.interrupt();this.clearSelection();},
  initCanvas(){if(!this.data.showSignals)return;wx.createSelectorQuery().in(this).select('#signals').fields({node:true,size:true}).exec(res=>{if(!res[0]||!res[0].node)return;const info=wx.getWindowInfo?wx.getWindowInfo():wx.getSystemInfoSync(),dpr=info.pixelRatio||1;const entry=res[0];entry.node.width=entry.width*dpr;entry.node.height=entry.height*dpr;const context=entry.node.getContext('2d');context.scale(dpr,dpr);this._canvas={ctx:context,width:entry.width,height:entry.height};this.drawSignals(this._lastResult);});},
  drawSignals(result){
    if(!this.data.showSignals||!this._canvas)return;const {ctx:c,width:w,height:h}=this._canvas;c.clearRect(0,0,w,h);
    const split=w*.69;c.strokeStyle='#203129';c.lineWidth=1;c.beginPath();c.moveTo(6,h/2);c.lineTo(split-10,h/2);c.moveTo(split,8);c.lineTo(split,h-8);c.stroke();
    const data=this._history;
    if(data.length>1){const first=data[0].t,last=data[data.length-1].t;for(let k=0;k<Math.min(5,result?result.nodes+1:0);k++){c.strokeStyle=COLORS[k];c.lineWidth=k?1.1:1.4;c.beginPath();for(let i=0;i<data.length;i++){const x=6+(split-22)*(data[i].t-first)/Math.max(.001,last-first),y=h/2-Math.max(-2,Math.min(2,data[i].signals[k]))*(h-16)/4;if(i)c.lineTo(x,y);else c.moveTo(x,y);}c.stroke();}}
    const x=split+(w-split)/2,y=h/2,r=Math.min((w-split)/2-12,h/2-9);c.strokeStyle='#304337';c.beginPath();c.arc(x,y,r,0,Math.PI*2);c.stroke();
    if(result&&result.nodes)for(let k=0;k<result.nodes;k++){const phase=result.phases[k],px=x+r*Math.cos(phase),py=y-r*Math.sin(phase);c.strokeStyle=COLORS[k+1];c.beginPath();c.moveTo(x,y);c.lineTo(px,py);c.stroke();c.fillStyle=COLORS[k+1];c.beginPath();c.arc(px,py,2.5,0,Math.PI*2);c.fill();}
  },
  onResize(){this.interrupt();this.clearSelection();const info=wx.getWindowInfo?wx.getWindowInfo():wx.getSystemInfoSync();this.setData({viewportHeight:info.windowHeight});if(this.data.cameraEnabled)this.measurePreview();this.initCanvas();},
  onHide(){this.stopCamera();},
  onUnload(){this._active=false;this.stopCamera();if(this._core)this._core.destroy();}
});
