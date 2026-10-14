const {loadCounter}=require('../../utils/counter');
const {mapping,prepare,previewPoint,LatestFrame,receiveClock}=require('../../utils/camera');
const {StreamDiagnostics}=require('../../utils/stream-diagnostics');
const COLORS=['#91a69b','#55aa92','#688bb8','#b5985e','#9982ab'];
const REASONS={1:'标定画面不足',2:'光屏跟踪失效',3:'相位信号不可靠',4:'接收时间异常',5:'有效帧间隔过大',6:'片尾仍有未确认事件',7:'帧数据无效',8:'标定未获得稳定的单向周期',9:'测量已中断'};
Page({
  data:{cameraEnabled:false,cameraReady:false,coreReady:false,roiSelected:false,selectionStyle:'',previewRegion:null,overlayStyle:'',points:[],showSignals:true,helpVisible:false,helpTitle:'使用方法',helpText:'开启相机，按住并拖动，框选整个光屏。\n\n点击标定，缓慢、连续地单向移动，使中心暗纹至少出现两次；标定最长10秒。标定方向定义为正方向。\n\n标定完成后，停下旋钮并记录起始读数，再点击“开始测量”。标定过程不计入测量。\n\n沿原方向旋转，正向增加、反向减少；短暂往返按净变化处理。停下旋钮后点击“停止”，再记录终止读数。信号仍有效时可直接“再次测量”。丢失跟踪或中断会冻结计数并标为不完整。\n\n曲线颜色对应画面采样点，相位图显示各路共同暗心零点的相位。点击“信号”可收起。\n\n画面仅在本地处理。',phase:'idle',status:'加载中',countText:'—',complete:true,canBegin:false,mainLabel:'标定',viewportHeight:600,safeBottom:0},
  onLoad(){
    this._active=true;this._source=0;this._history=[];this._lastUI=0;this._events=[];
    const info=wx.getWindowInfo?wx.getWindowInfo():wx.getSystemInfoSync();
    this.setData({viewportHeight:info.windowHeight,safeBottom:Math.max(0,info.screenHeight-(info.safeArea?info.safeArea.bottom:info.screenHeight)),helpVisible:!wx.getStorageSync('fringe-help-v2')});
    this.initializeCore();
  },
  async initializeCore(){
    let core;
    try{
      core=await loadCounter();const clock=await receiveClock();
      if(!this._active){core.destroy();return;}
      this._core=core;this._clock=clock;this._clockInfo={source:clock.source,unitsPerSecond:clock.unitsPerSecond};
      this.setData({coreReady:true,status:this.data.roiSelected?'待标定':'待圈选'});
    }catch(error){if(core)core.destroy();if(this._active){this.setData({coreReady:false,status:'初始化失败'});this.showMessage('无法启动',error.message);}}
  },
  createStreamDiagnostics(){
    let device={},account={};try{const d=wx.getDeviceInfo?wx.getDeviceInfo():wx.getSystemInfoSync();device={brand:d.brand,model:d.model,platform:d.platform,system:d.system};}catch(_){}
    try{account=wx.getAccountInfoSync().miniProgram;}catch(_){}
    this._streamCoreSignature=null;this._streamDiag=new StreamDiagnostics({build:'stream-diagnostics-2',clock:this._clockInfo,device,envVersion:account.envVersion,version:account.version,preview:this._previewBounds,frame:this._frameShape},{save:report=>wx.setStorageSync('fringe-stream-debug',report)});
    this._streamDiag.event('calibration-begin',{map:this._map});
  },
  copyStreamDiagnostics(){
    const report=this._streamDiag?this._streamDiag.snapshot():wx.getStorageSync('fringe-stream-debug');
    if(report)wx.setClipboardData({data:JSON.stringify(report)});
  },
  checkFrameWatchdog(){
    if(!this.isRunning())return;
    try{
      const now=this._clock(),age=now-this._lastCallbackTime;
      if(this._streamDiag){this._streamDiag.dropped=this._pipeline?this._pipeline.dropped:this._streamDiag.dropped;this._streamDiag.summary(this.data.phase);}
      const recovering=['calibrating','ready'].includes(this.data.phase)&&now<(this._fitGraceUntil||0);
      if(age>.35&&!recovering){
        if(this._streamDiag){this._streamDiag.event('watchdog-timeout',{phase:this.data.phase,callbackAgeMs:age*1000,processedTimestampAgeMs:(now-this._lastReceive)*1000,thresholdMs:350,pending:!!(this._pipeline&&this._pipeline.pending),core:this._lastResult?{state:this._lastResult.state,reason:this._lastResult.reason,calibrationFrames:this._lastResult.calibrationFrames}:null});this._streamDiag.persist();}
        this.interrupt();this.setData({status:'画面已中断 · 不完整'});
      }
    }catch(error){if(this._streamDiag)this._streamDiag.event('clock-error',{message:error.message});this.interrupt();this.showMessage('接收时间异常',error.message+'请重新打开小程序。');}
  },
  tryCalibration(){
    const started=Date.now();if(this._streamDiag)this._streamDiag.event('calibration-fit-start',{frames:this._core.snapshot().calibrationFrames});
    const success=this._core.start();const ended=Date.now();
    // Let callbacks queued during synchronous initialization get an event-loop turn.
    // This applies only to calibration, never an active measurement.
    this._fitGraceUntil=this._clock()+.15;
    if(this._streamDiag)this._streamDiag.event('calibration-fit-end',{success,fitMs:ended-started,coreState:this._core.snapshot().state,reason:this._core.snapshot().reason});
    return success;
  },
  onReady(){this.initCanvas();},
  // Invoked only by the development verification entry; never connected to product UI.
  verifyReplay(name){return require('../../utils/verify-replay').verify(name);},
  noop(){},
  closeHelp(){wx.setStorageSync('fringe-help-v2',true);this.setData({helpVisible:false});},
  showHelp(){if(['calibrating','measuring','stopping'].includes(this.data.phase))return;if(this.data.cameraEnabled)this.stopCamera();this.showMessage('使用方法',this.data.helpText);},
  showMessage(title,text){if(this.data.cameraEnabled)this.stopCamera();this.setData({helpTitle:title,helpBody:text,helpVisible:true});},
  showStatus(){
    if(['calibrating','measuring','stopping'].includes(this.data.phase))return;
    const last=wx.getStorageSync('fringe-last-session'),lost=wx.getStorageSync('fringe-last-incomplete');
    const history=lost?'\n\n此前有不完整测量段（净'+lost.count+'圈），其丢失信息没有被重新标定补回。当前计数只代表当前段。':'';
    this.showMessage('测量状态',(this._lastResult&&!this._hasMeasured?(REASONS[this._lastResult.reason]||this.data.status)+'。尚未开始有效测量。':this._lastResult?`${REASONS[this._lastResult.reason]||this.data.status}。当前净计数：${this._lastResult.count}圈。${this._lastResult.complete?'':'这一段存在丢失信息，重新标定不会补回。'}`:(last?`上一段净计数：${last.count}圈；${last.complete?'完整':'不完整'}。`:'圈选光屏后开始标定。'))+history);
  },
  isRunning(){return ['calibrating','measuring','stopping','ready','review'].includes(this.data.phase);},
  toggleCamera(){if(this.data.cameraEnabled)this.stopCamera();else this.setData({cameraEnabled:true,cameraReady:false,points:[]},()=>this.measurePreview());},
  onCameraReady(){
    if(!this.data.cameraEnabled)return;this.setData({cameraReady:true});this.measurePreview();
    this._listener=wx.createCameraContext().onCameraFrame(frame=>{
      this._frameShape={width:frame.width,height:frame.height};
      if(this._pipeline){try{const t=this._clock();this._lastCallbackTime=t;if(this._streamDiag)this._streamDiag.arrival(frame,t);this._pipeline.offer(frame,t,this._source++);}catch(error){this.interrupt();this.showMessage('接收时间异常',error.message+'请重新打开小程序。');}}
    });
    this._listener.start({fail:error=>{this.stopCamera();this.showMessage('相机帧不可用',error.errMsg||'请在手机微信中重试。');}});
  },
  measurePreview(done){
    wx.createSelectorQuery().in(this).select('.preview').boundingClientRect(rect=>{
      if(this.data.cameraEnabled&&rect&&rect.width>0&&rect.height>0){
        this._previewBounds=rect;this.setData({overlayStyle:`width:${rect.width}px;height:${rect.height}px;`});
      }
      if(done)done();
    }).exec();
  },
  stopFeed(){if(this._streamDiag&&this._pipeline){this._streamDiag.summary(this.data.phase,true);this._streamDiag.event('feed-stop',{phase:this.data.phase});this._streamDiag.persist();}if(this._pipeline)this._pipeline.stop();this._pipeline=null;if(this._watchdog)clearInterval(this._watchdog);this._watchdog=null;},
  remember(){if(this._lastResult&&this._hasMeasured){const saved={mode:'relative-start-phase',clock:this._clockInfo,startedAt:this._measurementStartedAt,endedAt:this._lastResult.stopTime>=0?this._lastResult.stopTime:this._lastResult.timestamp,cycles:this._lastResult.cycles,count:this._lastResult.count,complete:this._lastResult.complete,reason:this._lastResult.reason,events:this._events.slice(-512),eventsTruncated:!!this._eventsTruncated};wx.setStorageSync('fringe-last-session',saved);if(!saved.complete)wx.setStorageSync('fringe-last-incomplete',saved);}},
  interrupt(){
    const reviewing=this.data.phase==='review';this.stopFeed();if(this._core&&this.isRunning()){const result=this._core.interrupt();if(reviewing)this._hasMeasured=false;this.display(result,true);if(!reviewing)this.remember();}
  },
  stopCamera(){
    this.interrupt();if(this._listener)this._listener.stop();this._listener=null;this._dragStart=null;this._previewBounds=null;this._frameShape=null;
    this.setData({phase:'idle',mainLabel:'标定',overlayStyle:'',cameraEnabled:false,cameraReady:false,roiSelected:false,selectionStyle:'',previewRegion:null,points:[]});
  },
  onCameraError(event){if(this._streamDiag)this._streamDiag.event('camera-error',{detail:event.detail});this.stopCamera();this.showMessage('相机无法开启',(event.detail&&event.detail.errMsg)||'请检查相机权限后重试。');},
  onCameraStop(){if(this._streamDiag)this._streamDiag.event('camera-stop');if(this.data.cameraEnabled){this.stopCamera();this.setData({status:'相机已中断'});}},
  clearSelection(){
    this.interrupt();this._dragStart=null;this._dragCurrent=null;this.setData({phase:'idle',mainLabel:'标定',roiSelected:false,selectionStyle:'',previewRegion:null,points:[]});
    if(!this._lastResult||this._lastResult.state===1)this.setData({status:'待圈选'});
  },
  traceTouch(stage,event,point){
    const pick=t=>{if(!t)return null;const out={};for(const k of ['identifier','clientX','clientY','pageX','pageY','x','y'])if(t[k]!==undefined)out[k]=t[k];return out;};
    const row={stage,touches:Array.from(event.touches||[]).map(pick),changed:Array.from(event.changedTouches||[]).map(pick),detail:event.detail||null,point,bounds:this._previewBounds?{left:this._previewBounds.left,top:this._previewBounds.top,width:this._previewBounds.width,height:this._previewBounds.height}:null};
    this._touchTrace=(this._touchTrace||[]).concat(row).slice(-16);
    try{if(wx.getAccountInfoSync&&wx.getAccountInfoSync().miniProgram.envVersion==='develop'){
      console.info('[ROI]',JSON.stringify(row));wx.setStorageSync('fringe-roi-debug',this._touchTrace);
    }}catch(_){}
  },
  previewPoint(touch){
    const rect=this._previewBounds;if(!rect||!touch)return null;
    // A native/remote event may carry zero client coordinates with valid page/local
    // fields. (0,0) is outside this preview, so it must not become a valid corner.
    const candidates=[];
    for(const [kx,ky,local] of [['clientX','clientY',false],['pageX','pageY',false],['x','y',true]]){
      if(!Number.isFinite(touch[kx])||!Number.isFinite(touch[ky]))continue;
      const x=touch[kx],y=touch[ky];
      if(!local&&x===0&&y===0&&(rect.left>0||rect.top>0))continue;
      candidates.push({x:local?x:x-rect.left,y:local?y:y-rect.top});
    }
    if(!candidates.length)return null;
    const p=candidates.find(p=>p.x>=0&&p.y>=0&&p.x<=rect.width&&p.y<=rect.height)||candidates[0];
    return {x:Math.max(0,Math.min(rect.width,p.x)),y:Math.max(0,Math.min(rect.height,p.y))};
  },
  onSelectionStart(event){
    if(!this.data.cameraEnabled||this.isRunning())return;
    const touch=(event.touches&&event.touches[0])||(event.changedTouches&&event.changedTouches[0]);
    if(!this._previewBounds){this.measurePreview(()=>{if(this._previewBounds)this.onSelectionStart(event);});return;}
    const p=this.previewPoint(touch);this.traceTouch('start',event,p);if(!p){this.setData({status:'未收到有效触摸位置'});return;}
    this._touchInfo={keys:Object.keys(touch),point:p};
    this._dragStart=p;this._dragCurrent=p;
    this.setData({roiSelected:false,previewRegion:null,points:[]});
    this.drawCorner(p);
  },
  regionFromPoint(p){if(!p||!this._dragStart)return null;return {x:Math.min(p.x,this._dragStart.x),y:Math.min(p.y,this._dragStart.y),width:Math.abs(p.x-this._dragStart.x),height:Math.abs(p.y-this._dragStart.y)};},
  drawCorner(p){const b=this._previewBounds;this.drawRegion({x:Math.max(0,Math.min(b.width-8,p.x-4)),y:Math.max(0,Math.min(b.height-8,p.y-4)),width:8,height:8});},
  drawRegion(region){this.setData({selectionStyle:`left:${region.x}px;top:${region.y}px;width:${region.width}px;height:${region.height}px;`});},
  onSelectionMove(event){
    if(!this._dragStart)return;const p=this.previewPoint((event.touches&&event.touches[0])||(event.changedTouches&&event.changedTouches[0]));
    if(p){this._dragCurrent=p;this.drawRegion(this.regionFromPoint(p));}
  },
  onSelectionEnd(event){
    if(!this._dragStart){this.traceTouch('end-without-start',event,null);return;}
    const p=this.previewPoint(event.changedTouches&&event.changedTouches[0])||this._dragCurrent;
    this.traceTouch('end',event,p);const r=this.regionFromPoint(p);
    this._dragStart=null;this._dragCurrent=null;
    if(!r)return;

    if(r.width<32||r.height<32){this.clearSelection();return;}
    const b=this._previewBounds;this.drawRegion(r);this.setData({roiSelected:true,previewRegion:{x:r.x/b.width,y:r.y/b.height,width:r.width/b.width,height:r.height/b.height},status:'待标定'});
  },
  onSelectionCancel(event={}){
    if(this.isRunning())return;
    this.traceTouch('cancel',event,this._dragCurrent||null);
    this._dragStart=null;this._dragCurrent=null;
    if(!this.data.roiSelected)this.clearSelection();
  },
  mainAction(){
    if(this.data.phase==='stopping')return;
    if(this.data.phase==='ready'||this.data.phase==='review'){
      // Include the latest retained observation before latching the starting phase.
      if(this._pipeline&&this._pipeline.pending)this._pipeline.drain();
      if(!['ready','review'].includes(this.data.phase))return;
      let now;try{now=this._clock();}catch(error){this.interrupt();this.showMessage('接收时间异常',error.message+'请重新打开小程序。');return;}
      if(now-this._lastReceive>this._lastResult.gapLimit){this.interrupt();return;}
      this.remember();
      if(this._core.beginMeasurement()){
        this._measurementStartedAt=this._lastResult.timestamp;this._hasMeasured=true;this._events=[];this._eventsTruncated=false;
        this.display(this._core.snapshot(),true);
      }
      return;
    }
    if(this.data.phase==='measuring'){
      if(this._pipeline&&this._pipeline.pending)this._pipeline.drain();
      if(this.data.phase!=='measuring')return;
      if(this._core.stopMeasurement())this.display(this._core.snapshot(),true);return;
    }
    if(this.data.phase==='calibrating'){
      this.stopFeed();this.display(this._core.endCalibration(),true);return;
    }
    if(!this.data.coreReady||!this.data.roiSelected||!this._previewBounds||!this._frameShape)return;
    this.remember();this._map=mapping(this.data.previewRegion,this._previewBounds,this._frameShape);
    this._core.reset(this._map.offsetX,this._map.offsetY);this._hasMeasured=false;this._events=[];this._eventsTruncated=false;this._history=[];this._startedAt=null;this._lastAttempt=-1;
    this._lastReceive=this._clock();this._lastCallbackTime=this._lastReceive;this._fitGraceUntil=0;this.setData({phase:'calibrating',status:'标定中',countText:'—',complete:true,mainLabel:'停止',points:[]});
    this.createStreamDiagnostics();
    this._pipeline=new LatestFrame(item=>this.processFrame(item),error=>{this.interrupt();this.showMessage('测量已中断',error.message);});
    this._watchdog=setInterval(()=>this.checkFrameWatchdog(),100);
  },
  processFrame({frame,t,id}){
    if(!this.isRunning())return;const handlerStart=Date.now();const phaseBefore=this.data.phase;const reviewing=this.data.phase==='review',stopping=this.data.phase==='stopping';this._lastReceive=t;prepare(frame,this._map,this._core.input());const prepareEnd=Date.now();let result;
    if(this.data.phase==='calibrating'){
      if(this._startedAt===null)this._startedAt=t;result=this._core.collect(t,id);const elapsed=t-this._startedAt;
      if(elapsed>=2&&result.state===1&&elapsed-this._lastAttempt>=1){this._lastAttempt=elapsed;if(this.tryCalibration()){this._core.arm();result=this._core.snapshot();}}
      if(result.state===1&&(elapsed>=9.8||result.calibrationFrames>=256)){if(!this.tryCalibration())result=this._core.endCalibration();else {this._core.arm();result=this._core.snapshot();}}
    }else if(this.data.phase==='measuring'||this.data.phase==='stopping')result=this._core.push(t,id);
    else result=this._core.monitor(t,id);
    const coreEnd=Date.now();
    this.captureEvents();
    if(result.nodes&&[2,4,6].includes(result.state)){this._history.push({t,signals:result.signals.slice()});while(this._history.length>150||(this._history.length>1&&this._history[0].t<t-2))this._history.shift();}
    if(reviewing&&result.state===3)this._hasMeasured=false;
    this._lastResult=result;this.display(result,false);
    if(this._streamDiag){this._streamDiag.processedFrame({frame:id,timestamp:t,phase:phaseBefore,prepareMs:prepareEnd-handlerStart,coreMs:coreEnd-prepareEnd,totalMs:Date.now()-handlerStart,coreState:result.state,coreReason:result.reason,calibrationFrames:result.calibrationFrames,dropped:this._pipeline?this._pipeline.dropped:0});const signature=[result.state,result.reason,this.data.phase].join(':');if(signature!==this._streamCoreSignature){this._streamDiag.event('core-state',{frame:id,from:phaseBefore,to:this.data.phase,state:result.state,reason:result.reason});this._streamCoreSignature=signature;}this._streamDiag.summary(this.data.phase);}
    if(stopping&&result.state===4)this.remember();
    if(result.state===3||result.state===5){this.stopFeed();this.remember();}
  },
  captureEvents(){
    for(const row of this._core.rows())if(row[3])this._events.push({frame:row[0],time:row[1],direction:row[3],confirmedFrame:row[4],confirmedAt:row[5]});
    if(this._events.length>512){this._events=this._events.slice(-512);this._eventsTruncated=true;}
  },
  display(result,force){
    const nextPhase=result.stopping?'stopping':{1:'calibrating',2:'measuring',3:'stopped',4:'review',5:'stopped',6:'ready'}[result.state];
    if(this.data.phase!==nextPhase)force=true;this._lastResult=result;const now=Date.now()/1000;if(!force&&now-this._lastUI<.1&&[1,2,4,6].includes(result.state))return;this._lastUI=now;
    const names={1:'标定中',2:'计数中',3:'不完整',4:'已停止',5:'标定失败',6:'待开始'};
    let status=result.stopping?'确认中':names[result.state];if(result.state===1)status=`标定中 ${Math.min(10,Math.round((this._lastReceive-(this._startedAt===null?this._lastReceive:this._startedAt))*10)/10)}s`;
    const points=this._map&&result.nodes?result.points.map((p,k)=>{const q=previewPoint(p,this._map);return {id:k,color:COLORS[k],style:`left:${q.x-4}px;top:${q.y-4}px;background:${COLORS[k]};`,visible:q.x>=0&&q.y>=0&&q.x<=this._map.previewWidth&&q.y<=this._map.previewHeight};}):[];
    this.setData({phase:nextPhase,status,countText:result.state===6?'0':!this._hasMeasured||result.state===1||result.state===5?'—':String(result.count),complete:result.complete,canBegin:result.canBegin,mainLabel:result.stopping?'确认中':result.state===6?'开始测量':result.state===4?'再次测量':result.state<=2?'停止':'重新标定',points});
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
  onHide(){if(this._streamDiag)this._streamDiag.event('page-hide');this.stopCamera();},
  onUnload(){this._active=false;this.stopCamera();if(this._core)this._core.destroy();}
});
