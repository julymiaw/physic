// Metadata only: no image buffers, pixels or screenshots are retained.
class StreamDiagnostics {
  constructor(meta={},options={}){
    this.now=options.now||(()=>Date.now());this.write=options.write||(row=>console.info('[STREAM]',JSON.stringify(row)));
    this.save=options.save||(()=>{});this.meta=meta;this.started=this.now();this.rows=[];this.received=0;this.processed=0;this.dropped=0;this.lastArrival=null;this.lastProcessed=null;this.lastSummary=this.started;this.lastTiming=null;
    this.event('session',meta);
  }
  event(type,data={}){const row={type,elapsedMs:this.now()-this.started,...data};this.rows.push(row);if(this.rows.length>96)this.rows.shift();try{this.write(row);}catch(_){}return row;}
  arrival(frame,t){
    const now=this.now(),gap=this.lastArrival===null?null:now-this.lastArrival;this.received++;this.lastArrival=now;
    this.lastArrivalTimestamp=t;this.shape={width:frame.width,height:frame.height,bytes:frame.data?frame.data.byteLength:null};
    if(this.received===1)this.event('first-frame',{...this.shape,timestamp:t});
    if(gap!==null&&gap>350)this.event('callback-gap',{gapMs:gap,timestamp:t});
  }
  processedFrame(data){this.processed++;this.lastProcessed=this.now();this.lastTiming=data;this.dropped=data.dropped||0;if(data.totalMs>100)this.event('slow-frame',data);}
  summary(phase,force=false){const now=this.now();if(!force&&now-this.lastSummary<1000)return;const dt=Math.max(1,now-this.lastSummary);const previous=this.previousReceived||0,processed=this.previousProcessed||0;
    this.event('summary',{phase,received:this.received,processed:this.processed,dropped:this.dropped,callbackFps:(this.received-previous)*1000/dt,processedFps:(this.processed-processed)*1000/dt,arrivalAgeMs:this.lastArrival===null?null:now-this.lastArrival,processedAgeMs:this.lastProcessed===null?null:now-this.lastProcessed,lastTiming:this.lastTiming});this.previousReceived=this.received;this.previousProcessed=this.processed;this.lastSummary=now;
  }
  snapshot(){return {schema:1,meta:this.meta,received:this.received,processed:this.processed,dropped:this.dropped,rows:this.rows.slice()};}
  persist(){try{this.save(this.snapshot());}catch(_){} }
}
module.exports={StreamDiagnostics};
