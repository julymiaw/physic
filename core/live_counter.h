#pragma once
#include "../research/phase_core.h"

namespace live {
constexpr int size=96;
enum State {calibrating=1,measuring=2,invalid=3,finished=4,init_failed=5,ready=6};
enum Reason {none=0,need_calibration=1,tracking_lost=2,weak_phase=3,bad_time=4,time_gap=5,tail_unconfirmed=6,bad_input=7,calibration_failed=8,interrupted=9};
struct EventRow {Frame frame;Feature feature;Row decision;double confirmation_time;uint32_t confirmation_source;};
class Counter {
 public:
 std::array<float,size*size> input{};
 std::array<double,64> result{};
 std::array<double,26> row_result{};
 std::vector<EventRow> emitted;
 std::vector<Frame> calibration,queue;
 std::vector<Feature> features;
 std::vector<Track> tracks;
 std::unique_ptr<CalibrationModel> model;
 double ox=16,oy=16,last_time=-1,gap_limit=.2,cycles=0,stop_time=-1;
 std::array<double,4> phase_origins{};
 std::array<bool,4> phase_initialized{};
 int state=calibrating,reason=none,confirmed=0,fused=0;
 bool complete=true,relative_measurement=false,has_observation=false,stop_requested=false;
 Feature last_feature;
 Matrix transform=screen::identity;
 Counter(){calibration.reserve(256);queue.reserve(5);features.reserve(5);publish();}
 void reset(double x,double y){
  calibration.clear();queue.clear();features.clear();tracks.clear();model.reset();emitted.clear();
  ox=x;oy=y;last_time=-1;gap_limit=.2;cycles=0;phase_origins.fill(0);phase_initialized.fill(false);
  state=calibrating;reason=none;confirmed=fused=0;complete=true;relative_measurement=false;has_observation=false;stop_requested=false;stop_time=-1;last_feature=Feature{};transform=screen::identity;publish();
 }
 void fail(int why){complete=false;state=invalid;reason=why;publish();}
 bool validate(double t){
  if(!std::isfinite(t)||t<0||t<=last_time){fail(bad_time);return false;}
  if((state==measuring||(state==ready&&has_observation)||state==finished)&&last_time>=0&&t-last_time>gap_limit){fail(time_gap);return false;}
  for(float v:input)if(!std::isfinite(v)||v<0||v>1.000001f){fail(bad_input);return false;}
  return true;
 }
 bool collect(double t,uint32_t source){
  emitted.clear();if(state!=calibrating)return false;if(!validate(t))return false;
  if(calibration.size()>=256||(!calibration.empty()&&t-calibration.front().t>10.)){
   state=init_failed;reason=calibration_failed;publish();return false;
  }
  Frame f{t,source,Image(input.begin(),input.end())};calibration.push_back(std::move(f));last_time=t;publish();return true;
 }
 bool start(){
  if(state!=calibrating)return false;
  if(calibration.size()<24){reason=need_calibration;publish();return false;}
  // Retry works on a copy. Failed attempts must not mutate stored raw calibration.
  auto aligned=calibration;
  try {
   auto learned=std::make_unique<CalibrationModel>(aligned,size,ox,oy);
   Image intervals;for(size_t i=1;i<calibration.size();++i)intervals.push_back(calibration[i].t-calibration[i-1].t);
   gap_limit=std::min(.2,3*screen::quantile(intervals,.5));
   model=std::move(learned);tracks.assign(model->nodes.size(),Track{});state=measuring;reason=none;publish();return true;
  }catch(const std::exception&){reason=calibration_failed;publish();return false;}
 }
 // Product flow: calibration learns once; monitoring does not accumulate counts.
 bool arm(){if(state!=measuring||!queue.empty())return false;state=ready;has_observation=false;publish();return true;}
 bool monitor(double t,uint32_t source){
  emitted.clear();if((state!=ready&&state!=finished)||!complete)return false;
  if(!validate(t))return false;
  try{
   Image pixels(input.begin(),input.end());auto o=model->tracker->push(pixels);transform=o.transform;
   if(!o.accepted){last_time=t;fail(tracking_lost);return false;}
   Frame f{t,source,std::move(pixels)};f.h=o.transform;f.valid=true;
   last_feature=observe(f,*model,size,ox,oy);int valid=0;
   for(size_t k=0;k<model->nodes.size();++k)valid+=last_feature.radius[k]>=model->nodes[k].min_radius;
   last_time=t;if(valid<2){fail(weak_phase);return false;}
   has_observation=true;publish();return true;
  }catch(const std::exception&){last_time=t;fail(tracking_lost);return false;}
 }
 bool begin_measurement(){
  if((state!=ready&&state!=finished)||!complete||!has_observation)return false;
  queue.clear();features.clear();emitted.clear();tracks.assign(model->nodes.size(),Track{});
  phase_initialized.fill(false);phase_origins.fill(0);
  for(size_t k=0;k<tracks.size();++k)if(last_feature.radius[k]>=model->nodes[k].min_radius){
   auto& tr=tracks[k];tr.previous=tr.unwrapped=last_feature.phase[k];tr.last=0;
   tr.integer=tr.origin=int(std::floor(tr.previous/(2*pi)));
   phase_origins[k]=tr.unwrapped;phase_initialized[k]=true;
  }
  relative_measurement=true;stop_requested=false;stop_time=-1;confirmed=fused=0;cycles=0;state=measuring;reason=none;publish();return true;
 }
 bool stop_measurement(){
  if(state!=measuring||!relative_measurement||stop_requested)return false;
  stop_requested=true;stop_time=last_time;
  if(queue.empty()){state=finished;stop_requested=false;}
  publish();return true;
 }
 // Whole cycles relative to the measured starting phase, with symmetric deadband.
 static int relative_count(double value,int prior){
  while(value>(prior<0?prior:prior+1)+.02)++prior;
  while(value<(prior>0?prior:prior-1)-.02)--prior;
  return prior;
 }
 void end_calibration(){if(state==calibrating){state=init_failed;reason=calibration_failed;publish();}}
 bool commit(bool eof,double received_time,uint32_t received_source){
  if(features.empty())return true;
  auto candidate=tracks;auto d=advance(model->nodes,features,candidate,0,int(features.size())-1,fused);
  int delta=d.count-(relative_measurement?fused:confirmed);
  Image proposed_phases;for(size_t k=0;k<candidate.size();++k)if(candidate[k].last>=0&&phase_initialized[k]&&features.front().radius[k]>=model->nodes[k].min_radius)proposed_phases.push_back((candidate[k].unwrapped-phase_origins[k])/(2*pi));
  double proposed_cycles=proposed_phases.empty()?cycles:screen::quantile(proposed_phases,.5);
  int proposed_count=relative_measurement?relative_count(proposed_cycles,confirmed):d.count;
  if(relative_measurement)delta=proposed_count-confirmed;
  bool pending=eof&&delta!=0;
  if(!queue.front().valid||d.valid<2||(relative_measurement&&proposed_phases.size()<2)||std::abs(delta)>1)pending=true;
  const auto& front=queue.front();
  if(pending){
   complete=false;reason=eof?tail_unconfirmed:weak_phase;
  }else{
   tracks=std::move(candidate);fused=d.count;confirmed=proposed_count;
   Image phases;for(size_t k=0;k<tracks.size();++k)if(tracks[k].last>=0){
    if(!phase_initialized[k]&&!relative_measurement){phase_origins[k]=tracks[k].unwrapped;phase_initialized[k]=true;}
    if(phase_initialized[k])phases.push_back((tracks[k].unwrapped-phase_origins[k])/(2*pi));
   }
   if(relative_measurement)cycles=proposed_cycles;else if(!phases.empty())cycles=screen::quantile(phases,.5);
  }
  Row row{int(front.source),confirmed,pending?0:delta,int(received_source),d.valid,d.repairs,d.ambiguous,complete,pending,d.pair};
  emitted.push_back({front,features.front(),row,received_time,received_source});
  queue.erase(queue.begin());features.erase(features.begin());
  if(pending&&!eof)state=invalid;
  return !pending;
 }
 bool push(double t,uint32_t source){
  emitted.clear();if(state!=measuring)return false;if(!validate(t))return false;
  Frame f{t,source,Image(input.begin(),input.end())};
  try {
   auto o=model->tracker->push(f.pixels);f.h=o.transform;f.valid=o.accepted;f.iou=o.iou_after;transform=f.h;
   if(!f.valid){last_time=t;fail(tracking_lost);return false;}
   last_feature=observe(f,*model,size,ox,oy);int valid=0;
   for(size_t k=0;k<model->nodes.size();++k)valid+=last_feature.radius[k]>=model->nodes[k].min_radius;
   if(valid<2){last_time=t;fail(weak_phase);return false;}
   f.pixels.clear();queue.push_back(std::move(f));features.push_back(last_feature);last_time=t;
   if(queue.size()==5)commit(false,t,source);
   if(state==measuring&&stop_requested&&(queue.empty()||queue.front().t>stop_time)){
    queue.clear();features.clear();stop_requested=false;state=finished;has_observation=true;
   }
   publish();return state==measuring||state==finished;
  }catch(const std::exception&){last_time=t;fail(tracking_lost);return false;}
 }
 void finish(){
  emitted.clear();if(state==calibrating){end_calibration();return;}
  if(state!=measuring)return;
  double t=last_time;uint32_t source=queue.empty()?0:queue.back().source;
  while(!queue.empty())commit(true,t,source);
  state=complete?finished:invalid;publish();
 }
 void interrupt(){emitted.clear();if(state==measuring)fail(interrupted);else if(state==ready||state==finished)fail(interrupted);else if(state==calibrating)end_calibration();}
 void publish(){
  result.fill(0);result[0]=state;result[1]=confirmed;result[2]=cycles;result[3]=complete;result[4]=reason;
  result[5]=queue.size();result[6]=calibration.size();result[7]=model?model->nodes.size():0;result[8]=last_time;
  result[36]=has_observation;result[37]=stop_requested;result[38]=relative_measurement?stop_time:0;result[9]=emitted.size();result[10]=last_feature.a;result[11]=gap_limit;
  if(model){
   result[12]=model->cx;result[13]=model->cy;
   for(size_t k=0;k<model->nodes.size();++k){
    // Aligned physical phase is used for every phase arrow, not each node's own brightness minimum.
    result[14+k]=last_feature.phase[k];result[18+k]=last_feature.radius[k];
    // Reconstruct normalized auxiliary brightness from the phase before common zero alignment.
    double raw=model->nodes[k].orientation*wrap(last_feature.phase[k]+model->nodes[k].zero);
    result[22+k]=last_feature.radius[k]*std::sin(raw);
   }
   for(size_t k=0;k<=model->nodes.size();++k){double x=k?model->nodes[k-1].x:model->cx,y=k?model->nodes[k-1].y:model->cy;
    auto p=screen::project(transform,x+ox,y+oy);result[26+2*k]=p[0];result[27+2*k]=p[1];
   }
  }
 }
 const double* row(size_t i){
  if(i>=emitted.size())return nullptr;row_result.fill(0);const auto& e=emitted[i];const auto& r=e.decision;
  row_result[0]=e.frame.source;row_result[1]=e.frame.t;row_result[2]=r.count;row_result[3]=r.delta;
  row_result[4]=e.confirmation_source;row_result[5]=e.confirmation_time;row_result[6]=r.valid;row_result[7]=r.repairs;
  row_result[8]=r.ambiguous;row_result[9]=r.complete;row_result[10]=r.pending;row_result[11]=e.feature.a;
  row_result[12]=e.frame.valid;row_result[13]=e.frame.iou;
  for(int k=0;k<4;++k){row_result[14+2*k]=r.pairs[k];row_result[15+2*k]=e.feature.phase[k];}
  return row_result.data();
 }
};
}
