#pragma once
// Shared physical-phase model and event decision logic for Native replay and live WASM.
#include "screen_tracker.h"
#include <sstream>
#include <string>
#include <memory>
using screen::Image;
using screen::Matrix;
using screen::Point;
constexpr double pi=3.14159265358979323846;
using Clock=std::chrono::steady_clock;
struct Frame {double t;uint32_t source;Image pixels;Matrix h=screen::identity;bool valid=true;double iou=0;};
struct Node {double x,y,mid,scale,orientation,score,min_radius,zero=0;Image calibration;};
struct Feature {double a=0;std::array<double,4> phase{},radius{};bool valid=true;};
struct Track {double previous=0,unwrapped=0;int integer=0,origin=0,last=-1,outlier=0;};
struct Decision {int count=0,valid=0,repairs=0,ambiguous=0,direction=0;std::array<int,4> pair{};};
struct Row {int index,count,delta,confirmation,valid,repairs,ambiguous;bool complete,pending;std::array<int,4> pairs;};
inline double wrap(double x){return std::remainder(x,2*pi);}
inline double sample(const Frame& f,int n,double x,double y,double ox,double oy){auto p=screen::project(f.h,x+ox,y+oy);if(p[0]<0||p[1]<0||p[0]>n-1||p[1]>n-1)throw std::runtime_error("sample outside padded image");return screen::at(f.pixels,n,p[0],p[1]);}
inline double patch(const Frame& f,int n,double x,double y,double ox,double oy){double a=0;for(int j=-1;j<=1;++j)for(int i=-1;i<=1;++i)a+=sample(f,n,x+i,y+j,ox,oy);return a/9;}
inline double mean(const Image& a){return std::accumulate(a.begin(),a.end(),0.)/a.size();}
inline double correlation(const Image& a,const Image& b){double am=mean(a),bm=mean(b),aa=0,bb=0,ab=0;for(size_t i=0;i<a.size();++i){double x=a[i]-am,y=b[i]-bm;aa+=x*x;bb+=y*y;ab+=x*y;}return ab/std::max(1e-12,std::sqrt(aa*bb));}
inline bool solve4(double m[4][5],std::array<double,4>& v){for(int k=0;k<4;++k){int pivot=k;for(int j=k+1;j<4;++j)if(std::abs(m[j][k])>std::abs(m[pivot][k]))pivot=j;if(std::abs(m[pivot][k])<1e-10)return false;for(int j=k;j<5;++j)std::swap(m[k][j],m[pivot][j]);double d=m[k][k];for(int j=k;j<5;++j)m[k][j]/=d;for(int i=0;i<4;++i)if(i!=k){double a=m[i][k];for(int j=k;j<5;++j)m[i][j]-=a*m[k][j];}}for(int j=0;j<4;++j)v[j]=m[j][4];return true;}
inline Image gradient_smooth(const Image& a){std::array<double,5> k{};double total=0;for(int i=-2;i<=2;++i){k[i+2]=std::exp(-i*i/(2*.7*.7));total+=k[i+2];}for(double& v:k)v/=total;Image tmp(4096),b(4096);for(int y=0;y<64;++y)for(int x=0;x<64;++x)for(int j=-2;j<=2;++j)tmp[y*64+x]+=k[j+2]*a[y*64+std::clamp(x+j,0,63)];for(int y=0;y<64;++y)for(int x=0;x<64;++x)for(int j=-2;j<=2;++j)b[y*64+x]+=k[j+2]*tmp[std::clamp(y+j,0,63)*64+x];return b;}
inline Point gradient_center(const std::vector<Frame>& frames,int n,double ox,double oy,double& residual){
 struct Equation {std::array<double,4> a;double b,weight;};std::vector<Equation> equations;
 // Center fitting must ignore time-invariant screen texture and illumination.
 std::vector<Image> aligned;Image temporal_mean(4096);
 for(const auto& f:frames){Image image(4096);for(int y=0;y<64;++y)for(int x=0;x<64;++x){int p=y*64+x;image[p]=sample(f,n,x,y,ox,oy);temporal_mean[p]+=image[p]/frames.size();}aligned.push_back(std::move(image));}
 for(auto image:aligned){for(int p=0;p<4096;++p)image[p]-=temporal_mean[p];image=gradient_smooth(image);
  for(int y=10;y<54;++y)for(int x=10;x<54;++x){double r=std::hypot(x-31.5,y-31.5);if(r<4||r>22)continue;double gx=(image[y*64+x+1]-image[y*64+x-1])/2,gy=(image[(y+1)*64+x]-image[(y-1)*64+x])/2;double magnitude=std::hypot(gx,gy);if(magnitude<.02)continue;double xx=(x-31.5)/31.5,yy=(y-31.5)/31.5;equations.push_back({{2*gy*xx+2*gx*yy,2*gy*yy-2*gx*xx,gy,-gx},2*gx*yy,1});}
 }
 if(equations.size()<200)throw std::runtime_error("insufficient calibration gradients");std::array<double,4> v{};
 for(int iteration=0;iteration<6;++iteration){double m[4][5]{};for(auto e:equations)for(int i=0;i<4;++i){for(int j=0;j<4;++j)m[i][j]+=e.weight*e.a[i]*e.a[j];m[i][4]+=e.weight*e.a[i]*e.b;}if(!solve4(m,v))throw std::runtime_error("ill-conditioned gradient center");Image errors;for(auto e:equations){double err=-e.b;for(int j=0;j<4;++j)err+=e.a[j]*v[j];errors.push_back(std::abs(err));}residual=screen::quantile(errors,.5);double threshold=std::max(.003,2.2*residual);for(size_t j=0;j<equations.size();++j)equations[j].weight=std::min(1.,threshold/std::max(1e-12,errors[j]));}
 double a=v[0],b=v[1],c=1-a,det=a*c-b*b;if(a<=0||c<=0||det<.02)throw std::runtime_error("non-elliptic calibration gradients");double cx=31.5-15.75*(c*v[2]-b*v[3])/det,cy=31.5-15.75*(a*v[3]-b*v[2])/det;
 if(cx<16||cx>48||cy<16||cy>48)throw std::runtime_error("gradient center outside screen interior");return {cx,cy};
}

struct CalibrationModel {
 double cx=0,cy=0,mid0=0,scale0=0,center_residual=0,anchor_spread=0;
 Point center{};Image primary;std::vector<Node> nodes;std::vector<size_t> anchors;
 std::unique_ptr<screen::Tracker> tracker;
 CalibrationModel(std::vector<Frame>& calibration,int n,double ox,double oy){
  if(calibration.size()<24)throw std::runtime_error("calibration too short");
  Image reference(n*n);for(const auto& f:calibration)for(int p=0;p<n*n;++p)reference[p]+=f.pixels[p];
  for(double& v:reference)v/=calibration.size();tracker=std::make_unique<screen::Tracker>(reference,n);
  for(auto& f:calibration){auto o=tracker->push(f.pixels);f.h=o.transform;f.valid=o.accepted;if(!f.valid)throw std::runtime_error("calibration screen tracking failed");}
 center=gradient_center(calibration,n,ox,oy,center_residual);cx=center[0];cy=center[1];for(const auto& f:calibration)primary.push_back(patch(f,n,cx,cy,ox,oy));double lo0=screen::quantile(primary,.1),hi0=screen::quantile(primary,.9);mid0=(lo0+hi0)/2;scale0=(hi0-lo0)/2;if(scale0<.02)throw std::runtime_error("flat center signal");
 std::vector<Node> candidates;
 for(int direction=0;direction<8;++direction)for(int radius=2;radius<=16;++radius){double x=cx+radius*std::cos(direction*pi/4),y=cy+radius*std::sin(direction*pi/4);Image signal,radii;for(const auto& f:calibration)signal.push_back(patch(f,n,x,y,ox,oy));double lo=screen::quantile(signal,.1),hi=screen::quantile(signal,.9),mid=(lo+hi)/2,scale=(hi-lo)/2;if(scale<.02)continue;double previous_phase=0,net=0,absolute=0;for(size_t j=0;j<signal.size();++j){double a=(primary[j]-mid0)/scale0,b=(signal[j]-mid)/scale,ph=std::atan2(b,a);radii.push_back(std::hypot(a,b));if(j){double d=wrap(ph-previous_phase);net+=d;absolute+=std::abs(d);}previous_phase=ph;}if(std::abs(net)<3*pi||std::abs(net)<.9*absolute||*std::min_element(radii.begin(),radii.end())<.15)continue;double score=(1-std::abs(correlation(primary,signal)))*std::min(scale0,scale);candidates.push_back({x,y,mid,scale,net>=0?1.:-1.,score,.5*screen::quantile(radii,.1),0,signal});}
 while(nodes.size()<4){double best=-1;int which=-1;for(size_t j=0;j<candidates.size();++j){double diversity=1;bool near=false;for(auto q:nodes){near|=std::hypot(candidates[j].x-q.x,candidates[j].y-q.y)<3;diversity=std::min(diversity,1-std::abs(correlation(candidates[j].calibration,q.calibration)));}double score=candidates[j].score*std::sqrt(std::max(0.,diversity));if(!near&&score>best){best=score;which=int(j);}}if(which<0)break;nodes.push_back(candidates[which]);candidates.erase(candidates.begin()+which);}if(nodes.size()<2)throw std::runtime_error("insufficient phase pairs");
 // Common physical anchor: minima of the SAME center signal, never each node's minimum.
 double threshold=screen::quantile(primary,.25);anchors.clear();
 for(size_t i=0;i<primary.size();){if(primary[i]>threshold){++i;continue;}size_t first=i;while(i<primary.size()&&primary[i]<=threshold)++i;size_t at=first;for(size_t j=first;j<i;++j)if(primary[j]<primary[at])at=j;if(first>0&&i<primary.size())anchors.push_back(at);}
 // Multiple low-intensity fragments inside one revolution are one dark anchor.
 Image anchor_progress(primary.size());double previous_angle=0;for(size_t j=0;j<primary.size();++j){auto q=nodes[0];double a=q.orientation*std::atan2((q.calibration[j]-q.mid)/q.scale,(primary[j]-mid0)/scale0);anchor_progress[j]=j?anchor_progress[j-1]+wrap(a-previous_angle):a;previous_angle=a;}
 std::vector<size_t> unique;for(size_t j:anchors){if(!unique.empty()&&anchor_progress[j]-anchor_progress[unique.back()]<pi){if(primary[j]<primary[unique.back()])unique.back()=j;}else unique.push_back(j);}anchors=unique;
 if(anchors.size()<2)throw std::runtime_error("fewer than two complete dark-center calibration anchors");
 anchor_spread=0;for(auto& node:nodes){double sx=0,sy=0;Image angles;for(size_t j:anchors){double ph=node.orientation*std::atan2((node.calibration[j]-node.mid)/node.scale,(primary[j]-mid0)/scale0);angles.push_back(ph);sx+=std::cos(ph);sy+=std::sin(ph);}node.zero=std::atan2(sy,sx);for(double ph:angles)anchor_spread=std::max(anchor_spread,std::abs(wrap(ph-node.zero))/(2*pi));}if(anchor_spread>.2)throw std::runtime_error("inconsistent calibration dark phase");
 }
};

inline Feature observe(const Frame& f,const CalibrationModel& model,int n,double ox,double oy){
 Feature feature;feature.valid=f.valid;if(f.valid){feature.a=(patch(f,n,model.cx,model.cy,ox,oy)-model.mid0)/model.scale0;for(size_t k=0;k<model.nodes.size();++k){const auto& q=model.nodes[k];double b=(patch(f,n,q.x,q.y,ox,oy)-q.mid)/q.scale;feature.radius[k]=std::hypot(feature.a,b);feature.phase[k]=wrap(q.orientation*std::atan2(b,feature.a)-q.zero);}}return feature;
}
inline Decision advance(const std::vector<Node>& nodes,const std::vector<Feature>& features,std::vector<Track>& states,int i,int limit,int prior){Decision out;double pos=0,neg=0,fp=0,fn=0;for(size_t k=0;k<nodes.size();++k){auto st=states[k];if(features[i].radius[k]>=nodes[k].min_radius&&st.last>=0){double d=wrap(features[i].phase[k]-st.previous);if(std::abs(d)>.15&&std::abs(d)<.8*pi){if(d>0)++pos;else ++neg;}}for(int j=i;j<limit;++j)if(features[j].valid&&features[j+1].valid&&features[j].radius[k]>=nodes[k].min_radius&&features[j+1].radius[k]>=nodes[k].min_radius){double d=wrap(features[j+1].phase[k]-features[j].phase[k]);if(std::abs(d)>.15&&std::abs(d)<.8*pi){if(d>0)fp+=1./(j-i+1);else fn+=1./(j-i+1);}}}int direction=pos>=2&&pos>neg?1:(neg>=2&&neg>pos?-1:(fp>fn+1?1:(fn>fp+1?-1:0)));out.direction=direction;
  std::vector<int> values,used;for(size_t k=0;k<nodes.size();++k){auto& st=states[k];if(!features[i].valid||features[i].radius[k]<nodes[k].min_radius)continue;double ph=features[i].phase[k];if(st.last<0){st.unwrapped=ph;st.integer=st.origin=int(std::floor(ph/(2*pi)));}else{double d=wrap(ph-st.previous);if(std::abs(d)>.8*pi){++out.ambiguous;if(direction*d<0){d+=direction*2*pi;++out.repairs;}}st.unwrapped+=d;}st.previous=ph;st.last=i;double boundary=st.unwrapped/(2*pi);while(boundary>st.integer+1+.02)++st.integer;while(boundary<st.integer-.02)--st.integer;values.push_back(st.integer-st.origin);used.push_back(int(k));}
  int best_votes=0;out.count=prior;for(int value:values){int votes=int(std::count(values.begin(),values.end(),value));if(votes>best_votes||(votes==best_votes&&std::abs(value-prior)<std::abs(out.count-prior))){best_votes=votes;out.count=value;}}if(best_votes<2)out.count=prior;out.valid=int(values.size());
  if(best_votes>=3)for(size_t j=0;j<used.size();++j){auto& st=states[used[j]];if(values[j]!=out.count){if(++st.outlier>=4){int correction=out.count-values[j];st.integer+=correction;st.unwrapped+=correction*2*pi;st.outlier=0;++out.repairs;}}else st.outlier=0;}
  for(size_t k=0;k<nodes.size();++k)out.pair[k]=states[k].integer-states[k].origin;return out;
}
