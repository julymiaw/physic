#pragma once
// Native research replay: track the illumination footprint, excluding ring texture.
// Fits a reference outline to each current outline with a planar homography.
// No OpenCV dependency. Calibration-only reference; no measurement lookahead.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace screen {
using Image=std::vector<double>;
using Mask=std::vector<uint8_t>;
using Matrix=std::array<double,9>;
using Point=std::array<double,2>;
constexpr Matrix identity={1,0,0,0,1,0,0,0,1};
template<class T>T read(std::ifstream& f){T x;f.read(reinterpret_cast<char*>(&x),sizeof x);if(!f)throw std::runtime_error("truncated fixture");return x;}
double quantile(Image v,double q){std::sort(v.begin(),v.end());return v[size_t(q*(v.size()-1))];}
double at(const Image& a,int n,double x,double y){x=std::clamp(x,0.,double(n-1));y=std::clamp(y,0.,double(n-1));int ix=int(x),iy=int(y),jx=std::min(ix+1,n-1),jy=std::min(iy+1,n-1);double u=x-ix,v=y-iy;return (1-v)*((1-u)*a[iy*n+ix]+u*a[iy*n+jx])+v*((1-u)*a[jy*n+ix]+u*a[jy*n+jx]);}
Point project(const Matrix& h,double x,double y){double w=h[6]*x+h[7]*y+h[8];return {(h[0]*x+h[1]*y+h[2])/w,(h[3]*x+h[4]*y+h[5])/w};}
Matrix multiply(const Matrix& a,const Matrix& b){Matrix c{};for(int y=0;y<3;++y)for(int x=0;x<3;++x)for(int k=0;k<3;++k)c[y*3+x]+=a[y*3+k]*b[k*3+x];return c;}
Matrix pixels(const Matrix& h,int n){double s=(n-1)/2.;Matrix p={s,0,s,0,s,s,0,0,1},q={1/s,0,-1,0,1/s,-1,0,0,1};auto result=multiply(multiply(p,h),q);double w=result[8];for(double& v:result)v/=w;return result;}

Image blur(const Image& a,int n){
 std::array<double,9> k{};double sum=0;for(int j=-4;j<=4;++j){k[j+4]=std::exp(-j*j/(2*1.3*1.3));sum+=k[j+4];}for(double& v:k)v/=sum;
 Image tmp(n*n),out(n*n);for(int y=0;y<n;++y)for(int x=0;x<n;++x)for(int j=-4;j<=4;++j)tmp[y*n+x]+=k[j+4]*a[y*n+std::clamp(x+j,0,n-1)];
 for(int y=0;y<n;++y)for(int x=0;x<n;++x)for(int j=-4;j<=4;++j)out[y*n+x]+=k[j+4]*tmp[std::clamp(y+j,0,n-1)*n+x];return out;
}
Mask segment(const Image& a,int n){
 auto s=blur(a,n);double lo=quantile(s,.1),hi=quantile(s,.9);if(hi-lo<.08)throw std::runtime_error("screen contrast too low");
 Mask initial(n*n),visited(n*n),mask(n*n);for(int i=0;i<n*n;++i)initial[i]=s[i]>lo+.4*(hi-lo);
 std::vector<int> biggest,queue;queue.reserve(n*n);
 auto neighbors=[&](int p,auto visit){int x=p%n,y=p/n;if(x)visit(p-1);if(x+1<n)visit(p+1);if(y)visit(p-n);if(y+1<n)visit(p+n);};
 for(int seed=0;seed<n*n;++seed)if(initial[seed]&&!visited[seed]){queue.clear();queue.push_back(seed);visited[seed]=1;for(size_t j=0;j<queue.size();++j)neighbors(queue[j],[&](int p){if(initial[p]&&!visited[p]){visited[p]=1;queue.push_back(p);}});if(queue.size()>biggest.size())biggest=queue;}
 for(int p:biggest)mask[p]=1;
 // Fill enclosed dark fringe holes; only the outer footprint is a landmark.
 std::fill(visited.begin(),visited.end(),0);queue.clear();
 for(int p=0;p<n*n;++p)if((p%n==0||p%n==n-1||p/n==0||p/n==n-1)&&!mask[p]&&!visited[p]){visited[p]=1;queue.push_back(p);}
 for(size_t j=0;j<queue.size();++j)neighbors(queue[j],[&](int p){if(!mask[p]&&!visited[p]){visited[p]=1;queue.push_back(p);}});
 for(int p=0;p<n*n;++p)if(!visited[p])mask[p]=1;
 return mask;
}
Image distance(const Mask& m,int n,bool inside){
 Image d(n*n);for(int i=0;i<n*n;++i)d[i]=(bool(m[i])==inside)?1e6:0;
 const double diagonal=std::sqrt(2.);
 for(int y=0;y<n;++y)for(int x=0;x<n;++x){int p=y*n+x;if(x)d[p]=std::min(d[p],d[p-1]+1);if(y)d[p]=std::min(d[p],d[p-n]+1);if(x&&y)d[p]=std::min(d[p],d[p-n-1]+diagonal);if(y&&x+1<n)d[p]=std::min(d[p],d[p-n+1]+diagonal);}
 for(int y=n-1;y>=0;--y)for(int x=n-1;x>=0;--x){int p=y*n+x;if(x+1<n)d[p]=std::min(d[p],d[p+1]+1);if(y+1<n)d[p]=std::min(d[p],d[p+n]+1);if(x+1<n&&y+1<n)d[p]=std::min(d[p],d[p+n+1]+diagonal);if(x&&y+1<n)d[p]=std::min(d[p],d[p+n-1]+diagonal);}
 return d;
}
Image field(const Mask& m,int n){auto a=distance(m,n,true),b=distance(m,n,false);for(int i=0;i<n*n;++i)a[i]=std::clamp(a[i]-b[i],-8.,8.);return a;}
std::vector<Point> boundary(const Mask& m,int n){std::vector<Point> all,result;for(int y=1;y<n-1;++y)for(int x=1;x<n-1;++x){int p=y*n+x;if(m[p]!=m[p+1])all.push_back({x+.5,double(y)});if(m[p]!=m[p+n])all.push_back({double(x),y+.5});}if(all.size()<32)throw std::runtime_error("screen contour too small");size_t count=std::min(size_t(192),all.size());for(size_t j=0;j<count;++j)result.push_back(all[j*all.size()/count]);return result;}
bool solve(double a[8][9],std::array<double,8>& x){for(int k=0;k<8;++k){int pivot=k;for(int j=k+1;j<8;++j)if(std::abs(a[j][k])>std::abs(a[pivot][k]))pivot=j;if(std::abs(a[pivot][k])<1e-9)return false;for(int j=k;j<=8;++j)std::swap(a[k][j],a[pivot][j]);double d=a[k][k];for(int j=k;j<=8;++j)a[k][j]/=d;for(int i=0;i<8;++i)if(i!=k){double v=a[i][k];for(int j=k;j<=8;++j)a[i][j]-=v*a[k][j];}}for(int i=0;i<8;++i)x[i]=a[i][8];return true;}

Matrix fit(const std::vector<Point>& points,const Image& current,int n,Matrix h,int dimensions=8){
 double s=(n-1)/2.;
 for(int iteration=0;iteration<18;++iteration){double normal[8][9]{};int used=0;
  for(auto p:points){double x=(p[0]-s)/s,y=(p[1]-s)/s,w=h[6]*x+h[7]*y+1;auto uv=project(h,x,y);double px=uv[0]*s+s,py=uv[1]*s+s;
   if(px<1||py<1||px>n-2||py>n-2)continue;
   double r=at(current,n,px,py),gx=(at(current,n,px+.5,py)-at(current,n,px-.5,py)),gy=(at(current,n,px,py+.5)-at(current,n,px,py-.5));
   double gradient=std::hypot(gx,gy);if(gradient<.1)continue;
   double weight=std::min(1.,1./std::max(1e-9,std::abs(r)));double a=gx*s/w,b=gy*s/w;
   double j[8]={a*x,a*y,a,b*x,b*y,b,-(a*uv[0]+b*uv[1])*x,-(a*uv[0]+b*uv[1])*y};
   for(int row=0;row<dimensions;++row){for(int col=0;col<dimensions;++col)normal[row][col]+=weight*j[row]*j[col];normal[row][8]-=weight*j[row]*r;}++used;
  }
  if(used<int(points.size()*.8))break;
  for(int j=0;j<8;++j)normal[j][j]+=.01*std::max(1.,normal[j][j]);
  std::array<double,8> delta{};if(!solve(normal,delta))break;
  double max_update=0;for(double x:{-1.,1.})for(double y:{-1.,1.}){auto before=project(h,x,y);Matrix next=h;for(int k=0;k<8;++k)next[k]+=delta[k];auto after=project(next,x,y);max_update=std::max(max_update,s*std::hypot(after[0]-before[0],after[1]-before[1]));}
  double damping=std::min(1.,2./std::max(1e-9,max_update));for(int j=0;j<8;++j)h[j]+=delta[j]*damping;
  if(max_update<.005)break;
 }
 return h;
}
double overlap(const Mask& reference,const Mask& current,int n,const Matrix& h){size_t intersection=0,united=0;for(int y=0;y<n;++y)for(int x=0;x<n;++x){auto p=project(h,x,y);int xx=int(std::round(p[0])),yy=int(std::round(p[1]));bool a=reference[y*n+x],b=xx>=0&&yy>=0&&xx<n&&yy<n&&current[yy*n+xx];intersection+=a&&b;united+=a||b;}return double(intersection)/std::max(size_t(1),united);}
double area(const Mask& m){return std::accumulate(m.begin(),m.end(),0.);}
Image residuals(const std::vector<Point>& points,const Image& current,int n,const Matrix& hp){Image errors;for(auto p:points){auto q=project(hp,p[0],p[1]);errors.push_back(std::abs(at(current,n,q[0],q[1])));}return errors;}


struct Observation {
 Matrix transform=identity;bool accepted=false,projective=false;
 double area_ratio=0,frame_area_ratio=0,iou_before=0,iou_after=0,median=0,p95=0,dx=0,dy=0;
};
class Tracker {
 int n;Mask reference_mask;std::vector<Point> points;double reference_area,previous_area;Matrix h=identity;
 public:
 explicit Tracker(const Image& reference,int size):n(size),reference_mask(segment(reference,n)),points(boundary(reference_mask,n)),reference_area(area(reference_mask)),previous_area(reference_area){}
 Observation push(const Image& image){
  auto mask=segment(image,n);auto current=field(mask,n);
  Matrix initial=h;initial[6]=initial[7]=0;Matrix candidate=fit(points,current,n,initial,6),hp=pixels(candidate,n);auto errors=residuals(points,current,n,hp);bool projective=false;
  double affine_p95=quantile(errors,.95);
  if(affine_p95>1.25){auto perspective=fit(points,current,n,candidate,8);auto perspective_hp=pixels(perspective,n);auto perspective_errors=residuals(points,current,n,perspective_hp);if(quantile(perspective_errors,.95)<.8*affine_p95){candidate=perspective;hp=perspective_hp;errors=perspective_errors;projective=true;}}
  double a=area(mask),ratio=a/previous_area;
  double before=overlap(reference_mask,mask,n,identity),after=overlap(reference_mask,mask,n,hp),median=quantile(errors,.5),p95=quantile(errors,.95);
  auto center=project(hp,(n-1)/2.,(n-1)/2.);double min_denominator=1e9;for(double x:{-1.,1.})for(double y:{-1.,1.})min_denominator=std::min(min_denominator,candidate[6]*x+candidate[7]*y+1);
  bool accepted=ratio>.8&&ratio<1.25&&after>.9&&median<1&&p95<2&&min_denominator>.4;
  if(accepted){h=candidate;previous_area=a;}else hp=pixels(h,n);
  return {hp,accepted,projective,a/reference_area,ratio,before,after,median,p95,center[0]-(n-1)/2.,center[1]-(n-1)/2.};
 }
};
} // namespace screen
