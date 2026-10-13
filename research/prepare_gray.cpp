// Desktop data preparation only: exact area integration, no counting model.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
template<class T> void put(std::ostream& f,T v){f.write(reinterpret_cast<const char*>(&v),sizeof(v));}
int main(int argc,char** argv){try{
 if(argc!=11)throw std::runtime_error("width height roi_x roi_y roi_w roi_h size count rate output.frc; stdin: double timestamp + gray bytes per frame");
 int width=std::stoi(argv[1]),height=std::stoi(argv[2]),x=std::stoi(argv[3]),y=std::stoi(argv[4]),w=std::stoi(argv[5]),h=std::stoi(argv[6]),n=std::stoi(argv[7]),count=std::stoi(argv[8]);
 if(width<=0||height<=0||width>16384||height>16384||x<0||y<0||w<=0||h<=0||x>width-w||y>height-h||n<4||n>256||count<1)throw std::runtime_error("invalid dimensions");
 std::ofstream out(argv[10],std::ios::binary);if(!out)throw std::runtime_error("cannot open fixture output");out.write("FRA2",4);put(out,uint32_t(n));put(out,uint32_t(1));put(out,uint32_t(7));put(out,uint32_t(count));put(out,std::stod(argv[9]));
 std::vector<uint8_t> input(size_t(width)*height);std::vector<float> pixels(n*n);
 for(int frame=0;frame<count;++frame){double timestamp;std::cin.read(reinterpret_cast<char*>(&timestamp),8);std::cin.read(reinterpret_cast<char*>(input.data()),input.size());if(!std::cin)throw std::runtime_error("truncated gray input");
  for(int oy=0;oy<n;++oy)for(int ox=0;ox<n;++ox){double left=x+ox*double(w)/n,right=x+(ox+1)*double(w)/n,top=y+oy*double(h)/n,bottom=y+(oy+1)*double(h)/n,sum=0;
   for(int iy=int(top);iy<std::ceil(bottom);++iy)for(int ix=int(left);ix<std::ceil(right);++ix){double weight=(std::min(right,double(ix+1))-std::max(left,double(ix)))*(std::min(bottom,double(iy+1))-std::max(top,double(iy)));sum+=input[iy*width+ix]*weight;}
   pixels[oy*n+ox]=float(sum/((right-left)*(bottom-top)*255));
  }
  put(out,uint32_t(2));put(out,timestamp);out.write(reinterpret_cast<const char*>(pixels.data()),pixels.size()*sizeof(float));
 }
 if(!out)throw std::runtime_error("fixture write failed");return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
