#include "../core/live_counter.h"
int main(int argc,char** argv){
 if(argc!=4)return 1;
 std::ifstream f(argv[1],std::ios::binary);f.seekg(16);auto n=screen::read<uint32_t>(f);screen::read<double>(f);
 std::vector<Image> frames;for(uint32_t i=0;i<n;++i){screen::read<uint32_t>(f);screen::read<double>(f);Image a(96*96);for(auto& v:a)v=screen::read<float>(f);frames.push_back(std::move(a));}
 live::Counter c;c.reset(16,58.*96/345);std::ifstream ops(argv[2]);std::ofstream out(argv[3]);out<<std::setprecision(17);
 int op,id;double t;while(ops>>op>>id>>t){if(id>=0)std::copy(frames.at(id).begin(),frames.at(id).end(),c.input.begin());
  switch(op){case 0:c.collect(t,id);break;case 1:if(!c.start()||!c.arm())return 2;break;case 2:c.monitor(t,id);break;case 3:if(!c.begin_measurement())return 3;break;case 4:c.push(t,id);break;case 5:c.finish();break;case 6:c.interrupt();break;case 7:if(!c.stop_measurement())return 4;break;}
  for(int k=0;k<64;++k){if(k)out<<',';out<<c.result[k];}out<<'\n';
 }
}
