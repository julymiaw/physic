#include "live_counter.h"
int main(int argc,char** argv){try{
 if(argc!=8)throw std::runtime_error("fixture cal_start cal_end end offset_x offset_y output.csv");
 std::ifstream input(argv[1],std::ios::binary);char magic[4];input.read(magic,4);
 auto n=screen::read<uint32_t>(input);screen::read<uint32_t>(input);screen::read<uint32_t>(input);auto count=screen::read<uint32_t>(input);screen::read<double>(input);
 if(n!=96)throw std::runtime_error("expected 96px input");
 live::Counter c;c.reset(std::stod(argv[5]),std::stod(argv[6]));double start=std::stod(argv[2]),cal_end=std::stod(argv[3]),end=std::stod(argv[4]);bool started=false;
 std::ofstream out(argv[7]);out<<std::setprecision(17)<<"source_frame,timestamp,count,event_direction,confirmation_frame,confirmed_at,valid_pairs,repairs,ambiguous,complete,tail_pending,primary,tracking_valid,tracking_iou,pair0,phase0,pair1,phase1,pair2,phase2,pair3,phase3\n";
 std::ofstream state_out(std::string(argv[7])+".state.csv");state_out<<std::setprecision(17)<<"source_frame";for(int k=0;k<64;++k)state_out<<",state"<<k;state_out<<'\n';
 auto state_write=[&](int source){state_out<<source;for(double v:c.result)state_out<<','<<v;state_out<<'\n';};
 auto write=[&](){for(size_t j=0;j<c.emitted.size();++j){auto r=c.row(j);for(int k=0;k<22;++k){if(k)out<<',';out<<r[k];}out<<'\n';}};
 for(uint32_t i=0;i<count;++i){screen::read<uint32_t>(input);auto source=i;double t=screen::read<double>(input);for(float& v:c.input)v=screen::read<float>(input);if(t<start||t>end)continue;
  if(t<=cal_end){if(!c.collect(t,source))throw std::runtime_error("calibration collection failed");}
  else{if(!started){if(!c.start())throw std::runtime_error("calibration model failed");started=true;}c.push(t,source);write();state_write(source);}
 }
 c.finish();write();state_write(-1);std::cout<<"count="<<c.confirmed<<" complete="<<c.complete<<'\n';return c.complete?0:3;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
