#include "screen_tracker.h"
using namespace screen;
int main(int argc,char** argv){try{
 if(argc!=5&&argc!=7){std::cerr<<"fixture calibration_start calibration_end motion.csv [count_offset_x count_offset_y]\n";return 2;}
 double offset_x=argc==7?std::stod(argv[5]):0,offset_y=argc==7?std::stod(argv[6]):0;
 double start=std::stod(argv[2]),end=std::stod(argv[3]);std::ifstream input(argv[1],std::ios::binary);char magic[4];input.read(magic,4);int n=read<uint32_t>(input);read<uint32_t>(input);read<uint32_t>(input);int count=read<uint32_t>(input);read<double>(input);
 if(n<32||n>128)throw std::runtime_error("unsupported tracking size");
 std::vector<Image> images;Image reference(n*n);std::vector<double> times;int calibration_frames=0;
 for(int i=0;i<count;++i){read<uint32_t>(input);double t=read<double>(input);Image image(n*n);for(double& v:image)v=read<float>(input);if(t>=start&&t<=end){for(int p=0;p<n*n;++p)reference[p]+=image[p];++calibration_frames;}times.push_back(t);images.push_back(std::move(image));}
 if(calibration_frames<24)throw std::runtime_error("calibration too short");for(double& v:reference)v/=calibration_frames;
 screen::Tracker tracker(reference,n);int rejected=0;std::vector<double> durations;
 std::ofstream out(argv[4]);out<<std::setprecision(17)<<"source_frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,accepted,timestamp,area_ratio,frame_area_ratio,iou_before,iou_after,boundary_median_px,boundary_p95_px,center_dx_px,center_dy_px,core_us,projective\n";
 for(int i=0;i<count;++i)if(times[i]>=start){auto begin=std::chrono::steady_clock::now();auto o=tracker.push(images[i]);if(!o.accepted)++rejected;
 double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count();durations.push_back(us);
 Matrix to_tracking={1,0,offset_x,0,1,offset_y,0,0,1},to_counting={1,0,-offset_x,0,1,-offset_y,0,0,1};auto counting=multiply(multiply(to_counting,o.transform),to_tracking);
 out<<i;for(double v:counting)out<<','<<v;out<<','<<o.accepted<<','<<times[i]<<','<<o.area_ratio<<','<<o.frame_area_ratio<<','<<o.iou_before<<','<<o.iou_after<<','<<o.median<<','<<o.p95<<','<<o.dx<<','<<o.dy<<','<<us<<','<<o.projective<<'\n';
 }
 std::cout<<"frames="<<durations.size()<<" rejected="<<rejected<<" tracking_p95_ms="<<quantile(durations,.95)/1000<<'\n';return rejected?3:0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
