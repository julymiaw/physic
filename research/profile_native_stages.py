"""Controlled real-data C++/C++ stage profiling; production sources stay unchanged."""
import ctypes as C,csv,hashlib,json,re,subprocess,time
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/results/native-stage-profile';WORK=ROOT/'experiments/runs/native-stage-profile'
OLD_COMMIT='c3ae479a0999f263b9365f1f94845e691e55d953'
PROFILE_COMMIT='436a20b' # Frozen pre-streaming profiling source; timings belong to this revision.
OLD_NAMES=['red_extract_copy','histograms','screen_segment','roi_views','binary_morphology','edges_find_contours','ring_filter_classification','center_brightness','area_vote_count','embedded_drawing','other']
NEW_NAMES=['screen_segment','distance_field','motion_fit','tracking_quality','patch_sampling','phase_angles','confirmation','other']
TIMER=r'''#pragma once
#include <array>
#include <chrono>
namespace prof {
using Clock=std::chrono::steady_clock; inline bool enabled=false; inline int current=0; inline std::array<double,12> bins{}; inline Clock::time_point start,last;
inline void begin(int id,bool flag=true){enabled=flag;current=id;bins.fill(0);if(flag)start=last=Clock::now();}
inline void mark(int id){if(!enabled)return;auto now=Clock::now();bins[current]+=std::chrono::duration<double,std::micro>(now-last).count();last=now;current=id;}
inline double finish(){if(!enabled)return 0;mark(current);enabled=false;double total=0;for(auto v:bins)total+=v;return total;}
struct Scoped {int previous;Scoped(int id):previous(current){mark(id);}~Scoped(){mark(previous);}};
}
'''

def instrument_old():
    raw=subprocess.check_output(['git','show',f'{OLD_COMMIT}:轮廓检测.cpp'],cwd=ROOT).decode('utf-8-sig')
    source=raw[:raw.index('int main()')].replace('private:','public:')
    start=source.index('cv::Mat ContourDetection::process_frame');end=source.index('void ContourDetection::print_result')
    body=source[start:end]
    marks=[('cv::Mat red_channel = frame.clone();',0),('int threshold = calculate_threshold(red_channel);',1),('auto [bbox, mask] = segment_image',2),('float bbox_area = bbox.area();',10),('red_channel = red_channel(bbox);',3),('threshold = calculate_threshold(red_channel, mask);',1),('cv::Mat binary_image;',4),('cv::Mat edges;',5),('int max_radius =',6),('if (contours_info.empty()) {',10),('int cx =',7),('int scale_count = 0;',8),('state_info.contours_info = contours_info;',10)]
    for needle,index in marks:
        assert needle in body,needle;body=body.replace(needle,f'prof::mark({index}); '+needle,1)
    for needle in ['cv::drawContours(cropped_frame,','cv::copyMakeBorder(cropped_frame,']:
        pattern=re.escape(needle)+r'[^\n]+;'
        body,n=re.subn(pattern,lambda m:'{ prof::Scoped display(9); '+m[0]+' }',body);assert n==1,(needle,n)
    source=source[:start]+body+source[end:]
    return '#include "profile_timer.h"\n#include <opencv2/geometry.hpp>\n#include <memory>\n#include <map>\n#include <iomanip>\n#include <algorithm>\n#include <stdexcept>\n'+source+r'''
extern "C" {
const char* legacy_version(){return CV_VERSION;}
void* legacy_create(){cv::setNumThreads(1);return new ContourDetection;}
void legacy_destroy(void* p){delete static_cast<ContourDetection*>(p);}
int legacy_push(void* handle,unsigned char* bytes,int width,int height,int stride,int sample,double* result){try{
 auto p=static_cast<ContourDetection*>(handle);cv::Mat frame(height,width,CV_8UC3,bytes,stride);
 std::unique_ptr<ContourDetection> snapshot;cv::Mat clone;
 if(sample){snapshot=std::make_unique<ContourDetection>(*p);clone=frame.clone();}
 prof::enabled=false;auto begin=prof::Clock::now();p->process_frame(frame);result[0]=std::chrono::duration<double,std::micro>(prof::Clock::now()-begin).count();
 result[1]=p->get_count_value();result[2]=p->state_info.center_color;result[3]=p->state_info.status;result[4]=p->state_info.contours_info.size();result[5]=p->state_info.bbox_area;
 if(sample){prof::begin(10);snapshot->process_frame(clone);result[6]=prof::finish();for(int j=0;j<11;++j)result[7+j]=prof::bins[j];
  if(snapshot->get_count_value()!=p->get_count_value()||snapshot->state_info.center_color!=p->state_info.center_color||snapshot->state_info.status!=p->state_info.status||snapshot->state_info.contours_info!=p->state_info.contours_info)return -2;
 }
 return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return -1;}}
// Exact area formula used by prepare_gray.cpp, applied to existing red preview.
double prep_preview(unsigned char* input,int width,float* out){auto begin=prof::Clock::now();int x=68,y=132,w=240,h=345,n=96;
 for(int oy=0;oy<n;++oy)for(int ox=0;ox<n;++ox){double left=x+ox*double(w)/n,right=x+(ox+1)*double(w)/n,top=y+oy*double(h)/n,bottom=y+(oy+1)*double(h)/n,sum=0;
 for(int iy=int(top);iy<std::ceil(bottom);++iy)for(int ix=int(left);ix<std::ceil(right);++ix){double weight=(std::min(right,double(ix+1))-std::max(left,double(ix)))*(std::min(bottom,double(iy+1))-std::max(top,double(iy)));sum+=input[iy*width+ix]*weight;}
 out[oy*n+ox]=float(sum/((right-left)*(bottom-top)*255));}
 return std::chrono::duration<double,std::micro>(prof::Clock::now()-begin).count();}
}
''',raw

def instrument_new():
    header=(ROOT/'research/screen_tracker.h').read_text()
    header=header.replace('auto mask=segment(image,n);auto current=field(mask,n);','prof::mark(0);auto mask=segment(image,n);prof::mark(1);auto current=field(mask,n);prof::mark(2);')
    header=header.replace('double a=area(mask),ratio=a/previous_area;','prof::mark(3);double a=area(mask),ratio=a/previous_area;')
    (WORK/'screen_tracker.h').write_text(header)
    code=subprocess.check_output(['git','show',f'{PROFILE_COMMIT}:research/real_pair.cpp'],cwd=ROOT).decode();code='#include "profile_timer.h"\n'+code
    code=code.replace(' if(calibration.size()<24)', ' auto init_begin=Clock::now();if(calibration.size()<24)',1)
    code=code.replace(' double center_residual=0;',' auto init_track_end=Clock::now();double center_residual=0;',1)
    code=code.replace('double cx=center[0],cy=center[1];','auto init_center_end=Clock::now();double cx=center[0],cy=center[1];',1)
    code=code.replace(' // Common physical anchor:', ' auto init_signals_end=Clock::now();\n // Common physical anchor:',1)
    code=code.replace(' std::ofstream cal_log(', ' auto init_end=Clock::now();std::ofstream cal_log(',1)
    code=code.replace(' std::vector<Feature> features;', ' std::vector<std::array<double,15>> stage_rows;std::vector<Feature> features;',1)
    code=code.replace('for(size_t i=0;i<frames.size();++i){auto tick=', 'for(size_t i=0;i<frames.size();++i){prof::begin(7,i>=cal_count);auto tick=',1)
    code=code.replace('Feature feature;feature.valid=f.valid;','prof::mark(4);Feature feature;feature.valid=f.valid;',1)
    code=code.replace('feature.radius[k]=std::hypot(feature.a,b);','prof::mark(5);feature.radius[k]=std::hypot(feature.a,b);',1)
    code=code.replace('feature.phase[k]=wrap(q.orientation*std::atan2(b,feature.a)-q.zero);','feature.phase[k]=wrap(q.orientation*std::atan2(b,feature.a)-q.zero);prof::mark(4);',1)
    code=code.replace('  if(i>=cal_count){if(int(i)>=next+4)', '  prof::mark(6);if(i>=cal_count){if(int(i)>=next+4)',1)
    code=code.replace('double us=timing?std::chrono::duration<double,std::micro>(Clock::now()-tick).count():0;', 'double total=prof::finish();std::array<double,15> trace{};trace[0]=f.source;trace[1]=f.t;trace[2]=total;for(int j=0;j<12;++j)trace[3+j]=prof::bins[j];stage_rows.push_back(trace);double us=timing?std::chrono::duration<double,std::micro>(Clock::now()-tick).count():0;',1)
    code=code.replace(' auto end_tick=timing?', ' prof::begin(6);auto end_tick=timing?',1)
    needle=' std::ofstream out(argv[5]),events'
    addition=''' double eof_us=prof::finish();if(!stage_rows.empty()){stage_rows.back()[2]+=eof_us;for(int j=0;j<12;++j)stage_rows.back()[3+j]+=prof::bins[j];}
 std::ofstream stages(std::string(argv[5])+".stages.csv");stages<<"source_frame,timestamp,total_us,screen_segment,distance_field,motion_fit,tracking_quality,patch_sampling,phase_angles,confirmation,other";for(int j=8;j<12;++j)stages<<",unused"<<j;stages<<'\\n';for(auto r:stage_rows){for(size_t j=0;j<r.size();++j){if(j)stages<<',';stages<<std::setprecision(17)<<r[j];}stages<<'\\n';}
 auto ms=[](auto a,auto b){return std::chrono::duration<double,std::milli>(b-a).count();};std::ofstream init(std::string(argv[5])+".init.json");init<<"{\\"reference_and_calibration_tracking_ms\\":"<<ms(init_begin,init_track_end)<<",\\"gradient_center_ms\\":"<<ms(init_track_end,init_center_end)<<",\\"signal_selection_ms\\":"<<ms(init_center_end,init_signals_end)<<",\\"zero_alignment_ms\\":"<<ms(init_signals_end,init_end)<<",\\"total_ms\\":"<<ms(init_begin,init_end)<<"}\\n";
'''
    assert needle in code;code=code.replace(needle,addition+needle,1);return code

def build():
    OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True);(WORK/'profile_timer.h').write_text(TIMER)
    code,raw=instrument_old();(WORK/'legacy.cpp').write_text(code);(WORK/'legacy-original.cpp').write_text(raw)
    prefix=ROOT/'.tools/opencv-stage-profile/install';include=prefix/'include/opencv5'
    if not include.exists():include=prefix/'include/opencv4'
    subprocess.run(['clang++','-std=c++17','-O3','-DNDEBUG','-shared','-fPIC','-I',str(include),'-L',str(prefix/'lib'),f'-Wl,-rpath,{prefix/"lib"}',str(WORK/'legacy.cpp'),'-lopencv_core','-lopencv_imgproc','-lopencv_geometry','-o',str(WORK/'legacy.dylib')],check=True)
    (WORK/'native.cpp').write_text(instrument_new());subprocess.run(['clang++','-std=c++17','-O3','-DNDEBUG',str(WORK/'native.cpp'),'-o',str(WORK/'native-profile')],check=True)
    return hashlib.sha256(raw.encode()).hexdigest()

def main():
    old_hash=build();cv2.setNumThreads(1);lib=C.CDLL(str(WORK/'legacy.dylib'));lib.legacy_create.restype=C.c_void_p;lib.legacy_destroy.argtypes=[C.c_void_p];lib.legacy_version.restype=C.c_char_p
    lib.legacy_push.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_int,C.POINTER(C.c_double)];lib.prep_preview.argtypes=[C.c_void_p,C.c_int,C.c_void_p];lib.prep_preview.restype=C.c_double
    handle=lib.legacy_create();data=np.load(ROOT/'data/derived/original.npz');previews,times=data['frames'],data['timestamps'];samples=set(np.rint(np.linspace(279,1436,128)).astype(int).tolist());cap=cv2.VideoCapture(str(ROOT/'data/raw/michelson-original.mp4'));cap.set(cv2.CAP_PROP_ORIENTATION_AUTO,1)
    fields=['source_frame','timestamp','count','center_class','status','rings','bbox_pixels','decode_us','preview_to_96_us','baseline_total_us','profile_total_us']+OLD_NAMES
    result=np.zeros(18,np.float64);prepared=np.empty((96,96),np.float32);records=[];started=time.perf_counter();fixture=ROOT/'experiments/results/real-track/padded-screen.frc'
    import struct
    with fixture.open('rb') as expected,(OUT/'legacy-cpp.csv').open('w') as stream:
        expected.read(28);writer=csv.DictWriter(stream,fields,lineterminator='\n');writer.writeheader()
        for i,t in enumerate(times):
            tick=time.perf_counter_ns();ok,frame=cap.read();decode=(time.perf_counter_ns()-tick)/1000
            if not ok:raise RuntimeError(f'Decode failed at {i}')
            assert frame.shape[:2]==(7680,4320)
            _,ft=struct.unpack('<Id',expected.read(12));target=np.frombuffer(expected.read(96*96*4),'<f4').reshape(96,96);assert abs(ft-t)<1e-8
            prep=lib.prep_preview(previews[i].ctypes.data,432,prepared.ctypes.data);assert np.array_equal(prepared,target),'Preprocessing differs from frozen fixture'
            result.fill(0);error=lib.legacy_push(handle,frame.ctypes.data,4320,7680,frame.strides[0],int(i in samples),result.ctypes.data_as(C.POINTER(C.c_double)))
            if error:raise RuntimeError(f'Historical C++ failed at frame{i}, code{error}')
            row=dict(source_frame=i,timestamp=float(t),count=int(result[1]),center_class=int(result[2]),status=int(result[3]),rings=int(result[4]),bbox_pixels=int(result[5]),decode_us=decode,preview_to_96_us=prep,baseline_total_us=float(result[0]),profile_total_us=float(result[6]),**{k:float(result[7+j]) for j,k in enumerate(OLD_NAMES)})
            writer.writerow(row);records.append(row)
            if (i+1)%96==0:stream.flush();print(f'C++ legacy {i+1}/1437 count={row["count"]} elapsed={time.perf_counter()-started:.1f}s',flush=True)
    cap.release();lib.legacy_destroy(handle)
    native=OUT/'native-profile.csv';subprocess.run([str(WORK/'native-profile'),str(fixture),'9.6','11.6','60',str(native),'--timing'],check=True)
    assert native.read_bytes()==(ROOT/'experiments/results/bilibili-adapt/original/frames.csv').read_bytes()
    control=OUT/'native-control.csv';subprocess.run([str(ROOT/'build-native/fringe_count'),str(fixture),'9.6','11.6','60',str(control),'--timing'],check=True);assert native.read_bytes()==control.read_bytes()
    protocol=dict(machine=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string']).decode().strip(),old_cpp_commit=OLD_COMMIT,old_cpp_sha256=old_hash,new_cpp_sha256=hashlib.sha256(subprocess.check_output(['git','show',f'{PROFILE_COMMIT}:research/real_pair.cpp'],cwd=ROOT)).hexdigest(),profile_commit=PROFILE_COMMIT,opencv=lib.legacy_version().decode(),compiler='AppleClang, C++17 -O3 -DNDEBUG',opencv_threads=1,measurement_frame_interval=[279,1436],measurement_frames=1158,stage_sample_frames=sorted(samples),sample_count=128,old_timing='Full C++ processing on every decoded4320x7680 BGR frame. Fixed128 measurement frames additionally profile a copied pre-frame state; profile and baseline state/count/contours must agree.',new_timing='Current unmodified core control plus generated timing-instrumented copy; both full outputs byte-identical to verified results.',preprocessing='Actual second-stage area resize from existing432x768 red preview to96x96; matches frozen fixture bit-for-bit. Does not include producing that preview.',decode='Separately measured OpenCV full-resolution BGR decode; not an end-to-end Native deployment benchmark.',initialization='Native initialization is compute-only, excluding data acquisition, fixture read, output writes.',wall_seconds=time.perf_counter()-started)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');print('Profiling complete',flush=True)
if __name__=='__main__':main()
