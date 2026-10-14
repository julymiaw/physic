#include "live_counter.h"
extern "C" {
void* fc_create(){try{return new live::Counter;}catch(...){return nullptr;}}
void fc_destroy(void* h){delete static_cast<live::Counter*>(h);}
float* fc_input(void* h){return h?static_cast<live::Counter*>(h)->input.data():nullptr;}
const double* fc_result(void* h){return h?static_cast<live::Counter*>(h)->result.data():nullptr;}
const double* fc_row(void* h,int index){return h&&index>=0?static_cast<live::Counter*>(h)->row(index):nullptr;}
int fc_reset(void* h,double x,double y){if(!h||!std::isfinite(x)||!std::isfinite(y)||x<0||y<0||x+64>96||y+64>96)return 0;static_cast<live::Counter*>(h)->reset(x,y);return 1;}
int fc_calibrate(void* h,double t,unsigned source){return h&&static_cast<live::Counter*>(h)->collect(t,source);}
int fc_start(void* h){return h&&static_cast<live::Counter*>(h)->start();}
int fc_push(void* h,double t,unsigned source){return h&&static_cast<live::Counter*>(h)->push(t,source);}
void fc_finish(void* h){if(h)static_cast<live::Counter*>(h)->finish();}
void fc_interrupt(void* h){if(h)static_cast<live::Counter*>(h)->interrupt();}
void fc_calibration_end(void* h){if(h)static_cast<live::Counter*>(h)->end_calibration();}
}
