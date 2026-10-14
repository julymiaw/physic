"""Separate decoder/preparation context and historical C++ Bili replay; run after profile_native_stages.py."""
import ctypes as C,csv,json,subprocess,time
from pathlib import Path
import numpy as np,cv2
R=Path(__file__).resolve().parents[1];O=R/'experiments/results/native-stage-profile';W=R/'experiments/runs/native-stage-profile'
# Historical C++ counting logic, with timing probes disabled. Reference remains provisional.
lib=C.CDLL(str(W/'legacy.dylib'));lib.legacy_create.restype=C.c_void_p;lib.legacy_push.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_int,C.POINTER(C.c_double)];lib.legacy_destroy.argtypes=[C.c_void_p]
cap=cv2.VideoCapture(str(R/'data/raw/michelson-bilibili.mp4'));h=lib.legacy_create();times=np.load(R/'data/derived/bilibili.npz')['timestamps'];a=np.zeros(18);rows=[];error=None
for i,t in enumerate(times):
 ok,frame=cap.read();assert ok
 code=lib.legacy_push(h,frame.ctypes.data,frame.shape[1],frame.shape[0],frame.strides[0],0,a.ctypes.data_as(C.POINTER(C.c_double)))
 if code:error=dict(frame=i,timestamp=float(t),code=code);break
 rows.append(dict(source_frame=i,timestamp=float(t),count=int(a[1]),center_class=int(a[2]),status=int(a[3]),rings=int(a[4])))
cap.release();lib.legacy_destroy(h)
with (O/'legacy-cpp-bilibili.csv').open('w') as f:
 w=csv.DictWriter(f,rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
result=dict(frames=len(rows),error=error,origin=rows[255]['count'] if len(rows)>255 else None,final=rows[-1]['count'],measurement_delta=rows[-1]['count']-rows[255]['count'] if len(rows)==637 else None)
# Existing preparation path, received through Python pipe. This measures average offline throughput, not individual frame latency.
cmd=['ffmpeg','-v','error','-i',str(R/'data/raw/michelson-original.mp4'),'-an','-vf','scale=432:768:flags=area,format=gbrp,extractplanes=r','-fps_mode','passthrough','-pix_fmt','gray','-f','rawvideo','-']
tick=time.perf_counter();proc=subprocess.Popen(cmd,stdout=subprocess.PIPE);n=0
while True:
 b=proc.stdout.read(432*768)
 if not b:break
 assert len(b)==432*768;n+=1
assert proc.wait()==0 and n==1437
elapsed=time.perf_counter()-tick
r=dict(legacy_bilibili=result,ffmpeg_preview=dict(frames=n,total_seconds=elapsed,average_ms_per_frame=elapsed*1000/n,scope='FFmpeg decode, rotation, area resize, red channel extraction and Python pipe reception; offline throughput, not latency; excludes npz compression and Native processing.'))
(O/'context.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
