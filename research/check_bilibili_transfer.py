"""Reproduce the frozen Bilibili transfer failure and render its real-data diagnostics.
Generated diagnostic instrumentation prints calibration metrics; it does not change gates.
"""
import csv,hashlib,json,struct,subprocess,tempfile
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'experiments/runs/bilibili-frozen-baseline'
P.mkdir(parents=True,exist_ok=True)
protocol=json.loads((ROOT/'experiments/results/bilibili-transfer/protocol.json').read_text())
source_text=subprocess.check_output(['git','show','self-recorded-v1:research/real_pair.cpp'],cwd=ROOT).decode()
assert hashlib.sha256(source_text.encode()).hexdigest()==protocol['tested_core_sha256'],'Frozen Git baseline does not match the recorded implementation.'
assert hashlib.sha256((ROOT/'research/screen_tracker.h').read_bytes()).hexdigest()==protocol['hashes']['research/screen_tracker.h']
assert hashlib.sha256((ROOT/'data/derived/bilibili.npz').read_bytes()).hexdigest()==protocol['hashes']['data/derived/bilibili.npz']
data=np.load(ROOT/'data/derived/bilibili.npz');frames,times=data['frames'],data['timestamps']
with (P/'padded-screen.frc').open('wb') as f:
    f.write(struct.pack('<4sIIIId',b'FRC1',96,1,1,len(frames),30.))
    for i,(image,time) in enumerate(zip(frames,times)):
        padded=np.zeros((270,246),np.uint8);padded[45:]=image[:225,243:489]
        small=cv2.resize(padded.astype(np.float32)/255,(96,96),interpolation=cv2.INTER_AREA)
        f.write(struct.pack('<Id',i,float(time)));f.write(small.astype('<f4').tobytes())
with tempfile.TemporaryDirectory(prefix='physic-bilibili-') as td:
    td=Path(td);binary=td/'counter';tracker=td/'tracker'
    def build(src,dest):subprocess.run(['clang++','-std=c++17','-O3','-DNDEBUG','-I',str(ROOT/'research'),str(src),'-o',str(dest)],check=True)
    source=td/'frozen.cpp';source.write_text(source_text)
    build(source,binary);build(ROOT/'research/screen_track.cpp',tracker)
    args=[str(P/'padded-screen.frc'),'0','8.5','22',str(P/'frames.csv'),'--count-offset-x','16','--count-offset-y','16']
    result=subprocess.run([str(binary),*args],capture_output=True,text=True)
    (P/'replay.log').write_text(result.stdout+result.stderr)
    assert result.returncode==1 and 'insufficient phase pairs' in result.stderr
    tracking=subprocess.run([str(tracker),str(P/'padded-screen.frc'),'0','8.5',str(P/'motion.csv'),'16','16'],check=True,capture_output=True,text=True)
    (P/'tracking.log').write_text(tracking.stdout)
    diagnostic=source_text
    diagnostic=diagnostic.replace(' std::vector<Node> candidates;', ' std::cout<<"center,"<<cx<<","<<cy<<",gradient_residual,"<<center_residual<<",primary_scale,"<<scale0<<"\\n";std::cout<<"candidate,x,y,contrast_scale,net_cycles,direction_dominance,min_radius,accepted\\n"; std::vector<Node> candidates;')
    needle='if(std::abs(net)<3*pi||std::abs(net)<.9*absolute||*std::min_element(radii.begin(),radii.end())<.15)continue;'
    diagnostic=diagnostic.replace(needle,'std::cout<<"candidate,"<<x<<","<<y<<","<<scale<<","<<net/(2*pi)<<","<<std::abs(net)/absolute<<","<<*std::min_element(radii.begin(),radii.end())<<","<<(std::abs(net)>=3*pi&&std::abs(net)>=.9*absolute&&*std::min_element(radii.begin(),radii.end())>=.15)<<"\\n";'+needle)
    generated=td/'diagnostic.cpp';generated.write_text(diagnostic);build(generated,binary)
    diag=subprocess.run([str(binary),*args],capture_output=True,text=True)
    (P/'calibration-diagnostic.txt').write_text(diag.stdout+diag.stderr)
from pathlib import Path
import json,csv,struct
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
P=ROOT/'experiments/runs/bilibili-frozen-baseline'
P.mkdir(parents=True,exist_ok=True)
raw=np.load(ROOT/'data/derived/bilibili.npz')['frames']
with (P/'padded-screen.frc').open('rb') as f:
 _,n,_,_,count,_=struct.unpack('<4sIIIId',f.read(28));pixels=[];times=[]
 for i in range(count):
  _,t=struct.unpack('<Id',f.read(12));times.append(t);pixels.append(np.frombuffer(f.read(n*n*4),'<f4').reshape(n,n))
pixels=np.array(pixels);times=np.array(times);motion=list(csv.DictReader((P/'motion.csv').open()))
lines=(P/'calibration-diagnostic.txt').read_text().splitlines();a=np.array([[float(v) for v in r.split(',')[1:]] for r in lines if r.startswith('candidate,') and r.split(',')[1]!='x']);best=a[np.argmax(a[:,4])];center=[float(lines[0].split(',')[i]) for i in [1,2]]
tr=np.eye(3);tr[:2,2]=[16,16]
def patch(frame,h,x,y):
 out=0
 for dy in [-1,0,1]:
  for dx in [-1,0,1]:
   q=h@np.array([x+16+dx,y+16+dy,1.]);xx,yy=q[:2]/q[2];ix,iy=int(xx),int(yy);u,v=xx-ix,yy-iy
   out+=(1-v)*((1-u)*frame[iy,ix]+u*frame[iy,ix+1])+v*((1-u)*frame[iy+1,ix]+u*frame[iy+1,ix+1])
 return out/9
signals=[]
for i in range(256):
 hc=np.array([float(motion[i][f'h{x}{y}']) for x in range(3) for y in range(3)]).reshape(3,3);h=tr@hc@np.linalg.inv(tr)
 signals.append([patch(pixels[i],h,*center),patch(pixels[i],h,*best[:2])])
signals=np.array(signals);order=np.sort(signals,axis=0);lo,hi=order[int(.1*255)],order[int(.9*255)];norm=(signals-(lo+hi)/2)/((hi-lo)/2)
phase=np.unwrap(np.arctan2(norm[:,1],norm[:,0]));steps=np.diff(phase);dominance=abs(steps.sum())/abs(steps).sum();assert abs(dominance-best[4])<1e-5
report=dict(initialized=False,counting_started=False,measured_coverage=0,evaluation_interval=[float(times[256]),float(times[-1])],evaluation_frames=381,calibration_frames=256,candidate_positions_examined=120,sufficient_contrast_candidates=len(a),accepted_phase_pairs=int(sum(a[:,-1])),direction_dominance_required=.9,maximum_direction_dominance=float(best[4]),winding_gate_passes=int(sum(abs(a[:,3])>=1.5)),radius_gate_passes=int(sum(a[:,5]>=.15)),estimated_interference_center=center,gradient_residual=float(lines[0].split(',')[4]),tracking_frames=len(motion),tracking_rejected_frames=sum(int(r['accepted'])==0 for r in motion),tracking_iou_median=float(np.median([float(r['iou_after']) for r in motion])),best_rejected_candidate=dict(x=float(best[0]),y=float(best[1]),net_cycles=float(best[3]),direction_dominance=float(best[4])),conclusion='Current unchanged counting rules fail calibration on this clip. No net count or event accuracy is claimed. Passing contour quality gates does not independently prove physical screen registration.')
(P/'result.json').write_text(json.dumps(report,indent=2)+'\n')
font='/System/Library/Fonts/STHeiti Medium.ttc';fonts={s:ImageFont.truetype(font,s) for s in [18,20,24,30,50]};im=Image.new('RGB',(1440,850),'#0d1625');d=ImageDraw.Draw(im)
def txt(x,y,s,size=20,color='#edf3fa'):d.text((x,y),s,font=fonts[size],fill=color)
txt(30,20,'B站真实视频：跟踪通过，但相位标定失败',30)
txt(30,68,'固定算法与筛选规则 · 开头256帧标定 · 未放宽门槛、未用参考标签调参',20,'#a4b4c9')
# Show a raw calibration crop, not a generated or cleaned picture.
im.paste(Image.fromarray(raw[100,0:180,284:448]).convert('RGB').resize((410,450),Image.Resampling.LANCZOS),(30,130))
hc=np.array([float(motion[100][f'h{x}{y}']) for x in range(3) for y in range(3)]).reshape(3,3);h=tr@hc@np.linalg.inv(tr)
for p,color,label in [(center,'#ffa468','C'),(best[:2],'#54dec5','A')]:
 q=h@np.array([p[0]+16,p[1]+16,1.]);q=q[:2]/q[2];source=np.array([243,-45])+(q+.5)*np.array([246,270])/96-.5
 x=30+(source[0]-284)*2.5;y=130+source[1]*2.5;d.ellipse((x-6,y-6,x+6,y+6),outline=color,width=3);txt(x+8,y-10,label,20,color)
txt(30,595,'原始第100帧（3.333秒）',20)
txt(30,627,'C：拟合中心   A：方向一致性最高的候选',18,'#a4b4c9')
txt(485,125,'51组候选，合格0组',30,'#ffa468')
txt(485,171,f'最佳方向一致性 {dominance:.3f}  <  要求0.900',24)
txt(485,212,'下图仅为被拒绝候选的诊断，不是计数输出',18,'#a4b4c9')
# Normalized brightness traces.
x0,x1,y0,y1=490,1380,265,415
d.rectangle((x0,y0,x1,y1),outline='#354860')
for k,col in enumerate(['#ffa468','#54dec5']):
 pts=[(x0+t/8.5*(x1-x0),(y0+y1)/2-norm[j,k]*50) for j,t in enumerate(times[:256])]
 d.line(pts,fill=col,width=2)
txt(490,425,'中心 / 辅助块归一化亮度，0–8.5秒',18,'#a4b4c9')
# Net unwrapped angle, diagnostic only. Repeated jagged reversals explain dominance loss.
y0,y1=495,640;progress=(phase-phase[0])/(2*np.pi);lo,hi=progress.min()-.1,progress.max()+.1
d.rectangle((x0,y0,x1,y1),outline='#354860');d.line([(x0+t/8.5*(x1-x0),y1-(progress[j]-lo)/(hi-lo)*(y1-y0)) for j,t in enumerate(times[:256])],fill='#b29cff',width=2)
txt(490,653,'候选角度展开：总体推进约3.6周期，但有大量局部折返',18,'#a4b4c9')
txt(30,720,'当前结论：不能直接使用；算法在初始化阶段明确拒绝，没有可信计数。',24,'#ffa468')
txt(30,769,'可能成因：散斑与静态纹理、中心定位偏差、局部明暗偏离共同相位模型；尚未逐项证实。',20,'#a4b4c9')
im.save(P/'failure-overview.jpg');print(json.dumps(report,indent=2))
