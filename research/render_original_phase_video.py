"""Explain original-video phase from saved Native observations; no Python counter.
Brightness reconstruction is checked against every saved Native phase and primary value.
"""
import csv,hashlib,json,shutil,struct,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/results/bilibili-adapt';RESULT=OUT/'original'
W,H,FPS=1600,1000,24
FONT='/System/Library/Fonts/STHeiti Medium.ttc'
FONTS={s:ImageFont.truetype(FONT,s) for s in [16,18,20,22,26,32,42,84]}
BG='#0d1625';FG='#edf3fa';GRAY='#9cadc4';GREEN='#54dec5';ORANGE='#ffa468'
COLORS=['#ffffff','#54dec5','#ffa468','#b29cff','#6fbaff']
def text(d,p,s,size=20,color=FG):d.text(p,s,font=FONTS[size],fill=color)
def line(d,p,color,width=2):
    if len(p)>1:d.line(p,fill=color,width=width)
def main():
    data=np.load(ROOT/'data/derived/original.npz');raw,times=data['frames'],data['timestamps']
    rows=list(csv.DictReader((RESULT/'frames.csv').open()));byframe={int(r['source_frame']):r for r in rows}
    online={int(r['source_frame']):r for r in csv.DictReader((RESULT/'frames.csv.online.csv').open())}
    model=json.loads((RESULT/'frames.csv.model.json').read_text());points=np.array([model['center']]+[[q['x'],q['y']] for q in model['nodes']])
    motion_path=ROOT/'experiments/results/physical-count/motion.csv'
    if not motion_path.exists():
        motion_path=ROOT/'experiments/runs/phase-video/motion.csv';motion_path.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['cmake','-S',str(ROOT),'-B',str(ROOT/'build-native'),'-DCMAKE_BUILD_TYPE=Release'],check=True)
        subprocess.run(['cmake','--build',str(ROOT/'build-native'),'--target','fringe_track'],check=True)
        subprocess.run([str(ROOT/'build-native/fringe_track'),str(ROOT/'experiments/results/real-track/padded-screen.frc'),'9.6','11.6',str(motion_path),'16',str(58*96/345)],check=True)
    motion={int(r['source_frame']):r for r in csv.DictReader(motion_path.open())}
    matrices=np.tile(np.eye(3),(len(times),1,1))
    for i,r in motion.items():
        assert abs(float(r['timestamp'])-times[i])<1e-8
        matrices[i]=np.array([float(r[f'h{a}{b}']) for a in range(3) for b in range(3)]).reshape(3,3)
    with (ROOT/'experiments/results/real-track/padded-screen.frc').open('rb') as f:
        _,n,_,_,count,_=struct.unpack('<4sIIIId',f.read(28));assert n==96 and count==len(times)
        pixels=np.empty((count,n,n),np.float32)
        for i in range(count):
            _,t=struct.unpack('<Id',f.read(12));assert abs(t-times[i])<1e-8
            pixels[i]=np.frombuffer(f.read(n*n*4),'<f4').reshape(n,n)
    # Same pixel centers, nine bilinear samples, double arithmetic and quantiles as Native.
    signal=np.zeros((len(times),len(points)));offset=np.array([16.,58*96/345]);ids=np.arange(len(times))
    for k,(x,y) in enumerate(points):
        for dy in [-1,0,1]:
            for dx in [-1,0,1]:
                p=matrices@np.array([x+dx,y+dy,1.]);q=p[:,:2]/p[:,2:]+offset
                ix=q[:,0].astype(int);iy=q[:,1].astype(int);u=q[:,0]-ix;v=q[:,1]-iy
                assert ix.min()>=0 and iy.min()>=0 and ix.max()<95 and iy.max()<95
                signal[:,k]+=(1-v)*((1-u)*pixels[ids,iy,ix].astype(float)+u*pixels[ids,iy,ix+1]) + v*((1-u)*pixels[ids,iy+1,ix].astype(float)+u*pixels[ids,iy+1,ix+1])
    signal/=9;cal=np.where((times>=9.6)&(times<=11.6))[0];order=np.sort(signal[cal],axis=0)
    lo,hi=order[int(.1*(len(cal)-1))],order[int(.9*(len(cal)-1))];norm=(signal-(lo+hi)/2)/((hi-lo)/2)
    orientation=np.array([q['orientation'] for q in model['nodes']]);zeros=np.array([q['zero'] for q in model['nodes']])
    phase=orientation*np.arctan2(norm[:,1:],norm[:,0,None])-zeros
    phase=np.arctan2(np.sin(phase),np.cos(phase));p_error=a_error=0.
    for r in rows:
        i=int(r['source_frame']);a_error=max(a_error,abs(norm[i,0]-float(r['primary'])))
        for k in range(4):p_error=max(p_error,abs(np.arctan2(np.sin(phase[i,k]-float(r[f'phase{k}'])),np.cos(phase[i,k]-float(r[f'phase{k}'])))))
    assert a_error<1e-7 and p_error<1e-7,(a_error,p_error)
    events=[r for r in rows if int(r['event_direction'])];first,last=min(byframe),max(byframe)
    assert first==279 and last==1436 and len(rows)==1158 and int(rows[-1]['count'])==84
    a=norm[:,0];b=norm[:,1]*orientation[0];extent=max(1.4,float(max(abs(a[cal]).max(),abs(b[cal]).max())*1.15))
    def render(i,chapter,caption,retrospective=False,end=False):
        t=float(times[i]);im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
        text(d,(28,19),chapter,32)
        text(d,(28,68),'自录原片 · 同一适配分支算法 · 位置不同的明暗组合，给出周期进度',20,GRAY)
        text(d,(1320,28),f'原片 {t:06.3f}s',26,GREEN)
        text(d,(28,113),'5个采样块：固定在光屏坐标',22)
        im.paste(Image.fromarray(raw[i,132:477,68:308]).convert('RGB').resize((400,575),Image.Resampling.LANCZOS),(28,150))
        p=np.c_[points,np.ones(5)]@matrices[i].T;p=p[:,:2]/p[:,2:]+offset
        for k,q in enumerate(p):
            u=28+(q[0]+.5)*400/96;v=150+(q[1]+.5)*575/96
            d.rectangle((u-5,v-5,u+5,v+5),outline=COLORS[k],width=2);text(d,(u+7,v-12),'C' if k==0 else str(k),16,COLORS[k])
        text(d,(28,741),'白色 C：中心亮度 a',20)
        text(d,(28,775),'绿色 1：辅助亮度 b（此处展开说明）',18,GREEN)
        # Show two actual normalized brightness traces; only their observed history is drawn.
        left,right,top,bottom=485,1005,160,315;text(d,(455,113),'两处亮度错开：组合后才能判断进度',22)
        d.rectangle((left,top,right,bottom),outline='#354860')
        wave_ids=np.where((times>=max(9.6,t-2))&(times<=t))[0]
        scale=max(1.4,float(np.max(abs(norm[wave_ids,:2])))*1.1) if len(wave_ids) else 1.4
        for val in [-1,0,1]:
            y=(top+bottom)/2-val*(bottom-top)/(2*scale);d.line([(left,y),(right,y)],fill='#26374b');text(d,(458,y-10),str(val),16,GRAY)
        for k,col in [(0,FG),(1,GREEN)]:
            line(d,[(left+(times[j]-(t-2))/2*(right-left),(top+bottom)/2-norm[j,k]*(bottom-top)/(2*scale)) for j in wave_ids],col)
        text(d,(485,326),'各自按标定亮暗范围归一化；辅助信号可翻转方向。',18,GRAY)
        # Real photometric trajectory, not an ideal manufactured circle.
        cx,cy,rad=745,582,170
        def plane(x,y):return (cx+x/extent*rad,cy-y/extent*rad)
        d.rectangle((cx-rad,cy-rad,cx+rad,cy+rad),outline='#354860')
        d.line([(cx-rad,cy),(cx+rad,cy)],fill='#354860');d.line([(cx,cy-rad),(cx,cy+rad)],fill='#354860')
        text(d,(485,371),'真实亮度轨迹：横轴 a，纵轴方向统一后的 b',18)
        line(d,[plane(a[j],b[j]) for j in cal],'#42766d',2)
        # A shared dark-center instant determines the zero direction for this pair.
        zero=zeros[0];ray=plane(np.cos(zero)*extent*.92,np.sin(zero)*extent*.92)
        d.line([(cx,cy),ray],fill=ORANGE,width=2)
        for j in model['dark_anchor_frames']:
            u,v=plane(a[j],b[j]);d.ellipse((u-4,v-4,u+4,v+4),fill=ORANGE)
        recent=np.where((times>=max(9.6,t-.45))&(times<=t))[0]
        line(d,[plane(a[j],b[j]) for j in recent],GREEN,3)
        u,v=plane(a[i],b[i]);d.line([(cx,cy),(u,v)],fill=FG,width=2);d.ellipse((u-6,v-6,u+6,v+6),fill=FG)
        text(d,(485,765),'橙点：同一组中心暗纹时刻；橙线：本路零点',18,ORANGE)
        text(d,(485,797),'真实轨迹会偏离椭圆，当前没有完整椭圆校正。',18,GRAY)
        # All four zero-compensated phase observations, as saved by the Native replay.
        text(d,(1070,113),'各路减去自己的零点角度',22)
        if retrospective:
            text(d,(1090,162),'标定回顾',42,ORANGE);text(d,(1070,221),'此处展示已完成标定后的模型',18,GRAY)
            text(d,(1070,260),'共同暗心帧：236 / 250 / 262 / 272',18)
        else:
            text(d,(1070,157),'已确认净计数',22,GRAY);text(d,(1090,190),str(int(online[i]['count_available_now'])),84,GREEN);text(d,(1250,235),'圈',26)
            confirmed=[e for e in events if int(e['confirmation_frame'])<=i]
            if confirmed:
                e=confirmed[-1];text(d,(1070,316),f'最近事件 {int(e["event_direction"]):+d}  发生 {float(e["timestamp"]):.3f}s',20)
                text(d,(1070,350),f'确认 {float(e["confirmed_at"]):.3f}s  延迟 {e["delay_frames"]}帧',18,GRAY)
        pcx,pcy,pr=1260,566,122;d.ellipse((pcx-pr,pcy-pr,pcx+pr,pcy+pr),outline='#354860',width=2);d.line([(pcx,pcy),(pcx+pr,pcy)],fill=GRAY)
        text(d,(1390,550),'共同零点',18,ORANGE)
        for k in range(4):
            angle=phase[i,k] if retrospective else float(byframe[i][f'phase{k}']);u=pcx+pr*np.cos(angle);v=pcy-pr*np.sin(angle)
            d.line([(pcx,pcy),(u,v)],fill=COLORS[k+1],width=2);d.ellipse((u-5,v-5,u+5,v+5),fill=COLORS[k+1])
        text(d,(1070,718),'箭头不必完全重合：它们是近似相位。',18,GRAY)
        text(d,(1070,754),'跨越共同边界后，多路证据确认计数。',18,GRAY)
        if 824<=i<=843 and not retrospective and not end:caption='34秒小亮点有物理歧义：允许暂时减一、加一，区间结束净值一致。'
        text(d,(28,859),caption,22,GREEN if end else FG)
        text(d,(28,905),'周期进度 = atan2(辅助亮度, 中心亮度)，统一方向后减去各路共同暗心零点。',20,GRAY)
        text(d,(28,956),'当前开发参考：88个确定事件匹配，另有一对可选往返；片尾净84圈。不是精确光学相位测量。',18,GRAY)
        return im
    ffmpeg=shutil.which('ffmpeg');assert ffmpeg
    dest=OUT/'original-phase-explained.mp4';proc=subprocess.Popen([ffmpeg,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],stdin=subprocess.PIPE)
    emitted=0
    def emit(im):
        nonlocal emitted
        proc.stdin.write(im.tobytes());emitted+=1
    intro=render(300,'01  两处亮度，为什么能给出相位？','先看原画面：不同采样位置明暗错开，二维亮度组合给出周期进度。')
    intro.save(OUT/'original-phase-intro.jpg')
    for _ in range(5*FPS):emit(intro)
    for t in np.linspace(9.6,11.6,8*FPS):
        i=max(cal[0],min(cal[-1],int(np.searchsorted(times,t,side='right')-1)))
        emit(render(i,'02  标定回顾：同一暗心时刻，为四路分别定零点','这里是完成标定后的慢放回顾；不能把各个采样点自己的最暗时刻当成共同零点。',True))
    for t in np.linspace(times[first],15.632622,8*FPS):
        i=max(first,int(np.searchsorted(times,t,side='right')-1))
        emit(render(i,'03  从真实亮度轨迹，到共同计圈边界','中间显示真实亮度轨迹；右侧显示四路扣除各自零点后的近似相位。'))
    for t in np.arange(times[first],times[last]+.5/FPS,1/FPS):
        i=max(first,min(last,int(np.searchsorted(times,t,side='right')-1)))
        emit(render(i,'04  全测量段回放：跟踪、相位观测与已确认计数','箭头表示当前观测；计数显示已经确认的结果，最多滞后4帧。'))
    outro=render(last,'05  同一新版算法，自录测量段净84圈','1158帧完整处理；88个确定事件全部对应，一对可选往返保留局部歧义。',end=True)
    outro.save(OUT/'original-phase-end.jpg')
    for _ in range(4*FPS):emit(outro)
    proc.stdin.close();assert proc.wait()==0
    manifest=dict(video=str(dest.relative_to(ROOT)),frames=emitted,fps=FPS,duration_seconds=emitted/FPS,measurement_source_frames=[first,last],final_confirmed_count=int(online[last]['count_available_now']),phase_reconstruction_max_error_radians=p_error,normalized_primary_max_error=a_error,calibration_replay_is_retrospective=True,trajectory_is_actual_brightness_not_ideal_circle=True,complete_measurement_replayed=True,source_results_sha256=hashlib.sha256((RESULT/'frames.csv').read_bytes()).hexdigest(),model_sha256=hashlib.sha256((RESULT/'frames.csv.model.json').read_bytes()).hexdigest(),video_sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
    (OUT/'original-phase-video.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
