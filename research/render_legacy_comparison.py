"""Replay historical undergraduate detector unchanged and compare same-frame displays.
Only video I/O and visual drawing are replaced; old per-frame detection/state are original.
The historical process_video skips frame0 and changes output FPS, so it is not used.
"""
import ast,csv,hashlib,importlib.util,json,shutil,subprocess,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
import cv2 as cv
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/results/legacy-side-by-side'
OLD_COMMIT='c3ae479a0999f263b9365f1f94845e691e55d953'
W,H,FPS=1600,1000,24
FONTS={s:ImageFont.truetype('/System/Library/Fonts/STHeiti Medium.ttc',s) for s in [16,18,20,22,26,32,42,80]}
FG='#edf3fa';GRAY='#9cadc4';GREEN='#54dec5';ORANGE='#ffa468';BG='#0d1625'
COLORS=['#ffffff','#54dec5','#ffa468','#b29cff','#6fbaff']
def text(d,p,s,size=20,color=FG):d.text(p,s,font=FONTS[size],fill=color)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def main():
    OUT.mkdir(parents=True,exist_ok=True);run=ROOT/'experiments/runs/legacy-side-by-side';run.mkdir(parents=True,exist_ok=True)
    code=subprocess.check_output(['git','show',f'{OLD_COMMIT}:轮廓检测.py'],cwd=ROOT)
    source_hash=hashlib.sha256(code).hexdigest()
    assert source_hash=='0b0174987b88cccad8c8ca260b6aa2a84864b6cea8b0e690fac110c01d09625b'
    archived=ROOT/'archive/undergraduate-senior-autumn/轮廓检测.py'
    def core_ast(s):return [ast.dump(n) for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and getattr(n,'name',None)!='main']
    if archived.exists():assert core_ast(code.decode())==core_ast(archived.read_text())
    source=run/'historical_contour.py';source.write_bytes(code)
    spec=importlib.util.spec_from_file_location('historical_contour',source);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    cv.setNumThreads(1);counter=old.ContourDetection()
    data=np.load(ROOT/'data/derived/original.npz');times=data['timestamps'];count=len(times)
    # Fresh Native replay, with current code fingerprint checked against published evidence.
    summary=json.loads((ROOT/'experiments/results/bilibili-adapt/summary.json').read_text())
    assert sha(ROOT/'research/real_pair.cpp')==summary['implementation_sha256']
    build=ROOT/'build-native';subprocess.run(['cmake','-S',str(ROOT),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release'],check=True)
    subprocess.run(['cmake','--build',str(build),'--target','fringe_count','fringe_track'],check=True)
    native=run/'new.csv';fixture=ROOT/'experiments/results/real-track/padded-screen.frc'
    subprocess.run([str(build/'fringe_count'),str(fixture),'9.6','11.6','60',str(native)],check=True)
    assert native.read_bytes()==(ROOT/'experiments/results/bilibili-adapt/original/frames.csv').read_bytes()
    fresh=list(csv.DictReader(native.open()));rows={int(r['source_frame']):r for r in fresh}
    online={int(r['source_frame']):r for r in csv.DictReader(Path(str(native)+'.online.csv').open())}
    model=json.loads(Path(str(native)+'.model.json').read_text());points=np.array([model['center']]+[[q['x'],q['y']] for q in model['nodes']])
    motion_path=ROOT/'experiments/results/physical-count/motion.csv'
    if not motion_path.exists():
        motion_path=run/'motion.csv';subprocess.run([str(build/'fringe_track'),str(fixture),'9.6','11.6',str(motion_path),'16',str(58*96/345)],check=True)
    motion={int(r['source_frame']):r for r in csv.DictReader(motion_path.open())}
    first,last=min(rows),max(rows);assert (first,last)==(279,1436)
    baseline_path=ROOT/'experiments/results/legacy-reference-v1/contour/frames.csv'
    baseline=list(csv.DictReader(baseline_path.open())) if baseline_path.exists() else None
    observed={};original_process=old.process_frame;original_segment=old.segment_image;original_draw=cv.drawContours;original_circle=cv.circle
    def observe_segment(*args,**kwargs):
        result=original_segment(*args,**kwargs);observed['bbox']=result[0];return result
    def observe_draw(image,contours,index,color,thickness=1,*args,**kwargs):
        if image.ndim==3 and thickness==2 and tuple(color) in [(0,255,0),(255,0,0)]:observed['contours'].extend([(c.copy(),tuple(color)) for c in contours])
        return original_draw(image,contours,index,color,thickness,*args,**kwargs)
    def observe_circle(image,center,radius,color,thickness=1,*args,**kwargs):
        if image.ndim==3:observed['center_xy']=center
        return original_circle(image,center,radius,color,thickness,*args,**kwargs)
    def observe_process(*args,**kwargs):
        result=original_process(*args,**kwargs);observed['rings']=len(result[2]);observed['center_class']={255:'bright',0:'dark'}.get(result[3],'unknown');return result
    old.segment_image=observe_segment;old.process_frame=observe_process;cv.drawContours=observe_draw;cv.circle=observe_circle
    video=OUT/'undergraduate-vs-current.mp4';ffmpeg=shutil.which('ffmpeg');assert ffmpeg
    encoder=subprocess.Popen([ffmpeg,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
    cap=cv.VideoCapture(str(ROOT/'data/raw/michelson-original.mp4'));cap.set(cv.CAP_PROP_ORIENTATION_AUTO,1)
    old_origin=None;all_records=[];comparisons=[];emitted=0;started=time.perf_counter();last_image=None
    def render(i,preview,record,outro=False):
        im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
        text(d,(28,18),'同一自录原片：本科轮廓法  vs  当前相位法',32)
        text(d,(28,65),f'原始帧 {i}  ·  时间 {times[i]:.3f}s  ·  共同计数起点 11.633s',20,GRAY)
        d.line([(800,106),(800,917)],fill='#354860',width=2)
        text(d,(28,109),'本科旧方法 · 历史 Python 轮廓实现',26,ORANGE)
        text(d,(828,109),'当前新方法 · Native C++ 相位计数',26,GREEN)
        # Both panes show exactly the same source ROI and source frame.
        im.paste(Image.fromarray(cv.cvtColor(preview,cv.COLOR_BGR2RGB)).resize((400,575),Image.Resampling.LANCZOS),(28,155))
        # Right image is the clean red-channel crop saved before old update mutates frame.
        im.paste(Image.fromarray(clean_preview).convert('RGB').resize((400,575),Image.Resampling.LANCZOS),(828,155))
        h=np.array([float(motion[i][f'h{a}{b}']) for a in range(3) for b in range(3)]).reshape(3,3)
        p=np.c_[points,np.ones(5)]@h.T;p=p[:,:2]/p[:,2:]+[16,58*96/345]
        for k,q in enumerate(p):
            x=828+(q[0]+.5)*400/96;y=155+(q[1]+.5)*575/96
            d.rectangle((x-5,y-5,x+5,y+5),outline=COLORS[k],width=2);text(d,(x+7,y-11),'C' if k==0 else str(k),16,COLORS[k])
        old_count=int(record['count'])-old_origin;new_count=int(online[i]['count_available_now'])
        text(d,(459,166),'相对起点净计数',22,GRAY);text(d,(475,205),str(old_count),80,ORANGE);text(d,(650,245),'圈',22)
        text(d,(1259,166),'已确认净计数',22,GRAY);text(d,(1275,205),str(new_count),80,GREEN);text(d,(1450,245),'圈',22)
        status={'zoom_in':'收缩','zoom_out':'扩张'}.get(record['status'],'未确定')
        center={'bright':'亮心','dark':'暗心','unknown':'未判断'}.get(record['center_class'],'未判断')
        text(d,(459,329),f'筛选轮廓：{record["rings"]}条',22)
        text(d,(459,371),f'面积变化投票：{status}',20)
        text(d,(459,413),f'中心判断：{center}',20)
        text(d,(459,455),f'本帧计数变化：{int(record["delta"]):+d}',20)
        text(d,(459,512),f'旧原始累计：{record["count"]}',18,GRAY)
        text(d,(459,545),f'展示时减去起点累计：{old_origin}',18,GRAY)
        if center=='未判断':text(d,(459,596),'本帧无中心判断，沿用旧状态',18,ORANGE)
        pcx,pcy,rad=1400,483,100;d.ellipse((pcx-rad,pcy-rad,pcx+rad,pcy+rad),outline='#354860',width=2);d.line([(pcx,pcy),(pcx+rad,pcy)],fill=GRAY)
        for k in range(4):
            ph=float(rows[i][f'phase{k}']);x=pcx+rad*np.cos(ph);y=pcy-rad*np.sin(ph);d.line([(pcx,pcy),(x,y)],fill=COLORS[k+1],width=2)
        text(d,(1259,617),'4路近似相位，共同暗心零点',18,GRAY)
        text(d,(1259,651),'显示值最多滞后4帧确认',18,GRAY)
        text(d,(28,758),'红通道分割 → 二值化 → 筛选圆环轮廓',20)
        text(d,(28,798),'同色轮廓面积变化 → 中心明暗切换 → 加减一',20)
        text(d,(828,758),'跟踪光屏 → 5块局部亮度 → 4路相位',20)
        text(d,(828,798),'统一计圈零点 → 多路证据 → 最多4帧确认',20)
        if outro:msg=f'片尾两侧净值均为{new_count}圈；总值相同不能证明逐事件一致，需检查局部事件。'
        elif 1088<=i<=1105:msg='45.6秒附近：观察旧方法中心判断的短时切换，与新方法的计数确认。'
        elif 1223<=i<=1239:msg='51.25秒附近：观察内侧条纹推进，比较两侧是否记录这次周期。'
        elif 708<=i<=731:msg='29秒附近：观察短暂往返；不同边界和确认延迟会影响事件显示时刻。'
        elif 824<=i<=843:msg='34秒小亮点存在物理歧义：新参考允许暂时往返，但区间净值必须一致。'
        else:msg='左右严格按同一原始帧播放；旧算法未重置，只对显示累计减去共同起点值。'
        text(d,(28,862),msg,22,GREEN if outro else FG)
        text(d,(28,911),'两法周期边界和确认时刻有差异，瞬时计数差不能直接视为误差。',18,GRAY)
        text(d,(28,955),'同帧离线回放；播放速度不是计算速度。轮廓线仅为可视化加粗，不改变检测或计数。',18,GRAY)
        return im
    try:
        for i in range(count):
            ok,frame=cap.read()
            if not ok:raise RuntimeError(f'Video ended at source frame {i}')
            assert frame.shape[:2]==(7680,4320)
            clean_preview=cv.resize(frame[1320:4770,680:3080,2],(240,345),interpolation=cv.INTER_AREA)
            previous=counter.count_value
            if i==first:old_origin=previous
            observed.clear();observed['contours']=[];tick=time.perf_counter();counter.update(frame);ms=(time.perf_counter()-tick)*1000
            record=dict(source_frame=i,timestamp=float(times[i]),count=counter.count_value,delta=counter.count_value-previous,rings=observed.get('rings',0),center_class=observed.get('center_class','unknown'),status=counter.status or 'unknown',compute_ms=ms)
            if baseline:
                prior=baseline[i]
                for key in ['count','delta','rings','center_class','status']:assert str(record[key])==prior[key],(i,key,record[key],prior[key])
            all_records.append(record)
            if i>=first:
                preview=cv.cvtColor(clean_preview,cv.COLOR_GRAY2BGR);bbox=observed.get('bbox',(0,0,0,0));bx,by=bbox[:2]
                for c,color in observed['contours']:
                    q=c.astype(float);q[:,:,0]=(q[:,:,0]+bx-680)/10;q[:,:,1]=(q[:,:,1]+by-1320)/10
                    cv.polylines(preview,[np.round(q).astype(np.int32)],True,color,1,cv.LINE_AA)
                if 'center_xy' in observed:
                    x,y=observed['center_xy'];q=(round((x+bx-680)/10),round((y+by-1320)/10));cv.circle(preview,q,4,(0,180,255),1,cv.LINE_AA)
                current=render(i,preview,record);last_image=current
                if i==first:
                    current.save(OUT/'comparison-intro.jpg')
                    for _ in range(3*FPS):encoder.stdin.write(current.tobytes());emitted+=1
                encoder.stdin.write(current.tobytes());emitted+=1
                comparisons.append(dict(source_frame=i,timestamp=float(times[i]),old_raw_count=counter.count_value,old_relative_count=counter.count_value-old_origin,new_confirmed_count=int(online[i]['count_available_now']),new_occurrence_count=int(rows[i]['count']),old_delta=record['delta'],new_occurrence_delta=int(rows[i]['event_direction']),old_center=record['center_class'],old_status=record['status'],old_rings=record['rings']))
                if i in [1096,1231]:current.save(OUT/f'comparison-frame-{i}.jpg')
            if (i+1)%96==0:print(f'Historical replay {i+1}/{count}; old_raw={counter.count_value}; elapsed={time.perf_counter()-started:.1f}s',flush=True)
        ok,_=cap.read();assert not ok
        last_image=render(last,preview,all_records[-1],True);last_image.save(OUT/'comparison-end.jpg')
        for _ in range(4*FPS):encoder.stdin.write(last_image.tobytes());emitted+=1
    finally:
        old.process_frame=original_process;old.segment_image=original_segment;cv.drawContours=original_draw;cv.circle=original_circle;cap.release();encoder.stdin.close()
    assert encoder.wait()==0
    for filename,records in [('old-frames.csv',all_records),('comparison.csv',comparisons)]:
        with (OUT/filename).open('w') as f:
            writer=csv.DictWriter(f,records[0].keys(),lineterminator='\n');writer.writeheader();writer.writerows(records)
    assert [r['source_frame'] for r in comparisons]==list(range(first,last+1))
    assert sha(source)==source_hash
    manifest=dict(video=str(video.relative_to(ROOT)),fps=FPS,frames=emitted,duration_seconds=emitted/FPS,old_source_commit=OLD_COMMIT,old_source_path='轮廓检测.py',old_source_sha256=source_hash,old_archived_core_ast_equal=True,old_source_resolution=[4320,7680],old_parameters='Historical defaults; zoom_in positive; process_frame and ContourDetection.update unchanged',old_io_changes='Read every source frame from0; do not use historical driver that skips first frame and rewrites playback FPS.',measurement_frame_interval=[first,last],measurement_frame_count=len(comparisons),old_warmup_frames=first,old_display_origin=old_origin,old_final_raw_count=counter.count_value,old_final_relative_count=counter.count_value-old_origin,new_final_confirmed_count=int(online[last]['count_available_now']),old_measurement_positive_events=sum(r['old_delta']>0 for r in comparisons),old_measurement_negative_events=sum(r['old_delta']<0 for r in comparisons),new_measurement_positive_events=sum(r['new_occurrence_delta']>0 for r in comparisons),new_measurement_negative_events=sum(r['new_occurrence_delta']<0 for r in comparisons),legacy_reference_all_frames_identical=bool(baseline),new_replay_byte_identical_to_verified_results=True,new_source_sha256=summary['implementation_sha256'],raw_source_sha256=sha(ROOT/'data/raw/michelson-original.mp4'),video_sha256=sha(video),note='Same-frame offline algorithm demonstration, not a controlled speed benchmark. Equal endpoint totals do not establish equal event accuracy. New display uses confirmation timeline; old count updates immediately at its own threshold boundary.')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n');print(json.dumps(manifest,indent=2,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
