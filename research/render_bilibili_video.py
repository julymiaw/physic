"""Visualize saved Bilibili Native results, without implementing a second counter."""
import csv,hashlib,json,shutil,subprocess
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/results/bilibili-adapt';RESULT=OUT/'bilibili'
W,H,FPS=1440,940,30
FONT='/System/Library/Fonts/STHeiti Medium.ttc'
FONTS={s:ImageFont.truetype(FONT,s) for s in [16,18,20,22,26,32,42,80]}
BG='#0d1625';FG='#edf3fa';GRAY='#9cadc4';GREEN='#54dec5';ORANGE='#ffa468'
COLORS=['#ffffff','#54dec5','#ffa468','#b29cff','#6fbaff']
def text(d,p,s,size=20,color=FG):d.text(p,s,font=FONTS[size],fill=color)
def main():
    data=np.load(ROOT/'data/derived/bilibili.npz');raw,times=data['frames'],data['timestamps']
    rows=list(csv.DictReader((RESULT/'frames.csv').open()));byframe={int(r['source_frame']):r for r in rows}
    online={int(r['source_frame']):r for r in csv.DictReader((RESULT/'frames.csv.online.csv').open())}
    motion={int(r['source_frame']):r for r in csv.DictReader((ROOT/'experiments/results/bilibili-transfer/motion.csv').open())}
    model=json.loads((RESULT/'frames.csv.model.json').read_text());points=np.array([model['center']]+[[q['x'],q['y']] for q in model['nodes']])
    old=np.array([33.9194,31.3056]);events=[r for r in rows if int(r['event_direction'])]
    assert len(raw)==637 and len(rows)==381 and len(events)==6 and int(rows[-1]['count'])==6
    assert [int(r['source_frame']) for r in rows]==list(range(256,637))
    phase=np.array([[float(r[f'phase{k}'])/(2*np.pi) for k in range(4)] for r in rows]);rowtimes=np.array([float(r['timestamp']) for r in rows])
    def transform(i):return np.array([float(motion[i][f'h{a}{b}']) for a in range(3) for b in range(3)]).reshape(3,3)
    def mapped(i,p):
        q=np.c_[p,np.ones(len(p))]@transform(i).T
        return q[:,:2]/q[:,2:]
    def aligned(i):
        yy,xx=np.mgrid[:180,:164];p=np.stack([(xx+.5)*64/164-.5,(yy+.5)*64/180-.5,np.ones_like(xx)],axis=-1)@transform(i).T
        mx=(284+(p[:,:,0]/p[:,:,2]+.5)*164/64-.5).astype(np.float32)
        my=((p[:,:,1]/p[:,:,2]+.5)*180/64-.5).astype(np.float32)
        return cv2.remap(raw[i],mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    # This is an explanatory reconstruction of the saved geometry, not a model input.
    mean=np.mean([aligned(i).astype(float) for i in range(256)],axis=0)
    mean_image=np.uint8(np.clip(mean,0,255))
    # Native threshold's footprint, drawn as a display-only outline.
    pad=np.zeros((270,246),np.float32)
    for i in range(256):pad[45:]+=raw[i,:225,243:489]/(255*256)
    small=cv2.resize(pad,(96,96),interpolation=cv2.INTER_AREA);blur=cv2.GaussianBlur(small,(9,9),1.3,borderType=cv2.BORDER_REPLICATE)
    sorted_pixels=np.sort(blur.ravel());lo,hi=sorted_pixels[int(.1*(len(sorted_pixels)-1))],sorted_pixels[int(.9*(len(sorted_pixels)-1))]
    contours,_=cv2.findContours((blur>lo+.4*(hi-lo)).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    outline=max(contours,key=cv2.contourArea)[:,0,:].astype(float)-[16,16]
    def pane(im,d,x,image,title,positions=None,boundary=None,old_point=None):
        text(d,(x,109),title,20)
        im.paste(Image.fromarray(image).convert('RGB').resize((430,472),Image.Resampling.LANCZOS),(x,145))
        def xy(p):return (x+(float(p[0])+.5)*430/64,145+(float(p[1])+.5)*472/64)
        if boundary is not None:
            chain=[xy(p) for p in boundary];d.line(chain+[chain[0]],fill=GREEN,width=2)
        if positions is not None:
            for k,p in enumerate(positions):
                u,v=xy(p);d.rectangle((u-4,v-4,u+4,v+4),outline=COLORS[k],width=2)
            # Labels are kept outside the clustered auxiliary patches.
            u,v=xy(positions[0]);d.line([(u,v),(u-38,v+40)],fill=FG,width=1);text(d,(u-70,v+40),'新中心',18)
        if old_point is not None:
            u,v=xy(old_point);d.line([(u-5,v-5),(u+5,v+5)],fill=ORANGE,width=3);d.line([(u-5,v+5),(u+5,v-5)],fill=ORANGE,width=3);text(d,(u+10,v-25),'旧中心',18,ORANGE)
    def render(i,stage='run'):
        t=float(times[i]);active=i>=256;im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
        text(d,(28,20),'B站场景：去除静态背景，让中心标定恢复',32)
        text(d,(28,67),'同一套计数规则  ·  只修改中心拟合的标定输入  ·  不设置B站专用阈值',20,GRAY)
        text(d,(1150,28),f'原片 {t:05.2f} 秒',26,GREEN)
        if stage=='intro':
            pane(im,d,28,raw[i,:180,284:448],'原画面：纹理使原拟合中心偏上',mapped(i,points),None,mapped(i,old[None])[0])
            diff=np.uint8(np.clip(128+(aligned(i).astype(float)-mean)*2.5,0,255))
            pane(im,d,488,diff,'减去标定平均图：变化条纹更突出',points,None,old)
            text(d,(953,130),'仅改变标定',26,GREEN)
            for y,s in [(183,'1  对齐标定画面'),(222,'2  减去时间平均图'),(261,'3  拟合条纹几何中心'),(323,'测量时仍采样5个小块'),(361,'无需逐帧识别完整圆环')]:text(d,(953,y),s,20)
            text(d,(953,439),'解释定格',26,ORANGE);text(d,(953,481),'模型取自完整标定段',18,GRAY)
            text(d,(28,660),'相减图以灰色表示零，明暗表示正负变化；放大对比度仅为展示。',22,GRAY)
            text(d,(28,711),'随后回放完整原片：前8.5秒标定，之后开始计数。',26)
            text(d,(28,762),'这些图只解释保存的Native结果，不提供另一份计数实现。',20,GRAY)
        else:
            pane(im,d,28,raw[i,:180,284:448],'原画面：光屏内局部放大',mapped(i,points) if active else None,mapped(i,outline) if active else None)
            if active:
                pane(im,d,488,aligned(i),'光屏坐标：跟踪后读取5个小块',points,outline,old)
                r=byframe[i];shown=int(online[i]['count_available_now'])
                text(d,(953,122),'已确认净计数',22,GRAY);text(d,(962,161),str(shown),80,GREEN);text(d,(1090,199),'圈',26)
                confirmed=[e for e in events if int(e['confirmation_frame'])<=i]
                if confirmed:
                    e=confirmed[-1];text(d,(953,286),f'发生 {float(e["timestamp"]):.3f}s   +1',20)
                    text(d,(953,321),f'确认 {float(e["confirmed_at"]):.3f}s',20,GRAY)
                    text(d,(953,356),f'确认延迟 {e["delay_frames"]} 帧',20,GRAY)
                cx,cy,rad=1090,488,85;d.ellipse((cx-rad,cy-rad,cx+rad,cy+rad),outline='#354860',width=2);d.line([(cx,cy),(cx+rad,cy)],fill=GRAY)
                for k in range(4):
                    angle=float(r[f'phase{k}']);q=(cx+rad*np.cos(angle),cy-rad*np.sin(angle));d.line([(cx,cy),q],fill=COLORS[k+1],width=2);d.ellipse((q[0]-4,q[1]-4,q[0]+4,q[1]+4),fill=COLORS[k+1])
                text(d,(953,588),'4路相位，零点统一',20,GRAY)
                msg='6次事件匹配临时视觉参考；测量段净6圈。' if stage=='outro' else '原圈心上方的静态纹理被抑制；采样点固定在光屏坐标。'
                text(d,(28,651),msg,22,GREEN if stage=='outro' else FG)
                text(d,(28,699),'最近4秒相位观测 / 周期（跳边为角度回绕；计数取已确认值）',18,GRAY)
                left,right,top,bottom=60,1380,738,866;d.rectangle((left,top,right,bottom),outline='#354860')
                for value in [-.5,0,.5]:
                    y=(top+bottom)/2-value*(bottom-top);d.line([(left,y),(right,y)],fill='#26374b');text(d,(29,y-9),str(value),16,GRAY)
                ids=np.where((rowtimes>=t-4)&(rowtimes<=t))[0]
                for k in range(4):
                    chain=[]
                    for j in ids:
                        q=(left+(rowtimes[j]-(t-4))/4*(right-left),(top+bottom)/2-phase[j,k]*(bottom-top))
                        if chain and abs(phase[j,k]-phase[j-1,k])>.5:
                            if len(chain)>1:d.line(chain,fill=COLORS[k+1],width=2)
                            chain=[]
                        chain.append(q)
                    if len(chain)>1:d.line(chain,fill=COLORS[k+1],width=2)
            else:
                # Do not display future-selected nodes or future alignment as live calibration.
                collecting=np.uint8(np.mean(raw[:i+1,:180,284:448],axis=0))
                pane(im,d,488,collecting,'采集画面的累计平均（未对齐示意）')
                text(d,(953,130),'标定中',42,ORANGE);text(d,(953,204),f'已采集 {i+1}/256 帧',22)
                text(d,(953,254),'此段不累计圈数',22,GRAY)
                text(d,(28,651),'先收集0–8.5秒标定画面；收集完成前不展示未来选出的采样点。',22)
                text(d,(28,709),'光屏内的固定纹理与变化条纹混合在一起；时间平均用于抑制固定部分。',20,GRAY)
                progress=(i+1)/256;d.rectangle((28,775,1380,806),outline='#354860');d.rectangle((29,776,29+1350*progress,805),fill=ORANGE)
                text(d,(28,831),'计数从8.533秒开始；不会把标定阶段的圈数混入测量结果。',20,GRAY)
        text(d,(28,903),'B站测量段8.533–21.2秒；参考为AI视觉复核，未经独立人工确认；陌生场景仍未验证。',18,GRAY)
        return im
    dest=OUT/'bilibili-count-explained.mp4';ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:raise RuntimeError('FFmpeg is required')
    proc=subprocess.Popen([ffmpeg,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],stdin=subprocess.PIPE)
    intro=render(100,'intro');intro.save(OUT/'bilibili-video-intro.jpg')
    for _ in range(4*FPS):proc.stdin.write(intro.tobytes())
    for i in range(len(raw)):proc.stdin.write(render(i).tobytes())
    last=render(636,'outro');last.save(OUT/'bilibili-video-end.jpg')
    for _ in range(3*FPS):proc.stdin.write(last.tobytes())
    proc.stdin.close();assert proc.wait()==0
    manifest=dict(video=str(dest.relative_to(ROOT)),fps=FPS,frames=4*FPS+len(raw)+3*FPS,duration_seconds=(4*FPS+len(raw)+3*FPS)/FPS,source_time_interval=[float(times[0]),float(times[-1])],measurement_frame_interval=[256,636],final_confirmed_count=int(online[636]['count_available_now']),explanation_intro_seconds=4,calibration_not_counted=True,derived_data_sha256=hashlib.sha256((ROOT/'data/derived/bilibili.npz').read_bytes()).hexdigest(),result_sha256=hashlib.sha256((RESULT/'frames.csv').read_bytes()).hexdigest(),reference_status='provisional visual reference, not independently human verified')
    (OUT/'bilibili-video.json').write_text(json.dumps(manifest,indent=2)+'\n');print(dest)
if __name__=='__main__':main()
