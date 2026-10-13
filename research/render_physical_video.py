"""Render the saved Native physical-count replay; no counting is performed here."""
import csv,json,subprocess
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/results/physical-count'
W,H,FPS=1440,940,24
FONT='/System/Library/Fonts/STHeiti Medium.ttc'
fonts={s:ImageFont.truetype(FONT,s) for s in [16,18,20,22,26,32,40,76]}
BG='#0d1625';FG='#eef4fc';GRAY='#9cadc4';GREEN='#54dec5';ORANGE='#ffae6e'
COLORS=['#ffffff','#54dec5','#ffa468','#b29cff','#6fbaff']
def text(d,p,s,size=20,color=FG):d.text(p,s,font=fonts[size],fill=color)
def main():
    data=np.load(ROOT/'data/derived/original.npz');raw,times=data['frames'],data['timestamps']
    rows=list(csv.DictReader((OUT/'frames.csv').open()));byframe={int(r['source_frame']):r for r in rows}
    online={int(r['source_frame']):r for r in csv.DictReader((OUT/'frames.csv.online.csv').open())}
    motion={int(r['source_frame']):r for r in csv.DictReader((OUT/'motion.csv').open())}
    model=json.loads((OUT/'frames.csv.model.json').read_text());points=np.array([model['center']]+[[n['x'],n['y']] for n in model['nodes']])
    phases=np.array([[float(r[f'phase{k}'])/(2*np.pi) for k in range(4)] for r in rows]);rowtimes=np.array([float(r['timestamp']) for r in rows])
    events=[r for r in rows if int(r['event_direction'])]
    offset=np.array([16.,58*96/345]);tr=np.eye(3);tr[:2,2]=offset
    # Reference outline is a display overlay only. Native transforms come from C++.
    reference=raw[(times>=9.6)&(times<=11.6),132:477,68:308].mean(0).astype(np.float32)/255
    small=cv2.resize(reference,(96,96),interpolation=cv2.INTER_AREA)
    blur=cv2.GaussianBlur(small,(9,9),1.3,borderType=cv2.BORDER_REPLICATE)
    ordered=np.sort(blur.ravel());lo,hi=ordered[int(.1*(len(ordered)-1))],ordered[int(.9*(len(ordered)-1))]
    contours,_=cv2.findContours((blur>lo+.4*(hi-lo)).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    outline=max(contours,key=cv2.contourArea)[:,0,:].astype(np.float64)
    def render(i,stage):
        r=byframe[i];t=times[i];m=motion[i]
        hc=np.array([float(m[f'h{a}{b}']) for a in range(3) for b in range(3)]).reshape(3,3);h=tr@hc@np.linalg.inv(tr)
        def mapped(p):
            q=np.c_[p,np.ones(len(p))]@h.T
            return q[:,:2]/q[:,2:]
        current=mapped(points+offset);boundary=mapped(outline)
        a=raw[i,132:477,68:308]
        # Warp the actual frame to reference geometry using the saved Native transform.
        scale=np.diag([240/96,345/96,1]);hp=scale@h@np.linalg.inv(scale)
        stable=cv2.warpPerspective(a,hp,(240,345),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
        im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
        text(d,(28,18),'跟着光屏采样，用相位变化计数',32)
        text(d,(28,65),'真实自录视频  ·  9.6–11.6秒标定  ·  暗纹到达中心为统一周期边界',20,GRAY)
        text(d,(1120,27),f'原片 {t:06.3f} 秒',26,GREEN)
        text(d,(28,105),'相机画面：采样点随光屏移动',20)
        text(d,(412,105),'光屏坐标：位置对齐，明暗保留',20)
        for x,image,p,b in [(28,a,current,boundary),(412,stable,points+offset,outline)]:
            im.paste(Image.fromarray(image).convert('RGB').resize((360,518),Image.Resampling.LANCZOS),(x,143))
            def xy(q):return (x+(float(q[0])+.5)*360/96,143+(float(q[1])+.5)*518/96)
            line=[xy(q) for q in b];d.line(line+[line[0]],fill=GREEN,width=2)
            for k,q in enumerate(p):
                u,v=xy(q);d.rectangle((u-5,v-7,u+5,v+7),outline=COLORS[k],width=2)
                text(d,(u+7,v-10),'C' if k==0 else str(k),16,COLORS[k])
        text(d,(804,108),'已确认净计数',22,GRAY)
        shown=int(online[i]['count_available_now']);text(d,(810,140),str(shown),76,GREEN)
        text(d,(1000,164),'圈',26,GRAY)
        text(d,(804,239),'正向 +1   /   反向 −1',22)
        confirmed=[v for v in events if int(v['confirmation_frame'])<=i]
        if confirmed:
            e=confirmed[-1];direction=int(e['event_direction'])
            text(d,(804,279),f'最近事件 {direction:+d}：发生 {float(e["timestamp"]):.3f}s',20)
            text(d,(804,310),f'确认 {float(e["confirmed_at"]):.3f}s  ·  延迟 {e["delay_frames"]} 帧',20,GRAY)
        # Four phase observations, all zeroed at the same calibration dark-center times.
        cx,cy,radius=940,485,102
        d.ellipse((cx-radius,cy-radius,cx+radius,cy+radius),outline='#354860',width=2)
        d.line((cx,cy,cx+radius,cy),fill=GRAY,width=1);text(d,(1050,470),'共同零点',18,GRAY)
        for k in range(4):
            angle=float(r[f'phase{k}']);u=cx+radius*np.cos(angle);v=cy-radius*np.sin(angle)
            d.line((cx,cy,u,v),fill=COLORS[k+1],width=2)
            d.ellipse((u-5,v-5,u+5,v+5),fill=COLORS[k+1])
        text(d,(804,610),'4路相位观测，多路一致后确认跨越',20,GRAY)
        if 824<=i<=843:
            msg='34秒小亮点存在歧义：可计短暂往返；区间结束净值一致';color=ORANGE
        elif 708<=i<=731:
            msg='29秒：原画面确认两次短暂回穿，真实的减一、加一保留';color=GREEN
        elif 1108<=i<=1140:
            msg='46–47秒：局部往返不等于整圈；以暗纹是否到达中心判断';color=FG
        elif stage=='intro':
            msg='先对齐外轮廓，再读取5个小块的亮度；无需逐帧找完整圆环';color=FG
        elif stage=='outro':
            msg='88个确定事件全部对应；另有2个可选往返事件；片尾净84圈';color=GREEN
        else:
            msg='采样点固定在光屏坐标；每路中心与辅助亮度共同形成相位观测';color=GRAY
        text(d,(28,685),msg,22,color)
        text(d,(28,727),'最近2秒的相位观测 / 周期（曲线跳边为角度回绕，并非丢圈）',18,GRAY)
        left,right,top,bottom=60,1380,766,879
        d.rectangle((left,top,right,bottom),outline='#354860')
        for value in [-.5,0,.5]:
            y=(top+bottom)/2-value*(bottom-top)
            d.line((left,y,right,y),fill='#26374b');text(d,(29,y-9),str(value),16,GRAY)
        indices=np.where((rowtimes>=t-2)&(rowtimes<=t))[0]
        for k in range(4):
            chain=[]
            for j in indices:
                q=(left+(rowtimes[j]-(t-2))/2*(right-left),(top+bottom)/2-phases[j,k]*(bottom-top))
                if chain and abs(phases[j,k]-phases[j-1,k])>.5:
                    if len(chain)>1:d.line(chain,fill=COLORS[k+1],width=2)
                    chain=[]
                chain.append(q)
            if len(chain)>1:d.line(chain,fill=COLORS[k+1],width=2)
        text(d,(28,903),'自录开发参考；34秒局部歧义由用户认可。核心P95 0.608 ms；不代表手机或其他场景验收。',18,GRAY)
        return im
    dest=OUT/'physical-count-explained.mp4'
    proc=subprocess.Popen(['/opt/homebrew/bin/ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],stdin=subprocess.PIPE)
    first,last=min(byframe),max(byframe)
    intro=render(first,'intro');intro.save(OUT/'video-preview.jpg')
    for _ in range(4*FPS):proc.stdin.write(intro.tobytes())
    for t in np.arange(times[first],times[last]+.5/FPS,1/FPS):
        i=min(last,max(first,int(np.searchsorted(times,t,side='right')-1)))
        proc.stdin.write(render(i,'run').tobytes())
    outro=render(last,'outro')
    for _ in range(4*FPS):proc.stdin.write(outro.tobytes())
    proc.stdin.close();assert proc.wait()==0
    render(830,'run').save(OUT/'video-ambiguity-preview.jpg')
    print(dest)
if __name__=='__main__':main()
