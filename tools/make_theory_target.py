"""Controlled display target; not a real-scene accuracy benchmark."""
from pathlib import Path
import json,subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/results/theory-camera-target';OUT.mkdir(parents=True,exist_ok=True)
W,H,FPS=720,900,30;SIDE=600;LEFT,TOP=60,125
# Common-phase cosine and quadratic spatial phase, as in the original simulator.
y,x=np.mgrid[:SIDE,:SIDE];r=np.hypot((x+.5-SIDE/2)/(SIDE/2),(y+.5-SIDE/2)/(SIDE/2));psi=28*(r/.90)**2
mask=np.clip((.92-r)/.02,0,1)
font=ImageFont.truetype('/System/Library/Fonts/STHeiti Medium.ttc',25)
def phase_cycles(t):
 if t<6:return 0
 if t<12:return t-6
 if t<18:return 6
 return 6+min(10.25,t-18)
ff=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{W}x{H}','-framerate',str(FPS),'-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'theory-camera-test.mp4')],stdin=subprocess.PIPE)
for i in range(34*FPS):
 t=i/FPS;c=phase_cycles(t);values=np.rint(255*(.008+( .55+.33*np.cos(psi+2*np.pi*c)-.008)*mask)).clip(0,255).astype(np.uint8)
 frame=Image.new('RGB',(W,H),'#080b09');rgb=np.zeros((SIDE,SIDE,3),dtype=np.uint8);rgb[:,:,0]=values;frame.paste(Image.fromarray(rgb),(LEFT,TOP));d=ImageDraw.Draw(frame)
 stage='圈选光屏，等待标定' if t<6 else '点击标定 · 连续单向 1 圈/秒' if t<12 else '图样暂停 · 点击开始测量' if t<18 else '测量 · 推进 10.25 圈' if t<28.25 else '图样保持 · 预期完整净周期 10'
 d.text((LEFT,35),'理论干涉图样 · 相机管线排查',font=font,fill='#a6b7aa');d.text((LEFT,760),stage,font=font,fill='#a6b7aa');d.text((LEFT,807),f'{t:05.2f} s',font=font,fill='#6f8274');ff.stdin.write(np.asarray(frame).tobytes())
 if i==0:frame.save(OUT/'theory-preview.jpg')
ff.stdin.close();assert ff.wait()==0
spec=dict(color='red',redChannelUnchanged=True,formula='I=.55+.33*cos(28*(r/.9)^2+2*pi*C(t)) within fixed circular aperture; outside .008',fps=FPS,resolution=[W,H],roi=[LEFT,TOP,SIDE,SIDE],stages=[dict(start=0,end=6,action='select',cycles=0),dict(start=6,end=12,action='calibrate',speed_cycles_per_second=1),dict(start=12,end=18,action='pause and manually start measurement'),dict(start=18,end=28.25,action='advance',cycles=10.25),dict(start=28.25,end=34,action='hold and stop',expected_whole_cycles=10)],scope='Controlled camera-pipeline experiment; not evidence for real optical robustness. Quarter-cycle remainder intentionally avoids exact-integer hysteresis endpoint ambiguity.')
(OUT/'spec.json').write_text(json.dumps(spec,indent=2)+'\n');print(OUT/'theory-camera-test.mp4')
