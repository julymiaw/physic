"""Render current compact app layout using actual WASM observations, not another counter."""
from pathlib import Path
import json,hashlib,subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/results/wechat-core';RUN=ROOT/'experiments/runs/wechat-core'
W,H,FPS=540,1170,24
FONT='/System/Library/Fonts/STHeiti Medium.ttc';fonts={s:ImageFont.truetype(FONT,s) for s in [15,17,18,20,22,24,28,52]}
BG='#080e0a';COLORS=['#91a69b','#55aa92','#688bb8','#b5985e','#9982ab']
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 cache=np.load(ROOT/'data/derived/original.npz');images,times=cache['frames'],cache['timestamps']
 replay=json.loads((RUN/'original-camera.csv.state.json').read_text());rows=replay['measurement'];lookup={int(r['source_frame']):r for r in rows}
 assert len(rows)==1158 and replay['final']['count']==84 and replay['final']['complete']
 display={};last_display=None;last_update=-1
 for row in rows:
  if last_display is None or row['timestamp']-last_update>=.1:
   last_display=row;last_update=row['timestamp']
  display[int(row['source_frame'])]=last_display
 ids=np.where(times>=9.6)[0];first,last=int(ids[0]),len(times)-1;start,end=float(times[first]),float(times[last])
 rect=(12,142,528,828);width,height=rect[2]-rect[0],rect[3]-rect[1]
 scale=max(width/432,height/768);sw,sh=round(432*scale),round(768*scale);cropx=(sw-width)//2;cropy=(sh-height)//2
 def txt(d,xy,value,size=20,color='#91aa99'):d.text(xy,str(value),font=fonts[size],fill=color)
 def render(source,finished=False):
  t=float(times[source]);r=lookup.get(source) if finished else display.get(source);display_source=int(r['source_frame']) if r else source;im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
  txt(d,(22,28),'15:36',17);txt(d,(210,77),'干涉条纹计数',22);d.rounded_rectangle((407,64,521,104),radius=20,outline='#26392b',width=1);txt(d,(423,68),'•••   ○',22)
  picture=Image.fromarray(images[source]).convert('RGB').resize((sw,sh),Image.Resampling.LANCZOS).crop((cropx,cropy,cropx+width,cropy+height));im.paste(picture,rect[:2]);d=ImageDraw.Draw(im)
  d.rounded_rectangle((22,154,113,182),radius=6,fill=BG);txt(d,(28,158),'录像回放',15,'#6f917c')
  d.rounded_rectangle((399,154,514,182),radius=6,fill=BG);txt(d,(407,158),f'{t:06.3f}s',15,'#6f917c')
  def project(x,y):return (rect[0]+x*scale-cropx,rect[1]+y*scale-cropy)
  left,top=project(108,190);right,bottom=project(268,420)
  d.rectangle((left,top,right,bottom),outline='#61977f',width=1)
  if r:
   for k,p in enumerate(r['points']):
    x,y=project(68+(p['x']+.5)*240/96,132+(p['y']+.5)*345/96)
    if rect[0]<=x<=rect[2] and rect[1]<=y<=rect[3]:d.ellipse((x-4,y-4,x+4,y+4),fill=COLORS[k])
  # Same five normalized signals and shared-zero phases exposed by the app adapter.
  x0,x1,yt,yb=18,366,847,974;center=(yt+yb)/2;d.line((x0,center,x1,center),fill='#203129');d.line((382,yt,382,yb),fill='#203129')
  recent=[q for q in rows if int(q['source_frame'])<=display_source and q['timestamp']>=(r['timestamp'] if r else t)-2][-150:]
  if len(recent)>1:
   a,b=recent[0]['timestamp'],recent[-1]['timestamp']
   for k in range(5):
    path=[(x0+(x1-x0)*(q['timestamp']-a)/max(.001,b-a),center-np.clip(q['signals'][k],-2,2)*(yb-yt-16)/4) for q in recent]
    d.line(path,fill=COLORS[k],width=2)
  cx,cy,rad=451,center,51;d.ellipse((cx-rad,cy-rad,cx+rad,cy+rad),outline='#304337',width=1)
  if r:
   for k,phase in enumerate(r['phases'][:r['nodes']]):
    x,y=cx+rad*np.cos(phase),cy-rad*np.sin(phase);d.line((cx,cy,x,y),fill=COLORS[k+1],width=2);d.ellipse((x-3,y-3,x+3,y+3),fill=COLORS[k+1])
  count=replay['final']['count'] if finished else (r['count'] if r else '—')
  txt(d,(22,986),count,52,'#9bb5a5');txt(d,(115,1023),'圈',20,'#687e70')
  status='已停止' if finished else ('计数中' if r else f'标定中 {max(0,t-start):.1f}s')
  d.rounded_rectangle((357,1000,523,1042),radius=10,fill='#111c16');txt(d,(377,1010),status,18,'#779b83')
  buttons=[(12,90,'圈选',False),(102,345,'新测量' if finished else '停止',True),(357,445,'信号',False),(457,528,'?',False)]
  for left,right,label,main in buttons:
   d.rounded_rectangle((left,1066,right,1123),radius=12,fill='#254a35' if main else '#152219');box=d.textbbox((0,0),label,font=fonts[22]);txt(d,((left+right-(box[2]-box[0]))/2,1082),label,22,'#b2c8b9' if main else '#89a18e')
  d.rounded_rectangle((180,1150,360,1156),radius=3,fill='#3c5042')
  return im
 dest=OUT/'miniprogram-original-replay.mp4'
 cmd=['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)]
 proc=subprocess.Popen(cmd,stdin=subprocess.PIPE);frames=int(np.ceil((end-start)*FPS))+1
 for index in range(frames):
  t=min(end,start+index/FPS);source=min(last,int(np.searchsorted(times,t,side='right')-1));proc.stdin.write(render(source).tobytes())
 for _ in range(FPS*2):proc.stdin.write(render(last,True).tobytes())
 proc.stdin.close();assert proc.wait()==0
 for t,label in [(12.8,'preview'),(29.3,'reversal'),(59.831322,'end')]:
  source=min(last,int(np.searchsorted(times,t,side='right')-1));render(source,label=='end').save(OUT/f'miniprogram-{label}.jpg',quality=92)
 manifest=dict(scope='App-layout simulation, not a phone capture or IDE screen recording. Camera display uses original-video cached red-channel previews. All count, signals, phase and tracked points come from production RGBA camera adapter followed by actual streaming WASM. Its96px images match the frozen input bit-for-bit; calibration is9.6–11.6s. Confirmation uses at most4 processed observations; UI diagnostics/count are refreshed on a simulated0.1s schedule; no future values are displayed.',video_input_sha256=hashlib.sha256((ROOT/'data/raw/michelson-original.mp4').read_bytes()).hexdigest(),wasm_sha256=hashlib.sha256((ROOT/'miniprogram/wasm/fringe.wasm').read_bytes()).hexdigest(),rows=len(rows),final_count=replay['final']['count'],complete=replay['final']['complete'],resolution=[W,H],fps=FPS,output_frames=frames+FPS*2,duration_seconds=(frames+FPS*2)/FPS)
 (OUT/'video-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');print(json.dumps(manifest,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
