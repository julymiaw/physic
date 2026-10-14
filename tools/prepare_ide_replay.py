"""Temporary JSON fixed inputs for IDE verification; excluded from Git and release."""
from pathlib import Path
import base64,csv,json,struct,sys
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'miniprogram/verification'
def main():
 OUT.mkdir(parents=True,exist_ok=True);meta={}
 for name,fixture,start,cal,end,x,y in [('original',ROOT/'experiments/results/real-track/padded-screen.frc',9.6,11.6,60,16,58*96/345),('bilibili',ROOT/'experiments/runs/wechat-core/bilibili.frc',0,8.5,22,16,16)]:
  b=fixture.read_bytes();records=[];stride=12+96*96*4
  for i in range(struct.unpack_from('<I',b,16)[0]):
   offset=28+i*stride;t=struct.unpack_from('<d',b,offset+4)[0]
   if start<=t<=end:records.append([i,t,base64.b64encode(b[offset+12:offset+stride]).decode()])
  files=[]
  for i in range(0,len(records),16):
   filename=f'{name}-{i//16}.json';(OUT/filename).write_text(json.dumps(records[i:i+16],separators=(',',':')));files.append(filename)
  rows=list(csv.DictReader((ROOT/f'experiments/runs/wechat-core/{name}-native.csv').open()));keys=list(rows[0]);expected=[[float(r[k]) for k in keys] for r in rows]
  (OUT/f'{name}-expected.json').write_text(json.dumps(expected,separators=(',',':')))
  meta[name]=dict(start=start,cal=cal,end=end,x=x,y=y,files=files,expected=f'{name}-expected.json')
 (OUT/'manifest.json').write_text(json.dumps(meta,separators=(',',':')))
 print('Prepared local JSON fixtures; remove verification/ before shipping.')
if __name__=='__main__':main()
