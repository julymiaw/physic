"""Prepare real cached red planes and fixed manually selected ROIs for RGBA adapter replay."""
from pathlib import Path
import json,numpy as np
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'experiments/runs/wechat-core'
RUN.mkdir(parents=True,exist_ok=True)
for name,roi,start,cal,end in [('original',[108,190,160,230],9.6,11.6,60),('bilibili',[284,0,164,180],0,8.5,22)]:
 data=np.load(ROOT/f'data/derived/{name}.npz');frames=data['frames'];frames.tofile(RUN/f'{name}-red.raw')
 (RUN/f'{name}-camera.json').write_text(json.dumps(dict(width=frames.shape[2],height=frames.shape[1],timestamps=data['timestamps'].tolist(),roi=roi,start=start,cal=cal,end=end)))
