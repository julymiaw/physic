"""Generate local-only IDE replay from tests/measurement_flow.cjs outputs."""
from pathlib import Path
import base64,json,struct
root=Path(__file__).resolve().parents[1];run=root/'experiments/runs/measurement-flow';out=root/'miniprogram/verification';out.mkdir(exist_ok=True)
name='stop-cutoff-with-four-confirmation-frames';fixture=(root/'experiments/results/real-track/padded-screen.frc').read_bytes();packet=[];files=[]
for line in (run/f'{name}.ops').read_text().splitlines():
 op,frame,t=line.split();frame=int(frame);pixels=None
 if frame>=0:
  offset=28+frame*(12+96*96*4)+12;pixels=base64.b64encode(fixture[offset:offset+96*96*4]).decode()
 packet.append([int(op),frame,float(t),pixels])
 if len(packet)==16:
  file=f'measurement-{len(files):03}.json';(out/file).write_text(json.dumps(packet));files.append(file);packet=[]
if packet:
 file=f'measurement-{len(files):03}.json';(out/file).write_text(json.dumps(packet));files.append(file)
expected=[[float(v) for v in line.split(',')] for line in (run/f'{name}.native').read_text().splitlines()]
(out/'measurement-expected.json').write_text(json.dumps(expected));(out/'measurement-manifest.json').write_text(json.dumps(dict(x=16,y=58*96/345,files=files)))
print(f'Generated {len(expected)} local verification steps; remove miniprogram/verification after IDE check.')
