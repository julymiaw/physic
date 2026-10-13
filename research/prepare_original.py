"""Decode real source and reproduce the frozen area-integrated Native input."""
import argparse,hashlib,json,struct,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'experiments/results/real-track/padded-screen.frc');args=parser.parse_args()
    video=ROOT/'data/raw/michelson-original.mp4'
    expected=json.loads((ROOT/'data/videos.json').read_text())['videos'][0]['sha256']
    if not video.exists():raise SystemExit(f'Place the unchanged self-recorded video at {video}; see data/raw/README.md')
    if hashlib.sha256(video.read_bytes()).hexdigest()!=expected:raise SystemExit('Raw-video SHA-256 mismatch; do not reuse frozen reference for another source.')
    subprocess.run([sys.executable,str(ROOT/'experiments/prepare_video.py'),'original'],check=True)
    build=ROOT/'build-native';subprocess.run(['cmake','-S',str(ROOT),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release'],check=True)
    subprocess.run(['cmake','--build',str(build),'--target','fringe_prepare_gray'],check=True)
    data=np.load(ROOT/'data/derived/original.npz');frames,times=data['frames'],data['timestamps'];args.output.parent.mkdir(parents=True,exist_ok=True)
    process=subprocess.Popen([str(build/'fringe_prepare_gray'),'432','768','68','132','240','345','96',str(len(frames)),'3',str(args.output)],stdin=subprocess.PIPE)
    for image,time in zip(frames,times):process.stdin.write(struct.pack('<d',float(time)));process.stdin.write(image.tobytes())
    process.stdin.close()
    if process.wait():raise SystemExit('Native area preprocessing failed')
    actual=hashlib.sha256(args.output.read_bytes()).hexdigest();frozen=json.loads((ROOT/'experiments/results/physical-count/timing.json').read_text())['hashes']['experiments/results/real-track/padded-screen.frc']
    if actual!=frozen:raise SystemExit(f'Derived input differs from frozen baseline: {actual}; investigate decoder/resize differences before claiming regression accuracy.')
    print('Frozen real-video fixture reproduced:',args.output)
if __name__=='__main__':main()
