"""Decode untouched sources, retain presentation timestamps and inspection panels."""
import argparse
import json
import subprocess
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def prepare(name, width, height):
    source = ROOT / 'data/raw' / f'michelson-{name}.mp4'
    directory = ROOT / 'data/derived'
    directory.mkdir(exist_ok=True)
    output = directory / f'{name}.npz'
    if output.exists():
        print('Already prepared:', name, flush=True)
        return
    packets = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_packets',
        '-show_entries', 'packet=pts_time', '-of', 'json', str(source)]))['packets']
    timestamps = np.sort([float(p['pts_time']) for p in packets])
    timestamps -= timestamps[0]
    if np.any(np.diff(timestamps) <= 0):
        raise RuntimeError('Nonunique presentation timestamps')
    frames = np.empty((len(timestamps), height, width), dtype=np.uint8)
    cmd = ['ffmpeg', '-v', 'error', '-i', str(source), '-an', '-vf',
           f'scale={width}:{height}:flags=area,format=gbrp,extractplanes=r',
           '-fps_mode', 'passthrough', '-pix_fmt', 'gray', '-f', 'rawvideo', '-']
    begin = time.perf_counter()
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    for i in range(len(frames)):
        buf = process.stdout.read(width * height)
        if len(buf) != width*height:
            raise RuntimeError(f'Truncated decoded frame {i}')
        frames[i] = np.frombuffer(buf, np.uint8).reshape(height,width)
        if i % 200 == 0:
            print(name, i, '/', len(frames), flush=True)
    extra = process.stdout.read()
    if process.wait() or extra:
        raise RuntimeError('Decoder/frame timestamp mismatch')
    elapsed = time.perf_counter()-begin
    np.savez_compressed(output, frames=frames, timestamps=timestamps)
    (directory/f'{name}-metadata.json').write_text(json.dumps({
        'source': str(source.relative_to(ROOT)), 'frames': len(frames),
        'shape': [height,width], 'duration_video_seconds':float(timestamps[-1]),
        'decode_and_ffmpeg_resize_seconds':elapsed,
        'median_frame_interval':float(np.median(np.diff(timestamps))),
        'min_frame_interval':float(np.min(np.diff(timestamps))),
        'max_frame_interval':float(np.max(np.diff(timestamps))),
        'timestamp_method':'sorted H264 packet presentation timestamps, count checked against decoded frames',
        'rotation':'FFmpeg automatic metadata rotation before scaling',
        'channel':'red, 8-bit camera encoded intensity; not assumed radiometrically linear',
    },indent=2)+'\n')
    # Uniform-time contact sheet for visual ROI and quality assessment, not labels.
    indices = [int(np.argmin(abs(timestamps-t))) for t in np.linspace(0,timestamps[-1],12)]
    tw,th = (144,256) if height>width else (256,144)
    sheet = Image.new('RGB',(tw*4,(th+24)*3),(240,240,240))
    draw = ImageDraw.Draw(sheet)
    for j,i in enumerate(indices):
        im = Image.fromarray(frames[i]).convert('RGB').resize((tw,th))
        x,y=(j%4)*tw,(j//4)*(th+24)
        sheet.paste(im,(x,y));draw.text((x+4,y+th+4),f'{i}: {timestamps[i]:.2f}s',fill=(0,0,0))
    sheet.save(directory/f'{name}-contact.jpg')
    print('Prepared',name,'seconds',round(elapsed,2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('name',choices=['original','bilibili','all'])
    args=parser.parse_args()
    if args.name in ('original','all'):prepare('original',432,768)
    if args.name in ('bilibili','all'):prepare('bilibili',640,360)
