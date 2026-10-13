"""Paired real-video regression: preserve frozen evidence and use the same Native rules."""
import argparse,csv,hashlib,json,re,struct,subprocess
from pathlib import Path
import cv2,numpy as np
from check_real import evaluate,prefixes,read
ROOT=Path(__file__).resolve().parents[1]

def replay(binary,fixture,calibration,end,path,offset=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    cmd=[str(binary),str(fixture),str(calibration[0]),str(calibration[1]),str(end),str(path)]
    if offset:cmd+=['--count-offset-x',str(offset[0]),'--count-offset-y',str(offset[1])]
    result=subprocess.run(cmd,capture_output=True,text=True)
    path.with_suffix('.log').write_text(result.stdout+result.stderr)
    if result.returncode not in [0,3]:raise RuntimeError(result.stdout+result.stderr)

def bili_fixture(path):
    data=np.load(ROOT/'data/derived/bilibili.npz');frames,times=data['frames'],data['timestamps']
    protocol=json.loads((ROOT/'experiments/results/bilibili-transfer/protocol.json').read_text())
    assert hashlib.sha256((ROOT/'data/derived/bilibili.npz').read_bytes()).hexdigest()==protocol['hashes']['data/derived/bilibili.npz']
    with path.open('wb') as f:
        f.write(struct.pack('<4sIIIId',b'FRC1',96,1,1,len(frames),30.))
        for i,(image,time) in enumerate(zip(frames,times)):
            padded=np.zeros((270,246),np.uint8);padded[45:]=image[:225,243:489]
            small=cv2.resize(padded.astype(np.float32)/255,(96,96),interpolation=cv2.INTER_AREA)
            f.write(struct.pack('<Id',i,float(time)));f.write(small.astype('<f4').tobytes())
    return times

def bili_evaluate(path,times,output):
    reference_path=ROOT/'data/annotations/bilibili-provisional.json'
    reference=json.loads(reference_path.read_text());expected=[e for e in reference['events'] if e['timestamp_estimate']>8.5]
    rows=read(path);events=[r for r in rows if int(r['event_direction'])];pairs=[]
    passed=len(events)==len(expected)
    for i,(e,r) in enumerate(zip(expected,events)):
        t=e['timestamp_estimate'];lo=max(t-.2,(expected[i-1]['timestamp_estimate']+t)/2 if i else -1)
        hi=min(t+.2,(expected[i+1]['timestamp_estimate']+t)/2 if i+1<len(expected) else 100)
        matched=lo<=float(r['timestamp'])<=hi and int(r['event_direction'])==e['direction']
        pairs.append(dict(reference_id=e['id'],reference_time=t,event_time=float(r['timestamp']),direction=int(r['event_direction']),matched=matched,difference_seconds=float(r['timestamp'])-t));passed &= matched
    checkpoints=[]
    for i,e in enumerate(expected):
        t=(e['timestamp_estimate']+expected[i+1]['timestamp_estimate'])/2 if i+1<len(expected) else times[-1]
        r=next(r for r in reversed(rows) if float(r['timestamp'])<=t)
        wanted=sum(x['direction'] for x in expected[:i+1]);checkpoints.append(dict(timestamp=float(t),expected=wanted,actual=int(r['count'])));passed &= int(r['count'])==wanted
    covered=[int(r['source_frame']) for r in rows]==list(range(256,len(times)))
    complete=all(r['complete']=='1' and r['tracking_valid']=='1' and int(r['valid_pairs'])>=2 for r in rows)
    delay=all(0<=int(r['delay_frames'])<=4 for r in rows);tail=any(r['tail_pending']=='1' for r in rows)
    result=dict(passed=bool(passed and covered and complete and delay and not tail),reference_status=reference['status'],human_verified=False,
        accuracy_scope='Six post-calibration provisional visual reference events, reviewed in raw spacetime strips and local frame panels. No independent human ground truth; not unseen-scene validation.',
        reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),expected_events=len(expected),observed_events=len(events),pairs=pairs,checkpoints=checkpoints,
        final_expected=sum(e['direction'] for e in expected),final_actual=int(rows[-1]['count']),frames=len(rows),all_source_frames_covered=covered,all_frames_complete=complete,
        confirmation_delay_within_four_frames=delay,max_confirmation_delay_frames=max(int(r['delay_frames']) for r in rows),unconfirmed_tail=tail,max_reference_time_difference_seconds=max(abs(e['difference_seconds']) for e in pairs))
    (output/'accuracy.json').write_text(json.dumps(result,indent=2)+'\n');return result

def bili_prefixes(binary,fixture,path):
    full=read(path);checks=[]
    for end in [10.5,13.5,18.1,20.5]:
        short_path=path.parent/f'prefix-{end}.csv';replay(binary,fixture,(0,8.5),end,short_path,(16,16));short=read(short_path)
        lookup={int(r['source_frame']):r for r in short};last=int(short[-1]['source_frame']);compared=0
        for row in full:
            frame=int(row['source_frame'])
            if frame>last-4:break
            for key in ['count','event_direction','confirmed_at','confirmation_frame','complete','tail_pending']:assert row[key]==lookup[frame][key],(end,frame,key)
            compared+=1
        checks.append(dict(prefix_end=end,identical=True,compared_confirmed_frames=compared,pending_tail_frames=sum(int(r['tail_pending']) for r in short)))
    (path.parent/'causality.json').write_text(json.dumps(checks,indent=2)+'\n');return checks

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',default='current');a=p.parse_args();assert re.fullmatch('[a-zA-Z0-9_-]+',a.name)
    out=ROOT/'experiments/runs/paired'/a.name;out.mkdir(parents=True,exist_ok=True)
    build=ROOT/'build-native';subprocess.run(['cmake','-S',str(ROOT),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release'],check=True);subprocess.run(['cmake','--build',str(build),'--target','fringe_count'],check=True);binary=build/'fringe_count'
    original=out/'original/frames.csv';replay(binary,ROOT/'experiments/results/real-track/padded-screen.frc',(9.6,11.6),60,original)
    original_result=evaluate(original,original.parent);prefixes(binary,original)
    fixture=out/'bilibili.frc';times=bili_fixture(fixture);bili=out/'bilibili/frames.csv';replay(binary,fixture,(0,8.5),22,bili,(16,16))
    bili_result=bili_evaluate(bili,times,bili.parent);bili_prefixes(binary,fixture,bili)
    result=dict(passed=original_result['passed'] and bili_result['passed'],baseline='self-recorded-v1',
        implementation_sha256=hashlib.sha256((ROOT/'research/real_pair.cpp').read_bytes()).hexdigest(),
        tracker_sha256=hashlib.sha256((ROOT/'research/screen_tracker.h').read_bytes()).hexdigest(),
        original=dict(passed=original_result['passed'],certain_matched=original_result['matched_events'],optional_events=original_result['optional_ambiguous_events'],final=original_result['final_actual'],frames=original_result['source_frame_count']),
        bilibili=dict(passed=bili_result['passed'],provisional_matched=sum(e['matched'] for e in bili_result['pairs']),final=bili_result['final_actual'],frames=bili_result['frames']),
        warning='Both clips are development data. Passing is relative to the documented references, not proof of generalization or independent human accuracy.')
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)
if __name__=='__main__':main()
