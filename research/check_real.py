"""Check only the real recording against the visually reviewed development reference."""
import csv
import json
import subprocess
import argparse
import struct
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/results/physical-count'
FIXTURE=ROOT/'experiments/results/real-track/padded-screen.frc'
CURRENT_REFERENCE=ROOT/'experiments/results/physical-count/reference-v4.json'


def expected_source_frames():
    # Check coverage against the input, rather than against the output's own ends.
    with FIXTURE.open('rb') as f:
        _,size,_,_,count,_=struct.unpack('<4sIIIId',f.read(28))
        source=[]
        for i in range(count):
            _,timestamp=struct.unpack('<Id',f.read(12))
            if timestamp>11.6:source.append(i)
            f.seek(4*size*size,1)
    return source


def read(path):
    return [r for r in csv.DictReader(path.open()) if r['calibrating']=='0']


def evaluate(path, output_dir=None, reference_path=None):
    reference_path=(CURRENT_REFERENCE if CURRENT_REFERENCE.exists() else OUT/'reference.json') if reference_path is None else Path(reference_path)
    output_dir=Path(path).parent/'evaluation' if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    rows=read(path);reference=json.loads(reference_path.read_text())
    digest=reference_path.with_suffix('.sha256')
    if digest.exists():assert hashlib.sha256(reference_path.read_bytes()).hexdigest()==digest.read_text().split()[0], 'Reference hash mismatch'
    expected=reference['events'];all_observed=[r for r in rows if int(r['event_direction'])]
    episodes=reference.get('ambiguous_episodes',[]);optional=[];episode_checks=[]
    for episode in episodes:
        lo,hi=episode['frame_interval']
        before=next(r for r in reversed(rows) if int(r['source_frame'])<lo)
        inside=[r for r in rows if lo<=int(r['source_frame'])<=hi]
        events=[r for r in inside if int(r['event_direction'])]
        optional.extend(events);base=int(before['count'])
        directions=[int(r['event_direction']) for r in events]
        relative=sorted(set(int(r['count'])-base for r in inside))
        valid=(len(events)//2 in episode['allowed_pairs'] and len(events)%2==0 and
               directions==[-1,1]*(len(events)//2) and
               set(relative)<=set(episode['allowed_relative_counts']) and
               int(inside[-1]['count'])-base==episode['expected_net_change'])
        episode_checks.append(dict(id=episode['id'],passed=valid,base_count=base,
            end_count=int(inside[-1]['count']),events=events,relative_count_values=relative,
            exact_physical_events_known=False,policy=episode['human_policy']))
    observed=[r for r in all_observed if r not in optional]
    pairs=[];missing=[];extra=[];wrong_direction=[];i=j=0
    # Ordered one-to-one matching, no per-event timing correction or count offsets.
    while i<len(expected) and j<len(observed):
        e=expected[i];r=observed[j];t=float(r['timestamp']);a=e['timestamp_estimate']
        lo=max(a-.2,(expected[i-1]['timestamp_estimate']+a)/2 if i else -1)
        hi=min(a+.2,(a+expected[i+1]['timestamp_estimate'])/2 if i+1<len(expected) else 100)
        if t<lo:extra.append(r);j+=1
        elif t>hi:missing.append(e);i+=1
        elif int(r['event_direction'])!=e['direction']:
            wrong_direction.append(dict(reference_id=e['id'],event_frame=int(r['source_frame'])))
            extra.append(r);missing.append(e);i+=1;j+=1
        else:
            pairs.append(dict(reference_id=e['id'],reference_time=a,event_time=t,
                              event_frame=int(r['source_frame']),confirmed_at=float(r['confirmed_at']),
                              confirmation_frame=int(r['confirmation_frame']),direction=e['direction'],
                              difference_seconds=t-a,delay_frames=int(r['delay_frames'])));i+=1;j+=1
    missing.extend(expected[i:]);extra.extend(observed[j:])
    checkpoints=[]
    for i,e in enumerate(expected):
        t=(e['timestamp_estimate']+expected[i+1]['timestamp_estimate'])/2 if i+1<len(expected) else float(rows[-1]['timestamp'])
        rr=next((r for r in reversed(rows) if float(r['timestamp'])<=t),rows[0])
        net=sum(v['direction'] for v in expected[:i+1])
        ambiguity=next((e for e in episodes if e['timestamp_interval'][0]<=t<=e['timestamp_interval'][1]),None)
        if ambiguity:
            # Keep the original checkpoint visible with an explicit uncertainty band;
            # also enforce its exact net value after the user-accepted episode.
            allowed=[net+v for v in ambiguity['allowed_relative_counts']]
            checkpoints.append(dict(timestamp=t,expected_range=allowed,actual=int(rr['count']),
                error=0 if int(rr['count']) in allowed else 1,ambiguity_id=ambiguity['id']))
            t=ambiguity['timestamp_interval'][1]
            rr=next(r for r in reversed(rows) if float(r['timestamp'])<=t)
        checkpoints.append(dict(timestamp=t,expected=net,actual=int(rr['count']),error=int(rr['count'])-net))
    source=[int(r['source_frame']) for r in rows]
    coverage=source==expected_source_frames()
    initial=int(rows[0]['count'])==0
    delays=all(0<=int(r['confirmation_frame'])-int(r['source_frame'])<=4 for r in rows)
    tail=any(int(r['tail_pending']) for r in rows)
    final_expected=sum(e['direction'] for e in expected)
    result=dict(expected_events=len(expected),observed_events=len(all_observed),certain_observed_events=len(observed),optional_ambiguous_events=len(optional),ambiguous_episodes=episode_checks,matched_events=len(pairs),
                missing=missing,extra=extra,wrong_direction=wrong_direction,checkpoints=checkpoints,final_expected=final_expected,
                final_actual=int(rows[-1]['count']),final_error=int(rows[-1]['count'])-final_expected,
                source_frame_count=len(rows),source_frame_begin=source[0],source_frame_end=source[-1],
                all_source_frames_covered=coverage,initial_zero=initial,
                confirmation_delay_within_four_frames=delays,unconfirmed_tail=tail,
                max_reference_time_difference_seconds=max(abs(p['difference_seconds']) for p in pairs),
                max_confirmation_delay_frames=max(p['delay_frames'] for p in pairs),
                zero_quorum_frames=sum(int(r['valid_pairs'])==0 for r in rows),
                below_two_valid_pairs_frames=sum(int(r['valid_pairs'])<2 for r in rows),
                matching_note='Ordered, same direction, <=0.2s, clipped at adjacent reference midpoints.',
                reference_type=reference['status'],human_verified=False)
    result['reference_file']=str(reference_path.relative_to(ROOT)) if reference_path.is_relative_to(ROOT) else str(reference_path)
    result['unresolved_reference_intervals']=reference['unresolved']
    result['all_frames_complete']=all(r.get('complete','1')=='1' and r.get('tracking_valid','1')=='1' for r in rows)
    result['reference_sha256']=hashlib.sha256(reference_path.read_bytes()).hexdigest()
    result['passed']=bool(result['all_frames_complete'] and all(e['passed'] for e in episode_checks) and not missing and not extra and all(c['error']==0 for c in checkpoints) and
                          result['final_error']==0 and coverage and initial and delays and not tail and not reference['unresolved'])
    (output_dir/'events.json').write_text(json.dumps(pairs,indent=2)+'\n')
    (output_dir/'accuracy.json').write_text(json.dumps(result,indent=2)+'\n')
    columns=['classification','reference_id','reference_time','event_time','event_frame',
             'confirmed_at','confirmation_frame','direction','difference_seconds','delay_frames']
    records=[dict(classification='certain_match',**e) for e in pairs]
    for episode in episode_checks:
        for e in episode['events']:
            records.append(dict(classification='user_accepted_optional',event_time=e['timestamp'],
                event_frame=e['source_frame'],confirmed_at=e['confirmed_at'],
                confirmation_frame=e['confirmation_frame'],direction=e['event_direction'],
                delay_frames=e['delay_frames']))
    with (output_dir/'event-comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,columns);writer.writeheader()
        writer.writerows(sorted(records,key=lambda e:int(e['event_frame'])))
    with (output_dir/'checkpoints.csv').open('w') as f:
        writer=csv.DictWriter(f,['timestamp','expected_min','expected_max','actual','error','ambiguity_id'])
        writer.writeheader()
        for c in checkpoints:
            band=c.get('expected_range',[c.get('expected')])
            writer.writerow(dict(timestamp=c['timestamp'],expected_min=min(band),expected_max=max(band),
                actual=c['actual'],error=c['error'],ambiguity_id=c.get('ambiguity_id','')))
    return result


def prefixes(binary,full_path):
    full=read(full_path);checks=[]
    for end in [17.2,29.2,47.2,59.7]:
        path=full_path.parent/f'prefix-{end}.csv'
        replay=subprocess.run([str(binary),str(FIXTURE),'9.6','11.6',str(end),str(path)],check=False,capture_output=True,text=True)
        assert replay.returncode in [0,3],replay.stderr
        short=read(path);lookup={int(r['source_frame']):r for r in short};last=int(short[-1]['source_frame'])
        compared=0
        for row in full:
            frame=int(row['source_frame'])
            if frame>last-4:break
            other=lookup[frame]
            for key in ['count','event_direction','confirmed_at','confirmation_frame','complete','tail_pending']:
                assert row[key]==other[key],(end,frame,key,row[key],other[key])
            compared+=1
        checks.append(dict(prefix_end=end,compared_confirmed_frames=compared,identical=True,pending_tail_frames=sum(int(r['tail_pending']) for r in short),tail_complete=short[-1]['complete']))
    # Stable checkpoints after the physical dark-center boundary, with postroll.
    # 17.173144s was tight to the changed physical boundary; the 31-event check is unchanged.
    assert sum(int(r['event_direction']) for r in full if float(r['timestamp'])<=17.4)==12
    assert sum(int(r['event_direction']) for r in full if float(r['timestamp'])<=29.170833)==31
    (full_path.parent/'causality.json').write_text(json.dumps(checks,indent=2)+'\n')
    return checks


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,default=Path('/tmp/physic-physical-count'))
    p.add_argument('--csv',type=Path,default=ROOT/'experiments/runs/self-recorded/frames.csv');p.add_argument('--prefixes',action='store_true')
    p.add_argument('--reference',type=Path);p.add_argument('--output-dir',type=Path);a=p.parse_args()
    a.csv.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run([str(a.binary),str(FIXTURE),'9.6','11.6','60',str(a.csv)],check=True)
    result=evaluate(a.csv,a.output_dir,a.reference)
    if a.prefixes:prefixes(a.binary,a.csv)
    print(json.dumps({k:result[k] for k in ['passed','expected_events','observed_events','matched_events','final_actual','final_error','source_frame_count','max_confirmation_delay_frames']},indent=2))
    if not result['passed']:raise SystemExit(1)
