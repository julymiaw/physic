"""Reproduce both real-video Native/WASM parity checks and online failure tests."""
from pathlib import Path
import csv,hashlib,json,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'experiments/runs/wechat-core';OUT=ROOT/'experiments/results/wechat-core'
sys.path.insert(0,str(ROOT/'research'))
from compare_scenes import bili_fixture

def main():
 RUN.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
 subprocess.run(['cmake','-S',str(ROOT),'-B',str(ROOT/'build-native'),'-DCMAKE_BUILD_TYPE=Release'],check=True)
 subprocess.run(['cmake','--build',str(ROOT/'build-native'),'--target','fringe_count','fringe_live_replay','-j4'],check=True)
 subprocess.run(['bash',str(ROOT/'tools/build_live_wasm.sh')],check=True,cwd=ROOT)
 bili=RUN/'bilibili.frc';bili_fixture(bili)
 results=[]
 for name,fixture,start,cal,end,x,y in [('original',ROOT/'experiments/results/real-track/padded-screen.frc',9.6,11.6,60,16,58*96/345),('bilibili',bili,0,8.5,22,16,16)]:
  native=RUN/f'{name}-native.csv';wasm=RUN/f'{name}-wasm.csv';args=list(map(str,[fixture,start,cal,end,x,y]))
  subprocess.run([str(ROOT/'build-native/fringe_live_replay'),*args,str(native)],check=True)
  subprocess.run(['node',str(ROOT/'wasm/replay_live.cjs'),str(ROOT/'miniprogram/wasm/fringe.wasm'),*args,str(wasm)],check=True,cwd=ROOT,stdout=subprocess.DEVNULL)
  def read(p):return list(csv.DictReader(p.open()))
  a,b=read(native),read(wasm);frozen=read(ROOT/f'experiments/results/bilibili-adapt/{name}/frames.csv');assert len(a)==len(b)==len(frozen)
  differences={k:max(abs(float(u[k])-float(v[k])) for u,v in zip(a,b)) for k in a[0]}
  ref_differences={k:max(abs(float(u[k])-float(v[k])) for u,v in zip(a,frozen)) for k in a[0]}
  exact=['source_frame','count','event_direction','confirmation_frame','confirmed_at','complete','tail_pending','valid_pairs']
  assert all(differences[k]==ref_differences[k]==0 for k in exact)
  assert max(differences.values())<1e-6 and max(ref_differences.values())==0
  ns=np.genfromtxt(str(native)+'.state.csv',delimiter=',',skip_header=1);ws=np.genfromtxt(str(wasm)+'.state.csv',delimiter=',',skip_header=1)
  state_difference=float(np.max(abs(ns-ws)));assert ns.shape==ws.shape and state_difference<1e-6
  results.append(dict(name=name,frames=len(a),final=int(a[-1]['count']),events=sum(int(r['event_direction'])!=0 for r in a),all_reference_fields_exact=True,integer_and_state_fields_exact=True,max_wasm_row_difference=max(differences.values()),max_wasm_state_difference=state_difference,continuous_cycle_difference=float(np.max(abs(ns[:,3]-ws[:,3]))),confirmation_delay_max=max(int(r['confirmation_frame'])-int(r['source_frame']) for r in a)))
  (OUT/f'{name}-events.csv').write_text('event_frame,event_time,direction,confirmed_frame,confirmed_at\n'+''.join(f'{r["source_frame"]},{r["timestamp"]},{r["event_direction"]},{r["confirmation_frame"]},{r["confirmed_at"]}\n' for r in a if int(r['event_direction'])))
 tests=subprocess.check_output(['node',str(ROOT/'tests/live_pipeline.cjs')],cwd=ROOT,text=True);test_result=json.loads(tests)
 report=dict(passed=True,emscripten='6.0.12',wasm_bytes=(ROOT/'miniprogram/wasm/fringe.wasm').stat().st_size,wasm_sha256=hashlib.sha256((ROOT/'miniprogram/wasm/fringe.wasm').read_bytes()).hexdigest(),core_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['research/phase_core.h','research/screen_tracker.h','core/live_counter.h','core/live_api.cpp']},clips=results,failure_and_adapter_tests=test_result,scope='Fixed-input development replays, not new scene accuracy or phone real-time acceptance. Four processed observation lookahead; gaps are checked using received timestamps. Changes in camera geometry interrupt the segment rather than silently remapping it.')
 (OUT/'parity.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
