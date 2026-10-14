"""Create reproducible timing figures and evidence-bounded quality comparisons."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
from profile_native_stages import OUT,OLD_NAMES,NEW_NAMES,ROOT
FONT='/System/Library/Fonts/STHeiti Medium.ttc';font_manager.fontManager.addfont(FONT)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=FONT).get_name(),'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
OLD_LABELS=['红通道整图复制／提取','两次直方图与阈值','光屏区域分割','区域视图裁剪','二值化与形态学','边缘与轮廓发现','圆环筛选与内外分类','圆心与明暗判断','面积投票与加减计数','原代码内置绘图／边框','其余状态与临时对象']
NEW_LABELS=['小图光屏分割','距离场构建','相机位移拟合','跟踪质量检查','45次亮度采样','4路角度计算','多路确认与计数','其余']
def read(p):return list(csv.DictReader(p.open()))
def stats(a):
    a=np.asarray(a,dtype=float);return dict(mean_ms=float(a.mean()),median_ms=float(np.median(a)),p95_ms=float(np.percentile(a,95)),p99_ms=float(np.percentile(a,99)),max_ms=float(a.max()),frames=len(a))
def match(expected,observed):
    pairs=[];missing=[];extra=[];wrong=[];i=j=0
    while i<len(expected) and j<len(observed):
        e=expected[i];r=observed[j];t=r['timestamp'];a=e['timestamp_estimate']
        lo=max(a-.2,(expected[i-1]['timestamp_estimate']+a)/2 if i else -1)
        hi=min(a+.2,(a+expected[i+1]['timestamp_estimate'])/2 if i+1<len(expected) else 100)
        if t<lo:extra.append(r);j+=1
        elif t>hi:missing.append(e);i+=1
        elif r['direction']!=e['direction']:
            wrong.append(dict(reference_id=e['id'],frame=r['source_frame']));extra.append(r);missing.append(e);i+=1;j+=1
        else:pairs.append(dict(reference_id=e['id'],reference_time=a,event_time=t,direction=r['direction']));i+=1;j+=1
    missing.extend(expected[i:]);extra.extend(observed[j:]);return dict(matched=len(pairs),missing=missing,extra=extra,wrong_direction=wrong,pairs=pairs)
def old_events(rows,start):
    return [dict(source_frame=int(r['source_frame']),timestamp=float(r['timestamp']),direction=int(r['count'])-int(rows[i-1]['count'])) for i,r in enumerate(rows) if i>=start and int(r['count'])!=int(rows[i-1]['count'])]
def main():
    old_all=read(OUT/'legacy-cpp.csv');old=old_all[279:];native=read(OUT/'native-profile.csv.stages.csv');control=read(OUT/'native-control.csv.core.csv')
    sampled=[r for r in old if float(r['profile_total_us'])>0];ids={int(r['source_frame']) for r in sampled};new_sample=[r for r in native if int(r['source_frame']) in ids];assert len(sampled)==len(new_sample)==128
    for r in sampled:assert abs(sum(float(r[k]) for k in OLD_NAMES)-float(r['profile_total_us']))<1e-6
    for r in native:assert abs(sum(float(r[k]) for k in NEW_NAMES)-float(r['total_us']))<1e-6
    omean=np.array([np.mean([float(r[k])/1000 for r in sampled]) for k in OLD_NAMES]);nmean=np.array([np.mean([float(r[k])/1000 for r in new_sample]) for k in NEW_NAMES])
    result=dict(old_cpp=stats([float(r['baseline_total_us'])/1000 for r in old]),new_cpp=stats([float(r['core_us'])/1000 for r in control]),old_stage_sample=stats([float(r['profile_total_us'])/1000 for r in sampled]),new_stage_sample=stats([float(r['total_us'])/1000 for r in new_sample]),decode_full_bgr=stats([float(r['decode_us'])/1000 for r in old]),preview_to_96=stats([float(r['preview_to_96_us'])/1000 for r in old]),initialization=json.loads((OUT/'native-profile.csv.init.json').read_text()),old_stages=[dict(key=k,label=l,mean_ms=float(v),share=float(v/omean.sum())) for k,l,v in zip(OLD_NAMES,OLD_LABELS,omean)],new_stages=[dict(key=k,label=l,mean_ms=float(v),share=float(v/nmean.sum())) for k,l,v in zip(NEW_NAMES,NEW_LABELS,nmean)],context=json.loads((OUT/'context.json').read_text()))
    result['mean_core_ratio_current_routes']=result['old_cpp']['mean_ms']/result['new_cpp']['mean_ms'];result['input_pixel_ratio']=4320*7680/(96*96)
    result['old_profile_vs_same_frames_control_mean_ratio']=omean.sum()/np.mean([float(r['baseline_total_us'])/1000 for r in sampled])
    result['new_tracking_share']=float(nmean[:4].sum()/nmean.sum());result['stage_scope']='Exclusive stage times; same fixed128 source frames for both stage charts. Full1158-frame distributions shown separately. Old includes embedded drawing; no video encoding in benchmarks.'
    # Independent frozen physical reference; optional episode is handled explicitly.
    reference=json.loads((ROOT/'experiments/results/physical-count/reference-v4.json').read_text());expected=reference['events'];observed=old_events(old_all,279)
    optional=[]
    for ep in reference['ambiguous_episodes']:
        lo,hi=ep['frame_interval'];inside=[r for r in observed if lo<=r['source_frame']<=hi]
        dirs=[r['direction'] for r in inside];valid=len(dirs)%2==0 and len(dirs)//2 in ep['allowed_pairs'] and dirs==[-1,1]*(len(dirs)//2)
        if valid:optional+=inside
    old_match=match(expected,[r for r in observed if r not in optional]);origin=int(old_all[278]['count'])
    new_rows=read(ROOT/'experiments/results/bilibili-adapt/original/frames.csv');new_native=[]
    for r in new_rows:
        if int(r['event_direction']) and not any(a['frame_interval'][0]<=int(r['source_frame'])<=a['frame_interval'][1] for a in reference['ambiguous_episodes']):new_native.append(dict(source_frame=int(r['source_frame']),timestamp=float(r['timestamp']),direction=int(r['event_direction'])))
    new_match=match(expected,new_native);assert new_match['matched']==88 and not new_match['missing'] and not new_match['extra']
    stable=[]
    for i,e in enumerate(expected[:-1]):
        nxt=expected[i+1];t=(e['timestamp_estimate']+nxt['timestamp_estimate'])/2
        if nxt['timestamp_estimate']-e['timestamp_estimate']<.8 or any(ep['timestamp_interval'][0]<=t<=ep['timestamp_interval'][1] for ep in reference['ambiguous_episodes']):continue
        wanted=sum(v['direction'] for v in expected[:i+1]);o=next(r for r in reversed(old_all) if float(r['timestamp'])<=t);n=next(r for r in reversed(new_rows) if float(r['timestamp'])<=t)
        stable.append(dict(timestamp=t,expected=wanted,old=int(o['count'])-origin,new=int(n['count'])))
    stable.append(dict(timestamp=float(old[-1]['timestamp']),expected=84,old=int(old[-1]['count'])-origin,new=int(new_rows[-1]['count'])))
    bili=read(OUT/'legacy-cpp-bilibili.csv');bref=json.loads((ROOT/'data/annotations/bilibili-provisional.json').read_text());be=[e for e in bref['events'] if e['timestamp_estimate']>8.5];bmatch=match(be,old_events(bili,256))
    quality=dict(reference_sha256=hashlib.sha256((ROOT/'experiments/results/physical-count/reference-v4.json').read_bytes()).hexdigest(),reference_human_verified=False,old_origin=origin,old_final_relative=int(old[-1]['count'])-origin,new_final=84,old_event_comparison=old_match,new_event_comparison=new_match,stable_checkpoints=stable,stable_selection='All gaps >=0.8s in fixed reference, midpoints, outside accepted ambiguity episode; plus EOF. Same rule for both algorithms, no per-event time shifts.',old_optional_ambiguous_events=optional,bilibili=dict(old_event_comparison=bmatch,old_center_unknown_frames=sum(r['center_class']=='-1' for r in bili),old_frames=len(bili),old_measurement_delta=int(bili[-1]['count'])-int(bili[255]['count']),new_provisional_matched=6,new_final=6,reference_status=bref['status']),matching_note='Same direction, ordered, one-to-one; <=0.2s clipped by adjacent reference midpoints. Missing/extra include wrong-direction mismatches; categories overlap. Old threshold boundary differs from the physical dark-center boundary, so event-window mismatch is not automatically a whole-cycle error; stable checkpoint errors provide separate evidence.')
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');(OUT/'quality.json').write_text(json.dumps(quality,ensure_ascii=False,indent=2)+'\n')
    # Timing: common-scale total, then separately expanded exclusive stage distributions.
    fig=plt.figure(figsize=(16,11),facecolor='white');gs=fig.add_gridspec(2,2,height_ratios=[1,2.8],hspace=.52,wspace=.52)
    ax=fig.add_subplot(gs[0,:]);values=[result['old_cpp']['mean_ms'],result['new_cpp']['mean_ms']]
    ax.barh([1,0],values,color=['#c36d34','#167c68'],height=.43);ax.set_yticks([1,0],['旧 C++：原尺寸轮廓路线','新 C++：96像素跟踪＋相位路线']);ax.set_xlim(0,max(values)*1.24);ax.set_xlabel('每帧处理均值 / ms（1158帧；不含解码与新法预处理）')
    for y,v,s in zip([1,0],values,[result['old_cpp'],result['new_cpp']]):ax.text(v+.5,y,f'{v:.3f} ms  |  P95 {s["p95_ms"]:.3f} ms',va='center')
    ax.set_title(f'当前两条 C++ 核心路线：均值相差约 {result["mean_core_ratio_current_routes"]:.0f} 倍；输入像素数相差3600倍',loc='left',fontsize=15,pad=16)
    for col,values,labels,color,title in [(0,omean,OLD_LABELS,'#c36d34','旧 C++：时间主要消耗在大图直方图、复制和分割'),(1,nmean,NEW_LABELS,'#167c68','新 C++：时间主要消耗在小图跟踪，而非算相位')]:
        ax=fig.add_subplot(gs[1,col]);order=np.argsort(values)[::-1];ys=np.arange(len(values));ax.barh(ys,values[order],color=color,height=.63);ax.set_yticks(ys,[labels[i] for i in order]);ax.invert_yaxis();ax.set_xlim(0,max(values)*1.25);ax.set_xlabel('平均步骤耗时 / ms（固定同一批128帧；两图横轴尺度不同）');ax.set_title(title,fontsize=12,pad=16,loc='left');ax.xaxis.set_major_locator(MaxNLocator(5));ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
        for y,i in enumerate(order):
            v=values[i];label=f'{v:.3f} ms' if v>=.001 else '<0.001 ms'
            ax.text(v+max(values)*.025,y,label,va='center',fontsize=10)
    fig.suptitle('同一台 Apple M4 · C++17 Release · 图像处理单线程 · 真实自录视频',fontsize=16,y=.995)
    fig.text(.02,.052,f'新法一次性标定计算 {result["initialization"]["total_ms"]:.1f} ms，另需收集2秒标定画面；新法光屏跟踪占逐帧核心约 {result["new_tracking_share"]*100:.2f}%。',fontsize=11)
    fig.text(.02,.025,'这是不同输入规模与处理路线的实测差，不是“同分辨率换算法”的纯因果倍率；旧法内置绘图单列，已计入旧法总时间。',fontsize=10,color='#555555')
    fig.subplots_adjust(left=.19,right=.97,bottom=.12,top=.91);fig.savefig(OUT/'timing-breakdown.png',dpi=170);fig.savefig(OUT/'timing-breakdown.svg');plt.close(fig)
    # Quality: reference-defined stable checkpoints avoid most threshold/confirmation timing ambiguity.
    fig,axes=plt.subplots(2,1,figsize=(14,9),gridspec_kw={'height_ratios':[1.5,1]},facecolor='white');ax=axes[0]
    ts=[x['timestamp'] for x in stable];ax.plot(ts,[x['old']-x['expected'] for x in stable],'o-',color='#c36d34',label='本科旧 C++');ax.plot(ts,[x['new']-x['expected'] for x in stable],'s-',color='#167c68',label='当前新 C++');ax.axhline(0,color='#888888',lw=.8);ax.set_xlabel('原片时间 / s');ax.set_ylabel('相对固定参考的净计数误差 / 圈');ax.yaxis.set_major_locator(MaxNLocator(integer=True));ax.set_title('参考预先定义的稳定检查点：旧 C++ 片尾82圈，新 C++ 84圈',loc='left',fontsize=14);ax.legend();ax.grid(alpha=.17)
    ax=axes[1];ax.axis('off');rows_table=[['自录测量段','旧 C++','新 C++'],['确定参考事件匹配',str(old_match['matched'])+' / 88','88 / 88'],['片尾净计数',str(quality['old_final_relative']),'84'],['B站测量段事件输出',str(len(old_events(bili,256))),'6（匹配临时视觉参考）'],['B站状态与覆盖',f'中心未知{quality["bilibili"]["old_center_unknown_frames"]}/637帧','标定后381帧完整计数']]
    table=ax.table(cellText=rows_table[1:],colLabels=rows_table[0],cellLoc='center',loc='center',colWidths=[.44,.23,.33]);table.auto_set_font_size(False);table.set_fontsize(12);table.scale(1,1.8)
    for (row,col),cell in table.get_celld().items():cell.set_edgecolor('#dddddd');cell.set_facecolor('#f1f3f2' if row==0 else 'white')
    fig.suptitle('“更好”的证据来自事件与稳定计数，不来自速度倍率',fontsize=16)
    fig.text(.025,.06,'旧 C++与上一段视频的旧Python版不同，不能混用其结果。事件匹配采用同向、顺序、一对一及±0.2秒窗口。',fontsize=10,color='#555555')
    fig.text(.025,.025,'自录参考为AI视觉复核；34秒可选往返单列。B站参考尚未经独立人工确认，两段均为开发数据，陌生场景未验证。',fontsize=10,color='#555555')
    fig.tight_layout(rect=[.03,.10,.98,.94]);fig.savefig(OUT/'quality-checkpoints.png',dpi=170);fig.savefig(OUT/'quality-checkpoints.svg');plt.close(fig)
    for path in OUT.glob('*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    print(json.dumps(dict(ratio=result['mean_core_ratio_current_routes'],old_match=old_match['matched'],old_missing=len(old_match['missing']),old_extra=len(old_match['extra']),old_wrong_direction=len(old_match['wrong_direction']),stable=stable),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
