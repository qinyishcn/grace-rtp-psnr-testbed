import csv,gzip,hashlib,io,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('results');summary=[];allrows=[]
for folder in ['pilot_q1','pilot_q2','controlled_0.2','controlled_0.5']:
 with (root/folder/'frames.csv').open(newline='') as f:rows=list(csv.DictReader(f))
 for r in rows:r.setdefault('final_capture_loss',r['capture_loss'])
 allrows.extend(rows);groups=defaultdict(list)
 for r in rows:groups[(r['profile'],r['q'])].append(r)
 for (profile,q),rs in groups.items():
  avg=lambda k:float(np.mean([float(r[k]) for r in rs]))
  total=sum(int(r['packet_count']) for r in rs)
  summary.append(dict(dataset=folder,profile=profile,q=q,n=len(rs),psnr=avg('psnr'),clean_psnr=avg('clean_psnr'),
                      block_loss=avg('block_loss'),capture_loss=avg('capture_loss'),
                      final_capture_loss=avg('final_capture_loss'),
                      random_drop_fraction=sum(int(r['random_drops']) for r in rs)/total,
                      queue_drop_fraction=sum(int(r['queue_drops']) for r in rs)/total,
                      late_fraction=sum(int(r['late_packets']) for r in rs)/total,
                      mean_rtp_packets=avg('packet_count'),mean_wire_bytes=avg('wire_bytes')))
 raw=root/folder/'packet_trace.jsonl'
 if raw.stat().st_size:
  with raw.open('rb') as source,(root/folder/'packet_trace.jsonl.gz').open('wb') as target:
   with gzip.GzipFile(filename='',mode='wb',fileobj=target,mtime=0) as compressed:
    import shutil
    shutil.copyfileobj(source,compressed)
with (root/'condition_summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
with (root/'frame_dataset.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
rc=[]
for p in sorted(root.glob('*netem_*.json')):
 e=json.loads(p.read_text())
 if not isinstance(e,dict) or 'frames' not in e:continue
 frames=e['frames'];rc.append(dict(run=p.stem,profile=e['testbed_profile'],n=len(frames),
                                  psnr=float(np.mean([r['psnr'] for r in frames])),
                                  block_loss=float(np.mean([sum(map(int,r['mask']))/10 for r in frames])),
                                  sent=e['scheduled_packets'],received=e['rtp_received'],
                                  iframe_verified=e['iframe_verified'],
                                  full_feature_hash_matches=sum(r['original_sha256']==r['received_sha256'] for r in frames)))
with (root/'real_netem_summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rc[0]));w.writeheader();w.writerows(rc)
profiles=list(dict.fromkeys(r['profile'] for r in summary if r['dataset']=='pilot_q1'))
fig,axes=plt.subplots(1,2,figsize=(13,5))
for q in ['1.0','2.0']:
 values=[next(r for r in summary if float(r['q'])==float(q) and r['profile']==name) for name in profiles]
 axes[0].plot(profiles,[r['psnr'] for r in values],marker='o',label=f'q={q}')
 axes[1].plot(profiles,[r['block_loss']*100 for r in values],marker='o',label=f'q={q}')
for ax,title,ylabel in zip(axes,['GRACE reconstruction under packet impairment','Effective block erasure at deadline'],['Mean RGB PSNR (dB)','Missing feature blocks (%)']):
 ax.set_title(title);ax.set_ylabel(ylabel);ax.tick_params(axis='x',rotation=65);ax.grid(alpha=.3);ax.legend()
fig.tight_layout();fig.savefig(root/'network_sensitivity.png',dpi=180);plt.close(fig)
print(json.dumps(dict(frame_rows=len(allrows),real_frames=sum(r['n'] for r in rc),summary=summary),indent=2))
