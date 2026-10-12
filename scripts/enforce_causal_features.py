"""Recompute capture features from saved packet events without rerunning labels."""
import csv,hashlib,json
from collections import defaultdict
from pathlib import Path
from grace_rtp.network import capture_stats

proof=[]
for folder in ['pilot_q1','pilot_q2']:
 root=Path('results')/folder;groups=defaultdict(list)
 for line in (root/'packet_trace.jsonl').open():
  r=json.loads(line);groups[(str(r['video_id']),r['profile'],str(r['seed']),str(r['frame_id']))].append(r)
 rows=list(csv.DictReader((root/'frames.csv').open(newline='')))
 signature=lambda:hashlib.sha256(json.dumps([(r['video_id'],r['profile'],r['seed'],r['frame_id'],r['psnr'],r['mask']) for r in rows]).encode()).hexdigest()
 before=signature()
 for row in rows:
  trace=groups[(row['video_id'],row['profile'],row['seed'],row['frame_id'])]
  generation=int(row['frame_id'])*1000/30
  row.update(capture_stats(trace,generation+float(row['deadline_ms'])))
  row['final_capture_loss']=capture_stats(trace)['capture_loss']
 assert signature()==before
 with (root/'frames.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 p=json.loads((root/'provenance.json').read_text());p['feature_observation_boundary']='Only arrivals no later than frame generation + deadline; final_capture_loss excluded from prediction'
 (root/'provenance.json').write_text(json.dumps(p,indent=2))
 proof.append(dict(dataset=folder,rows=len(rows),label_mask_sha256=before,labels_unchanged=True,observation='frame deadline',fps=30))
Path('results/causal_feature_audit.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof,indent=2))
