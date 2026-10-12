import csv,hashlib,json
from pathlib import Path
import numpy as np

def stats(code,shape):
 n=int(np.prod(shape['mv']));mv,res=code[:n],code[n:]
 return dict(mv_abs=float(np.mean(abs(mv))),mv_rms=float(np.sqrt(np.mean(mv**2))),res_abs=float(np.mean(abs(res))),
             res_rms=float(np.sqrt(np.mean(res**2))),mv_nonzero=float(np.mean(mv!=0)),res_nonzero=float(np.mean(res!=0)))
telemetry={};iframe_rows=[];hashes=[]
for q in [1,2]:
 root=Path(f'cache/q{q}');m=json.loads((root/'manifest.json').read_text())
 for r in m['records']:
  d=np.load(root/f"v{r['video_id']}_f{r['frame_id']}.npz")
  if r['frame_pos']:
   r['latent']=stats(d['code'],r['shape']);telemetry[(q,str(r['video_id']),str(r['frame_id']))]=r['latent']
   hashes.append(dict(q=q,video_id=r['video_id'],frame_id=r['frame_id'],source_sha256=hashlib.sha256(d['source'].tobytes()).hexdigest(),
                      feature_sha256=hashlib.sha256(d['code'].tobytes()).hexdigest()))
  else:
   mse=float(np.mean((d['source'].astype(float)-d['clean'])**2));iframe_rows.append(dict(q=q,video_id=r['video_id'],frame_id=r['frame_id'],psnr=-10*np.log10(mse),jpeg_bytes=len(d['iframe']),network_loss=0))
 (root/'manifest.json').write_text(json.dumps(m,indent=2))
 Path(f'results/source_manifest_q{q}.json').write_text(json.dumps(m,indent=2))
for folder in ['pilot_q1','pilot_q2','controlled_0.2','controlled_0.5']:
 p=Path('results')/folder/'frames.csv';rows=list(csv.DictReader(p.open(newline='')))
 for r in rows:r.update(telemetry[(int(float(r['q'])),r['video_id'],r['frame_id'])])
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for name,rows in [('iframe_labels.csv',iframe_rows),('source_feature_hashes.csv',hashes)]:
 with (Path('results')/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
