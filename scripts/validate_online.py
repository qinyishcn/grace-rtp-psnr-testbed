"""Check causal stateful inference against saved batch predictions on held-out videos."""
import csv,json
from pathlib import Path
import numpy as np
from grace_rtp.infer import Predictor

root=Path('results');predictor=Predictor(root/'prediction/model.joblib')
expected={tuple(r[k] for k in ['video_id','q','profile','seed','frame_id']):float(r['two_stage_paired_capture'])
          for r in csv.DictReader((root/'prediction/predictions.csv').open(newline=''))}
errors=[];mismatches=[]
for r in csv.DictReader((root/'frame_dataset.csv').open(newline='')):
 if r['mode']!='packet_network' or int(r['video_id']) not in [9,10,11,12]:continue
 r['stream_id']=':'.join(r[k] for k in ['video_id','q','profile','seed'])
 key=tuple(r[k] for k in ['video_id','q','profile','seed','frame_id'])
 errors.append(abs(predictor.predict(r)['psnr_estimate']-expected[key]))
 if errors[-1]>1e-9:mismatches.append(dict(key=key,difference=errors[-1]))
print(dict(frames=len(errors),maximum_difference=max(errors),first_mismatches=mismatches[:5]))
assert len(errors)==3696 and max(errors)<1e-9
result=dict(status='passed',held_out_frames=len(errors),maximum_absolute_difference_db=float(np.max(errors)),
            history_entries=len(predictor.history),note='One previous/current GOP state per stream; no future arrivals or labels used')
(root/'online_validation.json').write_text(json.dumps(result,indent=2));print(result)
