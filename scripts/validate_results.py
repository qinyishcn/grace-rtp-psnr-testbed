"""Independent published-result checks; requires only numpy, no GRACE assets."""
import csv,json
from pathlib import Path
import numpy as np
root=Path('results');rows=list(csv.DictReader((root/'frame_dataset.csv').open(newline='')))
assert len(rows)==12096
keys=[(r['mode'],r['q'],r['profile'],r['seed'],r['video_id'],r['frame_id'],r['block_loss'] if r['mode']=='controlled_mask' else '') for r in rows]
assert len(keys)==len(set(keys))
assert all(np.isfinite(float(r['psnr'])) for r in rows)
assert all(abs(sum(map(int,r['mask']))/10-float(r['block_loss']))<1e-9 for r in rows)
assert max(abs(float(r['psnr'])-float(r['clean_psnr'])) for r in rows if r['profile']=='ideal')<.002
assert all(float(r['block_loss'])==0 for r in rows if r['profile'] in ['ideal','delay_50','deadline_200'])
assert all(float(r['block_loss'])==1 and float(r['final_capture_loss'])==0 for r in rows if r['profile']=='delay_120')
preds=list(csv.DictReader((root/'prediction/predictions.csv').open(newline='')))
for metric in csv.DictReader((root/'prediction/prediction_metrics.csv').open(newline='')):
 subset=[r for r in preds if r['split']==metric['split']]
 error=np.array([float(r[metric['model']])-float(r['psnr']) for r in subset])
 assert abs(np.mean(abs(error))-float(metric['mae']))<1e-9
 assert abs(np.sqrt(np.mean(error**2))-float(metric['rmse']))<1e-9
p=json.loads((root/'prediction/prediction_provenance.json').read_text())
assert not set(p['train_videos'])&set(p['test_videos'])
assert not set(p['validation_videos'])&set(p['test_videos'])
for prefix,profiles in [('official_netem',['no_emulation','5g_urban','5qi_7','congested']),
                        ('final_netem',['no_emulation','lossy','5qi_7','congested'])]:
 for profile in profiles:
  e=json.loads((root/f'{prefix}_{profile}.json').read_text())
  assert e['iframe_verified'] and len(e['received_layout'])==7
  assert e['rtp_received']==len(e['events'])
  if profile=='no_emulation':assert all(r['original_sha256']==r['received_sha256'] for r in e['frames'])
assert json.loads((root/'rtp_bytes_legacy.json').read_text())['rtp_byte_stream_sha256']==json.loads((root/'rtp_bytes_official.json').read_text())['rtp_byte_stream_sha256']
assert all(r['capture_application_sequence_sets_equal'] for r in json.loads((root/'pcap_audit.json').read_text()))
print('PASS: 12096 labels, masks, causal profile behavior, source split, independently recomputed prediction metrics, real SRTP evidence')
