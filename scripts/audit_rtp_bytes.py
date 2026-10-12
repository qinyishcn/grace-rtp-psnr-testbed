"""Hash the same cached P-frame RTP serialization under different aiortc runtimes."""
import argparse,hashlib,json,struct
from pathlib import Path
import aiortc
import numpy as np
from grace_rtp.blocks import compress_features,packetize

p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
digest=hashlib.sha256();frames=packets=0
for cache in [Path('cache/q1'),Path('cache/q2')]:
 manifest=json.loads((cache/'manifest.json').read_text());seq=0
 for r in manifest['records']:
  if r['frame_id']==0:seq=0
  if r['frame_pos']==0:continue
  with np.load(cache/f"v{r['video_id']}_f{r['frame_id']}.npz") as data:
   encoded=packetize(compress_features(data['code']),r['frame_id'],seq,fps=manifest['fps'])
  for packet in encoded:digest.update(struct.pack('!I',len(packet)));digest.update(packet)
  seq=(seq+len(encoded))%65536;frames+=1;packets+=len(encoded)
result=dict(aiortc_version=aiortc.__version__,frames=frames,packets=packets,rtp_byte_stream_sha256=digest.hexdigest(),
            scope='All 336 cached P frames; length-delimited serialized RTP bytes before virtual network impairment')
Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
