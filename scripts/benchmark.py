import argparse,json,time
from pathlib import Path
import numpy as np
from grace_rtp.codec import GraceCodec
from grace_rtp.blocks import compress_features,restore_features
p=argparse.ArgumentParser();p.add_argument('--grace',required=True);p.add_argument('--checkpoint',required=True);a=p.parse_args()
c=GraceCodec(a.grace,a.checkpoint,1,1);data=np.load('cache/q1/v1_f1.npz');ref=np.load('cache/q1/v1_f0.npz')['clean'];torch=c.torch
result={k:[] for k in ['encode_ms','decode_ms','entropy_encode_ms','entropy_decode_ms']}
for i in range(25):
 torch.cuda.synchronize();t=time.perf_counter();code,shape=c.encode(data['source'],ref);torch.cuda.synchronize();enc=(time.perf_counter()-t)*1000
 t=time.perf_counter();bs=compress_features(code);ec=(time.perf_counter()-t)*1000
 t=time.perf_counter();restored=restore_features(bs,len(code));ed=(time.perf_counter()-t)*1000
 torch.cuda.synchronize();t=time.perf_counter();c.decode(restored,ref,shape);torch.cuda.synchronize();dec=(time.perf_counter()-t)*1000
 if i>=5:
  for key,value in zip(result,[enc,dec,ec,ed]):result[key].append(value)
out=dict(warmup=5,samples=20,resolution=[256,192],batch_size=1,cuda_synchronized=True,
         scope='Serial same-frame microbenchmark; CPU/GPU copies included; no network, capture, live input or end-to-end FPS claim',
         metrics={k:dict(mean=float(np.mean(v)),p95=float(np.percentile(v,95))) for k,v in result.items()})
Path('results/codec_benchmark.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
