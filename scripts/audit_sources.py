import argparse,hashlib,inspect,json,platform,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--testbed',required=True);a=p.parse_args()
sys.path.insert(0,str(Path(a.testbed).resolve()/'netemu/src'))
import torch,torchvision,aiortc,av,numpy,netemu
import aiortc.rtcrtpsender,aiortc.rtcrtpreceiver,aiortc.jitterbuffer,aiortc.rtcdtlstransport,aiortc.codecs
modules=[aiortc.rtcrtpsender,aiortc.rtcrtpreceiver,aiortc.jitterbuffer,aiortc.rtcdtlstransport,aiortc.codecs,netemu]
data=dict(python=platform.python_version(),platform=platform.platform(),torch=torch.__version__,torchvision=torchvision.__version__,
          aiortc=aiortc.__version__,av=av.__version__,numpy=numpy.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),modules=[])
for mod in modules:
 f=Path(inspect.getsourcefile(mod));data['modules'].append(dict(module=mod.__name__,filename=f.name,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
Path('results/runtime_sources.json').write_text(json.dumps(data,indent=2));print(json.dumps(data,indent=2))
