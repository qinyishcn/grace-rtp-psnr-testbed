"""Map the Testbed configuration contract; reject unsupported semantic conversions."""
import argparse,json,hashlib
from pathlib import Path
import yaml
p=argparse.ArgumentParser();p.add_argument("source");p.add_argument("output");a=p.parse_args()
raw=Path(a.source).read_bytes();src=yaml.safe_load(raw)["profiles"];dst={};unsupported={}
for name in ["no_emulation","lossy","5qi_7","congested"]:
 c=src[name]; reasons=[]
 for field in ["loss_correlation_pct","reorder_pct","duplicate_pct"]:
  if c.get(field,0):reasons.append(field)
 if c.get("delay_distribution","fixed") not in ["fixed","normal"]:reasons.append("delay_distribution")
 if reasons:unsupported[name]=reasons;continue
 dst[name]={k:c[k] for k in ["delay_ms","jitter_ms","loss_pct","rate_mbit","limit_packets"] if k in c}
Path(a.output).write_text(yaml.safe_dump(dict(profiles=dst)),encoding="utf-8")
Path(a.output+".provenance.json").write_text(json.dumps(dict(source_sha256=hashlib.sha256(raw).hexdigest(),unsupported=unsupported,
 direction="one-way",deadline_ms=100,note="Virtual emulator differs from tc-netem; unsupported profiles explicitly excluded"),indent=2))
print(dst,unsupported)
