import requests,concurrent.futures,pathlib,json,hashlib,tarfile,time,argparse
from bs4 import BeautifulSoup
s=requests.Session();r=s.get('https://drive.google.com/uc?export=download&id=1IWD-VUc0RPXXhBzoH5j9YD6bl8kzYYJ1',timeout=40);f=BeautifulSoup(r.text,'html.parser').find('form');params={x['name']:x['value'] for x in f.select('input[name]')}
u=requests.Request('GET',f['action'],params=params).prepare().url
p=argparse.ArgumentParser();p.add_argument('--output-root',required=True);args=p.parse_args()
root=pathlib.Path(args.output_root).resolve();root.mkdir(parents=True,exist_ok=True)
def fetch(start,end):
 for attempt in range(3):
  try:
   r=requests.get(u,headers={'Range':f'bytes={start}-{end}'},timeout=90);r.raise_for_status();assert r.status_code==206 and r.headers['Content-Range'].startswith(f'bytes {start}-{end}/');assert len(r.content)==end-start+1;return r.content
  except requests.RequestException:
   if attempt==2:raise
   time.sleep(1)
offset=0;entries=[]
while offset<521768960:
 h=fetch(offset,offset+511)
 if h==b'\0'*512:break
 m=tarfile.TarInfo.frombuf(h,'utf-8','strict');start=offset+512;entries.append(dict(name=m.name,size=m.size,offset=start));print(m.name,m.size,flush=True)
 if m.size and (m.name in ['grace/128_freeze.model','grace/4096_freeze.model'] or m.name.startswith('other/')):
  p=root/m.name;p.parent.mkdir(parents=True,exist_ok=True)
  if not p.resolve().is_relative_to(root):raise ValueError('Archive path escapes output root')
  if p.exists() and p.stat().st_size==m.size:
   offset=start+((m.size+511)//512)*512
   continue
  ranges=[(a,min(a+1024*1024-1,start+m.size-1)) for a in range(start,start+m.size,1024*1024)]
  with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:data=b''.join(pool.map(lambda ab:fetch(*ab),ranges))
  p.write_bytes(data);print('SAVED',m.name,hashlib.sha256(data).hexdigest(),flush=True)
 offset=start+((m.size+511)//512)*512
pathlib.Path('results/model_archive_inventory.json').write_text(json.dumps(entries,indent=2))
