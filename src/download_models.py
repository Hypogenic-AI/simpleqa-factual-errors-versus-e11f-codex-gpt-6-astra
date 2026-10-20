"""Fetch pinned checkpoint files into this workspace; record failures and bytes."""
import concurrent.futures as cf
import json, time
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1]
assert Path.cwd()==ROOT
pins=json.loads((ROOT/'artifacts/resource_finder/model_revisions.json').read_text())

def fetch(job):
    mid,hf,rev,name=job
    dest=ROOT/'models'/mid/name
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists(): return {'model':mid,'file':name,'bytes':dest.stat().st_size,'cached':True}
    url=f'https://huggingface.co/{hf}/resolve/{rev}/{name}'
    start=time.time()
    for attempt in range(3):
        try:
            with requests.get(url,stream=True,timeout=(20,90)) as r:
                r.raise_for_status()
                with dest.with_suffix(dest.suffix+'.part').open('wb') as f:
                    for chunk in r.iter_content(8*1024*1024): f.write(chunk)
            dest.with_suffix(dest.suffix+'.part').rename(dest)
            result={'model':mid,'file':name,'bytes':dest.stat().st_size,'seconds':time.time()-start}
            print(json.dumps(result),flush=True)
            return result
        except Exception as e:
            print(mid,name,'attempt',attempt,str(e)[:200],flush=True)
    return {'model':mid,'file':name,'error':str(e) if 'e' in locals() else 'download failed'}

jobs=[]
for p in sorted(pins,key=lambda p: ['qwen3-8b','deepseek-r1-distill-llama-8b','moonlight-16b-a3b-instruct','qwen2.5-7b-judge'].index(p['id'])):
    meta=requests.get(f'https://huggingface.co/api/models/{p["hf_id"]}/revision/{p["revision"]}',timeout=30).json()
    files=[x['rfilename'] for x in meta['siblings']]
    for name in files:
        if name.endswith(('.safetensors','.json','.py','.model','.tiktoken')) or name in ['merges.txt','vocab.json','chat_template.jinja']:
            jobs.append((p['id'],p['hf_id'],p['revision'],name))
with cf.ThreadPoolExecutor(max_workers=8) as pool:
    out=list(pool.map(fetch,jobs))
(ROOT/'results/runtime/downloads.json').write_text(json.dumps(out,indent=2))
