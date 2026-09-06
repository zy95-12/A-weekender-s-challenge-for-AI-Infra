from check_model import check_model
"""Pinned public auxiliary training data; the target prompt is never used for training."""
import argparse,hashlib,json,sys,urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'cache/deps'))
import pyarrow.parquet as pq
from transformers import AutoTokenizer


def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);a=p.parse_args();root=Path(__file__).resolve().parent
 check_model(a.model)
 cache=root/'cache';cache.mkdir(exist_ok=True)
 revision='b08601e04326c79dfdd32d625aee71d232d685c3';tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
 manifest={'dataset':'Salesforce/wikitext','config':'wikitext-2-raw-v1','revision':revision,'license':'CC-BY-SA-3.0 / GFDL','context_tokens':4096,'splits':{}}
 result={}
 for split,limit in [('train',65536),('validation',8192),('test',8192)]:
  filename=f'wikitext-2-raw-v1/{split}-00000-of-00001.parquet';dest=cache/(split+'.parquet')
  url=f'https://huggingface.co/datasets/Salesforce/wikitext/resolve/{revision}/{filename}'
  if not dest.exists():urllib.request.urlretrieve(url,dest)
  texts=pq.read_table(dest)['text'].to_pylist();ids=[];used=[]
  for i,text in enumerate(texts):
   if not text.strip():continue
   ids+=tok.encode(text,add_special_tokens=False);used.append(i)
   if len(ids)>=limit:break
  ids=ids[:limit];assert len(ids)==limit
  result[split]=[ids[i:i+4096] for i in range(0,len(ids),4096)]
  manifest['splits'][split]={'tokens':len(ids),'source_row_indices':used,'parquet_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'token_ids_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
 (cache/'public_tokens.json').write_text(json.dumps(result))
 (root/'data/public_manifest.json').write_text(json.dumps(manifest,indent=2))
 print(json.dumps({k:v['tokens'] for k,v in manifest['splits'].items()}))

if __name__=='__main__':main()
