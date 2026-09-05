"""Isolated synchronized copy/codec costs, never interpreted as serving latency."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from split_poc.wire import pack,unpack


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    parser.add_argument("--repeats",type=int,default=20)
    args=parser.parse_args()
    torch.set_num_threads(4)
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for size in (8192,32768,4*1024**2,16*1024**2,64*1024**2):
        source=torch.full((2,size//8192,2048),.25,device="cuda",dtype=torch.float16)
        for pinned in (False,True):
            host=torch.empty(source.shape,dtype=torch.float16,pin_memory=pinned)
            target=torch.empty_like(source)
            for repeat in range(-1,args.repeats):
                torch.cuda.synchronize()
                t=time.perf_counter()
                host.copy_(source,non_blocking=pinned)
                torch.cuda.synchronize()
                d2h=time.perf_counter()
                encoded=pack({},[host[0].numpy(),host[1].numpy()])
                packed=time.perf_counter()
                _,arrays=unpack(encoded)
                parsed=time.perf_counter()
                target.copy_(host,non_blocking=pinned)
                torch.cuda.synchronize()
                end=time.perf_counter()
                assert torch.equal(source,target)
                assert all((a==.25).all() for a in arrays)
                if repeat>=0:
                    rows.append({"bytes_each_direction":size,"pinned":pinned,"repeat":repeat,
                        "d2h_ms":(d2h-t)*1000,"pack_ms":(packed-d2h)*1000,
                        "unpack_ms":(parsed-packed)*1000,"h2d_ms":(end-parsed)*1000})
    summary=[]
    for size in sorted({r['bytes_each_direction'] for r in rows}):
        for pinned in (False,True):
            subset=[r for r in rows if r['bytes_each_direction']==size and r['pinned']==pinned]
            summary.append({"bytes_each_direction":size,"pinned":pinned,"n":len(subset),
                            **{k:statistics.mean(r[k] for r in subset) for k in ('d2h_ms','pack_ms','unpack_ms','h2d_ms')}})
    (out/'measurement.json').write_text(json.dumps({"result":"PASS","command":sys.argv,
        "gpu":torch.cuda.get_device_name(),"rows":rows,"summary":summary,
        "scope":"Isolated synchronized costs; reuses destination buffers in BOTH modes. H2D copies the same original host buffer; codec roundtrip validated separately. No overlap or model inference."},indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
