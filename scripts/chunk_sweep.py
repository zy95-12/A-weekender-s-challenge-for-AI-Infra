"""Representative candidate scan; not a capacity benchmark.

Each candidate must pass unchanged same-TP correctness before timings. Keeps
serial chunk and scheduler-policy ablations separate; no automatic winner.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

from baseline_matrix import ROOT, PYTHON, run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    parser.add_argument("--reference", help="Existing native reference, useful from an isolated worktree")
    parser.add_argument("--tp",type=int,choices=[1,2],default=2)
    parser.add_argument("--sizes",default="0,256,512,1024,2048")
    parser.add_argument("--policies",default="legacy,decode-first")
    parser.add_argument("--repeats",type=int,default=1,help="Representative repeats; no full-matrix requirement")
    parser.add_argument("--mixed-repeats",type=int,default=3)
    parser.add_argument("--ipc-mode",choices=["pipe","shm"],default="pipe")
    parser.add_argument("--wire-fast",action="store_true")
    parser.add_argument("--tcp-buffer-mib",type=int,default=0)
    args = parser.parse_args()
    sizes = [int(x) for x in args.sizes.split(",")]
    policies = args.policies.split(",")
    if (len(set(sizes)) != len(sizes) or any(not 0 <= x <= 16384 for x in sizes) or
            len(set(policies)) != len(policies) or set(policies)-{"legacy","decode-first"} or
            min(args.repeats,args.mixed_repeats) < 1):
        parser.error("Invalid scan dimensions")
    output = Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    manifest = {"result":"RUNNING","config":vars(args),"completed":[],"started_unix":time.time(),
                "scope":"8k/256 C1+C4 candidate scan and controlled mixed arrivals; not full regression"}
    status = output/"sweep.json"
    status.write_text(json.dumps(manifest,indent=2))
    common = ["bash","poc","up","--wan","--enterprise-tp",str(args.tp),"--cloud-tp",str(args.tp),
              "--ipc-mode",args.ipc_mode,"--tcp-buffer-mib",str(args.tcp_buffer_mib)]
    if args.wire_fast:
        common.append("--wire-fast")
    reference = Path(args.reference) if args.reference else ROOT/("results/baseline_native_tp1" if args.tp==1 else "results/validation_final/reference")
    try:
        for size in sizes:
            for policy in policies:
                label = f"chunk_{size}_{policy}"
                folder = output/label
                folder.mkdir()
                run("up",common+["--prefill-chunk-size",str(size),"--scheduler-policy",policy],folder)
                correctness = folder/"correctness"
                run("correctness",[PYTHON,"scripts/correctness.py","--reference",str(reference),
                                   "--steps","33","--output",str(correctness)],folder)
                for repeat in range(args.repeats):
                    for c in (1,4):
                        point = f"isl_8192_osl_256_c_{c}_r_{repeat}"
                        run(point,[PYTHON,"scripts/benchmark.py","--baseline","--rates","inf",
                            "--max-concurrency",str(c),"--requests",str(max(8,4*c)),"--input","8192",
                            "--output-tokens","256","--seed",str(repeat),"--output",str(folder/point),
                            "--correctness-report",str(correctness/"summary.json")],folder)
                run("mixed",[PYTHON,"scripts/mixed_arrival.py","--output",str(folder/"mixed"),
                             "--repeats",str(args.mixed_repeats),"--correctness-report",str(correctness/"summary.json")],folder)
                manifest["completed"].append(label)
                status.write_text(json.dumps(manifest,indent=2))
        # Return to the same data-path variant, with stage-2 switches OFF.
        run("restore_serial_unchunked",common,output)
        manifest.update(result="PASS",finished_unix=time.time())
    except BaseException as error:
        manifest.update(result="FAIL",error=repr(error))
        raise
    finally:
        status.write_text(json.dumps(manifest,indent=2))


if __name__ == "__main__":
    main()
