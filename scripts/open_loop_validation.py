"""Repeat representative open-loop validation points with independent arrival seeds."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from open_loop_sweep import ROOT, run

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
out = a.out.resolve()
out.mkdir(parents=True, exist_ok=False)
files = list((ROOT / "split_poc").glob("*.py")) + [
    ROOT / "scripts" / n
    for n in [
        "open_loop_benchmark.py",
        "open_loop_validation.py",
        "manage.py",
        "network_state.py",
    ]
]
manifest = {
    "commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip(),
    "source_sha256": {
        str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in files
    },
    "seeds": [29, 43],
    "costs_refitted": False,
}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2))
for f in files:
    d = out / "source" / f.relative_to(ROOT)
    d.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(f, d)
for n in ["current_prompt.json", "current_reference.json"]:
    shutil.copyfile(ROOT / "Q1/data" / n, out / n)
summaries = []
try:
    for system, rates in [
        ("baseline", [0.75]),
        ("optimized", [1.0, 4.440035098628993, 4.744246360930501]),
    ]:
        folder = out / system
        folder.mkdir()
        run(
            folder / "startup.log",
            [
                "./poc",
                "up",
                "--preset",
                system,
                "--wan",
                "--max-active",
                "96",
                "--kv-blocks",
                "32768",
            ],
        )
        live = Path((ROOT / "run/current_results").read_text().strip())
        for rate in rates:
            for seed in [29, 43]:
                label = f"rate-{rate:.3f}-seed-{seed}"
                dest = folder / label
                print("START", system, label, flush=True)
                run(
                    folder / (label + ".log"),
                    [
                        str(ROOT / ".venv/bin/python"),
                        "scripts/open_loop_benchmark.py",
                        "--system",
                        system,
                        "--rate",
                        str(rate),
                        "--seed",
                        str(seed),
                        "--seconds",
                        "120" if rate >= 4 else "60",
                        "--label",
                        system + "-" + label,
                        "--out",
                        str(dest),
                    ],
                )
                s = json.loads((dest / "summary.json").read_text())
                s["path"] = str(dest.relative_to(out))
                summaries.append(s)
                (out / "summaries.json").write_text(json.dumps(summaries, indent=2))
                print("RESULT", json.dumps(s), flush=True)
        for f in [
            *live.glob("*config.json"),
            *live.glob("*trace.jsonl"),
            live / "environment.json",
        ]:
            if f.exists():
                shutil.copyfile(f, folder / f.name)
finally:
    run(out / "restore.log", ["./poc", "up", "--wan"])
