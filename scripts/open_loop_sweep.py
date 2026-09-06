"""Compare presets with identical Poisson schedules at shared arrival rates."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RATES = [0.5, 1.0, 1.5, 2.0, 3.0, 4.135823836327486, 4.744246360930501]


def run(log, command):
    with log.open("w") as f:
        subprocess.run(
            command, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    files = list((ROOT / "split_poc").glob("*.py")) + [
        ROOT / "scripts" / name
        for name in [
            "open_loop_benchmark.py",
            "open_loop_sweep.py",
            "pd_benchmark.py",
            "manage.py",
            "network_state.py",
        ]
    ]
    hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "source_sha256": hashes,
                "rates": RATES,
                "seed": 17,
                "coarse_seconds": 60,
                "high_rate_seconds": 120,
                "warmup_seconds": 30,
                "tail_seconds": 30,
            },
            indent=2,
        )
    )
    for p in files:
        dest = out / "source" / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
    for name in ["current_prompt.json", "current_reference.json"]:
        shutil.copyfile(ROOT / "Q1/data" / name, out / name)
    summaries = []
    try:
        for system in ["baseline", "optimized"]:
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
            (folder / "trace_source.txt").write_text(str(live))

            def point(rate, label):
                dest = folder / label
                print("START", system, label, rate, flush=True)
                run(
                    folder / (label + ".log"),
                    [
                        str(ROOT / ".venv/bin/python"),
                        str(ROOT / "scripts/open_loop_benchmark.py"),
                        "--system",
                        system,
                        "--rate",
                        str(rate),
                        "--label",
                        system + "-" + label,
                        "--out",
                        str(dest),
                        "--seconds",
                        "120" if rate >= 4 else "60",
                    ],
                )
                s = json.loads((dest / "summary.json").read_text())
                s["path"] = str(dest.relative_to(out))
                summaries.append(s)
                (out / "summaries.json").write_text(json.dumps(summaries, indent=2))
                print("RESULT", json.dumps(s), flush=True)
                return (
                    min(
                        s["arrival_cohort"]["joint_slo"],
                        s["completion_cohort"]["joint_slo"],
                    )
                    >= 0.99
                )

            last_pass = 0
            for i, rate in enumerate(RATES):
                if point(rate, f"rate-{rate:.3f}"):
                    last_pass = rate
                else:
                    # One midpoint makes the first failed interval reviewable, without an exhaustive search.
                    midpoint = (last_pass + rate) / 2
                    point(midpoint, f"refine-{midpoint:.3f}")
                    break
            for p in (
                list(live.glob("*config.json"))
                + list(live.glob("*trace.jsonl"))
                + [live / "environment.json"]
            ):
                if p.exists():
                    shutil.copyfile(p, folder / p.name)
    finally:
        run(out / "restore.log", ["./poc", "up", "--wan"])
    assert hashes == {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
