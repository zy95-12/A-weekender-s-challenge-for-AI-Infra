"""Download a pinned, public checkpoint; no tokens or credentials needed."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import shutil
import subprocess

REPO = "Qwen/Qwen2.5-3B-Instruct"
REV = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
DEST = Path(__file__).resolve().parents[1] / "models/qwen"


def download(entry):
    name = entry["rfilename"]
    path = DEST / name
    expected = entry.get("lfs", {}).get("sha256")
    expected_size = entry.get("size")
    if path.exists() and (expected_size is None or path.stat().st_size == expected_size):
        if not expected or hashlib.file_digest(path.open("rb"), "sha256").hexdigest() == expected:
            print(f"Verified {name}", flush=True)
            return
    part = path.with_suffix(path.suffix + ".part")
    for attempt in range(5):
        try:
            url = f"https://huggingface.co/{REPO}/resolve/{REV}/{name}?download=true"
            print(f"Downloading {name}", flush=True)
            if expected_size and expected_size > 100_000_000 and shutil.which("aria2c"):
                subprocess.run(["aria2c", "-x", "16", "-s", "16", "-k", "4M", "-c",
                    "--file-allocation=none", "--summary-interval=30", "--console-log-level=warn",
                    "--show-console-readout=false", "--enable-color=false",
                    "--download-result=hide", "--auto-file-renaming=false",
                    "-d", str(DEST), "-o", part.name, url], check=True)
            else:
                with urllib.request.urlopen(url, timeout=90) as response, part.open("wb") as out:
                    while chunk := response.read(8 * 1024 * 1024):
                        out.write(chunk)
            if expected_size and part.stat().st_size != expected_size:
                raise ValueError("Download size mismatch")
            if expected and hashlib.file_digest(part.open("rb"), "sha256").hexdigest() != expected:
                raise ValueError("Weight SHA256 mismatch")
            part.replace(path)
            print(f"Ready {name}", flush=True)
            return
        except Exception as exc:
            print(f"Retry {name}: {exc}", flush=True)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Cannot download {name}")


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/api/models/{REPO}/revision/{REV}?blobs=true"
    with urllib.request.urlopen(url, timeout=30) as response:
        info = json.load(response)
    files = [e for e in info["siblings"] if e["rfilename"].endswith((".safetensors", ".json", ".txt"))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, files))
    (DEST / "checksums.json").write_text(json.dumps({"repository": REPO, "revision": REV,
        "weights": {e["rfilename"]: e.get("lfs", {}).get("sha256")
                    for e in files if e["rfilename"].endswith(".safetensors")}}, indent=2))
    (DEST / "revision.txt").write_text(REV + "\n")


if __name__ == "__main__":
    main()
