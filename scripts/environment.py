"""Record the actual environment and source digests for reproducibility."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys


def run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    return {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def main():
    root = Path(__file__).resolve().parents[1]
    output = Path(sys.argv[1])
    import torch
    data = {"python": sys.version, "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "nccl": torch.cuda.nccl.version(), "vllm": importlib.metadata.version("vllm"),
            "gpu": run(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total", "--format=csv"]),
            "gpu_topology": run(["nvidia-smi", "topo", "-m"]), "kernel": run(["uname", "-a"]),
            "git": run(["git", "rev-parse", "HEAD"]), "git_status": run(["git", "status", "--short"]),
            "packages": run([sys.executable, "-m", "pip", "freeze"]),
            "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for folder in ("split_poc", "scripts") for p in (root / folder).glob("*.py")}}
    model_manifest = root / "models/qwen/checksums.json"
    if model_manifest.exists():
        data["model"] = json.loads(model_manifest.read_text())
    output.write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
