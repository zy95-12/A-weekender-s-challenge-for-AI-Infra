"""Create a Cloud model directory with no tokenizer artifacts.

Weights are read-only inputs shared on disk, not shared GPU activations. The
Cloud runner instantiates and loads only its owned transformer layers.
"""
from pathlib import Path
import shutil
import hashlib
import sys
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from split_poc import WEIGHT_SHA256, MODEL_ID, REVISION


def main():
    root = Path(__file__).resolve().parents[1]
    source, dest = root / "models/qwen", root / "models/cloud"
    for name, expected in WEIGHT_SHA256.items():
        with (source / name).open("rb") as file:
            if hashlib.file_digest(file, "sha256").hexdigest() != expected:
                raise RuntimeError(f"Checkpoint SHA256 mismatch: {name}")
    (source / "checksums.json").write_text(json.dumps({"repository": MODEL_ID, "revision": REVISION,
        "weights": WEIGHT_SHA256, "verified_sha256": True}, indent=2))
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "generation_config.json"):
        shutil.copyfile(source / name, dest / name)
    for name in WEIGHT_SHA256:
        path = source / name
        link = dest / path.name
        if not link.exists():
            link.symlink_to(Path("../qwen") / path.name)
    if any(dest.glob("tokenizer*")):
        raise RuntimeError("Cloud view unexpectedly contains tokenizer files")


if __name__ == "__main__":
    main()
