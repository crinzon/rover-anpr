"""Download pinned upstream LFS binaries; never replace existing weights."""

import argparse
import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

REVISION = "faa895e481b6538783184ad50d304aa453b4e794"
MODELS = {
    "plate_detector.pt": (
        5381573,
        "0ee9b29633e38d1ca444124791036665bbb28a563e3cdae765a72c7fd3ea1312",
    ),
    "plate_recognizer.pt": (
        8611055,
        "b115566f6894da055780959d1497f10ad0fd8b461a2ef533d08386cf135f6ac4",
    ),
}
BASE = (
    "https://media.githubusercontent.com/media/"
    f"shershunov/russian-license-plate-anpr/{REVISION}/checkpoints/"
)


def verify(path, name):
    size, digest = MODELS[name]
    data = Path(path).read_bytes()
    if data.startswith(b"version https://git-lfs.github.com/spec/"):
        raise ValueError(f"{name}: LFS pointer instead of weights")
    if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"{name}: size or SHA256 mismatch")


def download(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, (size, _) in MODELS.items():
        target = directory / name
        if target.exists():
            verify(target, name)
            continue
        fd, temporary = tempfile.mkstemp(dir=directory, suffix=".part")
        try:
            with os.fdopen(fd, "wb") as output:
                with urllib.request.urlopen(
                    BASE + name, timeout=60
                ) as response:
                    total = 0
                    while chunk := response.read(1024 * 1024):
                        total += len(chunk)
                        if total > size:
                            raise ValueError(f"{name}: download exceeds size")
                        output.write(chunk)
            verify(temporary, name)
            # Runtime containers use a different UID; weights are not secrets.
            os.chmod(temporary, 0o644)
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
        print(f"{name}: size and SHA256 verified")


def validate_loading(directory):
    # Use the production architecture and checkpoint loader, not torch.load
    # on unverified/untrusted bytes. No RTSP capture or recognition worker.
    from full_stand.config import StandConfig
    from full_stand.pipeline import PlatePipeline

    directory = Path(directory)
    for name in MODELS:
        verify(directory / name, name)
    pipeline = PlatePipeline(
        StandConfig(
            detector=directory / "plate_detector.pt",
            recognizer=directory / "plate_recognizer.pt",
            device="cpu",
            use_graphs=False,
        )
    )
    print(f"Both production models loaded on {pipeline.device}")
    return pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    download(args.directory)
    validate_loading(args.directory)


if __name__ == "__main__":
    main()
