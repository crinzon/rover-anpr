"""Integrity and safe installation of the runtime model weights."""

import hashlib
import sys
from unittest.mock import patch

import pytest

from download_models import download, verify


def test_reject_lfs_pointer(tmp_path):
    path = tmp_path / "weights"
    path.write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(ValueError, match="LFS pointer"):
        verify(path, "plate_detector.pt")


def test_reject_corruption_and_preserve_existing(tmp_path):
    path = tmp_path / "plate_detector.pt"
    path.write_bytes(b"existing local model")
    with pytest.raises(ValueError, match="SHA256"):
        download(tmp_path)
    assert path.read_bytes() == b"existing local model"


def test_atomic_verified_download(tmp_path):
    import io

    payload = b"verified binary"
    models = {"model.pt": (len(payload), hashlib.sha256(payload).hexdigest())}
    with patch("download_models.MODELS", models):
        with patch("urllib.request.urlopen", return_value=io.BytesIO(payload)):
            download(tmp_path)
    assert (tmp_path / "model.pt").read_bytes() == payload
    assert not list(tmp_path.glob("*.part"))
    if sys.platform != "win32":
        assert (tmp_path / "model.pt").stat().st_mode & 0o777 == 0o644


def test_failed_download_leaves_no_model(tmp_path):
    import io

    with patch("urllib.request.urlopen", return_value=io.BytesIO(b"bad")):
        with pytest.raises(ValueError):
            download(tmp_path)
    assert not list(tmp_path.iterdir())
