from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Artifact:
    path: Path
    sha256: str
    size: int


def checksum_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.sha256")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify_artifact(path: Path) -> bool:
    sidecar = checksum_path(path)
    if not path.is_file() or not sidecar.is_file():
        return False
    try:
        expected = sidecar.read_text(encoding="ascii").strip().split()[0]
    except (OSError, IndexError, UnicodeDecodeError):
        return False
    return len(expected) == 64 and sha256_file(path) == expected


def persist_verified(source: Path, destination: Path) -> Artifact:
    source_hash = sha256_file(source)
    if destination.exists():
        if verify_artifact(destination) and sha256_file(destination) == source_hash:
            return Artifact(destination, source_hash, destination.stat().st_size)
        raise FileExistsError(f"destination exists and is not reusable: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.partial")
    sidecar = checksum_path(destination)
    sidecar_partial = sidecar.with_name(f"{sidecar.name}.partial")
    if partial.exists() or sidecar_partial.exists():
        raise FileExistsError(f"partial destination exists: {destination}")
    try:
        with source.open("rb") as src, partial.open("xb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
        if sha256_file(partial) != source_hash:
            raise OSError(f"checksum mismatch while copying {destination}")
        with sidecar_partial.open("x", encoding="ascii") as handle:
            handle.write(f"{source_hash}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, destination)
        os.replace(sidecar_partial, sidecar)
        _fsync_directory(destination.parent)
    finally:
        partial.unlink(missing_ok=True)
        sidecar_partial.unlink(missing_ok=True)

    if not verify_artifact(destination):
        raise OSError(f"persisted artifact failed verification: {destination}")
    return Artifact(destination, source_hash, destination.stat().st_size)
