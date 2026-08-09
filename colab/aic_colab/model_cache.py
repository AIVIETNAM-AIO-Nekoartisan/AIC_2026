from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

import zstandard

from .persistence import sha256_file


@dataclass(frozen=True)
class ModelCacheArtifact:
    path: Path
    sha256: str
    manifest: Path


def create_model_cache(
    cache_root: Path,
    archive: Path,
    manifest: Path,
    models: dict[str, str],
) -> ModelCacheArtifact:
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(f"{archive.name}.partial")
    with temporary.open("wb") as raw:
        compressor = zstandard.ZstdCompressor(level=10, threads=0)
        with compressor.stream_writer(raw, closefd=False) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|") as output:
                for path in sorted(cache_root.rglob("*"), key=lambda item: item.as_posix()):
                    relative = path.relative_to(cache_root).as_posix()
                    info = tarfile.TarInfo(relative + ("/" if path.is_dir() else ""))
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mode = 0o755 if path.is_dir() else 0o644
                    if path.is_dir():
                        info.type = tarfile.DIRTYPE
                        output.addfile(info)
                    elif path.is_file():
                        info.size = path.stat().st_size
                        with path.open("rb") as source:
                            output.addfile(info, source)
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(temporary, archive)
    digest = sha256_file(archive)
    payload = {
        "schema_version": 1,
        "archive": archive.name,
        "sha256": digest,
        "models": dict(sorted(models.items())),
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return ModelCacheArtifact(archive, digest, manifest)


def ensure_model_cache(
    cache_root: Path, drive_archive: Path, manifest: Path
) -> Path:
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    if sha256_file(drive_archive) != metadata.get("sha256"):
        raise ValueError("model cache checksum mismatch")
    if cache_root.exists() and any(cache_root.iterdir()):
        return cache_root
    cache_root.mkdir(parents=True, exist_ok=True)
    try:
        with drive_archive.open("rb") as raw:
            with zstandard.ZstdDecompressor().stream_reader(raw) as decompressed:
                with tarfile.open(fileobj=decompressed, mode="r|") as archive:
                    for member in archive:
                        _extract_safe_member(archive, member, cache_root)
    except Exception:
        if cache_root.exists():
            shutil.rmtree(cache_root)
        raise
    return cache_root


def _extract_safe_member(
    archive: tarfile.TarFile, member: tarfile.TarInfo, root: Path
) -> None:
    relative = PurePosixPath(member.name)
    if relative.is_absolute() or ".." in relative.parts or member.issym() or member.islnk():
        raise ValueError(f"unsafe model cache member: {member.name}")
    target = root.joinpath(*relative.parts)
    if member.isdir():
        target.mkdir(parents=True, exist_ok=True)
        return
    if not member.isfile():
        raise ValueError(f"unsupported model cache member: {member.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    source = archive.extractfile(member)
    if source is None:
        raise ValueError(f"cannot read model cache member: {member.name}")
    with source, target.open("xb") as output:
        shutil.copyfileobj(source, output, length=1024 * 1024)
