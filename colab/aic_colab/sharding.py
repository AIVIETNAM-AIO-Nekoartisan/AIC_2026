from __future__ import annotations

import os
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from collections.abc import Sequence

from .extraction import ExtractedFrame


@dataclass(frozen=True)
class TarShard:
    path: Path
    count: int
    point_ids: tuple[str, ...]
    members: tuple[str, ...]


def partition_frames(
    frames: Sequence[ExtractedFrame], shard_size: int = 1024
) -> list[list[ExtractedFrame]]:
    if not 1 <= shard_size <= 1024:
        raise ValueError("shard_size must be between 1 and 1024")
    ordered = sorted(frames, key=lambda item: (item.record.video_id, item.record.ordinal))
    return [ordered[start : start + shard_size] for start in range(0, len(ordered), shard_size)]


def write_tar_shard(frames: Sequence[ExtractedFrame], tar_path: Path) -> TarShard:
    if not frames:
        raise ValueError("cannot write an empty tar shard")
    ordered = sorted(frames, key=lambda item: (item.record.video_id, item.record.ordinal))
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tar_path.with_name(f"{tar_path.name}.partial")
    if temporary.exists() or tar_path.exists():
        raise FileExistsError(f"tar destination exists: {tar_path}")
    members: list[str] = []
    try:
        with tarfile.open(temporary, "w", format=tarfile.USTAR_FORMAT) as archive:
            for item in ordered:
                name = item.member_name
                pure = PurePosixPath(name)
                if pure.is_absolute() or ".." in pure.parts:
                    raise ValueError(f"unsafe tar member: {name}")
                info = tarfile.TarInfo(name)
                info.size = item.path.stat().st_size
                info.mode = 0o644
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = 0
                with item.path.open("rb") as handle:
                    archive.addfile(info, handle)
                members.append(name)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, tar_path)
    finally:
        temporary.unlink(missing_ok=True)
    return TarShard(
        path=tar_path,
        count=len(ordered),
        point_ids=tuple(item.record.point_id for item in ordered),
        members=tuple(members),
    )
