from __future__ import annotations

import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from collections.abc import Sequence
from zipfile import ZipFile

import cv2

from .map_reader import KeyframeRecord


@dataclass(frozen=True)
class ExtractedFrame:
    record: KeyframeRecord
    path: Path

    @property
    def member_name(self) -> str:
        group = self.record.video_id.split("_", 1)[0]
        width = max(3, len(str(self.record.ordinal)))
        return f"{group}/{self.record.video_id}/{self.record.ordinal:0{width}d}.webp"


def _safe_member(name: str) -> bool:
    member = PurePosixPath(name)
    return not member.is_absolute() and ".." not in member.parts and bool(member.parts)


def safe_extract_zip(archive: Path, destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with ZipFile(archive) as zf:
        for info in zf.infolist():
            if not _safe_member(info.filename):
                raise ValueError(f"unsafe ZIP member: {info.filename}")
            unix_mode = info.external_attr >> 16
            if stat.S_ISLNK(unix_mode):
                raise ValueError(f"unsafe ZIP member: {info.filename}")
        for info in zf.infolist():
            target = destination.joinpath(*PurePosixPath(info.filename).parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            extracted.append(target)
    return extracted


def extract_mapped_frames(
    video: Path,
    records: Sequence[KeyframeRecord],
    destination: Path,
    webp_quality: int = 90,
) -> list[ExtractedFrame]:
    ordered = list(records)
    if not ordered:
        return []
    if any(record.video_id != video.stem for record in ordered):
        raise ValueError(f"map/video ID mismatch for {video.name}")
    if any(
        current.frame_index >= following.frame_index
        for current, following in zip(ordered, ordered[1:])
    ):
        raise ValueError("mapped frame indexes must be strictly increasing")
    if not 1 <= webp_quality <= 100:
        raise ValueError("webp_quality must be between 1 and 100")

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    requested = {record.frame_index: record for record in ordered}
    last_index = ordered[-1].frame_index
    extracted: list[ExtractedFrame] = []
    frame_index = 0
    try:
        while frame_index <= last_index:
            ok, image = capture.read()
            if not ok:
                missing = next(
                    (record for record in ordered if record.frame_index >= frame_index),
                    ordered[-1],
                )
                raise RuntimeError(
                    f"decode failed video={video.stem} ordinal={missing.ordinal} "
                    f"frame={missing.frame_index}"
                )
            record = requested.get(frame_index)
            if record is not None:
                observed_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
                observed = observed_ms / 1000.0 if observed_ms > 0 else 0.0
                tolerance = max(1.0 / record.fps, 0.05)
                if observed > 0 and abs(observed - record.timestamp_seconds) > tolerance:
                    raise RuntimeError(
                        f"timestamp mismatch video={video.stem} ordinal={record.ordinal} "
                        f"frame={record.frame_index} expected={record.timestamp_seconds} "
                        f"observed={observed}"
                    )
                group = record.video_id.split("_", 1)[0]
                width = max(3, len(str(record.ordinal)))
                output = (
                    destination
                    / group
                    / record.video_id
                    / f"{record.ordinal:0{width}d}.webp"
                )
                output.parent.mkdir(parents=True, exist_ok=True)
                temporary = output.with_name(f"{output.stem}.partial.webp")
                if not cv2.imwrite(
                    str(temporary), image, [cv2.IMWRITE_WEBP_QUALITY, webp_quality]
                ):
                    raise RuntimeError(
                        f"WebP write failed video={video.stem} ordinal={record.ordinal}"
                    )
                os.replace(temporary, output)
                extracted.append(ExtractedFrame(record, output))
            frame_index += 1
    finally:
        capture.release()
    return extracted
