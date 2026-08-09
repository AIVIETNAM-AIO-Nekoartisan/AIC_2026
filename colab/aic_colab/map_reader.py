from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .ids import stable_point_id


@dataclass(frozen=True)
class KeyframeRecord:
    video_id: str
    ordinal: int
    timestamp_seconds: float
    fps: float
    frame_index: int
    point_id: str


def read_map_csv(path: Path, dataset_id: str) -> list[KeyframeRecord]:
    required = {"n", "pts_time", "fps", "frame_idx"}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"required columns are {sorted(required)}")
        records: list[KeyframeRecord] = []
        previous: KeyframeRecord | None = None
        video_id = path.stem
        for raw in reader:
            ordinal = int(raw["n"])
            timestamp = float(raw["pts_time"])
            fps = float(raw["fps"])
            frame_index = int(raw["frame_idx"])
            if ordinal < 1 or timestamp < 0 or fps <= 0 or frame_index < 0:
                raise ValueError(f"invalid map value in {path.name}")
            record = KeyframeRecord(
                video_id=video_id,
                ordinal=ordinal,
                timestamp_seconds=timestamp,
                fps=fps,
                frame_index=frame_index,
                point_id=stable_point_id(dataset_id, video_id, ordinal),
            )
            if previous is not None and not (
                record.ordinal > previous.ordinal
                and record.frame_index > previous.frame_index
                and record.timestamp_seconds > previous.timestamp_seconds
            ):
                raise ValueError(f"map rows must be strictly monotonic: {path.name}")
            if previous is None and ordinal != 1:
                raise ValueError(f"first ordinal must be 1: {path.name}")
            previous = record
            records.append(record)
    return records
