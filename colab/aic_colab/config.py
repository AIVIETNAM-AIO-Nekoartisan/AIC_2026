from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    dataset_id: str
    expected_videos: int
    expected_keyframes: int
    archives: tuple[str, ...]
    siglip_model: str
    siglip_dimension: int
    jina_model: str
    jina_dimension: int
    shard_size: int
    webp_quality: int
    deadline_seconds: int


def load_config(path: Path) -> DatasetConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    archives = tuple(data["archives"])
    if not data.get("dataset_id"):
        raise ValueError("dataset_id is required")
    if len(set(archives)) != len(archives):
        raise ValueError("duplicate archive name")
    if not 1 <= int(data["shard_size"]) <= 4096:
        raise ValueError("shard_size must be between 1 and 4096")
    if (int(data["siglip_dimension"]), int(data["jina_dimension"])) != (
        1152,
        1024,
    ):
        raise ValueError("model dimensions must be 1152 and 1024")
    return DatasetConfig(
        dataset_id=str(data["dataset_id"]),
        expected_videos=int(data["expected_videos"]),
        expected_keyframes=int(data["expected_keyframes"]),
        archives=archives,
        siglip_model=str(data["siglip_model"]),
        siglip_dimension=int(data["siglip_dimension"]),
        jina_model=str(data["jina_model"]),
        jina_dimension=int(data["jina_dimension"]),
        shard_size=int(data["shard_size"]),
        webp_quality=int(data["webp_quality"]),
        deadline_seconds=int(data["deadline_seconds"]),
    )
