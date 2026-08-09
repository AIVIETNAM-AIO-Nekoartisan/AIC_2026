from __future__ import annotations

import uuid


def stable_point_id(dataset_id: str, video_id: str, ordinal: int) -> str:
    if not dataset_id or not video_id or ordinal < 1:
        raise ValueError("dataset_id, video_id, and positive ordinal are required")
    value = f"aic://{dataset_id}/{video_id}/{ordinal}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, value))
