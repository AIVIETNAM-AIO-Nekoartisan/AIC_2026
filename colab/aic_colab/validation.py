from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

import numpy as np


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    count: int
    errors: tuple[str, ...]


def write_embedding_shard(
    point_ids: Sequence[str], vectors: np.ndarray, path: Path
) -> Path:
    ids = np.asarray(list(point_ids), dtype=str)
    array = np.asarray(vectors)
    if array.ndim != 2 or len(ids) != array.shape[0]:
        raise ValueError("point IDs and vector rows must have equal length")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.partial")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, point_ids=ids, vectors=array)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def validate_embedding_shard(
    path: Path, expected_ids: Sequence[str], expected_dim: int
) -> ValidationResult:
    errors: list[str] = []
    count = 0
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"point_ids", "vectors"}:
                errors.append("NPZ must contain exactly point_ids and vectors")
            if "point_ids" not in archive.files or "vectors" not in archive.files:
                return ValidationResult(False, 0, tuple(errors))
            point_ids = archive["point_ids"]
            vectors = archive["vectors"]
    except (OSError, ValueError) as exc:
        return ValidationResult(False, 0, (f"cannot read NPZ: {exc}",))

    count = len(point_ids)
    expected = np.asarray(list(expected_ids), dtype=str)
    if point_ids.ndim != 1 or not np.array_equal(point_ids.astype(str), expected):
        errors.append("point ID order does not match manifest")
    if vectors.shape != (len(expected), expected_dim):
        errors.append(
            f"vector shape {vectors.shape} does not match {(len(expected), expected_dim)}"
        )
    if vectors.dtype != np.float32:
        errors.append(f"vectors must use float32, got {vectors.dtype}")
    if not np.all(np.isfinite(vectors)):
        errors.append("vectors contain NaN or infinite values")
    if vectors.ndim == 2 and vectors.shape[0] > 0 and np.all(np.isfinite(vectors)):
        norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
        if not np.all((norms >= 0.999) & (norms <= 1.001)):
            errors.append("vectors must be L2-normalized")
    return ValidationResult(not errors, count, tuple(errors))
