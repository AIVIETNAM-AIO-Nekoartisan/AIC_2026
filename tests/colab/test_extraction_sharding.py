import hashlib
import tarfile
from zipfile import ZipFile

import cv2
import numpy as np
import pytest

from aic_colab.extraction import ExtractedFrame, extract_mapped_frames, safe_extract_zip
from aic_colab.map_reader import KeyframeRecord
from aic_colab.sharding import partition_frames, write_tar_shard


def _record(ordinal: int, frame_index: int, timestamp: float) -> KeyframeRecord:
    return KeyframeRecord(
        video_id="L21_V001",
        ordinal=ordinal,
        timestamp_seconds=timestamp,
        fps=10.0,
        frame_index=frame_index,
        point_id=f"point-{ordinal}",
    )


@pytest.fixture
def extracted_frames(tmp_path):
    root = tmp_path / "frames"
    items = []
    for ordinal in (1, 2, 3):
        path = root / "L21" / "L21_V001" / f"{ordinal:03d}.webp"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"frame-{ordinal}".encode())
        items.append(ExtractedFrame(_record(ordinal, ordinal - 1, (ordinal - 1) / 10), path))
    return items


def test_safe_extract_zip_rejects_parent_traversal(tmp_path):
    archive = tmp_path / "bad.zip"
    with ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", "owned")

    with pytest.raises(ValueError, match="unsafe ZIP member"):
        safe_extract_zip(archive, tmp_path / "out")

    assert not (tmp_path / "escape.txt").exists()


def test_extract_mapped_frames_decodes_only_requested_indexes(tmp_path):
    video = tmp_path / "L21_V001.avi"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (320, 240)
    )
    assert writer.isOpened()
    for frame_index in range(30):
        image = np.full((240, 320, 3), frame_index * 8, dtype=np.uint8)
        writer.write(image)
    writer.release()
    records = [_record(1, 0, 0.0), _record(2, 10, 1.0), _record(3, 29, 2.9)]

    frames = extract_mapped_frames(video, records, tmp_path / "frames")

    assert [item.path.name for item in frames] == ["001.webp", "002.webp", "003.webp"]
    assert all(item.path.is_file() and item.path.stat().st_size > 0 for item in frames)


def test_extract_mapped_frames_rejects_unreachable_frame(tmp_path):
    video = tmp_path / "L21_V001.avi"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 64)
    )
    writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()

    with pytest.raises(RuntimeError, match="frame=2"):
        extract_mapped_frames(video, [_record(1, 2, 0.2)], tmp_path / "frames")


def test_tar_shards_are_bounded_and_have_stable_members(extracted_frames, tmp_path):
    shards = partition_frames(extracted_frames, shard_size=2)
    assert [len(shard) for shard in shards] == [2, 1]

    first = write_tar_shard(shards[0], tmp_path / "first.tar")
    second = write_tar_shard(list(reversed(shards[0])), tmp_path / "second.tar")

    assert hashlib.sha256(first.path.read_bytes()).digest() == hashlib.sha256(
        second.path.read_bytes()
    ).digest()
    with tarfile.open(first.path) as archive:
        assert archive.getnames() == [
            "L21/L21_V001/001.webp",
            "L21/L21_V001/002.webp",
        ]
        assert all(member.mtime == 0 for member in archive.getmembers())


def test_partition_frames_rejects_oversized_shard_setting(extracted_frames):
    with pytest.raises(ValueError, match="between 1 and 1024"):
        partition_frames(extracted_frames, shard_size=1025)
