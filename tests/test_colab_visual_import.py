import hashlib
import json
import tarfile
from io import BytesIO

import numpy as np
import pytest

from offline.backup_qdrant_collection import create_and_download_snapshot
from offline.import_colab_visual import (
    discover_visual_shards,
    import_visual_shards,
    verify_and_extract_keyframes,
)


def _sidecar(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name(path.name + ".sha256").write_text(digest + "\n")


def _result_fixture(tmp_path):
    root = tmp_path / "results"
    batch = "L21_a"
    (root / "keyframes" / batch).mkdir(parents=True)
    (root / "embeddings" / batch).mkdir(parents=True)
    (root / "manifests" / batch).mkdir(parents=True)
    tar_path = root / "keyframes" / batch / "000000.tar"
    with tarfile.open(tar_path, "w") as archive:
        for ordinal in (1, 2):
            data = f"frame-{ordinal}".encode()
            info = tarfile.TarInfo(f"L21/L21_V001/{ordinal:03d}.webp")
            info.size = len(data)
            archive.addfile(info, BytesIO(data))
    ids = np.asarray(["id-1", "id-2"])
    for model, dimension in (("siglip", 1152), ("jina", 1024)):
        vectors = np.zeros((2, dimension), dtype=np.float32)
        vectors[:, 0] = 1
        path = root / "embeddings" / batch / f"000000.{model}.npz"
        np.savez_compressed(path, point_ids=ids, vectors=vectors)
        _sidecar(path)
    manifest = root / "manifests" / batch / "000000.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps(
                {
                    "dataset_id": "aic25-b1",
                    "batch_id": batch,
                    "video_id": "L21_V001",
                    "ordinal": ordinal,
                    "timestamp_seconds": (ordinal - 1) * 0.1,
                    "fps": 10,
                    "frame_index": ordinal - 1,
                    "point_id": f"id-{ordinal}",
                    "keyframe_shard": f"keyframes/{batch}/000000.tar",
                    "keyframe_member": f"L21/L21_V001/{ordinal:03d}.webp",
                }
            )
            for ordinal in (1, 2)
        )
        + "\n"
    )
    _sidecar(tar_path)
    _sidecar(manifest)
    return root


class RecordingClient:
    def __init__(self, exists=False):
        self.exists = exists
        self.created = None
        self.upserts = []
        self.thresholds = []
        self.indexes = []

    def collection_exists(self, _name):
        return self.exists

    def create_collection(self, **kwargs):
        self.created = kwargs

    def update_collection(self, **kwargs):
        self.thresholds.append(kwargs)

    def upsert(self, **kwargs):
        self.upserts.extend(kwargs["points"])

    def create_payload_index(self, **kwargs):
        self.indexes.append(kwargs)


def test_import_creates_named_vectors_and_uses_manifest_ids(tmp_path):
    root = _result_fixture(tmp_path)
    shards = discover_visual_shards(root)
    keyframes = tmp_path / "keyframes"
    verify_and_extract_keyframes(shards, keyframes)
    client = RecordingClient()

    imported = import_visual_shards(
        client, "AIC_data_aic25_b1", shards, keyframes, batch_size=1
    )

    assert imported == 2
    assert set(client.created["vectors_config"]) == {"siglip", "jina"}
    assert [point.id for point in client.upserts] == ["id-1", "id-2"]
    assert set(client.upserts[0].vector) == {"siglip", "jina"}
    assert client.upserts[0].payload["frame"]["path"].endswith("001.webp")


def test_import_refuses_existing_collection_without_resume(tmp_path):
    shards = discover_visual_shards(_result_fixture(tmp_path))

    with pytest.raises(ValueError, match="already exists"):
        import_visual_shards(
            RecordingClient(exists=True),
            "AIC_data_aic25_b1",
            shards,
            tmp_path / "keyframes",
        )


def test_snapshot_download_writes_checksum_and_manifest(tmp_path):
    calls = []

    def transport(method, url, headers, data=None):
        calls.append((method, url))
        if method == "POST":
            return 200, json.dumps({"result": {"name": "snap.snapshot"}}).encode()
        if url.endswith("/collections/AIC_data_aic25_b1"):
            return 200, json.dumps({"result": {"points_count": 177321}, "version": "1.15"}).encode()
        return 200, b"snapshot-bytes"

    artifact = create_and_download_snapshot(
        "http://localhost:6333",
        "AIC_data_aic25_b1",
        tmp_path,
        bundle_version="1",
        transport=transport,
        timestamp="20260809T120000Z",
    )

    assert artifact.path.read_bytes() == b"snapshot-bytes"
    assert artifact.sha256 == hashlib.sha256(b"snapshot-bytes").hexdigest()
    manifest = json.loads(artifact.manifest.read_text())
    assert manifest["collection"] == "AIC_data_aic25_b1"
    assert manifest["points"] == 177321
    assert manifest["snapshot_name"] == "snap.snapshot"
