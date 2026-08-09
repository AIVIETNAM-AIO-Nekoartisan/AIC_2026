import hashlib
import json

import pytest

from aic_colab.persistence import persist_verified, verify_artifact
from aic_colab.state import StateStore


def test_persist_verified_writes_final_file_and_sidecar(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"abc")

    artifact = persist_verified(source, tmp_path / "drive" / "out.bin")

    expected = hashlib.sha256(b"abc").hexdigest()
    assert artifact.path.read_bytes() == b"abc"
    assert artifact.sha256 == expected
    assert artifact.path.with_name("out.bin.sha256").read_text().strip() == expected
    assert not list((tmp_path / "drive").glob("*.partial"))


def test_verify_artifact_rejects_tampering(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"abc")
    artifact = persist_verified(source, tmp_path / "out.bin")
    artifact.path.write_bytes(b"changed")

    assert not verify_artifact(artifact.path)


def test_persist_verified_does_not_replace_invalid_destination(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"new")
    destination = tmp_path / "out.bin"
    destination.write_bytes(b"old")

    with pytest.raises(FileExistsError, match="destination exists"):
        persist_verified(source, destination)

    assert destination.read_bytes() == b"old"


def test_state_store_restores_backup_and_resets_stale_running(tmp_path):
    store = StateStore(tmp_path / "state.json", "aic25-b1", "1")
    state = store.load()
    store.mark(state, "L21_a", "siglip", "000000", "running")
    store.save(state)
    store.mark(state, "L21_a", "siglip", "000000", "done")
    store.save(state)
    (tmp_path / "state.json").write_text("broken")

    restored = store.load()

    shard = restored["batches"]["L21_a"]["stages"]["siglip"]["shards"]["000000"]
    assert shard["status"] == "running"
    store.reset_stale_running(restored)
    assert shard["status"] == "pending"


def test_state_store_rejects_wrong_dataset(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bundle_version": "1",
                "dataset_id": "wrong",
                "batches": {},
            }
        )
    )

    with pytest.raises(ValueError, match="dataset"):
        StateStore(path, "aic25-b1", "1").load()
