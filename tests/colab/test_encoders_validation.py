import numpy as np
import pytest
import torch

from aic_colab.encoders import encode_with_oom_backoff
from aic_colab.validation import validate_embedding_shard, write_embedding_shard


class ThresholdEncoder:
    name = "fake"
    dimension = 4

    def encode(self, images):
        if len(images) > 2:
            raise torch.cuda.OutOfMemoryError("test threshold")
        values = np.zeros((len(images), 4), dtype=np.float32)
        values[:, 0] = 1.0
        return values


def test_oom_backoff_halves_batch_and_returns_every_row():
    vectors, used = encode_with_oom_backoff(
        ThresholdEncoder(), list(range(5)), initial_batch_size=4
    )

    assert vectors.shape == (5, 4)
    assert used == 2
    assert np.all(vectors[:, 0] == 1.0)


class AlwaysOOM:
    name = "fake"
    dimension = 4

    def encode(self, images):
        raise torch.cuda.OutOfMemoryError("always")


def test_oom_backoff_stops_after_three_reductions():
    with pytest.raises(torch.cuda.OutOfMemoryError):
        encode_with_oom_backoff(AlwaysOOM(), [1], initial_batch_size=8)


def test_embedding_validation_accepts_exact_schema(tmp_path):
    ids = ["a", "b"]
    vectors = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)

    path = write_embedding_shard(ids, vectors, tmp_path / "x.npz")
    result = validate_embedding_shard(path, ids, expected_dim=4)

    assert result.passed
    assert result.count == 2
    with np.load(path) as archive:
        assert set(archive.files) == {"point_ids", "vectors"}


def test_embedding_validation_rejects_nan(tmp_path):
    vectors = np.array([[1, 0, 0, 0], [np.nan, 1, 0, 0]], dtype=np.float32)
    path = write_embedding_shard(["a", "b"], vectors, tmp_path / "x.npz")

    result = validate_embedding_shard(path, ["a", "b"], expected_dim=4)

    assert not result.passed
    assert any("NaN or infinite" in error for error in result.errors)


def test_embedding_validation_rejects_wrong_order_dtype_and_norm(tmp_path):
    path = tmp_path / "bad.npz"
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            point_ids=np.asarray(["b", "a"]),
            vectors=np.asarray([[2, 0], [0, 2]], dtype=np.float64),
        )

    result = validate_embedding_shard(path, ["a", "b"], expected_dim=2)

    assert not result.passed
    assert any("point ID order" in error for error in result.errors)
    assert any("float32" in error for error in result.errors)
    assert any("L2-normalized" in error for error in result.errors)
