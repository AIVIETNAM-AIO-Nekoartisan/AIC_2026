from __future__ import annotations

import gc
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
import torch


class ImageEncoder(Protocol):
    name: str
    dimension: int

    def encode(self, images: Sequence[Any]) -> np.ndarray: ...


def _normalized_float32(values: Any, dimension: int) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().float().cpu().numpy()
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != dimension:
        raise ValueError(f"encoder returned {array.shape}, expected (*, {dimension})")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if not np.all(np.isfinite(norms)) or np.any(norms == 0):
        raise ValueError("encoder returned non-finite or zero vectors")
    return np.ascontiguousarray(array / norms, dtype=np.float32)


def encode_with_oom_backoff(
    encoder: ImageEncoder,
    images: Sequence[Any],
    initial_batch_size: int,
    max_reductions: int = 3,
) -> tuple[np.ndarray, int]:
    if initial_batch_size < 1:
        raise ValueError("initial_batch_size must be positive")
    if not images:
        return np.empty((0, encoder.dimension), dtype=np.float32), initial_batch_size
    batch_size = min(initial_batch_size, len(images))
    reductions = 0
    while True:
        output: list[np.ndarray] = []
        try:
            for start in range(0, len(images), batch_size):
                chunk = encoder.encode(images[start : start + batch_size])
                chunk = np.asarray(chunk, dtype=np.float32)
                if chunk.shape != (
                    min(batch_size, len(images) - start),
                    encoder.dimension,
                ):
                    raise ValueError(
                        f"{encoder.name} returned invalid batch shape {chunk.shape}"
                    )
                output.append(chunk)
            return np.concatenate(output, axis=0), batch_size
        except torch.cuda.OutOfMemoryError:
            output.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if batch_size == 1 or reductions >= max_reductions:
                raise
            batch_size = max(1, batch_size // 2)
            reductions += 1


class SiglipEncoder:
    name = "siglip"

    def __init__(
        self,
        model_id: str = "google/siglip2-so400m-patch14-384",
        dimension: int = 1152,
        cache_dir: str | None = None,
        revision: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.dimension = dimension
        self.cache_dir = cache_dir
        self.revision = revision
        self._model: Any = None
        self._processor: Any = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoProcessor

        kwargs = {"cache_dir": self.cache_dir, "revision": self.revision}
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        self._processor = AutoProcessor.from_pretrained(self.model_id, **kwargs)
        dtype = torch.float16 if self._device == "cuda" else torch.float32
        self._model = AutoModel.from_pretrained(
            self.model_id, torch_dtype=dtype, **kwargs
        ).to(self._device)
        self._model.eval()

    def encode(self, images: Sequence[Any]) -> np.ndarray:
        self._load()
        inputs = self._processor(images=list(images), return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with torch.inference_mode():
            features = self._model.get_image_features(**inputs)
        return _normalized_float32(features, self.dimension)

    def close(self) -> None:
        self._model = None
        self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class JinaEncoder:
    name = "jina"

    def __init__(
        self,
        model_id: str = "jinaai/jina-clip-v2",
        dimension: int = 1024,
        cache_dir: str | None = None,
        revision: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.dimension = dimension
        self.cache_dir = cache_dir
        self.revision = revision
        self._model: Any = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel

        kwargs = {
            "cache_dir": self.cache_dir,
            "revision": self.revision,
            "trust_remote_code": True,
        }
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        self._model = AutoModel.from_pretrained(self.model_id, **kwargs).to(self._device)
        if self._device == "cuda":
            self._model.half()
        self._model.eval()

    def encode(self, images: Sequence[Any]) -> np.ndarray:
        self._load()
        with torch.inference_mode():
            features = self._model.encode_image(list(images))
        return _normalized_float32(features, self.dimension)

    def close(self) -> None:
        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
