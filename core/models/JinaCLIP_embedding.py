# model/JinaCLIP_embedding.py
import io
import hashlib
import time
from typing import Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel


class MultimodalEmbeddingJinaCLIP:
    """
    Wrapper cho jinaai/jina-clip-v2 (đa ngôn ngữ, hỗ trợ tiếng Việt trực tiếp,
    text tối đa 8192 token — không bị giới hạn 64 token như SigLIP).
    Xuất vector 1024 chiều, đã L2-normalize, API giống MultimodalEmbeddingSigLIP.
    """

    DIM = 1024

    def __init__(self, fp16: bool = True, cache_size: int = 1000,
                 model_name: str = "jinaai/jina-clip-v2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if (fp16 and self.device.type == "cuda") else torch.float32

        print(f"Loading Jina-CLIP-v2 on {self.device} (dtype={self.dtype})...")
        start_time = time.time()
        # Không truyền torch_dtype: custom code của jina (viết cho transformers 4.x)
        # crash khi nhận torch.dtype object từ transformers 5.x — load xong cast sau
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(self.device, dtype=self.dtype).eval()
        print(f"✅ Jina-CLIP-v2 loaded in {time.time() - start_time:.2f}s")

        self._text_cache = {}
        self._image_cache = {}
        self.cache_size = cache_size

    @staticmethod
    def _get_cache_key(data: Union[str, bytes]) -> str:
        if isinstance(data, str):
            data = data.encode()
        return hashlib.md5(data).hexdigest()

    def _cleanup_cache(self, cache_dict: dict):
        if len(cache_dict) > self.cache_size:
            for key in list(cache_dict.keys())[:len(cache_dict) // 5]:
                del cache_dict[key]

    @staticmethod
    def _l2(v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)

    @torch.inference_mode()
    def get_text_embedding(self, text: str) -> np.ndarray:
        cache_key = self._get_cache_key(text)
        if cache_key in self._text_cache:
            return self._text_cache[cache_key]

        # task="retrieval.query" là LoRA adapter cho câu truy vấn ngắn
        vec = self.model.encode_text([text], task="retrieval.query")
        result = self._l2(np.asarray(vec, dtype=np.float32))[0]

        self._text_cache[cache_key] = result
        self._cleanup_cache(self._text_cache)
        return result

    @torch.inference_mode()
    def get_batch_text_embeddings(self, texts: list) -> np.ndarray:
        if not texts:
            return np.array([])
        vecs = self.model.encode_text(texts, task="retrieval.query",
                                      batch_size=min(8, len(texts)))
        return self._l2(np.asarray(vecs, dtype=np.float32))

    @torch.inference_mode()
    def get_image_embedding(self, image_input: Union[str, Image.Image, bytes, io.BytesIO]) -> np.ndarray:
        cache_key = None
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
            cache_key = self._get_cache_key(image_input)
        elif isinstance(image_input, bytes):
            image = Image.open(io.BytesIO(image_input)).convert("RGB")
            cache_key = self._get_cache_key(image_input)
        elif hasattr(image_input, "read"):
            data = image_input.getvalue() if hasattr(image_input, "getvalue") else image_input.read()
            image = Image.open(io.BytesIO(data)).convert("RGB")
            cache_key = self._get_cache_key(data)
        else:
            image = image_input.convert("RGB")

        if cache_key and cache_key in self._image_cache:
            return self._image_cache[cache_key]

        vec = self.model.encode_image([image])
        result = self._l2(np.asarray(vec, dtype=np.float32))[0]

        if cache_key:
            self._image_cache[cache_key] = result
            self._cleanup_cache(self._image_cache)
        return result

    @torch.inference_mode()
    def get_batch_image_embeddings(
        self,
        image_inputs: list[Union[str, Image.Image]],
        batch_size: int = 4,
    ) -> np.ndarray:
        """Embed local images in one Jina call for offline ingestion."""
        if not image_inputs:
            return np.empty((0, self.DIM), dtype=np.float32)

        images = []
        for image_input in image_inputs:
            if isinstance(image_input, str):
                image = Image.open(image_input).convert("RGB")
            else:
                image = image_input.convert("RGB")
            images.append(image)

        vectors = self.model.encode_image(
            images,
            batch_size=max(1, min(int(batch_size), len(images))),
            show_progress_bar=False,
        )
        return self._l2(np.asarray(vectors, dtype=np.float32))

    def clear_cache(self):
        self._text_cache.clear()
        self._image_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


_jina_model = None

def get_jina_model() -> MultimodalEmbeddingJinaCLIP:
    global _jina_model
    if _jina_model is None:
        _jina_model = MultimodalEmbeddingJinaCLIP()
    return _jina_model


if __name__ == "__main__":
    m = get_jina_model()
    t = m.get_text_embedding("người dẫn chương trình thời sự đang ngồi trong trường quay")
    print("text vec:", t.shape, "norm:", np.linalg.norm(t))
    import glob
    imgs = glob.glob("results/test_1/*.webp")[:2]
    for p in imgs:
        v = m.get_image_embedding(p)
        print(p, "->", v.shape, "cos:", float(np.dot(t, v)))
