# model/OpenCLIP_embedding_optimized.py
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from functools import lru_cache
import hashlib
from typing import Union, Optional
import io
import time

class MultimodalEmbeddingOpenCLIP:
    def __init__(self, fp16: bool = True, cache_size: int = 1000):
        self.dtype = torch.float16 if fp16 else torch.float32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 🚀 OPTIMIZATION 1: Model optimization
        print(f"Loading optimized CLIP model on {self.device}...")
        start_time = time.time()

        self.model = CLIPModel.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            torch_dtype=self.dtype,
            device_map="auto",  # Auto device mapping
            low_cpu_mem_usage=True,  # Reduce CPU memory usage
        ).to(self.device)

        # 🚀 OPTIMIZATION 2: Model compilation (PyTorch 2.0+)
        if hasattr(torch, 'compile') and torch.cuda.is_available():
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("✅ Model compiled with PyTorch 2.0")
            except:
                print("⚠️ PyTorch compilation not available")

        self.model.eval()

        self.processor = CLIPProcessor.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        )

        print(f"✅ Model loaded in {time.time() - start_time:.2f}s on {self.device} (dtype={self.dtype})")

        # 🚀 OPTIMIZATION 3: Advanced caching
        self._text_cache = {}
        self._image_cache = {}
        self.cache_size = cache_size

        # 🚀 OPTIMIZATION 4: Batch processing setup
        self.max_batch_size = 8

        # 🚀 OPTIMIZATION 5: Memory optimization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_cache_key(self, data: Union[str, bytes]) -> str:
        """Generate cache key for text or image data"""
        if isinstance(data, str):
            return hashlib.md5(data.encode()).hexdigest()
        else:
            return hashlib.md5(data).hexdigest()

    def _cleanup_cache(self, cache_dict: dict):
        """Clean up cache if it gets too large"""
        if len(cache_dict) > self.cache_size:
            # Remove oldest 20% of entries
            items_to_remove = len(cache_dict) // 5
            keys_to_remove = list(cache_dict.keys())[:items_to_remove]
            for key in keys_to_remove:
                del cache_dict[key]

    @torch.inference_mode()
    def _l2_normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Optimized L2 normalization"""
        return F.normalize(tensor, p=2, dim=-1, eps=1e-8)

    @torch.inference_mode()
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Optimized text embedding with caching"""
        # 🚀 OPTIMIZATION 6: Smart caching
        cache_key = self._get_cache_key(text)
        if cache_key in self._text_cache:
            return self._text_cache[cache_key]

        # 🚀 OPTIMIZATION 7: Optimized preprocessing
        with torch.cuda.amp.autocast(enabled=self.dtype == torch.float16):
            inputs = self.processor(
                text=text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77  # CLIP max length
            ).to(self.device, non_blocking=True)

            # 🚀 OPTIMIZATION 8: Efficient inference
            feats = self.model.get_text_features(**inputs)
            feats = self._l2_normalize(feats)

            # Convert to numpy efficiently
            result = feats.squeeze(0).cpu().numpy()

        # Cache result
        self._text_cache[cache_key] = result
        self._cleanup_cache(self._text_cache)

        return result

    @torch.inference_mode()
    def get_image_embedding(self, image_input: Union[str, Image.Image, bytes, io.BytesIO]) -> np.ndarray:
        """Highly optimized image embedding with multiple input types"""

        # 🚀 OPTIMIZATION 9: Smart image caching
        if isinstance(image_input, str):
            # File path
            with open(image_input, 'rb') as f:
                image_bytes = f.read()
            cache_key = self._get_cache_key(image_bytes)
        elif hasattr(image_input, 'read'):
            # BytesIO or similar
            if hasattr(image_input, 'getvalue'):
                image_bytes = image_input.getvalue()
            else:
                image_bytes = image_input.read()
                image_input.seek(0)  # Reset for reuse
            cache_key = self._get_cache_key(image_bytes)
        elif isinstance(image_input, bytes):
            image_bytes = image_input
            cache_key = self._get_cache_key(image_bytes)
        else:
            # PIL Image - convert to bytes for caching
            img_buffer = io.BytesIO()
            if hasattr(image_input, 'save'):
                image_input.save(img_buffer, format='PNG')
                image_bytes = img_buffer.getvalue()
                cache_key = self._get_cache_key(image_bytes)
            else:
                cache_key = None  # Skip caching for unknown types

        # Check cache
        if cache_key and cache_key in self._image_cache:
            return self._image_cache[cache_key]

        # 🚀 OPTIMIZATION 10: Efficient image processing
        try:
            if isinstance(image_input, str):
                image = Image.open(image_input).convert("RGB")
            elif hasattr(image_input, 'convert'):
                image = image_input.convert("RGB")
            else:
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # 🚀 OPTIMIZATION 11: Smart resizing
            # Resize large images to reduce memory and computation
            max_size = 512
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)

        except Exception as e:
            print(f"Error processing image: {e}")
            raise

        # 🚀 OPTIMIZATION 12: Optimized inference
        with torch.cuda.amp.autocast(enabled=self.dtype == torch.float16):
            inputs = self.processor(
                images=image,
                return_tensors="pt"
            ).to(self.device, non_blocking=True)

            feats = self.model.get_image_features(**inputs)
            feats = self._l2_normalize(feats)

            result = feats.squeeze(0).cpu().numpy()

        # Cache result
        if cache_key:
            self._image_cache[cache_key] = result
            self._cleanup_cache(self._image_cache)

        return result

    @torch.inference_mode()
    def get_batch_text_embeddings(self, texts: list) -> np.ndarray:
        """Batch text embedding for better efficiency"""
        if len(texts) == 1:
            return self.get_text_embedding(texts[0]).reshape(1, -1)

        # Process in batches
        all_embeddings = []
        for i in range(0, len(texts), self.max_batch_size):
            batch_texts = texts[i:i + self.max_batch_size]

            with torch.cuda.amp.autocast(enabled=self.dtype == torch.float16):
                inputs = self.processor(
                    text=batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77
                ).to(self.device, non_blocking=True)

                feats = self.model.get_text_features(**inputs)
                feats = self._l2_normalize(feats)

                batch_embeddings = feats.cpu().numpy()
                all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings)

    def clear_cache(self):
        """Clear all caches"""
        self._text_cache.clear()
        self._image_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        return {
            'text_cache_size': len(self._text_cache),
            'image_cache_size': len(self._image_cache),
            'memory_allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            'memory_cached': torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        }

# 🚀 OPTIMIZATION 13: Singleton pattern with lazy loading
_embedding_model = None

def get_embedding_model() -> MultimodalEmbeddingOpenCLIP:
    """Get singleton embedding model instance"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = MultimodalEmbeddingOpenCLIP()
    return _embedding_model

# For backward compatibility
embedding_model = get_embedding_model()

if __name__ == "__main__":
    # Performance test
    import os, time, numpy as np

    # ==== cấu hình nhanh (sửa 3 dòng dưới) ====
    IMAGES_DIR = "/media/tan/F/AI_Challenge_2025/data/small_img_test"          # thư mục ảnh
    QUERY      = "panda"     # câu truy vấn
    EXTS       = {"jpg", "jpeg", "png", "webp"}
    # ==========================================

    # gom danh sách ảnh (không đệ quy)
    image_paths = []
    for fname in sorted(os.listdir(IMAGES_DIR)):
        p = os.path.join(IMAGES_DIR, fname)
        if os.path.isfile(p) and "." in fname and fname.rsplit(".", 1)[-1].lower() in EXTS:
            image_paths.append(p)
    if not image_paths:
        raise SystemExit(f"❌ Không có ảnh trong: {IMAGES_DIR}")

    print(f"📁 Thư mục: {IMAGES_DIR} | Ảnh: {len(image_paths)}")
    print(f"🔎 Query  : {QUERY}\n")

    # ---- text embedding (1 lần) ----
    t0 = time.perf_counter()
    text_emb = embedding_model.get_text_embedding(QUERY)   # (D,)
    t1 = time.perf_counter()
    print(f"⏱️ text_emb: {(t1 - t0)*1000:.1f} ms\n")

    # ---- duyệt từng ảnh và tính similarity ----
    results = []
    for i, path in enumerate(image_paths, 1):
        s0 = time.perf_counter()
        img_emb  = embedding_model.get_image_embedding(path)   # (D,)
        score    = float(text_emb @ img_emb)                   # cosine đã chuẩn hoá
        s1 = time.perf_counter()
        results.append((path, score))
        print(f"[{i:>4}/{len(image_paths)}] {os.path.basename(path)}  sim={score:.4f}  ({(s1 - s0)*1000:.1f} ms)")

    # ---- in Top-k ----
    TOPK = min(10, len(results))
    results.sort(key=lambda x: x[1], reverse=True)
    print("\n🏆 Top kết quả:")
    for r, (p, s) in enumerate(results[:TOPK], 1):
        print(f"{r:>2}. {os.path.basename(p)}   sim={s:.4f}")