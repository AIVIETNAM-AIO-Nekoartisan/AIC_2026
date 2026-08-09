# model/SigLIP_embedding_optimized.py
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, AutoModel
import numpy as np
from functools import lru_cache
import hashlib
from typing import Union, Optional
import io
import time
import requests

class MultimodalEmbeddingSigLIP:
    def __init__(self, fp16: bool = True, cache_size: int = 1000):
        self.dtype = torch.float16 if fp16 else torch.float32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 🚀 OPTIMIZATION 1: Model optimization
        print(f"Loading optimized SigLIP model on {self.device}...")
        start_time = time.time()

        # SigLIP2: tokenizer Gemma đa ngôn ngữ (hiểu tiếng Việt tốt hơn v1),
        # cùng dim 1152. Text tối đa 64 token (giới hạn positional embedding
        # của kiến trúc SigLIP — luôn truyền max_length=64 để dùng đủ, không bị
        # truncate sớm hơn; câu dài hơn 64 token đã có Jina-CLIP-v2 xử lý).
        # LƯU Ý: SigLIP2 train trên text đã lowercase + pad đủ 64 token
        # (tokenizer Gemma phân biệt hoa/thường, không tự lowercase như v1) —
        # thiếu 1 trong 2 điều kiện là recall giảm mạnh (transformers#43054)
        model_name = "google/siglip2-so400m-patch14-384"
        self.max_text_tokens = 64

        self.model = AutoModel.from_pretrained(
            model_name,
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
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

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

    @staticmethod
    def _feats_tensor(feats):
        """transformers mới trả ModelOutput thay vì tensor từ get_*_features"""
        return feats.pooler_output if hasattr(feats, "pooler_output") else feats

    @torch.inference_mode()
    def _embed_long_text_windows(self, text: str) -> np.ndarray:
        """
        Câu dài hơn max_text_tokens: positional embedding của SigLIP chỉ có 64 vị trí
        nên không thể đưa thẳng câu dài vào. Thay vì truncate mất thông tin, chia
        token thành các cửa sổ 64 chồng lấp (stride 48), embed từng cửa sổ rồi
        average + L2-normalize — câu ngắn không đi qua đường này nên không đổi chất lượng.
        """
        tok = self.processor.tokenizer
        ids = tok(text, truncation=False, padding=False)["input_ids"]
        win, stride = self.max_text_tokens, self.max_text_tokens - 16
        chunks = [ids[i:i + win] for i in range(0, max(len(ids) - win, 0) + stride, stride)]
        chunks = [c for c in chunks if c] or [ids[:win]]

        pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        batch = torch.full((len(chunks), win), pad_id, dtype=torch.long)
        for r, c in enumerate(chunks):
            batch[r, :len(c)] = torch.tensor(c, dtype=torch.long)

        with torch.amp.autocast('cuda', enabled=self.dtype == torch.float16):
            feats = self._feats_tensor(self.model.get_text_features(input_ids=batch.to(self.device)))
            feats = self._l2_normalize(feats)                 # normalize từng cửa sổ
            mean = self._l2_normalize(feats.mean(dim=0))       # rồi average + normalize lại
        return mean.cpu().numpy()

    @torch.inference_mode()
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Optimized text embedding with caching"""
        # 🚀 OPTIMIZATION 6: Smart caching
        text = text.lower()  # SigLIP2 train trên text lowercase
        cache_key = self._get_cache_key(text)
        if cache_key in self._text_cache:
            return self._text_cache[cache_key]

        # Câu vượt 64 token -> chia cửa sổ + average thay vì truncate
        n_tokens = len(self.processor.tokenizer(text, truncation=False, padding=False)["input_ids"])
        if n_tokens > self.max_text_tokens:
            result = self._embed_long_text_windows(text)
            self._text_cache[cache_key] = result
            self._cleanup_cache(self._text_cache)
            return result

        # 🚀 OPTIMIZATION 7: Optimized preprocessing
        with torch.amp.autocast('cuda', enabled=self.dtype == torch.float16):
            inputs = self.processor(
                text=text,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_text_tokens,
                truncation=True
            ).to(self.device, non_blocking=True)

            # 🚀 OPTIMIZATION 8: Efficient inference
            feats = self._feats_tensor(self.model.get_text_features(**inputs))
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
            # File path or URL
            if image_input.startswith('http'):
                response = requests.get(image_input, stream=True)
                image_bytes = response.content
            else:
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
                if image_input.startswith('http'):
                    image = Image.open(requests.get(image_input, stream=True).raw).convert("RGB")
                else:
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
        with torch.amp.autocast('cuda', enabled=self.dtype == torch.float16):
            inputs = self.processor(
                images=image,
                return_tensors="pt"
            ).to(self.device, non_blocking=True)

            feats = self._feats_tensor(self.model.get_image_features(**inputs))
            feats = self._l2_normalize(feats)

            result = feats.squeeze(0).cpu().numpy()

        # Cache result
        if cache_key:
            self._image_cache[cache_key] = result
            self._cleanup_cache(self._image_cache)

        return result

    @torch.inference_mode()
    def get_batch_image_embeddings(
        self,
        image_inputs: list[Union[str, Image.Image]],
    ) -> np.ndarray:
        """Embed a batch of local images without filling the per-image cache."""
        if not image_inputs:
            return np.empty((0, 1152), dtype=np.float32)

        images = []
        for image_input in image_inputs:
            if isinstance(image_input, str):
                image = Image.open(image_input).convert("RGB")
            else:
                image = image_input.convert("RGB")
            images.append(image)

        with torch.amp.autocast("cuda", enabled=self.dtype == torch.float16):
            inputs = self.processor(images=images, return_tensors="pt").to(
                self.device,
                non_blocking=True,
            )
            features = self._feats_tensor(self.model.get_image_features(**inputs))
            features = self._l2_normalize(features)
        return features.cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def get_batch_text_embeddings(self, texts: list) -> np.ndarray:
        """
        Batch text embedding - FAST VERSION
        """
        if not texts:
            return np.array([])

        if len(texts) == 1:
            return self.get_text_embedding(texts[0]).reshape(1, -1)

        texts = [q.lower() for q in texts]  # SigLIP2 train trên text lowercase

        # Có câu vượt 64 token -> đi từng câu để dùng windowing (có cache sẵn)
        tok = self.processor.tokenizer
        if any(len(tok(q, truncation=False, padding=False)["input_ids"]) > self.max_text_tokens for q in texts):
            return np.vstack([self.get_text_embedding(q) for q in texts])

        # Process in batches of 8
        batch_size = 8
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # Process batch — SigLIP train với padding "max_length" nên phải giữ
            # nguyên khi inference, padding động (True) làm lệch embedding
            inputs = self.processor(
                text=batch,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_text_tokens,
                truncation=True
            ).to(self.device)

            with torch.amp.autocast('cuda', enabled=True):
                feats = self._feats_tensor(self.model.get_text_features(**inputs))
                feats = F.normalize(feats, p=2, dim=-1)

            all_embeddings.append(feats.cpu().numpy())

        return np.vstack(all_embeddings)

    @torch.inference_mode()
    def compute_similarity(self, texts: list, image: Union[str, Image.Image]) -> np.ndarray:
        """Compute SigLIP similarities between texts and image"""
        # Get embeddings
        text_embeddings = self.get_batch_text_embeddings(texts)
        image_embedding = self.get_image_embedding(image)

        # Compute cosine similarities
        similarities = np.dot(text_embeddings, image_embedding)

        # Convert to SigLIP probabilities using sigmoid
        probabilities = 1 / (1 + np.exp(-similarities))

        return probabilities

    def find_best_match(self, texts: list, image: Union[str, Image.Image]) -> tuple:
        """Find the best matching text for an image"""
        scores = self.compute_similarity(texts, image)

        best_idx = np.argmax(scores)
        best_text = texts[best_idx]
        best_score = scores[best_idx]

        return best_text, best_score, scores

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

def get_embedding_model() -> MultimodalEmbeddingSigLIP:
    """Get singleton embedding model instance"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = MultimodalEmbeddingSigLIP()
    return _embedding_model

class _LazyEmbeddingModel:
    """Backward-compatible proxy that avoids loading weights during import."""

    def __getattr__(self, name):
        return getattr(get_embedding_model(), name)


# For backward compatibility. Accessing an attribute triggers the actual load.
embedding_model = _LazyEmbeddingModel()

if __name__ == "__main__":
    # Simple embedding test like OpenCLIP
    model = get_embedding_model()

    # Test 1: Simple text-image similarity
    print("=== SigLIP Embedding Test ===")
    text_emb = model.get_text_embedding("a photo of a red bus")
    img_emb = model.get_image_embedding("bus.jpg")

    score = np.dot(text_emb, img_emb) / (np.linalg.norm(text_emb) * np.linalg.norm(img_emb))
    print(f"Similarity: {score:.4f}")

    # Test 2: Multiple queries
    print("\n=== Multiple Queries Test ===")
    queries = [
        "a photo of a red bus",
        "a photo of a cat",
        "a photo of a dog",
        "con gấu mèo đỏ"
    ]

    for query in queries:
        text_emb = model.get_text_embedding(query)
        score = np.dot(text_emb, img_emb) / (np.linalg.norm(text_emb) * np.linalg.norm(img_emb))
        print(f"'{query}': {score:.4f}")

    # Test 3: Different images
    print("\n=== Different Images Test ===")
    images = ["bus.jpg", "example.jpeg"]  # Add your image paths
    query = "a photo of a red bus"
    text_emb = model.get_text_embedding(query)

    for img_path in images:
        try:
            img_emb = model.get_image_embedding(img_path)
            score = np.dot(text_emb, img_emb) / (np.linalg.norm(text_emb) * np.linalg.norm(img_emb))
            print(f"'{img_path}': {score:.4f}")
        except:
            print(f"'{img_path}': [not found]")

    print(f"\nCache stats: {model.get_cache_stats()}")
