# model/BEiT3_embedding_optimized.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from timm.models.registry import register_model
from torchscale.model.BEiT3 import BEiT3
from torchscale.architecture.config import EncoderConfig
from transformers import XLMRobertaTokenizer   # .spm tokenizer (slow), no fallback
import timm
import math
import hashlib
import io
import time
from functools import lru_cache
from typing import Union, Optional, List, Tuple
import warnings


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


def _get_base_config(
    img_size=224, patch_size=16, drop_path_rate=0,
    checkpoint_activations=None, mlp_ratio=4, vocab_size=64010, **kwargs
):
    return EncoderConfig(
        img_size=img_size, patch_size=patch_size, vocab_size=vocab_size, multiway=True,
        layernorm_embedding=False, normalize_output=True, no_output_layer=True,
        drop_path_rate=drop_path_rate, encoder_embed_dim=768, encoder_attention_heads=12,
        encoder_ffn_embed_dim=int(768 * mlp_ratio), encoder_layers=12,
        checkpoint_activations=checkpoint_activations,
    )


def _get_large_config(
    img_size=224, patch_size=16, drop_path_rate=0,
    checkpoint_activations=None, mlp_ratio=4, vocab_size=64010, **kwargs
):
    return EncoderConfig(
        img_size=img_size, patch_size=patch_size, vocab_size=vocab_size, multiway=True,
        layernorm_embedding=False, normalize_output=True, no_output_layer=True,
        drop_path_rate=drop_path_rate, encoder_embed_dim=1024, encoder_attention_heads=16,
        encoder_ffn_embed_dim=int(1024 * mlp_ratio), encoder_layers=24,
        checkpoint_activations=checkpoint_activations,
    )


class BEiT3Wrapper(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        self.beit3 = BEiT3(args)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'beit3.encoder.embed_positions.A.weight', 'beit3.vision_embed.cls_token', 'logit_scale'}


class BEiT3ForRetrieval(BEiT3Wrapper):
    def __init__(self, args, **kwargs):
        super(BEiT3ForRetrieval, self).__init__(args=args)
        embed_dim = args.encoder_embed_dim
        self.language_head = nn.Linear(embed_dim, embed_dim, bias=False)
        self.vision_head = nn.Linear(embed_dim, embed_dim, bias=False)
        self.language_head.apply(self._init_weights)
        self.vision_head.apply(self._init_weights)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image=None, text_description=None, padding_mask=None, only_infer=True, **kwargs):
        if image is not None:
            outputs = self.beit3(
                textual_tokens=None,
                visual_tokens=image,
                text_padding_position=None,
            )
            x = outputs["encoder_out"]
            vision_cls = self.vision_head(x[:, 0, :])
            vision_cls = F.normalize(vision_cls, dim=-1)
        else:
            vision_cls = None

        if text_description is not None:
            outputs = self.beit3(
                textual_tokens=text_description,
                visual_tokens=None,
                text_padding_position=padding_mask,
            )
            x = outputs["encoder_out"]
            language_cls = self.language_head(x[:, 0, :])
            language_cls = F.normalize(language_cls, dim=-1)
        else:
            language_cls = None

        return vision_cls, language_cls


@register_model
def beit3_large_patch16_384_retrieval(pretrained=False, **kwargs):
    args = _get_large_config(img_size=384, **kwargs)
    model = BEiT3ForRetrieval(args, **kwargs)
    return model


@register_model
def beit3_base_patch16_384_retrieval(pretrained=False, **kwargs):
    args = _get_base_config(img_size=384, **kwargs)
    model = BEiT3ForRetrieval(args, **kwargs)
    return model


class MultimodalEmbeddingBEiT3:
    def __init__(self,
                 model_name: str = "beit3_large_patch16_384_retrieval",
                 checkpoint_path: str = None,
                 tokenizer_path: str = None,
                 device: str = None,
                 fp16: bool = True,
                 input_size: int = 384,
                 warmup: bool = False,
                 aggressive_optimization: bool = False):

        requested_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("BEiT-3 requested CUDA but CUDA is not available")
        self.device = torch.device(requested_device)
        self.dtype = (
            torch.float16
            if fp16 and self.device.type == "cuda"
            else torch.float32
        )
        try:
            from torch.backends.cuda import sdp_kernel
            sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
        except Exception:
            pass
        self.input_size = input_size
        self.aggressive_optimization = aggressive_optimization

        # 🚀 OPTIMIZATION 1: Model loading optimization with warnings suppressed
        print(f"Loading optimized BEiT3 model on {self.device}...")
        start_time = time.time()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self.model = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=0,  # Remove classification head
                drop_path_rate=0.0  # Disable drop path for inference
            ).to(self.device)

            # Load checkpoint if provided
            if checkpoint_path:
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(checkpoint['model'], strict=False)

        # Set to eval mode and freeze parameters
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # 🚀 OPTIMIZATION 2: Advanced PyTorch compilation
        if (
            hasattr(torch, 'compile')
            and self.device.type == "cuda"
            and aggressive_optimization
        ):
            try:
                gpu_props = torch.cuda.get_device_properties(0)
                sm_count = gpu_props.multi_processor_count

                if sm_count >= 80:  # A100, H100, RTX 4090
                    self.model = torch.compile(
                        self.model,
                        mode="max-autotune-no-cudagraphs",
                        dynamic=False,
                        fullgraph=True,
                        backend="inductor"
                    )
                    print(f"✅ Ultra compilation (max-autotune-no-cudagraphs) - {sm_count} SMs")
                elif sm_count >= 40:  # RTX 3080, RTX 4080
                    self.model = torch.compile(
                        self.model,
                        mode="reduce-overhead",
                        dynamic=False,
                        fullgraph=True
                    )
                    print(f"✅ Advanced compilation (reduce-overhead) - {sm_count} SMs")
                else:
                    self.model = torch.compile(self.model, mode="default")
                    print(f"✅ Standard compilation - {sm_count} SMs")

            except Exception as e:
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    print("✅ Model compiled with PyTorch 2.0 (reduce-overhead)")
                except:
                    print("⚠️ PyTorch compilation not available")

        # ✅ Load tokenizer (.spm) — no fallback
        if tokenizer_path:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # XLMRobertaTokenizer accepts path to sentencepiece model (.spm)
                self.tokenizer = XLMRobertaTokenizer(tokenizer_path)
            self.bos_token_id = self.tokenizer.bos_token_id
            self.eos_token_id = self.tokenizer.eos_token_id
            self.pad_token_id = self.tokenizer.pad_token_id
        else:
            print("⚠️ No tokenizer path provided, text embedding will not work")
            self.tokenizer = None
            self.bos_token_id = None
            self.eos_token_id = None
            self.pad_token_id = None

        print(f"✅ Model loaded in {time.time() - start_time:.2f}s on {self.device} (dtype={self.dtype})")

        # 🚀 OPTIMIZATION 3: Pre-setup transforms and tensors
        self._setup_optimized_transforms()
        self._pre_allocate_tensors()

        # 🚀 OPTIMIZATION 4: Advanced memory and CUDA optimizations
        self.max_batch_size = 32
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

            # Advanced CUDA optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # Pre-allocate memory pools
            torch.cuda.memory.set_per_process_memory_fraction(0.95)

            # Comprehensive warmup if requested
            if warmup:
                self._warmup_model()

    def _setup_optimized_transforms(self):
        """Pre-setup optimized image transforms"""
        self.image_transform = transforms.Compose([
            transforms.Resize(  # ✅ (5) dùng BILINEAR để nhanh hơn
                (self.input_size, self.input_size),
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ])

    def _pre_allocate_tensors(self):
        """Pre-allocate commonly used tensors"""
        if torch.cuda.is_available():
            self.dummy_text_tokens = torch.zeros(1, 64, dtype=torch.long, device=self.device)
            self.dummy_padding_mask = torch.zeros(1, 64, dtype=torch.long, device=self.device)
            self.dummy_image_tensor = torch.zeros(
                1, 3, self.input_size, self.input_size,
                dtype=self.dtype, device=self.device
            )

    def _warmup_model(self):
        """Comprehensive warmup with multiple passes"""
        print("🔥 Comprehensive warmup (this will improve performance significantly)...")
        try:
            with torch.amp.autocast('cuda', enabled=self.dtype == torch.float16):
                # Multiple warmup passes for better kernel optimization
                for i in range(5):
                    # Image warmup
                    dummy_img = torch.randn(1, 3, self.input_size, self.input_size,
                                          dtype=self.dtype, device=self.device)
                    self.model(image=dummy_img, only_infer=True)

                    # Text warmup (if tokenizer available)
                    if self.tokenizer:
                        dummy_text = torch.randint(1, 1000, (1, 64), device=self.device)
                        dummy_mask = torch.zeros(1, 64, device=self.device)
                        self.model(text_description=dummy_text, padding_mask=dummy_mask, only_infer=True)

                # Additional warmup for different sequence lengths
                if self.tokenizer:
                    for seq_len in [16, 32, 64]:
                        dummy_text = torch.randint(1, 1000, (1, seq_len), device=self.device)
                        dummy_mask = torch.zeros(1, seq_len, device=self.device)
                        self.model(text_description=dummy_text, padding_mask=dummy_mask, only_infer=True)

                torch.cuda.synchronize()

            print("✅ Comprehensive warmup completed")
        except Exception as e:
            print(f"⚠️ Warmup failed: {e}")

        torch.cuda.empty_cache()

    def _get_cache_key(self, data: Union[str, bytes]) -> str:
        """Generate cache key for text or image data"""
        pass  # Removed caching functionality

    def _cleanup_cache(self, cache_dict: dict):
        """Clean up cache if it gets too large"""
        pass  # Removed caching functionality

    def _to_text_tokens(self, text: str, max_len: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert text to tokens with padding mask — using .spm tokenizer's encode_plus
        Trả về:
          - input_ids: (1, L)
          - pad_mask : (1, L) với '1' tại vị trí PAD (phù hợp text_padding_position)
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer not initialized. Please provide path to .spm for XLMRobertaTokenizer.")

        enc = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].to(self.device, non_blocking=True)            # (1, L)
        attention_mask = enc["attention_mask"].to(self.device, non_blocking=True)  # (1, L)

        # Mô hình của bạn dùng '1' cho vị trí PAD → pad_mask = 1 - attention_mask
        pad_mask = (1 - attention_mask).long()                                     # (1, L)
        return input_ids, pad_mask

    def _to_image_tokens(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL image to tensor tokens - ultra optimized version"""
        # Convert to RGB if needed (faster than checking first)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply transforms
        image_tensor = self.image_transform(image)

        # Move to device with non_blocking for better performance
        return image_tensor.unsqueeze(0).to(self.device, dtype=self.dtype, non_blocking=True)

    @torch.inference_mode()
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Get text embedding - optimized version with encode_plus"""
        with torch.amp.autocast('cuda', enabled=self.dtype == torch.float16):
            text_tokens, padding_mask = self._to_text_tokens(text)
            _, text_embedding = self.model(
                text_description=text_tokens,
                padding_mask=padding_mask,
                only_infer=True
            )
            result = text_embedding.squeeze(0).cpu().numpy()
        return result

    @torch.inference_mode()
    def get_image_embedding(self, image_input: Union[str, Image.Image, bytes, io.BytesIO]) -> np.ndarray:
        """Get image embedding - optimized version"""
        # Process image input efficiently
        try:
            if isinstance(image_input, str):
                image = default_loader(image_input)
            elif hasattr(image_input, 'convert'):
                image = image_input
            else:
                image = Image.open(io.BytesIO(image_input))
        except Exception as e:
            print(f"Error processing image: {e}")
            raise

        # Optimized inference
        with torch.amp.autocast('cuda', enabled=self.dtype == torch.float16):
            image_tokens = self._to_image_tokens(image)
            image_embedding, _ = self.model(
                image=image_tokens,
                only_infer=True
            )
            result = image_embedding.squeeze(0).cpu().numpy()
        return result

    @torch.inference_mode()
    def get_batch_image_embeddings(
        self,
        image_paths: List[Union[str, Image.Image]],
    ) -> np.ndarray:
        """Batch image embedding for better efficiency"""
        if len(image_paths) == 1:
            return self.get_image_embedding(image_paths[0]).reshape(1, -1)

        all_embeddings = []
        for i in range(0, len(image_paths), self.max_batch_size):
            batch_paths = image_paths[i:i + self.max_batch_size]

            # Load and transform batch
            batch_images = []
            for image_input in batch_paths:
                image = (
                    default_loader(image_input)
                    if isinstance(image_input, str)
                    else image_input.convert("RGB")
                )
                # Transform on CPU and transfer the completed batch once.
                # The old implementation moved every image to CUDA before
                # stacking, which wasted memory and serialized host-to-device
                # copies.
                batch_images.append(self.image_transform(image))

            batch_tensor = torch.stack(batch_images).to(
                self.device,
                dtype=self.dtype,
                non_blocking=True,
            )

            with torch.amp.autocast('cuda', enabled=self.dtype == torch.float16):
                batch_embeddings, _ = self.model(image=batch_tensor, only_infer=True)
                embeddings_np = batch_embeddings.cpu().numpy()
                all_embeddings.append(embeddings_np)

        return np.vstack(all_embeddings)

    @torch.inference_mode()
    def get_batch_text_embeddings(self, texts: List[str]) -> np.ndarray:
        """Batch text embedding - optimized version"""
        if len(texts) == 1:
            return self.get_text_embedding(texts[0]).reshape(1, -1)

        # Process individually for now (encode_plus đã giúp gọn & nhanh hơn cách thủ công)
        all_embeddings = []
        for text in texts:
            embedding = self.get_text_embedding(text)
            all_embeddings.append(embedding)

        return np.array(all_embeddings)

    def similarity_search(self, query_embedding: np.ndarray,
                         embeddings: List[np.ndarray],
                         k: int = 10) -> List[Tuple[int, float]]:
        """Find top-k most similar embeddings - optimized version"""
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        similarities = []

        for i, embedding in enumerate(embeddings):
            emb_norm = embedding / np.linalg.norm(embedding)
            similarity = float(np.dot(query_norm, emb_norm))
            similarities.append((i, similarity))

        return sorted(similarities, key=lambda x: x[1], reverse=True)[:k]

    def clear_cache(self):
        """Clear all caches - GPU memory cleanup"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        return {
            'memory_allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            'memory_cached': torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        }


# 🚀 OPTIMIZATION: Singleton pattern with lazy loading
_embedding_model = None

def get_embedding_model(checkpoint_path: str = None,
                       tokenizer_path: str = None,
                       **kwargs) -> MultimodalEmbeddingBEiT3:
    """Get singleton embedding model instance"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = MultimodalEmbeddingBEiT3(
            checkpoint_path=checkpoint_path,
            tokenizer_path=tokenizer_path,
            **kwargs
        )
    return _embedding_model


if __name__ == "__main__":
    import os, time, numpy as np

    # ======= CẤU HÌNH TỐI GIẢN (sửa 4 dòng này) =======
    IMAGES_DIR = "/media/tan/F/AI_Challenge_2025/data/small_img_test"   # thư mục ảnh
    CHECKPOINT = "model_weights/beit3_large_patch16_384_coco_retrieval.pth"
    TOKENIZER  = "model_weights/beit3.spm"  # .spm
    QUERY      = "A red panda on a tree"  # câu truy vấn
    # ================================================

    MODEL_NAME = "beit3_large_patch16_384_retrieval"  # hoặc base nếu bạn có ckpt base
    INPUT_SIZE = 384
    EXTS = {"jpg", "jpeg", "png", "webp"}

    # Lấy danh sách ảnh (không đệ quy, đơn giản)
    image_paths = []
    for f in sorted(os.listdir(IMAGES_DIR)):
        p = os.path.join(IMAGES_DIR, f)
        if os.path.isfile(p) and "." in f and f.rsplit(".", 1)[-1].lower() in EXTS:
            image_paths.append(p)
    if not image_paths:
        raise SystemExit(f"❌ Không thấy ảnh trong: {IMAGES_DIR}")

    print(f"📁 Thư mục: {IMAGES_DIR} | Ảnh: {len(image_paths)}")
    print(f"🔎 Query  : {QUERY}")

    # Khởi tạo model (tắt compile & warmup để nhanh lần đầu)
    model = MultimodalEmbeddingBEiT3(
        model_name=MODEL_NAME,
        checkpoint_path=CHECKPOINT,
        tokenizer_path=TOKENIZER,
        fp16=True,
        input_size=INPUT_SIZE,
        warmup=False,
        aggressive_optimization=False,
    )

    # --- helper: cosine similarity ---
    def l2norm(x: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(x) + 1e-12
        return x / n

    # Embedding cho query (1 lần)
    t0 = time.perf_counter()
    q_emb = l2norm(model.get_text_embedding(QUERY))
    t1 = time.perf_counter()
    print(f"⏱️  Text embed: {(t1 - t0)*1000:.2f} ms")

    # Duyệt từng ảnh, tính similarity
    results = []
    t_img0 = time.perf_counter()
    for i, path in enumerate(image_paths, 1):
        e0 = time.perf_counter()
        img_emb = l2norm(model.get_image_embedding(path))
        sim = float(np.dot(q_emb, img_emb))
        e1 = time.perf_counter()
        results.append((path, sim))
        # In từng ảnh (tuỳ thích)
        print(f"[{i:>4}/{len(image_paths)}] {os.path.basename(path)}  sim={sim:.4f}  ({(e1-e0)*1000:.1f} ms)")
    t_img1 = time.perf_counter()

    # Sắp xếp và in Top-k
    TOPK = min(10, len(results))
    results.sort(key=lambda x: x[1], reverse=True)

    print("\n🏆 Top kết quả:")
    for rank, (p, s) in enumerate(results[:TOPK], 1):
        print(f"{rank:>2}. {os.path.basename(p)}   sim={s:.4f}")

    print(f"\n⏱️  Tổng thời gian ảnh: {(t_img1 - t_img0)*1000:.2f} ms  (~{len(image_paths)/max(1e-9,(t_img1-t_img0)):.2f} img/s)")
