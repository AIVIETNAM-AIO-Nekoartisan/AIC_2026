# model/BLIP2_rerank.py
import os
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Blip2ForImageTextRetrieval
from PIL import Image

def _to_device(batch: dict, device: torch.device, dtype: torch.dtype) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            # Pinned memory + non_blocking copy để tăng tốc chuyển CPU->GPU
            if device.type == "cuda":
                try:
                    v = v.pin_memory()
                except Exception:
                    pass
                out[k] = v.to(device=device, dtype=(dtype if v.is_floating_point() else None), non_blocking=True)
            else:
                out[k] = v.to(device=device, dtype=(dtype if v.is_floating_point() else None))
        else:
            out[k] = v
    return out

class BLIP2Reranker:
    """
    Reranker BLIP2 (batched). Hỗ trợ ITC (nhanh) và ITM (chính xác hơn trong một số bối cảnh).
    - GPU: float16 + SDPA (tránh Flash-Attn 2 lỗi ở vit-g), TF32 cho GEMM
    - CPU: float32
    """

    def __init__(self, model_path: str = "Salesforce/blip2-itm-vit-g", device: str = "cuda"):
        use_cuda = (device == "cuda") and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        # Tăng tốc kernel phía GPU
        if self.device.type == "cuda":
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
            except Exception:
                pass
            try:
                torch.backends.cudnn.benchmark = True
            except Exception:
                pass

        print(f"[BLIP2] Device: {self.device} | DType: {self.dtype}")
        print(f"[BLIP2] Loading: {model_path}")

        # Quan trọng: ép dùng SDPA để tránh FA2 không hỗ trợ trên model này
        # low_cpu_mem_usage: nạp lazy từ safetensors (mmap) thay vì dựng model fp32
        # trong RAM rồi mới cast — máy 14GB RAM bị OOM-kill nếu thiếu flag này
        self.model = Blip2ForImageTextRetrieval.from_pretrained(
            model_path,
            torch_dtype=(torch.float16 if self.device.type == "cuda" else torch.float32),
            attn_implementation="eager",     # <-- then chốt tốc độ/ổn định
            low_cpu_mem_usage=True,
        ).eval().to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        print("✅ BLIP2 reranker ready.")

    # ---------- Core scoring (batched) ----------
    @torch.no_grad()
    def _scores_for_batch(
        self,
        images: List[Image.Image],
        text: str,
        score_type: str
    ) -> torch.Tensor:
        """
        Trả về tensor điểm có shape (B,)
        """
        texts = [text] * len(images)

        # Chuẩn bị input
        inputs = self.processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True
        )
        inputs = _to_device(inputs, self.device, self.dtype)

        # Bật mem-efficient SDPA trong forward để thống nhất kernel nhanh
        sdp_ctx = getattr(torch.backends.cuda, "sdp_kernel", None)
        if sdp_ctx and self.device.type == "cuda":
            ctx = sdp_ctx(enable_flash=False, enable_math=False, enable_mem_efficient=True)
        else:
            # context giả để dùng with ... as no-op
            class _Noop:
                def __enter__(self): return None
                def __exit__(self, *args): return False
            ctx = _Noop()

        with ctx:
            if score_type == "itm":
                outputs = self.model(**inputs, use_image_text_matching_head=True)
                logits2 = None
                if hasattr(outputs, "logits_per_image") and outputs.logits_per_image is not None:
                    logits2 = outputs.logits_per_image
                elif hasattr(outputs, "logits") and outputs.logits is not None:
                    logits2 = outputs.logits
                elif hasattr(outputs, "itm_score") and outputs.itm_score is not None:
                    logits2 = outputs.itm_score
                if logits2 is None:
                    raise RuntimeError("BLIP2 ITM: no logits in outputs")
                if logits2.ndim == 1:
                    return logits2.float()
                probs = F.softmax(logits2, dim=-1)[..., 1]
                return probs.float()

            outputs = self.model(**inputs, use_image_text_matching_head=False)
            logits = outputs.logits_per_image
            if logits.ndim == 2:
                scores = logits.diag() if logits.size(0) == logits.size(1) else logits[:, 0]
            else:
                scores = logits.squeeze()
            return scores.float()

            # ITC: logits_per_image: [B, T] (thường T=B nếu text lặp) -> lấy diag hoặc cột 0
            # outputs = self.model(**inputs, use_image_text_matching_head=False)
            # logits = outputs.logits_per_image
            # if logits.ndim == 2:
            #     if logits.size(0) == logits.size(1):
            #         scores = logits.diag()
            #     elif logits.size(1) >= 1:
            #         scores = logits[:, 0]
            #     else:
            #         scores = logits.squeeze()
            # else:
            #     scores = logits.squeeze()
            # return scores.float()

    def _load_images_safe(self, paths: List[str]) -> List[Tuple[str, Image.Image]]:
        pairs = []
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
                pairs.append((p, img))
            except Exception:
                # skip ảnh lỗi để không chặn pipeline
                pass
        return pairs

    def rerank_batched(
        self,
        image_paths: List[str],
        query_text: str,
        score_type: str = "itc",
        batch_size: int = 16
    ) -> List[Dict]:
        """
        Rerank theo batch. Trả về: [{"path": ..., "score": ...}, ...] (đã sort desc)
        - Double-buffer: preload batch kế tiếp song song để tận dụng CPU/GPU
        - Non-blocking H2D copy + pinned mem => nhanh hơn rõ khi batch_size lớn
        """
        if not image_paths or not query_text:
            return []

        results: List[Dict] = []
        n = len(image_paths)
        i = 0
        bs = max(1, int(batch_size))

        # Double-buffer prefetch
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(self._load_images_safe, image_paths[i: i + bs])

            while i < n:
                pairs = future.result()
                i_next = i + bs
                if i_next < n:
                    future = ex.submit(self._load_images_safe, image_paths[i_next: i_next + bs])

                if not pairs:
                    i = i_next
                    continue

                try:
                    _, imgs = zip(*pairs)
                    scores = self._scores_for_batch(list(imgs), query_text, score_type)
                    for (p, _), s in zip(pairs, scores.tolist()):
                        results.append({"path": p, "score": float(s)})
                    i = i_next

                except torch.cuda.OutOfMemoryError:
                    # Giảm batch size và thử lại ngay trên batch này
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                    new_bs = max(1, bs // 2)
                    if new_bs == bs:
                        # không giảm thêm được => bỏ batch để không kẹt
                        i = i_next
                    else:
                        bs = new_bs
                        # nạp lại batch hiện tại với bs mới
                        future = ex.submit(self._load_images_safe, image_paths[i: i + bs])

                except Exception:
                    # Lỗi lẻ tẻ không mong muốn => bỏ batch
                    i = i_next

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    # API cũ để backward-compatible
    def rerank(self, image_paths: List[str], query_text: str, score_type: str = "itc") -> List[Dict]:
        # batch_size lớn hơn một chút để tận dụng GPU (tự giảm nếu OOM)
        return self.rerank_batched(image_paths, query_text, score_type=score_type, batch_size=36)

    @torch.no_grad()
    def score_single_pair(self, image_input, text_prompt: str, score_type: str = "itc") -> float:
        if isinstance(image_input, str):
            try:
                image = Image.open(image_input).convert("RGB")
            except Exception:
                return 0.0
        else:
            image = image_input
        s = self._scores_for_batch([image], text_prompt, score_type)
        return float(s.item())
