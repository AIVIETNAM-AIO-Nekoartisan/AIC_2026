"""KIS search: expand query (GPT) -> embed (SigLIP2/Jina/BGE) -> Qdrant -> SRRF -> BLIP2 rerank."""
import os
import math
import logging
from typing import List, Dict, Any, Optional

import numpy as np
from PIL import Image
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

from core.utils.parsing import extract_frame_id_from_payload as _extract_frame_id_from_payload
from online.backend.resources import resources
from online.backend.fusion import srrf_weighted_fuse

logger = logging.getLogger(__name__)


def _process_kis_search(
    text_query: Optional[str],
    image: Optional[Image.Image],
    num_results: int,
    score_threshold: float,
    query_weight: float,
    auto_expand: bool,
    num_expansions: int,
    use_rerank: bool = True,
    rerank_score_type: str = "itm",
    rerank_top_k: int = 100,
    rrf_k: int = 60,
    rrf_weights: Optional[Dict[str, float]] = None,
    use_siglip: bool = True,
    use_jina: bool = True,
    use_beit3: bool = True,
    use_bge: bool = True,
    video_filter: Optional[str] = None,
    object_class: Optional[str] = None,
    object_count: Optional[int] = None,
    object_count_tolerance: int = 3,
    object_filter_mode: str = "hard",
    object_score_floor: float = 0.25,
    object_filters: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    import time
    t0 = time.time()
    t = t0
    print("\n[KIS] ===== START =====")

    if not any((use_siglip, use_jina, use_beit3, use_bge)):
        use_siglip = True
    if image is not None and not any((use_siglip, use_jina, use_beit3)):
        # BGE is text-only; an image-only request still needs a visual encoder.
        use_siglip = True
    print(
        f"[KIS] models: siglip={use_siglip} jina={use_jina} "
        f"beit3={use_beit3} bge={use_bge}",
    )

    conds = []
    if video_filter:
        conds.append(FieldCondition(key="video.name", match=MatchValue(value=video_filter)))
        print(f"[KIS] video_filter: {video_filter}")
    # Lọc object detection: theo class cụ thể hoặc tổng số object, khoảng ±tolerance
    # Gộp dạng 1-class cũ và danh sách nhiều class thành 1 list spec
    specs: List[Dict[str, Any]] = []
    for s in (object_filters or []):
        specs.append({"cls": (s.get("cls") or "").strip().lower(),
                      "count": s.get("count") if s.get("count") is not None else None})
    if not specs and (object_class or object_count is not None):
        specs.append({"cls": (object_class or "").strip().lower(), "count": object_count})

    def _od_key(cls: str) -> str:
        return f"object_detection.class_counts.{cls}" if cls else "object_detection.object_count"

    if specs and object_filter_mode == "soft":
        # soft: không cắt ở tầng Qdrant, chấm lại điểm ở cuối pipeline
        print(f"[KIS] object_boost (soft): {specs} floor={object_score_floor}")
    else:
        for sp in specs:
            if sp["count"] is not None:
                lo = max(0, int(sp["count"]) - int(object_count_tolerance))
                hi = int(sp["count"]) + int(object_count_tolerance)
                conds.append(FieldCondition(key=_od_key(sp["cls"]), range=Range(gte=lo, lte=hi)))
                print(f"[KIS] object_filter: {_od_key(sp['cls'])} trong [{lo}, {hi}]")
            elif sp["cls"]:
                # chỉ nhập class, không nhập số lượng -> yêu cầu ảnh CÓ class đó
                conds.append(FieldCondition(key=_od_key(sp["cls"]), range=Range(gte=1)))
                print(f"[KIS] object_filter: có ít nhất 1 '{sp['cls']}'")
    flt = Filter(must=conds) if conds else None

    # 1) Query expansion
    all_queries = []
    if text_query:
        all_queries.append(text_query)
        if auto_expand and resources.gpt_service:
            try:
                expanded = resources.gpt_service.expand_query(text_query, num_expansions)
                all_queries = expanded
                text_query = all_queries[0]
            except Exception as e:
                print(f"[KIS] expand ERROR: {e}")
    print(f"[KIS] expand: {time.time() - t:.3f}s (n_queries={len(all_queries)})"); t = time.time()

    # 2) Embeddings
    sl_txt_vecs = []
    jn_txt_vecs = []
    bt_txt_vecs = []
    bge_txt_vecs = []

    if all_queries and use_siglip:
        # SigLIP
        if hasattr(resources.embedding_model_siglip, 'get_batch_text_embeddings'):
            sl_txt = resources.embedding_model_siglip.get_batch_text_embeddings(all_queries)
            sl_txt_vecs = [sl_txt[i] for i in range(len(all_queries))]
        else:
            sl_txt_vecs = [resources.embedding_model_siglip.get_text_embedding(q) for q in all_queries]
    print(f"[KIS] embed_siglip: {time.time() - t:.3f}s (n={len(sl_txt_vecs)})"); t = time.time()

    # Jina-CLIP-v2
    if all_queries and use_jina and resources.embedding_model_jina is not None:
        try:
            jn_txt = resources.embedding_model_jina.get_batch_text_embeddings(all_queries)
            jn_txt_vecs = [jn_txt[i] for i in range(len(all_queries))]
        except Exception as e:
            print(f"[KIS] embed_jina ERROR: {e}")
            jn_txt_vecs = []
    print(f"[KIS] embed_jina: {time.time() - t:.3f}s (n={len(jn_txt_vecs)})"); t = time.time()

    # Search bằng ảnh: embed ảnh với cả SigLIP lẫn Jina, tìm cùng không gian vector
    if image is not None:
        try:
            if use_siglip:
                sl_txt_vecs.append(resources.embedding_model_siglip.get_image_embedding(image))
            if use_jina and resources.embedding_model_jina is not None:
                jn_txt_vecs.append(resources.embedding_model_jina.get_image_embedding(image))
        except Exception as e:
            print(f"[KIS] embed_image ERROR: {e}")
    print(f"[KIS] embed_image: {time.time() - t:.3f}s"); t = time.time()

    # BEiT-3
    if all_queries and use_beit3 and resources.embedding_model_beit3 is not None:
        try:
            bt_txt = resources.embedding_model_beit3.get_batch_text_embeddings(all_queries)
            bt_txt_vecs = [bt_txt[i] for i in range(len(all_queries))]
        except Exception as e:
            print(f"[KIS] embed_beit3 ERROR: {e}")
            bt_txt_vecs = []
    if image is not None and use_beit3 and resources.embedding_model_beit3 is not None:
        try:
            bt_txt_vecs.append(resources.embedding_model_beit3.get_image_embedding(image))
        except Exception as e:
            print(f"[KIS] embed_beit3_image ERROR: {e}")
    print(f"[KIS] embed_beit3: {time.time() - t:.3f}s (n={len(bt_txt_vecs)})"); t = time.time()

    if all_queries and use_bge and resources.embedding_model_bge is not None:
        try:
            # encode trả numpy; chuẩn hóa L2 để hợp COSINE
            enc = resources.embedding_model_bge.encode(
                all_queries,
                # BGE's tqdm is disabled only when len < batch_size.
                # Search expansion is capped at 10 queries, so 16 is both a
                # single batch and quiet in backend logs.
                batch_size=16,
                max_length=128,
                return_dense=True, return_sparse=False, return_colbert_vecs=False,
            )
            dense = enc.get("dense_vecs", None)
            for i in range(len(dense)):
                v = np.asarray(dense[i], dtype=np.float32)
                v /= (np.linalg.norm(v) + 1e-12)
                bge_txt_vecs.append(v.tolist())
        except Exception as e:
            print(f"[KIS] embed_bge ERROR: {e}")
    print(f"[KIS] embed_bge: {time.time() - t:.3f}s (n={len(bge_txt_vecs)})"); t = time.time()

    # 3) Qdrant search
    res_sigclip, res_jina, res_beit3, res_caption = [], [], [], []
    max_candidates = max(num_results, rerank_top_k)

    for v in sl_txt_vecs:
        res = resources.qdrant.search_by("siglip", v, limit=max_candidates, score_threshold=score_threshold, filter_condition=flt)
        if res:
            res_sigclip.extend(res)
    print(f"[KIS] qdrant_siglip: {time.time() - t:.3f}s (cands={len(res_sigclip)})"); t = time.time()

    for v in jn_txt_vecs:
        try:
            res = resources.qdrant.search_by("jina", v, limit=max_candidates, score_threshold=score_threshold, filter_condition=flt)
            if res:
                res_jina.extend(res)
        except Exception as e:
            print(f"[KIS] qdrant_jina ERROR: {e}")
    print(f"[KIS] qdrant_jina: {time.time() - t:.3f}s (cands={len(res_jina)})"); t = time.time()

    for v in bt_txt_vecs:
        res = resources.qdrant.search_by("beit3", v, limit=max_candidates, score_threshold=score_threshold, filter_condition=flt)
        if res:
            res_beit3.extend(res)
    print(f"[KIS] qdrant_beit3: {time.time() - t:.3f}s (cands={len(res_beit3)})"); t = time.time()

    for v in bge_txt_vecs:
        if resources.qdrant_caption is not None:
            try:
                res = resources.qdrant_caption.search(
                    vector=v, limit=max_candidates, score_threshold=score_threshold,
                    filter_condition=flt
                )
                if res:
                    res_caption.extend(res)
            except Exception as e:
                print(f"[KIS] qdrant_caption ERROR: {e}")
    print(f"[KIS] qdrant_caption: {time.time() - t:.3f}s (cands={len(res_caption)})"); t = time.time()

    if not (res_sigclip or res_jina or res_beit3 or res_caption):
        print(f"[KIS] early_exit (no candidates). TOTAL: {time.time() - t0:.3f}s")
        return []

    # 4) SRRF fuse
    # fused = rrf_weighted_fuse(
    #     results_by_model={
    #         **({"siglip": res_sigclip} if res_sigclip else {}),
    #         **({"beit3":  res_beit3}  if res_beit3  else {}),
    #     },
    #     k=rrf_k,
    #     weights=rrf_weights or {"siglip": 1.0, "beit3": 1.0},
    #     topn=max_candidates
    # )
    # print(f"[KIS] rrf_fuse: {time.time() - t:.3f}s (out={len(fused)})"); t = time.time()



    fused = srrf_weighted_fuse(  # <--- Thay tên hàm
        results_by_model={
            **({"siglip": res_sigclip} if res_sigclip else {}),
            **({"jina":   res_jina}   if res_jina   else {}),
            **({"beit3":  res_beit3}  if res_beit3  else {}),
            **({"caption": res_caption} if res_caption else {})
        },
        k=rrf_k,
        weights=rrf_weights or {"siglip": 1.0, "jina": 1.0, "beit3": 1.0},
        topn=max_candidates,
        beta=0.6  # <--- Thêm tham số beta, có thể lấy từ request hoặc để mặc định
    )
    print(f"[KIS] srrf_fuse: {time.time() - t:.3f}s (out={len(fused)})"); t = time.time()
    # 5) BLIP2 rerank
    results_list = fused
    if (use_rerank and resources.reranker is not None
            and text_query is not None and len(results_list) > 0):
        candidates = results_list[:min(len(results_list), rerank_top_k)]
        paths = [c["file_path"] for c in candidates if c.get("file_path")]
        print(f"[KIS] prepare_rerank (top_k={len(paths)}): {time.time() - t:.3f}s"); t = time.time()
        try:
            batch_size = int(os.getenv("BLIP2_BATCH", "10"))
            rr = resources.reranker.rerank_batched(
                paths, text_query, score_type=rerank_score_type, batch_size=batch_size
            )
            cand_map = {c["file_path"]: dict(c) for c in candidates}
            for x in rr:
                p = x["path"]; s = float(x["score"])
                if p in cand_map:
                    cand_map[p]["rerank_score"] = s
            rr_order_paths = [x["path"] for x in rr if x["path"] in cand_map]
            rr_set = set(rr_order_paths)
            ordered_rr = [cand_map[p] for p in rr_order_paths]
            rest_cands = [c for c in candidates if c["file_path"] not in rr_set]
            outside = [it for it in results_list if it["file_path"] not in set(paths)]
            results_list = ordered_rr + rest_cands + outside

            for it in results_list:
                if "rerank_score" in it:
                    it["rrf_score"] = float(it.get("score", 0.0))
                    it["score"] = float(it["rerank_score"])
                    it["score_type"] = f"blip2_{rerank_score_type}"
                else:
                    it["score_type"] = "rrf"
        except Exception as e:
            print(f"[KIS] blip2_rerank ERROR: {e}")
            for it in results_list:
                it["score_type"] = "rrf"
        print(f"[KIS] blip2_rerank: {time.time() - t:.3f}s"); t = time.time()
    else:
        for it in results_list:
            it["score_type"] = "rrf"
        print(f"[KIS] skip_rerank: {time.time() - t:.3f}s"); t = time.time()

    # 5b) Soft object boost: đúng số lượng x1.0, lệch d thì nhân
    #     floor + (1-floor) * exp(-d / tolerance) — giảm dần về floor, không cắt ảnh nào.
    #     Nhiều spec -> lấy TRUNG BÌNH các hệ số (product sẽ tụt dưới floor).
    if specs and object_filter_mode == "soft":
        scale = max(1, int(object_count_tolerance))
        floor = float(object_score_floor)
        for it in results_list:
            od = (it.get("payload") or {}).get("object_detection") or {}
            counts = od.get("class_counts") or {}
            mults, actuals = [], {}
            for sp in specs:
                actual = int(counts.get(sp["cls"], 0)) if sp["cls"] else int(od.get("object_count", 0))
                actuals[sp["cls"] or "_total"] = actual
                if sp["count"] is None:
                    mults.append(1.0 if actual >= 1 else floor)   # chỉ cần có mặt
                else:
                    diff = abs(actual - int(sp["count"]))
                    mults.append(floor + (1.0 - floor) * math.exp(-diff / scale))
            mult = sum(mults) / len(mults) if mults else 1.0
            it["object_actual"] = actuals
            it["object_boost"] = round(mult, 4)
            it["score"] = float(it.get("score", 0.0)) * mult
        results_list.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        print(f"[KIS] object_boost áp cho {len(results_list)} kết quả"); t = time.time()

    # 6) Post-process
    for it in results_list:
        fid = _extract_frame_id_from_payload(it.get("payload") or {})
        if fid is not None:
            it["frame_id"] = fid
            it["primary_query"] = text_query                 # câu chính (sau expand)
            it["expanded_queries"] = all_queries
    print(f"[KIS] postprocess: {time.time() - t:.3f}s"); t = time.time()

    print(f"[KIS] TOTAL: {time.time() - t0:.3f}s\n")
    return results_list[:num_results]


def process_kis_search(*args, **kwargs) -> List[Dict[str, Any]]:
    """Bound concurrent GPU work so parallel requests cannot exhaust VRAM."""
    timeout = max(0.1, float(os.getenv("KIS_QUEUE_TIMEOUT_SECONDS", "30")))
    acquired = resources.kis_semaphore.acquire(timeout=timeout)
    if not acquired:
        raise RuntimeError(
            f"KIS đang bận; không lấy được GPU slot sau {timeout:.1f} giây",
        )
    try:
        return _process_kis_search(*args, **kwargs)
    finally:
        resources.kis_semaphore.release()
