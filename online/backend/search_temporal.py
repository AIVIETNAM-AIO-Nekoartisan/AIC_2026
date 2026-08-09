"""Temporal search: tách event (GPT) -> KIS/OCR/ASR per-event -> beam search theo thời gian."""
import os
import re
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

from core.utils.parsing import extract_frame_id_from_payload as _extract_frame_id_from_payload
from online.backend.schemas import EventSpec
from online.backend.resources import resources
from online.backend.search_kis import process_kis_search
from online.backend.utils import (
    enrich_asr_with_frames_batched,
    _pick_representative_frame_for_asr,
    _frame_obj_timestamp_seconds,
    _payload_timestamp_seconds,
    _normalize_weights,
    _norm_scores_per_component,
    _minmax_norm,
    _format_time,
    calculate_lambda,
    _group_candidates_by_video,
    parse_meta_from_path,
)

logger = logging.getLogger(__name__)


def _fetch_event_candidates_multimodal(
    ev_text: str,
    per_event_k: int,
    score_threshold: float,
    rrf_k: int,
    rrf_weights: Optional[Dict[str, float]],
    use_rerank_per_event: bool,
    rerank_score_type: str,
    rerank_top_k: int,
    same_video_only: bool,
    weights: Dict[str, float],
) -> List[Dict[str, Any]]:
    """
    Cho 1 event text:
      - chạy KIS (SRRF + optional BLIP2), OCR(ES), ASR(ES + enrich frames)
      - chuẩn hoá từng nguồn theo min-max (per-event)
      - gộp theo trọng số 'weights' thành combined_score
      - trả list ứng viên: mỗi item có 'ts','file_path','payload','score'(combined)
    """
    kis_res, ocr_res, asr_res = [], [], []

    # --- KIS ---
    kis_res = process_kis_search(
        text_query=ev_text, image=None,
        num_results=per_event_k,
        score_threshold=score_threshold,
        query_weight=1.0,
        auto_expand=True,
        num_expansions=3,
        use_rerank=use_rerank_per_event,
        rerank_score_type=rerank_score_type,
        rerank_top_k=min(per_event_k, rerank_top_k),
        rrf_k=rrf_k,
        rrf_weights=rrf_weights
    ) or []

    # --- OCR ---
    try:
        ocr_res = resources.es_ocr.search(
            query=ev_text, limit=per_event_k, score_threshold=score_threshold, filters={"content_type": "image"}
        ) or []
        # gắn payload video/ts nếu thiếu
        for r in ocr_res:
            if not ((r.get("payload") or {}).get("video") or {}).get("name"):
                meta = parse_meta_from_path(r.get("file_path",""))
                if meta:
                    r["payload"] = {**(r.get("payload") or {}), **meta}
    except Exception as e:
        logger.warning(f"OCR per-event failed: {e}")

    # --- ASR ---
    try:
        asr_res = resources.es_audio.search(
            query=ev_text, limit=per_event_k, score_threshold=score_threshold, filters={"content_type": "audio"}
        ) or []
        if asr_res:
            asr_res = enrich_asr_with_frames_batched(
                asr_res, context_seconds=2.0, max_frames_per_hit=6, merge_slack_seconds=1.0, max_frames_per_merged_range=300
            )
    except Exception as e:
        logger.warning(f"ASR per-event failed: {e}")

    # --- chuẩn hoá điểm từng nguồn ---
    kis_scores = _norm_scores_per_component(
        kis_res, key=("rerank_score" if any("rerank_score" in x for x in (kis_res or [])) else "score")
    ) if kis_res else {}
    ocr_scores = _norm_scores_per_component(ocr_res, key="score") if ocr_res else {}
    # với ASR: ta map về frame đại diện để có file_path đơn lẻ
    asr_frame_items = []
    path2asr_item = {}
    for a in asr_res or []:
        f = _pick_representative_frame_for_asr(a)
        if not f or not f.get("file_path"):
            continue
        # clone item để hành xử như 1 "frame-hit"
        pf = dict(f.get("payload") or {})
        # đảm bảo có video.name
        if not ((pf.get("video") or {}).get("name")):
            meta = parse_meta_from_path(f.get("file_path",""))
            if meta:
                pf = {**pf, **meta}
        it = {
            "file_path": f["file_path"],
            "payload": pf,
            "score": float(a.get("score", 0.0) or 0.0)
        }
        asr_frame_items.append(it)
        path2asr_item[f["file_path"]] = it
    asr_scores = _norm_scores_per_component(asr_frame_items, key="score") if asr_frame_items else {}

    # helper lấy video từ payload/path
    def _video_of_item(r: Dict[str, Any]) -> str:
        pay = (r.get("payload") or {})
        return ((pay.get("video") or {}).get("name")) or "unknown"

    # video-level max cho projection (nếu same_video_only và muốn “điền khuyết”)
    ocr_max_by_video, asr_max_by_video = defaultdict(float), defaultdict(float)
    if same_video_only:
        for r in ocr_res or []:
            p = r.get("file_path");
            if p and p in ocr_scores:
                ocr_max_by_video[_video_of_item(r)] = max(ocr_max_by_video[_video_of_item(r)], ocr_scores[p])
        for r in asr_frame_items or []:
            p = r.get("file_path")
            if p and p in asr_scores:
                asr_max_by_video[_video_of_item(r)] = max(asr_max_by_video[_video_of_item(r)], asr_scores[p])

    # unified path set + index để chọn item đại diện hiển thị (ưu tiên KIS -> OCR -> ASR)
    path2kis = {r["file_path"]: r for r in kis_res if r.get("file_path")}
    path2ocr = {r["file_path"]: r for r in ocr_res if r.get("file_path")}
    all_paths = set().union(kis_scores.keys(), ocr_scores.keys(), asr_scores.keys())

    # nếu ràng buộc cùng video và đang bật >=2 nguồn → chỉ giữ video có ≥2 nguồn đóng góp
    if same_video_only and sum(1 for v in (weights or {}).values() if v > 0) >= 2:
        vid2srcs = defaultdict(set)
        def _video_of_path(p: str) -> str:
            ref = path2kis.get(p) or path2ocr.get(p) or path2asr_item.get(p) or {}
            return _video_of_item(ref)
        for p in list(all_paths):
            v = _video_of_path(p)
            if p in kis_scores and weights.get("kis",0)>0: vid2srcs[v].add("kis")
            if p in ocr_scores and weights.get("ocr",0)>0: vid2srcs[v].add("ocr")
            if p in asr_scores and weights.get("asr",0)>0: vid2srcs[v].add("asr")
        keep_vids = {v for v,s in vid2srcs.items() if len(s)>=2}
        if keep_vids:
            all_paths = {p for p in all_paths if _video_of_path(p) in keep_vids}

    out = []
    for p in all_paths:
        base = path2kis.get(p) or path2ocr.get(p) or path2asr_item.get(p)
        if not base:
            continue
        pay = base.get("payload") or {}
        ts = _payload_timestamp_seconds(base)  # ưu tiên payload.frame; nếu fail sẽ thử index/fps
        if ts is None:
            # nếu là OCR parse hoặc ASR frame sẽ có timestamp_seconds trong payload
            ts = _frame_obj_timestamp_seconds(base)  # fallback nhẹ
        if ts is None:
            continue

        vid = ((pay.get("video") or {}).get("name")) or "unknown"
        ks = kis_scores.get(p, 0.0) if weights.get("kis",0)>0 else 0.0
        if weights.get("ocr",0)>0:
            if p in ocr_scores:
                os_ = ocr_scores[p]
            else:
                os_ = ocr_max_by_video.get(vid, 0.0) if same_video_only else 0.0
        else:
            os_ = 0.0
        if weights.get("asr",0)>0:
            if p in asr_scores:
                as_ = asr_scores[p]
            else:
                as_ = asr_max_by_video.get(vid, 0.0) if same_video_only else 0.0
        else:
            as_ = 0.0

        combined = ks*weights.get("kis",0.0) + os_*weights.get("ocr",0.0) + as_*weights.get("asr",0.0)
        if combined <= 0:
            continue

        cand = {
            "file_path": p,
            "payload": pay,
            "ts": float(ts),
            "score": float(combined),
            "original_score": float(combined),
            "event": ev_text,
            "component_scores": {"kis": ks if weights.get("kis",0)>0 else None,
                                 "ocr": os_ if weights.get("ocr",0)>0 else None,
                                 "asr": as_ if weights.get("asr",0)>0 else None},
            "weights_used": dict(weights)
        }
        if base and base.get("expanded_queries"):
            cand["primary_query"] = base.get("primary_query")
            cand["expanded_queries"] = base["expanded_queries"]

        fid = _extract_frame_id_from_payload(pay)
        if fid is not None:
            cand["frame_id"] = fid

        # nếu base (từ KIS) có srrf/blip giữ lại để debug UI
        if "srrf_score" in base: cand["srrf_score"] = base["srrf_score"]
        if "rrf_score" in base: cand["rrf_score"] = base["rrf_score"]
        if "rerank_score" in base: cand["rerank_score"] = base["rerank_score"]
        out.append(cand)

    # sort desc theo combined
    out.sort(key=lambda x: x["score"], reverse=True)
    return out

def _fetch_event_candidates_from_eventspec(
    ev_spec: EventSpec,
    per_event_k: int,
    score_threshold: float,
    rrf_k: int,
    rrf_weights: Optional[Dict[str, float]],
    use_rerank_per_event: bool,
    rerank_score_type: str,
    rerank_top_k: int,
    same_video_only: bool,
    weights: Dict[str, float],
) -> List[Dict[str, Any]]:
    kis_q = (ev_spec.kis or "").strip()
    ocr_q = (ev_spec.ocr or "").strip()
    asr_q = (ev_spec.asr or "").strip()

    kis_res, ocr_res, asr_res = [], [], []

    # --- KIS: KHÔNG expand GPT trong mode explicit ---
    if kis_q:
        kis_res = process_kis_search(
            text_query=kis_q, image=None,
            num_results=per_event_k,
            score_threshold=score_threshold,
            query_weight=1.0,
            auto_expand=True,                 # <— KHÓA: không GPT expand/dịch
            num_expansions=3,
            use_rerank=use_rerank_per_event,  # BLIP2 per-event (nếu bật)
            rerank_score_type=rerank_score_type,
            rerank_top_k=min(per_event_k, rerank_top_k),
            rrf_k=rrf_k,
            rrf_weights=rrf_weights
        ) or []

    # --- OCR: đẩy thẳng xuống ES, KHÔNG GPT ---
    if ocr_q:
        try:
            ocr_res = resources.es_ocr.search(
                query=ocr_q, limit=per_event_k, score_threshold=score_threshold, filters={"content_type": "image"}
            ) or []
            # đảm bảo có payload.video khi thiếu
            for r in ocr_res:
                if not ((r.get("payload") or {}).get("video") or {}).get("name"):
                    meta = parse_meta_from_path(r.get("file_path",""))
                    if meta:
                        r["payload"] = {**(r.get("payload") or {}), **meta}
        except Exception as e:
            logger.warning(f"OCR per-event failed: {e}")

    # --- ASR: đẩy thẳng xuống ES, KHÔNG GPT ---
    if asr_q:
        try:
            asr_res = resources.es_audio.search(
                query=asr_q, limit=per_event_k, score_threshold=score_threshold, filters={"content_type": "audio"}
            ) or []
            if asr_res:
                asr_res = enrich_asr_with_frames_batched(
                    asr_res, context_seconds=2.0, max_frames_per_hit=6, merge_slack_seconds=1.0, max_frames_per_merged_range=300
                )
        except Exception as e:
            logger.warning(f"ASR per-event failed: {e}")

    # --- Chuẩn hoá điểm từng nguồn (min-max) như Combination Retrieval ---
    kis_scores = _norm_scores_per_component(
        kis_res, key=("rerank_score" if any("rerank_score" in x for x in (kis_res or [])) else "score")
    ) if kis_res else {}
    ocr_scores = _norm_scores_per_component(ocr_res, key="score") if ocr_res else {}

    # ASR: quy về frame đại diện để có file_path đơn lẻ
    asr_frame_items, path2asr_item = [], {}
    for a in asr_res or []:
        f = _pick_representative_frame_for_asr(a)
        if not f or not f.get("file_path"):
            continue
        pf = dict(f.get("payload") or {})
        if not ((pf.get("video") or {}).get("name")):
            meta = parse_meta_from_path(f.get("file_path",""))
            if meta: pf = {**pf, **meta}
        it = {"file_path": f["file_path"], "payload": pf, "score": float(a.get("score") or 0.0)}
        asr_frame_items.append(it)
        path2asr_item[f["file_path"]] = it
    asr_scores = _norm_scores_per_component(asr_frame_items, key="score") if asr_frame_items else {}

    # --- Hợp nhất theo path + same_video_only như Combination Retrieval ---
    path2kis = {r["file_path"]: r for r in kis_res if r.get("file_path")}
    path2ocr = {r["file_path"]: r for r in ocr_res if r.get("file_path")}
    all_paths = set().union(kis_scores.keys(), ocr_scores.keys(), asr_scores.keys())

    if same_video_only and sum(1 for v in weights.values() if v > 0) >= 2:
        vid2srcs = defaultdict(set)
        def _video_of_path(p: str) -> str:
            ref = path2kis.get(p) or path2ocr.get(p) or path2asr_item.get(p) or {}
            pay = ref.get("payload") or {}
            return ((pay.get("video") or {}).get("name")) or "unknown"
        for p in list(all_paths):
            v = _video_of_path(p)
            if p in kis_scores and weights.get("kis",0)>0: vid2srcs[v].add("kis")
            if p in ocr_scores and weights.get("ocr",0)>0: vid2srcs[v].add("ocr")
            if p in asr_scores and weights.get("asr",0)>0: vid2srcs[v].add("asr")
        keep_vids = {v for v,s in vid2srcs.items() if len(s)>=2}
        if keep_vids:
            all_paths = {p for p in all_paths if _video_of_path(p) in keep_vids}

    # Project max theo video nếu thiếu component (y hệt combine_multimodal_results)
    def _video_of_item(r: Dict[str, Any]) -> str:
        pay = (r.get("payload") or {})
        return ((pay.get("video") or {}).get("name")) or "unknown"
    ocr_max_by_video, asr_max_by_video = defaultdict(float), defaultdict(float)
    if same_video_only:
        for r in ocr_res or []:
            p = r.get("file_path")
            if p and p in ocr_scores:
                ocr_max_by_video[_video_of_item(r)] = max(ocr_max_by_video[_video_of_item(r)], ocr_scores[p])
        for r in asr_frame_items or []:
            p = r.get("file_path")
            if p and p in asr_scores:
                asr_max_by_video[_video_of_item(r)] = max(asr_max_by_video[_video_of_item(r)], asr_scores[p])

    # Xuất candidates (score = weighted sum)
    out = []
    for p in all_paths:
        base = path2kis.get(p) or path2ocr.get(p) or path2asr_item.get(p)
        if not base:
            continue
        pay = base.get("payload") or {}
        ts = _payload_timestamp_seconds(pay)
        if ts is None:
            ts = _frame_obj_timestamp_seconds(base)
        if ts is None:
            continue

        vid = ((pay.get("video") or {}).get("name")) or "unknown"
        ks = kis_scores.get(p, 0.0) if weights.get("kis",0)>0 else 0.0
        os_ = (ocr_scores.get(p, 0.0) if (weights.get("ocr",0)>0 and p in ocr_scores)
               else (ocr_max_by_video.get(vid, 0.0) if (weights.get("ocr",0)>0 and same_video_only) else 0.0))
        as_ = (asr_scores.get(p, 0.0) if (weights.get("asr",0)>0 and p in asr_scores)
               else (asr_max_by_video.get(vid, 0.0) if (weights.get("asr",0)>0 and same_video_only) else 0.0))

        combined = ks*weights.get("kis",0.0) + os_*weights.get("ocr",0.0) + as_*weights.get("asr",0.0)
        if combined <= 0:
            continue

        cand = {
            "file_path": p,
            "payload": pay,
            "ts": float(ts),
            "score": float(combined),
            "original_score": float(combined),
            "event": { "kis": kis_q or None, "ocr": ocr_q or None, "asr": asr_q or None },
            "component_scores": {
                "kis": ks if weights.get("kis",0)>0 else None,
                "ocr": os_ if weights.get("ocr",0)>0 else None,
                "asr": as_ if weights.get("asr",0)>0 else None
            },
            "weights_used": dict(weights)
        }

        if base and base.get("expanded_queries"):
            cand["primary_query"] = base.get("primary_query")
            cand["expanded_queries"] = base["expanded_queries"]

        fid = _extract_frame_id_from_payload(pay)
        if fid is not None:
            cand["frame_id"] = fid

        if "srrf_score" in base: cand["srrf_score"] = base["srrf_score"]
        if "rrf_score" in base:  cand["rrf_score"]  = base["rrf_score"]
        if "rerank_score" in base: cand["rerank_score"] = base["rerank_score"]

        out.append(cand)

    out.sort(key=lambda x: x["score"], reverse=True)
    return out


def process_temporal_combinate_search(
    text_query: str,
    num_results: int = 5,
    per_event_k: int = 50,
    max_events: int = 5,
    score_threshold: float = 0.1,
    same_video_only: bool = True,
    max_gap_seconds: Optional[float] = None,
    decay_rate: float = 0.01,
    rrf_k: int = 60,
    rrf_weights: Optional[Dict[str, float]] = None,
    use_rerank_per_event: bool = True,
    rerank_score_type: str = "itm",
    rerank_top_k: int = 100,
    weights: Optional[Dict[str, float]] = None,
    events_explicit: Optional[List[EventSpec]] = None,   # <-- NEW
) -> List[Dict[str, Any]]:
    """
    Temporal Combinate = Combination Retrieval cho MỖI EVENT (KIS/OCR/ASR riêng) + beam λ-decay.
    - Nếu events_explicit có mặt: bỏ qua GPT; mỗi event là {kis, ocr, asr}.
    - Nếu không: fallback pipeline cũ (GPT tách event + 1 text cho cả 3 modal).
    """
    w = _normalize_weights(weights or {"kis": 0.5, "ocr": 0.3, "asr": 0.2})

    # ===== 1) Lấy danh sách events =====
    explicit_mode = False
    ev_specs: List[EventSpec] = []

    if events_explicit:
        # Lọc bỏ event trống cả 3 trường
        for ev in events_explicit:
            kis_q = (ev.kis or "").strip()
            ocr_q = (ev.ocr or "").strip()
            asr_q = (ev.asr or "").strip()
            if kis_q or ocr_q or asr_q:
                ev_specs.append(EventSpec(kis=kis_q or None, ocr=ocr_q or None, asr=asr_q or None))
        explicit_mode = len(ev_specs) >= 2

    cands_per_event: List[List[Dict[str, Any]]] = []

    if explicit_mode:
        # ===== 2A) COMBINATION RETRIEVAL THEO MỖI EVENT (KHÔNG GPT) =====
        for ev in ev_specs:
            cands = _fetch_event_candidates_from_eventspec(
                ev_spec=ev,
                per_event_k=per_event_k,
                score_threshold=score_threshold,
                rrf_k=rrf_k,
                rrf_weights=rrf_weights,
                use_rerank_per_event=use_rerank_per_event,
                rerank_score_type=rerank_score_type,
                rerank_top_k=rerank_top_k,
                same_video_only=same_video_only,
                weights=w
            )
            cands_per_event.append(cands)

        # Nếu event nào rỗng ứng viên → trả rỗng (giữ ràng buộc)
        if any(len(c) == 0 for c in cands_per_event):
            return []

    else:
        # ===== 2B) Fallback: pipeline cũ (GPT tách event + 1 text cho cả 3 modal) =====
        if not text_query or not text_query.strip():
            return []
        try:
            events = resources.gpt_service.extract_temporal_events(text_query, max_events=max_events)
        except Exception as e:
            logger.error(f"extract_temporal_events failed: {e}")
            return []
        if not events or len(events) < 2:
            return []

        # Mỗi event 1 text → multimodal như cũ
        for ev in events:
            cands = _fetch_event_candidates_multimodal(
                ev_text=ev,
                per_event_k=per_event_k,
                score_threshold=score_threshold,
                rrf_k=rrf_k,
                rrf_weights=rrf_weights,
                use_rerank_per_event=use_rerank_per_event,
                rerank_score_type=rerank_score_type,
                rerank_top_k=rerank_top_k,
                same_video_only=same_video_only,
                weights=w
            )
            cands_per_event.append(cands)

        if any(len(c)==0 for c in cands_per_event):
            return []

    # ===== 3) Nhóm theo video + sort ts tăng (giống enhanced) =====
    by_video = _group_candidates_by_video(cands_per_event)
    if not by_video:
        return []

    # Min-max per (video,event) trước khi beam (đảm bảo công bằng)
    for vid, ev_lists in by_video.items():
        for ei, cands in enumerate(ev_lists):
            base_scores = [c["score"] for c in cands] if cands else []
            norm = _minmax_norm(base_scores)
            for i, c in enumerate(cands):
                c["vis_norm"] = float(norm[i])
                c["score"] = float(norm[i])  # beam dùng "score"

    # ===== 4) Beam search theo thời gian với exponential λ-decay =====
    sequences = []
    for vid, ev_lists in by_video.items():
        seq = _enhanced_beam_sequence_for_video(
            cands_for_video=ev_lists,
            beam_size=8,
            max_gap_seconds=max_gap_seconds,
            decay_rate=decay_rate
        )
        if not seq:
            continue

        start_ts = seq[0]["ts"]; end_ts = seq[-1]["ts"]; total_dur = max(0.0, end_ts - start_ts)
        enh_sum = 0.0
        for step in seq:
            lam = float(step.get("lambda_factor", 1.0))
            vnorm = float(step.get("vis_norm", step.get("score", 0.0)))
            step["enhanced_score"] = float(vnorm * lam)

            payload = step.get("payload") or {}
            frame = (payload.get("frame") or {})
            step["timestamp_formatted"] = frame.get("timestamp_formatted", _format_time(step["ts"]))
            shot = (payload.get("shot") or {})
            step["shot_number"] = shot.get("number")
            fid = _extract_frame_id_from_payload(payload)
            if fid is not None:
                step["frame_id"] = fid

            enh_sum += step["enhanced_score"]

        sequences.append({
            "video": vid,
            "enhanced_score": float(enh_sum),
            "decay_rate": float(decay_rate),
            "total_duration": float(total_dur),
            "steps": seq
        })

    # ===== 5) BLIP2 post-rerank toàn chuỗi (giữ nguyên) =====
    try:
        _post_blip2_rerank_sequences(
            sequences, text_query=(text_query or ""), score_type="itm", recompute_enhanced=True
        )
    except Exception as e:
        logger.warning(f"post BLIP2 combine error: {e}")

    sequences.sort(key=lambda x: x["enhanced_score"], reverse=True)
    return sequences[:num_results] if num_results else sequences


def _post_blip2_rerank_sequences(
    sequences: List[Dict[str, Any]],
    text_query: str,
    score_type: str = "itm",
    batch_size_env: str = "BLIP2_BATCH",
    recompute_enhanced: bool = True,   # ← nếu True thì thay enhanced_score theo BLIP2
) -> None:
    """
    Rerank BLIP2 một lần sau beam-search cho toàn bộ steps trong các sequences.
    - Ghi step['rerank_score'] và step['score_type'] = 'blip2_post'
    - Nếu recompute_enhanced=True: recompute enhanced_score = vis_norm * lambda * blip_norm
    """
    if resources.reranker is None or not sequences or not text_query:
        return

    # 1) Gom tất cả paths duy nhất
    uniq_paths = []
    seen = set()
    for seq in sequences:
        for step in (seq.get("steps") or []):
            p = step.get("file_path")
            if p and p not in seen:
                uniq_paths.append(p); seen.add(p)
    if not uniq_paths:
        return

    # 2) BLIP2 batched
    try:
        batch_size = int(os.getenv(batch_size_env, "15"))
        rr = resources.reranker.rerank_batched(
            uniq_paths, text_query, score_type=score_type, batch_size=batch_size
        )
        path2blip = {x["path"]: float(x["score"]) for x in rr}
    except Exception as e:
        logger.warning(f"post BLIP2 rerank failed: {e}")
        return

    # 3) Gán điểm BLIP2 vào step
    blip_scores = []
    for seq in sequences:
        for step in (seq.get("steps") or []):
            p = step.get("file_path")
            if p in path2blip:
                s = path2blip[p]
                step["rerank_score"] = float(s)
                step["score_type"] = "blip2_post"
                blip_scores.append(s)

    if not blip_scores:
        return

    # 4) Normalize BLIP2 cho kết hợp (0..1)
    lo, hi = min(blip_scores), max(blip_scores)
    rng = max(hi - lo, 1e-9)

    if recompute_enhanced:
        # recompute per-step & per-seq
        for seq in sequences:
            enh_sum = 0.0
            for step in (seq.get("steps") or []):
                b = step.get("rerank_score", None)
                lam = float(step.get("lambda_factor", 1.0))
                vis = float(step.get("vis_norm", step.get("score", 0.0)))
                if b is None:
                    # Nếu step không có BLIP2, giữ nguyên đóng góp cũ
                    step["enhanced_score"] = float(vis * lam)
                else:
                    bnorm = (float(b) - lo) / rng
                    step["enhanced_score"] = float(vis * lam * bnorm)
                enh_sum += float(step["enhanced_score"])

            # cập nhật tổng sequence & cờ đánh dấu
            seq["enhanced_score"] = float(enh_sum)
            seq["blip2_post_applied"] = True
            seq["blip2_score_type"] = score_type
    else:
        for seq in sequences:
            seq["blip2_post_applied"] = True
            seq["blip2_score_type"] = score_type


def _enhanced_beam_sequence_for_video(
    cands_for_video: List[List[Dict[str, Any]]],
    beam_size: int = 8,
    max_gap_seconds: Optional[float] = None,
    decay_rate: float = 0.01
) -> Optional[List[Dict[str, Any]]]:
    """
    Tìm dãy ứng viên với scoring enhanced bằng exponential decay lambda.
    """
    if not cands_for_video:
        return None

    # Seed từ event 0
    firsts = cands_for_video[0][:max(1, min(10, len(cands_for_video[0])))]
    # Khởi tạo beams: (sequence, enhanced_score, lambda_factors)
    beams = [([c0], c0["score"], [1.0]) for c0 in firsts]

    for ei in range(1, len(cands_for_video)):
        nxt_cands = cands_for_video[ei]
        new_beams = []

        for seq, cum_score, lambda_factors in beams:
            prev_ts = seq[-1].get("ts", None)

            for c in nxt_cands:
                ts = c.get("ts", None)
                if ts is None or prev_ts is None:
                    continue
                if ts <= prev_ts:  # Đảm bảo thứ tự thời gian
                    continue

                # Tính gap và lambda
                gap = ts - prev_ts
                if max_gap_seconds is not None and gap > max_gap_seconds:
                    continue

                # Tính lambda factor cho event này
                lambda_factor = calculate_lambda(gap, decay_rate)

                # Enhanced scoring với exponential decay
                event_score = c["score"] * lambda_factor
                new_enhanced_score = cum_score + event_score

                # Lưu lambda factors để debug/display
                new_lambda_factors = lambda_factors + [lambda_factor]

                new_beams.append((
                    seq + [c],
                    new_enhanced_score,
                    new_lambda_factors
                ))

        if not new_beams:
            return None  # Không thể nối chuỗi hợp lệ

        # Giữ top beam_size theo enhanced score
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]

    # Chọn best beam
    best_seq, best_score, best_lambdas = beams[0]

    # Attach lambda info vào sequence
    for i, (cand, lf) in enumerate(zip(best_seq, best_lambdas)):
        cand["lambda_factor"] = lf
        if i > 0:
            prev_ts = best_seq[i-1].get("ts", 0)
            curr_ts = cand.get("ts", 0)
            cand["gap_seconds"] = curr_ts - prev_ts
        else:
            cand["gap_seconds"] = 0.0

    return best_seq

def _fetch_event_candidates(
    event_texts: List[str],
    per_event_k: int,
    score_threshold: float,
    rrf_k: int,
    rrf_weights: Optional[Dict[str, float]],
    parallel: bool = True,
    max_workers: Optional[int] = None,
    use_rerank_per_event: bool = False,   # 🚩 KHÓA NÀY: tránh BLIP2 tranh GPU
    rerank_score_type: str = "itm",
) -> List[List[Dict[str, Any]]]:
    """
    Mỗi event → gọi KIS (SigLIP + BEiT-3 + RRF [+ BLIP2 nếu bật]).
    Trả về list theo event, GIỮ NGUYÊN THỨ TỰ.
    """
    def _search_one(idx_ev: int, ev: str) -> (int, List[Dict[str, Any]]):
        ev_res = process_kis_search(
            text_query=ev,
            image=None,
            num_results=per_event_k,
            score_threshold=score_threshold,
            query_weight=1.0,
            auto_expand=True,          # tránh drift
            num_expansions=3,
            use_rerank=use_rerank_per_event,
            rerank_score_type=rerank_score_type,
            rerank_top_k=per_event_k,
            rrf_k=rrf_k,
            rrf_weights=rrf_weights
        ) or []

        ev_cands = []
        for r in ev_res:
            payload = r.get("payload") or {}
            ts = _payload_timestamp_seconds(payload)
            if ts is None:
                continue
            ev_cands.append({
                "file_path": r.get("file_path"),
                "payload": payload,
                "ts": float(ts),
                "score": float(r.get("score", 0.0)),       # sẽ normalize sau
                "original_score": float(r.get("score", 0.0)),
                "event": ev,
                "srrf_score": r.get("srrf_score"),
                "rrf_score": float(r.get("rrf_score", 0.0)),
                "rerank_score": float(r.get("rerank_score", 0.0)) if r.get("rerank_score") is not None else None,
            })
        return idx_ev, ev_cands

    # Không song song hoặc chỉ 1 event
    if not parallel or len(event_texts) <= 1:
        ordered = []
        for i, ev in enumerate(event_texts):
            _, cands = _search_one(i, ev)
            ordered.append(cands)
        return ordered

    # Song song
    max_workers = max_workers or min(len(event_texts), int(os.getenv("TEMPORAL_EVENT_WORKERS", "4")))
    results_buffer = [None] * len(event_texts)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_search_one, i, ev) for i, ev in enumerate(event_texts)]
        for fut in as_completed(futures):
            idx, cands = fut.result()
            results_buffer[idx] = cands

    # Bảo toàn thứ tự event đầu vào
    return results_buffer


def process_enhanced_temporal_search(
    text_query: str,
    num_results: int = 5,
    per_event_k: int = 30,
    max_events: int = 5,
    score_threshold: float = 0.1,
    same_video_only: bool = True,
    max_gap_seconds: Optional[float] = None,
    decay_rate: float = 0.01,
    rrf_k: int = 60,
    rrf_weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Enhanced Temporal Search:
      - Tách events bằng GPT
      - Mỗi event → KIS (SigLIP+BEiT-3, RRF, BLIP2 nếu sẵn)
      - Chuẩn hoá điểm theo (video,event) bằng min-max
      - Beam search theo thời gian với exponential decay λ = exp(-α·gap)
      - Không cross-video fallback (same_video_only=True)
    """
    if not text_query or not text_query.strip():
        return []

    # 1) GPT tách events
    try:
        events = resources.gpt_service.extract_temporal_events(text_query, max_events=max_events)
    except Exception as e:
        logger.error(f"extract_temporal_events failed: {e}")
        return []
    if not events or len(events) < 2:
        return []

    # 2) Lấy ứng viên cho từng event (KIS + RRF)
    cands_per_event = _fetch_event_candidates(
        events, per_event_k=per_event_k, score_threshold=score_threshold, parallel = True, max_workers = 3, use_rerank_per_event=True,
        rrf_k=rrf_k, rrf_weights=rrf_weights
    )
    # Nếu event nào rỗng → trả rỗng (đúng theo yêu cầu không fallback)
    if any(len(c) == 0 for c in cands_per_event):
        return []

    # 3) Gom theo video (và sort ts tăng trong helper _group_candidates_by_video)
    by_video = _group_candidates_by_video(cands_per_event)
    if not by_video:
        return []

    # 4) Chuẩn hoá điểm theo (video,event) để công bằng
    for vid, ev_lists in by_video.items():
        for ei, cands in enumerate(ev_lists):
            vis = [c["score"] for c in cands] if cands else []
            norm = _minmax_norm(vis)
            for i, c in enumerate(cands):
                c["vis_norm"] = float(norm[i])
                c["score"] = float(norm[i])  # beam sẽ dùng "score"

    # 5) Beam search với exponential decay
    sequences = []
    for vid, ev_lists in by_video.items():
        seq = _enhanced_beam_sequence_for_video(
            cands_for_video=ev_lists,
            beam_size=8,
            max_gap_seconds=max_gap_seconds,
            decay_rate=decay_rate
        )
        if not seq:
            continue

        # enrich + tính tổng điểm enhanced
        start_ts = seq[0]["ts"]; end_ts = seq[-1]["ts"]
        total_dur = max(0.0, end_ts - start_ts)
        enh_sum = 0.0
        for step in seq:
            lam = float(step.get("lambda_factor", 1.0))
            vnorm = float(step.get("vis_norm", step.get("score", 0.0)))
            step["enhanced_score"] = float(vnorm * lam)

            payload = step.get("payload") or {}
            frame = (payload.get("frame") or {})
            step["timestamp_formatted"] = frame.get("timestamp_formatted", _format_time(step["ts"]))
            shot = (payload.get("shot") or {})
            step["shot_number"] = shot.get("number")

            fid = _extract_frame_id_from_payload(payload)
            if fid is not None:
                step["frame_id"] = fid

            enh_sum += step["enhanced_score"]

        sequences.append({
            "video": vid,
            "enhanced_score": float(enh_sum),
            "decay_rate": float(decay_rate),
            "total_duration": float(total_dur),
            "steps": seq
        })
    try:
        _post_blip2_rerank_sequences(
            sequences,
            text_query=text_query,
            score_type="itm",          # hoặc "itc" nếu bạn muốn nhanh hơn
            recompute_enhanced=True,   # True = thay enhanced_score theo BLIP2
        )
    except Exception as e:
        logger.warning(f"post BLIP2 combine error: {e}")

    sequences.sort(key=lambda x: x["enhanced_score"], reverse=True)
    if num_results and num_results > 0:
        sequences = sequences[:num_results]
    return sequences
