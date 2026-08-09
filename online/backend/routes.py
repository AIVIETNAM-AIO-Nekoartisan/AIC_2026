"""Toàn bộ HTTP endpoint (APIRouter) — logic nằm ở các module search_*."""
import os
import logging
from typing import List, Dict, Any, Optional

import base64
import io
from PIL import Image
from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from fastapi.responses import JSONResponse

from online.backend.schemas import (
    KISSearchRequest, TextSearchRequest, ShotDetailRequest, SearchResponse,
    EnhancedTemporalSearchRequest, TemporalCombinateSearchRequest,
    EventSpec, IntelligentSearchRequest, AgentChatRequest,
)
from online.backend.agent import run_agent_turn
from online.backend.resources import resources
from online.backend.search_kis import process_kis_search
from online.backend.search_temporal import (
    process_enhanced_temporal_search,
    process_temporal_combinate_search,
)
from online.backend.search_intelligent import process_intelligent_search
from core.utils.parsing import extract_frame_id_from_payload as _extract_frame_id_from_payload
from online.backend.utils import (
    enrich_asr_with_frames_batched,
    _es_docs_around,
    decode_base64_image,
    parse_meta_from_path,
    _lookup_frame_by_index_qdrant,
    _lookup_nearest_frame_by_ts_qdrant,
    _point_to_frame_dict,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
def health_check():
    snapshot = resources.snapshot()
    return {
        "status": "healthy" if snapshot["kis_ready"] else snapshot["phase"],
        **snapshot,
    }


@router.get("/health/live")
def health_live():
    return {"status": "alive", "phase": resources.phase}


@router.get("/health/ready")
def health_ready():
    snapshot = resources.snapshot()
    status_code = 200 if snapshot["kis_ready"] else 503
    return JSONResponse(status_code=status_code, content=snapshot)

@router.get("/stats/{service}")
def get_stats(service: str):
    try:
        if service == "qdrant":
            if resources.qdrant is None:
                raise HTTPException(status_code=503, detail="Qdrant chưa sẵn sàng")
            points_count = 0
            vectors_count = 0
            try:
                count_res = resources.qdrant.client.count(
                    collection_name=resources.qdrant.collection_name,
                    exact=True
                )
                points_count = int(getattr(count_res, "count", 0))
            except Exception as e:
                logger.warning(f"Qdrant count failed: {e}")

            try:
                col_info = resources.qdrant.client.get_collection(
                    collection_name=resources.qdrant.collection_name
                )
                vc = getattr(col_info, "vectors_count", None)
                if vc is None:
                    try:
                        data = col_info.dict()
                    except Exception:
                        try:
                            data = col_info.model_dump()
                        except Exception:
                            data = {}
                    vc = data.get("vectors_count", None)

                if isinstance(vc, dict):
                    vectors_count = int(vc.get("total", points_count))
                elif vc is not None:
                    vectors_count = int(vc)
                else:
                    vectors_count = points_count
            except Exception as e:
                logger.warning(f"Qdrant get_collection failed: {e}")
                vectors_count = points_count

            return {
                "success": True,
                "stats": {
                    "collection_name": resources.qdrant.collection_name,
                    "points_count": points_count,
                    "vectors_count": vectors_count
                }
            }

        elif service == "es_audio":
            if resources.es_audio is None:
                raise HTTPException(status_code=503, detail="ASR index chưa sẵn sàng")
            stats = resources.es_audio.get_stats()
            return {"success": True, "stats": stats}

        elif service == "es_ocr":
            if resources.es_ocr is None:
                raise HTTPException(status_code=503, detail="OCR index chưa sẵn sàng")
            stats = resources.es_ocr.get_stats()
            return {"success": True, "stats": stats}

        else:
            raise HTTPException(status_code=404, detail=f"Service {service} not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting stats for {service}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/videos/{video_name}/shots")
def get_video_shots(video_name: str, start_shot: int = 1, end_shot: int = 30):
    """Lấy shots trong khoảng start_shot đến end_shot"""
    try:
        shots = resources.qdrant.get_video_shots_range(video_name, start_shot, end_shot)
        return {
            "success": True,
            "video_name": video_name,
            "start_shot": start_shot,
            "end_shot": end_shot,
            "shots": shots,
            "total_count": len(shots)
        }
    except Exception as e:
        logger.error(f"Get video shots error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search/kis", response_model=SearchResponse)
def search_kis(request: KISSearchRequest):
    import time
    start_time = time.time()

    try:
        if not resources.kis_ready:
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "KIS đang khởi tạo model",
                    "state": resources.snapshot(),
                },
            )

        image = None
        if request.image_base64:
            image = decode_base64_image(request.image_base64)

        if not request.text_query and not image:
            return SearchResponse(
                success=False,
                results=[],
                total_count=0,
                search_time=0,
                message="Please provide text query or image"
            )

        results = process_kis_search(
            text_query=request.text_query,
            image=image,
            num_results=request.num_results,
            score_threshold=request.score_threshold,
            query_weight=request.query_weight,
            auto_expand=request.auto_expand,
            num_expansions=request.num_expansions,
            use_rerank=request.use_rerank,
            rerank_score_type=request.rerank_score_type,
            rerank_top_k=request.rerank_top_k,
            rrf_k=request.rrf_k,                       # <-- NEW
            rrf_weights=request.rrf_weights,
            use_siglip=request.use_siglip,
            use_jina=request.use_jina,
            use_beit3=request.use_beit3,
            use_bge=request.use_bge,
            video_filter=request.video_filter,
            object_class=request.object_class,
            object_count=request.object_count,
            object_count_tolerance=request.object_count_tolerance,
            object_filter_mode=request.object_filter_mode,
            object_score_floor=request.object_score_floor,
            object_filters=request.object_filters
        )

        elapsed = time.time() - start_time

        return SearchResponse(
            success=True,
            results=results,
            total_count=len(results),
            search_time=elapsed
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"KIS search error: {e}")
        elapsed = time.time() - start_time
        return SearchResponse(
            success=False,
            results=[],
            total_count=0,
            search_time=elapsed,
            message=str(e)
        )

@router.post("/search/audio", response_model=SearchResponse)
def search_audio(request: TextSearchRequest):
    import time
    start_time = time.time()

    try:
        filters = request.filters or {}
        filters["content_type"] = "audio"

        results = resources.es_audio.search(
            query=request.query,
            limit=request.num_results,
            score_threshold=request.score_threshold,
            filters=filters
        ) or []

        # 👉 Gắn frames theo time range trước khi trả về
        results = enrich_asr_with_frames_batched(
                            results,
                            context_seconds=2.0,
                            max_frames_per_hit=10,
                            merge_slack_seconds=1.0,
                            max_frames_per_merged_range=300
                        )

        elapsed = time.time() - start_time

        return SearchResponse(
            success=True,
            results=results,
            total_count=len(results),
            search_time=elapsed
        )

    except Exception as e:
        logger.error(f"Audio search error: {e}")
        elapsed = time.time() - start_time
        return SearchResponse(
            success=False,
            results=[],
            total_count=0,
            search_time=elapsed,
            message=str(e)
        )

@router.post("/search/ocr", response_model=SearchResponse)
def search_ocr(request: TextSearchRequest):
    import time
    start_time = time.time()

    try:
        filters = request.filters or {}
        filters["content_type"] = "image"

        results = resources.es_ocr.search(
            query=request.query,
            limit=request.num_results,
            score_threshold=request.score_threshold,
            filters=filters
        )

        elapsed = time.time() - start_time

        return SearchResponse(
            success=True,
            results=results or [],
            total_count=len(results) if results else 0,
            search_time=elapsed
        )

    except Exception as e:
        logger.error(f"OCR search error: {e}")
        elapsed = time.time() - start_time
        return SearchResponse(
            success=False,
            results=[],
            total_count=0,
            search_time=elapsed,
            message=str(e)
        )

@router.post("/shot/detail")
def get_shot_detail(request: ShotDetailRequest):
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        filter_condition = Filter(
            must=[
                FieldCondition(
                    key="shot_id",
                    match=MatchValue(value=request.shot_id)
                )
            ]
        )

        results = resources.qdrant.client.scroll(
            collection_name=resources.qdrant.collection_name,
            scroll_filter=filter_condition,
            limit=request.limit,
            with_payload=True,
            with_vectors=False
        )

        frames = []
        if results and results[0]:
            for frame in results[0]:
                payload = frame.payload or {}
                fid = _extract_frame_id_from_payload(payload)
                frames.append({
                    "file_path": payload.get("file_path"),
                    "frame_id": fid,             # <-- NEW
                    "payload": payload
                })

            frames = sorted(frames, key=lambda x: ["start", "middle", "end"].index(
                x["payload"].get("shot", {}).get("position", "middle")
            ))

        return {
            "success": True,
            "shot_id": request.shot_id,
            "frames": frames,
            "total_count": len(frames)
        }

    except Exception as e:
        logger.error(f"Shot detail error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search/temporal/enhanced", response_model=SearchResponse)
def search_temporal_enhanced(request: EnhancedTemporalSearchRequest):
    import time
    start_time = time.time()

    try:
        results = process_enhanced_temporal_search(
            text_query=request.query,
            num_results=request.num_results,
            per_event_k=request.per_event_k,
            max_events=request.max_events,
            score_threshold=request.score_threshold,
            same_video_only=request.same_video_only,
            max_gap_seconds=request.max_gap_seconds,
            decay_rate=request.decay_rate,
            rrf_k=request.rrf_k,                        # NEW
            rrf_weights=request.rrf_weights
        )
        elapsed = time.time() - start_time
        return SearchResponse(
            success=True,
            results=results or [],
            total_count=len(results) if results else 0,
            search_time=elapsed
        )
    except Exception as e:
        logger.error(f"Enhanced temporal search error: {e}")
        elapsed = time.time() - start_time
        return SearchResponse(
            success=False,
            results=[],
            total_count=0,
            search_time=elapsed,
            message=str(e)
        )

@router.post("/search/temporal/combinate", response_model=SearchResponse)
def search_temporal_combinate(request: TemporalCombinateSearchRequest):
    import time
    start_time = time.time()
    try:
        results = process_temporal_combinate_search(
            text_query=request.query,
            num_results=request.num_results,
            per_event_k=request.per_event_k,
            max_events=request.max_events,
            score_threshold=request.score_threshold,
            same_video_only=request.same_video_only,
            max_gap_seconds=request.max_gap_seconds,
            decay_rate=request.decay_rate,
            rrf_k=request.rrf_k,
            rrf_weights=request.rrf_weights,
            use_rerank_per_event=request.use_rerank_per_event,
            rerank_score_type=request.rerank_score_type,
            rerank_top_k=request.rerank_top_k,
            weights=request.weights,
            events_explicit=request.events_explicit,
        )
        elapsed = time.time() - start_time
        return SearchResponse(
            success=True,
            results=results or [],
            total_count=len(results) if results else 0,
            search_time=elapsed
        )
    except Exception as e:
        logger.error(f"Temporal combinate search error: {e}")
        elapsed = time.time() - start_time
        return SearchResponse(
            success=False,
            results=[],
            total_count=0,
            search_time=elapsed,
            message=str(e)
        )

@router.post("/search/intelligent")
def search_intelligent(request: IntelligentSearchRequest):
    import time
    start_time = time.time()

    try:
        # Validate query
        if not request.query or not request.query.strip():
            return {
                "success": False,
                "results": [],
                "total_count": 0,
                "search_time": 0,
                "message": "Query cannot be empty",
                "analysis": None
            }

        # Gọi process function
        result = process_intelligent_search(
            query=request.query,
            num_results=request.num_results,
            score_threshold=request.score_threshold,
            same_video_only=request.same_video_only,
            enable_kis=request.enable_kis,
            enable_ocr=request.enable_ocr,
            enable_asr=request.enable_asr,
            custom_weights=request.custom_weights,
            rrf_k=request.rrf_k,                        # NEW
            rrf_weights=request.rrf_weights
        )

        elapsed = time.time() - start_time

        if result["success"]:
            response_data = {
                "success": True,
                "results": result["results"],
                "total_count": len(result["results"]),
                "search_time": elapsed,
                "analysis": result["analysis"] if request.show_analysis else None,
                "component_counts": result["component_counts"],
                "total_combined": result["total_combined"]
            }
        else:
            response_data = {
                "success": False,
                "results": [],
                "total_count": 0,
                "search_time": elapsed,
                "message": result.get("message", "Unknown error"),
                "analysis": None,
                "component_counts": result["component_counts"],
                "total_combined": 0
            }

        return response_data

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Intelligent search endpoint error: {e}")
        return {
            "success": False,
            "results": [],
            "total_count": 0,
            "search_time": elapsed,
            "message": str(e),
            "analysis": None,
            "component_counts": {"kis": 0, "ocr": 0, "asr": 0},
            "total_combined": 0
        }

@router.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        return {
            "success": True,
            "image_base64": img_base64,
            "filename": file.filename,
            "size": len(contents)
        }

    except Exception as e:
        logger.error(f"Image upload error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/agent/chat")
def agent_chat(request: AgentChatRequest):
    """Một lượt của chế độ Agent: GPT tự gọi tool search, hỏi lại người dùng khi cần."""
    try:
        return {"success": True, **run_agent_turn(request.messages)}
    except Exception as e:
        logger.error(f"agent_chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/frame/info")
def frame_info(
    video: str = Query(..., description="video name, vd L27_V001"),
    frame_index: Optional[int] = Query(None, ge=0),
    ts: Optional[float] = Query(None, description="timestamp giây (nếu không có frame_index)"),
    window: float = Query(10.0, ge=0.5, le=120.0, description="cửa sổ ± giây lấy ASR/OCR quanh frame"),
):
    """Thông tin đầy đủ của 1 keyframe: payload Qdrant (object detection, shot,
    GIF...) + ASR + OCR quanh timestamp — cho modal chi tiết trên UI."""
    try:
        frame = None
        if frame_index is not None:
            frame = _lookup_frame_by_index_qdrant(video, frame_index)
        if frame is None and ts is not None:
            frame = _lookup_nearest_frame_by_ts_qdrant(video, float(ts))
        eff_ts = ts
        if eff_ts is None and frame:
            eff_ts = frame.get("timestamp_seconds")

        asr = _es_docs_around(resources.es_audio, video, eff_ts, window=window)
        ocr = _es_docs_around(resources.es_ocr, video, eff_ts, window=window)
        return {
            "success": True,
            "video": video,
            "ts": eff_ts,
            "frame": frame,                     # payload Qdrant đầy đủ (có object_detection)
            "asr": asr,
            "ocr": ocr,
            "asr_available": resources.es_audio is not None,
            "ocr_available": resources.es_ocr is not None,
        }
    except Exception as e:
        logger.error(f"frame_info failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/frames/lookup")
def frames_lookup(
    video: str = Query(..., description="video name, vd L01_V02"),
    frame_index: Optional[int] = Query(None, ge=0, description="ưu tiên nếu có"),
    ts: Optional[float] = Query(None, description="timestamp (seconds) nếu không có frame_index"),
    window: float = Query(0.5, ge=0.01, description="cửa sổ tìm theo ts (±window)"),
    max_window: float = Query(5.0, ge=0.05, description="mở rộng tối đa"),
):
    """
    Trả về 1 frame (file_path + payload) từ Qdrant theo (video, frame_index) hoặc timestamp gần nhất.
    """
    try:
        ret = None
        if frame_index is not None:
            ret = _lookup_frame_by_index_qdrant(video, int(frame_index))
        elif ts is not None:
            ret = _lookup_nearest_frame_by_ts_qdrant(video, float(ts), window=window, max_window=max_window)
        else:
            raise HTTPException(status_code=400, detail="Provide either frame_index or ts")
        return {"success": True, "result": ret}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/frames/lookup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/videos/{video_name}/frames")
def get_frames_in_range(
    video_name: str,
    start_s: float = Query(..., description="start seconds"),
    end_s: float = Query(..., description="end seconds"),
    limit: int = Query(500, ge=1, le=5000),
    sample_stride: int = Query(1, ge=1, description="lấy 1 frame mỗi N mẫu"),
):
    """
    Public endpoint wrap lại resources.qdrant.get_frames_by_timerange để client dùng thẳng.
    """
    try:
        frames = resources.qdrant.get_frames_by_timerange(
            video_name=video_name,
            start_s=float(start_s),
            end_s=float(end_s),
            limit=int(limit),
            sample_stride=int(sample_stride),
        ) or []
        # đảm bảo sort theo thời gian
        frames = [f for f in frames if f.get("timestamp_seconds") is not None]
        frames.sort(key=lambda x: x["timestamp_seconds"])
        return {"success": True, "video": video_name, "frames": frames, "total_count": len(frames)}
    except Exception as e:
        logger.error(f"/videos/{video_name}/frames error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
