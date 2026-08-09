"""Pydantic request/response schemas cho backend API."""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class AgentChatRequest(BaseModel):
    """Lịch sử hội thoại do frontend giữ (stateless ở backend)."""
    messages: List[Dict[str, Any]]


class KISSearchRequest(BaseModel):
    text_query: Optional[str] = None
    image_base64: Optional[str] = None
    num_results: int = Field(default=20)
    score_threshold: float = Field(default=0.2)
    query_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    auto_expand: bool = True
    num_expansions: int = Field(default=4, ge=2, le=10)

    # Rerank
    use_rerank: bool = True
    rerank_top_k: int = Field(default=100, ge=5, le=500)
    rerank_score_type: str = "itm"   # "itc" nhanh hơn; "itm" chính xác hơn

    # RRF
    rrf_k: int = Field(default=60, ge=1, le=1000)
    rrf_weights: Optional[Dict[str, float]] = None  # {"siglip":1.0, "beit3":1.0}

    # Chọn từng kênh embedding; nếu tắt hết thì backend tự bật lại SigLIP.
    use_siglip: bool = True
    use_jina: bool = True
    use_beit3: bool = True
    use_bge: bool = True

    # Giới hạn tìm trong 1 video (video.name đã có payload index)
    video_filter: Optional[str] = None

    # Lọc theo object detection (payload object_detection.* do offline/ingest_object_detection.py nạp)
    # - object_class rỗng -> lọc theo TỔNG số object; có class -> lọc số object của class đó
    # - khớp trong khoảng [object_count - tolerance, object_count + tolerance]
    # Lọc nhiều class cùng lúc: [{"cls": "person", "count": 2}, {"cls": "car"}]
    # (cls rỗng = tổng số object; count rỗng = chỉ cần có mặt ít nhất 1)
    object_filters: Optional[List[Dict[str, Any]]] = None
    # Dạng 1 class (giữ cho tương thích, gộp chung vào object_filters khi xử lý)
    object_class: Optional[str] = None
    object_count: Optional[int] = Field(default=None, ge=0)
    object_count_tolerance: int = Field(default=3, ge=0, le=50)
    # "hard": cắt hẳn ngoài khoảng ±tolerance (Qdrant filter)
    # "soft": không cắt — đúng số lượng thì x1.0, lệch càng xa nhân điểm giảm dần
    #         về object_score_floor (tolerance làm thang decay)
    object_filter_mode: str = Field(default="hard", pattern="^(hard|soft)$")
    object_score_floor: float = Field(default=0.25, ge=0.0, le=1.0)

class TextSearchRequest(BaseModel):
    query: str
    num_results: int = Field(default=20)
    score_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    use_fuzzy: bool = False
    filters: Optional[Dict[str, Any]] = None

class ShotDetailRequest(BaseModel):
    shot_id: str
    limit: int = Field(default=10)

class SearchResponse(BaseModel):
    success: bool
    results: List[Dict[str, Any]]
    total_count: int
    search_time: float
    message: Optional[str] = None

# ---- Enhanced Temporal
class EnhancedTemporalSearchRequest(BaseModel):
    query: str
    num_results: int = Field(default=10)
    per_event_k: int = Field(default=30)
    max_events: int = Field(default=5)
    score_threshold: float = Field(default=0.1)
    same_video_only: bool = True
    max_gap_seconds: Optional[float] = None
    decay_rate: float = Field(default=0.01, ge=0.001, le=0.1)  # α parameter
    rrf_k: int = Field(default=60)
    rrf_weights: Optional[Dict[str, float]] = None  # {"siglip":1.0,"beit3":0.3}

class EventSpec(BaseModel):
    kis: Optional[str] = None   # mô tả hình ảnh/scene (KIS)
    ocr: Optional[str] = None   # text trong ảnh/biển số (OCR)
    asr: Optional[str] = None   # từ khoá lời nói (ASR)

class TemporalCombinateSearchRequest(BaseModel):
    query: str
    num_results: int = Field(default=5, ge=1, le=100)
    per_event_k: int = Field(default=50, ge=10, le=1000)
    max_events: int = Field(default=5, ge=2, le=8)
    score_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    same_video_only: bool = True
    max_gap_seconds: Optional[float] = None
    decay_rate: float = Field(default=0.01, ge=0.001, le=0.1)
    events_explicit: Optional[List[EventSpec]] = None

    # Trọng số multimodal cho MỖI EVENT
    weights: Optional[Dict[str, float]] = None  # {"kis":0.5,"ocr":0.3,"asr":0.2}

    # KIS (SRRF + BLIP2) cho MỖI EVENT
    rrf_k: int = Field(default=60, ge=10, le=200)
    rrf_weights: Optional[Dict[str, float]] = None   # {"siglip":1.0, "beit3":0.3}
    use_rerank_per_event: bool = True
    rerank_score_type: str = "itm"                  # "itc" nhanh hơn
    rerank_top_k: int = Field(default=100, ge=10, le=500)


class IntelligentSearchRequest(BaseModel):
    query: str
    image_base64: Optional[str] = None
    num_results: int = Field(default=20, ge=5, le=100)
    score_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    same_video_only: bool = True
    enable_kis: bool = True
    enable_ocr: bool = True
    enable_asr: bool = True

    # Advanced options
    show_analysis: bool = True
    custom_weights: Optional[Dict[str, float]] = None
    rrf_k: int = Field(default=60, ge=1, le=1000)
    rrf_weights: Optional[Dict[str, float]] = None
