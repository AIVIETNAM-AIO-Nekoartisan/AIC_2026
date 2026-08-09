"""Helper thuần cho backend: timestamp/frame utils, enrich ASR, Qdrant lookup, chuẩn hoá điểm."""
import os
import io
import re
import base64
import math
import logging
from collections import defaultdict

import torch
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

from core.utils.parsing import (
    parse_hms_to_seconds as _parse_hms_to_seconds,
    extract_frame_id_from_payload as _extract_frame_id_from_payload,
)
from online.backend.resources import resources

logger = logging.getLogger(__name__)


def _merge_intervals(intervals, slack: float = 0.0):
    """
    intervals: List[Tuple[start, end, idx]]  (idx = index của hit trong list ban đầu)
    Gộp các khoảng chồng lấn/tiếp giáp (có slack).
    Trả về: List[Tuple[start, end, [idx...]]]
    """
    if not intervals:
        return []

    intervals = sorted(intervals, key=lambda x: x[0])
    merged = []
    cur_s, cur_e, cur_idxs = intervals[0][0], intervals[0][1], [intervals[0][2]]

    for s, e, i in intervals[1:]:
        if s <= cur_e + slack:   # chồng lấn/tiếp giáp trong 'slack'
            cur_e = max(cur_e, e)
            cur_idxs.append(i)
        else:
            merged.append((cur_s, cur_e, cur_idxs))
            cur_s, cur_e, cur_idxs = s, e, [i]
    merged.append((cur_s, cur_e, cur_idxs))
    return merged

def _frame_obj_timestamp_seconds(f: Dict[str, Any]) -> Optional[float]:
    """
    Nhận một object frame từ Qdrant (có thể top-level hoặc trong 'frame')
    và trả về timestamp giây (float).
    Ưu tiên: f['timestamp_seconds'] -> f['frame']['timestamp_seconds'] -> f['frame']['timestamp'] ->
             f['frame']['timestamp_formatted'] -> (index/fps).
    """
    try:
        # top-level
        if f.get("timestamp_seconds") is not None:
            return float(f["timestamp_seconds"])
    except Exception:
        pass
    # nested
    fr = f.get("frame") or {}
    for k in ("timestamp_seconds", "timestamp"):
        v = fr.get(k)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
    tsf = fr.get("timestamp_formatted")
    if tsf:
        ts = _parse_hms_to_seconds(tsf)
        if ts is not None:
            return ts
    # index/fps fallback
    try:
        idx = int(fr.get("index"))
    except Exception:
        idx = None
    fps = ((f.get("video") or {}).get("fps")) or fr.get("fps")
    if idx is not None and fps:
        try:
            return float(idx) / float(fps)
        except Exception:
            pass
    return None



def _sample_frames(frames, k: int):
    """
    Lấy k frame phân bố đều, giữ nguyên thứ tự.
    frames: list đã sort theo timestamp
    """
    if not frames or k <= 0:
        return []
    if len(frames) <= k:
        return frames
    step = len(frames) / float(k)
    out, acc = [], 0.0
    for _ in range(k):
        idx = int(acc)
        out.append(frames[idx])
        acc += step
    return out

def enrich_asr_with_frames_batched(
    asr_results: List[Dict[str, Any]],
    context_seconds: float = 8.0,
    max_frames_per_hit: int = 12,
    merge_slack_seconds: float = 1.0,
    max_frames_per_merged_range: int = 2000,
) -> List[Dict[str, Any]]:
    """
    Enrich theo lô:
    - Gom hit theo video
    - Mở rộng mỗi hit ±context_seconds
    - Gộp khoảng (merge) để giảm số lần query Qdrant
    - Query Qdrant theo từng khoảng gộp, rồi phân bổ frames về từng hit
    """
    if not asr_results:
        return asr_results

    # 1) Chuẩn hoá thông tin hit + nhóm theo video
    by_video = {}
    for idx, item in enumerate(asr_results):
        try:
            s = float(item.get("start_time", 0.0))
            e = float(item.get("end_time", s))
        except Exception:
            s, e = 0.0, 0.0
        if e < s:
            s, e = e, s

        # Suy ra video_name (giống logic cũ)
        video_name = item.get("video_name")
        if not video_name:
            fn = (item.get("file_name") or "").strip()
            if fn:
                video_name = os.path.splitext(os.path.basename(fn))[0]
            else:
                video_name = "unknown"

        ext_s = max(0.0, s - context_seconds)
        ext_e = e + context_seconds

        # Lưu extended khoảng để dùng lại khi phân bổ
        item["_ext_start"] = ext_s
        item["_ext_end"] = ext_e
        item["_video_name"] = video_name

        by_video.setdefault(video_name, []).append((ext_s, ext_e, idx))

    # 2) Với mỗi video: gộp khoảng & query một lần
    video_frames_cache = {}   # video -> list frames (mỗi frame có ts đã tính)
    for video, intervals in by_video.items():
        merged = _merge_intervals(intervals, slack=merge_slack_seconds)
        def _fetch_range(ms, me):
            try:
                return resources.qdrant.get_frames_by_timerange(
                    video_name=video,
                    start_s=ms,
                    end_s=me,
                    limit=max_frames_per_merged_range,  # giảm còn ~300–500
                    sample_stride=1,                     # có thể tăng 2–3 nếu mật độ frame dày
                ) or []
            except Exception as e:
                logger.warning(f"get_frames_by_timerange failed for {video} [{ms},{me}]: {e}")
                return []

        all_frames = []
        # chạy song song các merged range của 1 video
        max_workers = min(4, len(merged))  # 2–4 là hợp lý
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_fetch_range, ms, me) for (ms, me, _idxs) in merged]
            for fut in as_completed(futures):
                all_frames.extend(fut.result())


        # sort theo thời gian để sampling mượt
        all_frames = [f for f in all_frames if f.get("timestamp_seconds") is not None]
        all_frames.sort(key=lambda x: x["timestamp_seconds"])
        video_frames_cache[video] = all_frames

    # 3) Phân bổ frames về từng hit theo extended khoảng
    out = []
    for item in asr_results:
        video = item.get("_video_name", "unknown")
        ms = float(item.get("_ext_start", 0.0))
        me = float(item.get("_ext_end", 0.0))

        pool = video_frames_cache.get(video, [])
        # Lọc frames thuộc khoảng extended
        sub = []
        if pool:
            # binary search có thể nhanh hơn, nhưng filter là đủ
            for f in pool:
                ts = f.get("timestamp_seconds", -1.0)
                if ms <= ts <= me:
                    sub.append(f)

        # Sampling để không quá nhiều
        sub = _sample_frames(sub, max_frames_per_hit)

        # Dọn metadata phụ
        item = dict(item)
        item["frames"] = sub
        item["extended_start"] = ms
        item["extended_end"] = me
        # cleanup key tạm
        item.pop("_video_name", None)
        item.pop("_ext_start", None)
        item.pop("_ext_end", None)
        out.append(item)

    return out


def _pick_representative_frame_for_asr(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Chọn frame trong khoảng ASR gần midpoint nhất để hiện thumbnail + lấy timestamp/file_path.
    """
    frames = item.get("frames") or []
    if not frames:
        return None
    s = float(item.get("start_time", 0.0) or 0.0)
    e = float(item.get("end_time", s) or s)
    mid = (s + e) / 2.0
    best, bestd = None, float("inf")
    for f in frames:
        ts = f.get("timestamp_seconds")
        if ts is None:
            continue
        d = abs(float(ts) - mid)
        if d < bestd:
            best, bestd = f, d
    return best or frames[len(frames)//2]



# đặt gần đầu file (chung chỗ cũ)
OCR_RE = re.compile(
    r'(?P<video>L\d+_V\d+).*?__shot_(?P<shot>\d+)__pos_(?P<pos>start|middle|end)__frame_(?P<frame>\d+)__ts_(?P<ts>\d+(?:\.\d+)?)',
    re.IGNORECASE
)
LEGACY_KEYFRAME_RE = re.compile(
    r'(?P<video>[A-Z]\d+_V\d+)_scene_(?P<shot>\d+)_(?P<pos>start|middle|end)',
    re.IGNORECASE,
)

def parse_meta_from_path(fp: str):
    base = os.path.basename(fp) if fp else ""
    m = OCR_RE.search(base)
    if not m:
        m = OCR_RE.search((fp or "").replace("\\", "/"))
    if not m:
        legacy = LEGACY_KEYFRAME_RE.search(base)
        if not legacy:
            return {}
        d = legacy.groupdict()
        return {
            "video": {"name": d["video"]},
            "shot": {"number": int(d["shot"]), "position": d["pos"].lower()},
        }
    d = m.groupdict()
    ts_raw = (d.get("ts") or "").rstrip(".")
    try:
        ts_val = float(ts_raw)
    except Exception:
        print(f"[DBG][parse_meta_from_path] BAD TS '{ts_raw}' from: {fp}")
        return {}
    meta = {
        "video": {"name": d["video"]},
        "shot": {"number": int(d["shot"]), "position": d["pos"]},
        "frame": {"index": int(d["frame"]), "timestamp_seconds": ts_val}
    }
    return meta


def _point_to_frame_dict(pt) -> Optional[Dict[str, Any]]:
    """Chuẩn hoá 1 point Qdrant -> dict khớp với enrich logic hiện có."""
    if not pt or not getattr(pt, "payload", None):
        return None
    payload = dict(pt.payload or {})
    # cố gắng lấy ts (giây)
    ts = None
    try:
        ts = payload.get("timestamp_seconds", None)
        if ts is None:
            ts = ((payload.get("frame") or {}).get("timestamp_seconds"))
        if ts is None:
            ts = ((payload.get("frame") or {}).get("timestamp"))
        if ts is not None:
            ts = float(ts)
    except Exception:
        ts = None

    out = {
        "file_path": payload.get("file_path"),
        "payload": payload,
        "timestamp_seconds": ts
    }
    fid = _extract_frame_id_from_payload(payload)
    if fid is not None:
        out["frame_id"] = fid
    return out

def _lookup_frame_by_index_qdrant(video_name: str, frame_index: int) -> Optional[Dict[str, Any]]:
    """
    Tìm đúng frame theo (video, frame.index) trong Qdrant bằng scroll + filter.
    """
    try:
        flt = Filter(must=[
            FieldCondition(key="video.name", match=MatchValue(value=video_name)),
            FieldCondition(key="frame.index", match=MatchValue(value=int(frame_index)))
        ])
        res = resources.qdrant.client.scroll(
            collection_name=resources.qdrant.collection_name,
            scroll_filter=flt,
            limit=1, with_payload=True, with_vectors=False
        )
        pts = (res[0] or [])
        if not pts:
            return None
        return _point_to_frame_dict(pts[0])
    except Exception as e:
        logger.warning(f"_lookup_frame_by_index_qdrant error: {e}")
        return None

def _lookup_nearest_frame_by_ts_qdrant(
    video_name: str,
    ts: float,
    window: float = 0.5,
    max_window: float = 5.0,
    step: float = 0.5,
) -> Optional[Dict[str, Any]]:
    """
    Tìm frame gần nhất theo timestamp_seconds với cửa sổ mở rộng dần.
    """
    try:
        ts = float(ts)
    except Exception:
        return None

    w = max(0.05, float(window))
    max_w = max(w, float(max_window))
    best = None
    best_d = float("inf")

    while w <= max_w:
        try:
            flt = Filter(must=[
                FieldCondition(key="video.name", match=MatchValue(value=video_name)),
                FieldCondition(key="frame.timestamp_seconds", range=Range(gte=ts - w, lte=ts + w))
            ])
            # lấy về "vừa đủ": 256 là thoải mái cho +/-5s với mật độ ~50fps đã sampling
            res = resources.qdrant.client.scroll(
                collection_name=resources.qdrant.collection_name,
                scroll_filter=flt, limit=256, with_payload=True, with_vectors=False
            )
            pts = (res[0] or [])
            for p in pts:
                d = _point_to_frame_dict(p)
                if not d or d.get("timestamp_seconds") is None:
                    continue
                dist = abs(float(d["timestamp_seconds"]) - ts)
                if dist < best_d:
                    best, best_d = d, dist
            if best is not None:
                break
        except Exception as e:
            logger.warning(f"_lookup_nearest_frame_by_ts_qdrant window={w} error: {e}")
            # tiếp tục tăng cửa sổ

        w += max(0.05, float(step))

    return best

def _es_docs_around(svc, video_name: str, ts: Optional[float], window: float = 10.0, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Lấy document ES (ASR/OCR) của một video quanh timestamp ts (±window giây).
    Schema ingest có thể khác nhau (video_name / file_name / file_path) nên match
    lỏng; nếu lọc theo khoảng thời gian ra rỗng thì fallback bỏ lọc thời gian.
    """
    if svc is None or not video_name:
        return []
    video_match = {"bool": {"should": [
        {"match_phrase": {"video_name": video_name}},
        {"match_phrase": {"file_name": video_name}},
        {"wildcard": {"file_path": f"*{video_name}*"}},
    ], "minimum_should_match": 1}}

    def _run(with_time: bool):
        q: Dict[str, Any] = {"bool": {"must": [video_match]}}
        if with_time and ts is not None:
            q["bool"]["filter"] = [
                {"range": {"start_time": {"lte": float(ts) + window}}},
                {"range": {"end_time": {"gte": float(ts) - window}}},
            ]
        res = svc.client.search(index=svc.index_name, query=q, size=limit,
                                sort=[{"start_time": {"order": "asc", "unmapped_type": "float"}}])
        return [h.get("_source", {}) for h in res.get("hits", {}).get("hits", [])]

    try:
        docs = _run(with_time=True)
        if not docs and ts is not None:
            docs = _run(with_time=False)
        return docs
    except Exception as e:
        logger.warning(f"_es_docs_around({svc.index_name}, {video_name}) lỗi: {e}")
        return []


def _normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    kis = max(float(w.get("kis", 0.0)), 0.0)
    ocr = max(float(w.get("ocr", 0.0)), 0.0)
    asr = max(float(w.get("asr", 0.0)), 0.0)
    total = kis + ocr + asr
    if total <= 0:
        return {"kis": 1.0, "ocr": 0.0, "asr": 0.0}
    return {"kis": kis/total, "ocr": ocr/total, "asr": asr/total}

def _norm_scores_per_component(items: List[Dict[str, Any]], key: str = "score", eps: float = 1e-9) -> Dict[str, float]:
    """
    Min-max per component → {path: score in [0,1]}.
    FIX: nếu tất cả điểm bằng nhau (hoặc chỉ có 1 hit) thì trả 1.0 thay vì 0.0.
    """
    if not items:
        return {}
    vals = [float(x.get(key, 0.0) or 0.0) for x in items]
    lo, hi = min(vals), max(vals)
    rng = hi - lo
    out = {}
    if rng < eps:
        # Trường hợp chỉ 1 hit hoặc tất cả bằng nhau → gán 1.0 cho tất cả
        for it in items:
            p = it.get("file_path")
            if p:
                out[p] = 1.0
        return out

    for it, v in zip(items, vals):
        p = it.get("file_path")
        if p:
            out[p] = (v - lo) / rng
    return out


def enrich_asr_with_frames(asr_results: List[Dict[str, Any]], max_frames: int = 10, context_seconds: float = 5.0) -> List[Dict[str, Any]]:
    """
    Gắn list frames vào mỗi kết quả ASR, bao gồm cả context trước/sau
    """
    enriched = []
    for item in asr_results or []:
        try:
            start_s = item.get("start_time")
            end_s = item.get("end_time")

            if start_s is None or end_s is None:
                enriched.append(item)
                continue

            start_s = float(start_s)
            end_s = float(end_s)

            # Suy ra video name
            video_name = item.get("video_name")
            if not video_name:
                fn = (item.get("file_name") or "").strip()
                if fn:
                    video_name = os.path.splitext(os.path.basename(fn))[0]
                else:
                    video_name = "unknown"

            # Mở rộng khoảng thời gian để lấy context
            extended_start = max(0, start_s - context_seconds)
            extended_end = end_s + context_seconds

            # Lấy tất cả frames trong khoảng mở rộng
            frames = resources.qdrant.get_frames_by_timerange(
                video_name=video_name,
                start_s=extended_start,
                end_s=extended_end,
                limit=max_frames * 2  # Tăng limit để có đủ frames
            )

            # Gắn vào item
            item = dict(item)
            item["frames"] = frames
            item["extended_start"] = extended_start
            item["extended_end"] = extended_end

        except Exception as e:
            logger.warning(f"Failed to enrich ASR with frames: {e}")

        enriched.append(item)

    return enriched


def _minmax_norm(values, eps: float = 1e-9):
    vals = [float(v) for v in values] if values else []
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    rng = hi - lo
    if rng < eps:
        return [0.5 for _ in vals]
    return [(v - lo) / rng for v in vals]

def _payload_timestamp_seconds(payload: Dict[str, Any]) -> Optional[float]:
    """Trả timestamp (giây) từ payload; hỗ trợ số, chuỗi HH:MM:SS.fff, và fallback index/fps."""
    frame = (payload or {}).get("frame", {}) or {}

    # numeric seconds fields trước
    for key in ("timestamp_seconds", "timestamp"):
        v = frame.get(key)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass

    # chuỗi formatted
    tsf = frame.get("timestamp_formatted")
    if tsf:
        ts = _parse_hms_to_seconds(tsf)
        if ts is not None:
            return ts

    # fallback: index / fps
    idx = None
    try:
        idx = int(frame.get("index"))
    except Exception:
        idx = None

    fps = ((payload.get("video") or {}).get("fps")) or frame.get("fps")
    if idx is not None and fps:
        try:
            return float(idx) / float(fps)
        except Exception:
            pass
    return None

def _format_time(seconds: float) -> str:
    """seconds -> HH:MM:SS"""
    try:
        seconds = int(seconds)
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"
    except Exception:
        return ""

def calculate_lambda(gap_seconds: float, decay_rate: float = 0.01) -> float:
    """
    Tính lambda decay factor theo công thức exponential decay.

    Args:
        gap_seconds: Khoảng cách thời gian giữa 2 events (seconds)
        decay_rate: Tốc độ decay (α), mặc định 0.01

    Returns:
        Lambda factor (0-1), với gap càng lớn thì lambda càng nhỏ
    """
    if gap_seconds <= 0:
        return 1.0

    # Exponential decay: λ = e^(-α * gap)
    lambda_factor = math.exp(-decay_rate * gap_seconds)

    # Giới hạn minimum để tránh quá nhỏ
    return max(lambda_factor, 0.001)

def _group_candidates_by_video(cands_per_event: List[List[Dict[str, Any]]]) -> Dict[str, List[List[Dict[str, Any]]]]:
    """
    Nhóm ứng viên theo video. Giữ nguyên thứ tự events.
    cands_per_event: [ [cand_e0...], [cand_e1...], ... ]
    Trả về: { video_name: [list_cands_e0, list_cands_e1, ...] }
    """
    grouped = defaultdict(lambda: [list() for _ in range(len(cands_per_event))])
    for ei, cands in enumerate(cands_per_event):
        for c in cands:
            vid = ((c.get("payload") or {}).get("video") or {}).get("name") or "unknown"
            grouped[vid][ei].append(c)
    # sort mỗi event-list theo timestamp tăng dần (ổn hơn cho ràng buộc thời gian)
    for vid, lists in grouped.items():
        for i in range(len(lists)):
            lists[i].sort(key=lambda x: x.get("ts", float("inf")))
    return grouped


def decode_base64_image(base64_string: str) -> Image.Image:
    try:
        image_data = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid base64 image: {str(e)}")

def _to_device_tensor(np_vec, device, dtype):
    t = torch.from_numpy(np_vec)
    if dtype is not None and t.is_floating_point():
        t = t.to(dtype=dtype)
    return t.to(device)
