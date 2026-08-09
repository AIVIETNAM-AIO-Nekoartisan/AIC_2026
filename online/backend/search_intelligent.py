"""Intelligent search: GPT phân tích query -> chạy KIS/OCR/ASR song song -> RRF đa modal."""
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional

from online.backend.resources import resources
from online.backend.search_kis import process_kis_search
from online.backend.fusion import rrf_weighted_fuse
from online.backend.utils import (
    enrich_asr_with_frames_batched,
    _pick_representative_frame_for_asr,
    _norm_scores_per_component,
    _normalize_weights,
    parse_meta_from_path,
)

logger = logging.getLogger(__name__)


def combine_multimodal_results(
    kis_results: List[Dict[str, Any]],
    ocr_results: List[Dict[str, Any]],
    asr_results: List[Dict[str, Any]],
    weights: Dict[str, float],
    same_video_only: bool = True
) -> List[Dict[str, Any]]:
    from collections import defaultdict

    # Bổ sung video name cho OCR/ASR nếu thiếu (để group theo video)
    for r in (ocr_results or []):
        if not ((r.get("payload") or {}).get("video") or {}).get("name"):
            meta = parse_meta_from_path(r.get("file_path",""))
            if meta:
                r["payload"] = {**(r.get("payload") or {}), **meta}
    for r in (asr_results or []):
        if not ((r.get("payload") or {}).get("video") or {}).get("name"):
            meta = parse_meta_from_path(r.get("file_path",""))
            if meta:
                r["payload"] = {**(r.get("payload") or {}), **meta}

    # Chuẩn hoá từng component (đảm bảo 1-hit = 1.0)
    kis_scores = _norm_scores_per_component(
        kis_results, key=("rerank_score" if any("rerank_score" in x for x in (kis_results or [])) else "score")
    ) if kis_results else {}
    ocr_scores = _norm_scores_per_component(ocr_results or [], key="score") if ocr_results else {}
    asr_scores = _norm_scores_per_component(asr_results or [], key="score") if asr_results else {}

    # Ranges để nhìn
    def _rng(d):
        if not d: return "-"
        vs = list(d.values())
        return f"({min(vs):.3f}, {max(vs):.3f}) len={len(vs)}"
    print("\n[DBG][MM] weights:", weights, "same_video_only:", same_video_only)
    print("[DBG][MM] norm ranges  KIS", _rng(kis_scores), " OCR", _rng(ocr_scores), " ASR", _rng(asr_scores))

    # all paths & map lấy payload theo path
    all_paths = set()
    all_paths.update(kis_scores.keys()); all_paths.update(ocr_scores.keys()); all_paths.update(asr_scores.keys())

    path_to_result = {}
    for r in (kis_results or []) + (ocr_results or []) + (asr_results or []):
        p = r.get("file_path")
        if p and p not in path_to_result:
            path_to_result[p] = r

    def _video_of(p: str) -> str:
        pay = (path_to_result.get(p, {}) or {}).get("payload") or {}
        return ((pay.get("video") or {}).get("name")) or "unknown"

    # Nhóm theo video và lọc theo same_video_only
    if same_video_only:
        video_groups = defaultdict(list)
        for p in list(all_paths):
            if p in path_to_result:
                video_groups[_video_of(p)].append(p)
        # debug: xem video nào có OCR/KIS/ASR
        print("[DBG][MM] video groups (show up to 6):")
        for i, (vid, paths) in enumerate(video_groups.items()):
            if i>=6: break
            c_kis = sum(1 for pp in paths if pp in kis_scores)
            c_ocr = sum(1 for pp in paths if pp in ocr_scores)
            c_asr = sum(1 for pp in paths if pp in asr_scores)
            print(f"  - {vid}: {len(paths)} paths | kis={c_kis} ocr={c_ocr} asr={c_asr}")
        enabled_cnt = sum(1 for w in weights.values() if w > 0)
        valid = set()
        for vid, paths in video_groups.items():
            has_kis = any(pp in kis_scores for pp in paths) and weights.get("kis", 0) > 0
            has_ocr = any(pp in ocr_scores for pp in paths) and weights.get("ocr", 0) > 0
            has_asr = any(pp in asr_scores for pp in paths) and weights.get("asr", 0) > 0
            comps = int(has_kis) + int(has_ocr) + int(has_asr)
            if comps >= 2 or enabled_cnt == 1:
                valid.update(paths)
        all_paths = valid
        print(f"[DBG][MM] valid paths after same_video_only: {len(all_paths)}")

    # Max OCR/ASR theo video để “project” nếu path không có OCR/ASR riêng
    ocr_max_by_video = defaultdict(float)
    asr_max_by_video = defaultdict(float)
    if same_video_only:
        for p, s in ocr_scores.items():
            ocr_max_by_video[_video_of(p)] = max(ocr_max_by_video[_video_of(p)], float(s))
        for p, s in asr_scores.items():
            asr_max_by_video[_video_of(p)] = max(asr_max_by_video[_video_of(p)], float(s))
        print("[DBG][MM] OCR max by video (up to 6):", dict(list(ocr_max_by_video.items())[:6]))
        print("[DBG][MM] ASR max by video (up to 6):", dict(list(asr_max_by_video.items())[:6]))

    combined_results = []
    debug_rows = []

    for path in all_paths:
        if path not in path_to_result:
            continue
        vid = _video_of(path)

        kis_s = kis_scores.get(path, 0.0) if weights.get("kis", 0) > 0 else 0.0

        if weights.get("ocr", 0) > 0:
            if path in ocr_scores:
                ocr_s, ocr_src = ocr_scores[path], "path"
            else:
                ocr_s, ocr_src = (ocr_max_by_video.get(vid, 0.0), "video_max") if same_video_only else (0.0, "-")
        else:
            ocr_s, ocr_src = 0.0, "-"

        if weights.get("asr", 0) > 0:
            if path in asr_scores:
                asr_s, asr_src = asr_scores[path], "path"
            else:
                asr_s, asr_src = (asr_max_by_video.get(vid, 0.0), "video_max") if same_video_only else (0.0, "-")
        else:
            asr_s, asr_src = 0.0, "-"

        combined_score = (
            float(kis_s) * weights.get("kis", 0.0) +
            float(ocr_s) * weights.get("ocr", 0.0) +
            float(asr_s) * weights.get("asr", 0.0)
        )
        if combined_score <= 0:
            continue

        item = dict(path_to_result[path])
        item["component_scores"] = {
            "kis": float(kis_s) if weights.get("kis", 0) > 0 else None,
            "ocr": float(ocr_s) if weights.get("ocr", 0) > 0 else None,
            "asr": float(asr_s) if weights.get("asr", 0) > 0 else None
        }
        item["component_sources"] = {"ocr": ocr_src, "asr": asr_src}
        item["weights_used"] = weights
        item["combined_score"] = float(combined_score)
        item["score"] = float(combined_score)
        combined_results.append(item)

        if len(debug_rows) < 6:
            debug_rows.append((vid, os.path.basename(path), kis_s, ocr_s, asr_s, combined_score, ocr_src, asr_src))

    combined_results.sort(key=lambda x: x["combined_score"], reverse=True)

    # In 6 dòng đầu sau khi combine
    print("[DBG][MM] TOP-6 combined rows:")
    for i, row in enumerate(debug_rows, 1):
        vid, base, ks, os_, as_, comb, o_src, a_src = row
        print(f"  [{i}] vid={vid} file={base}  KIS={ks:.3f}  OCR={os_:.3f}({o_src})  ASR={as_:.3f}({a_src})  -> combined={comb:.4f}")

    return combined_results



def process_intelligent_search(
    query: str,
    num_results: int = 20,
    score_threshold: float = 0.1,
    same_video_only: bool = True,
    enable_kis: bool = True,
    enable_ocr: bool = True,
    enable_asr: bool = True,
    custom_weights: Optional[Dict[str, float]] = None,
    rrf_k: int = 60,                                    # NEW
    rrf_weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Xử lý tìm kiếm thông minh đa phương thức
    """
    result = {
        "success": False,
        "results": [],
        "analysis": None,
        "component_counts": {"kis": 0, "ocr": 0, "asr": 0},
        "total_combined": 0
    }
    try:
        # 1. Phân tích query bằng GPT-4o; fallback local giữ đủ ba modality.
        if resources.gpt_service is not None:
            analysis = resources.gpt_service.analyze_intelligent_query(query)
        else:
            analysis = {
                "components": {
                    "kis": query,
                    "ocr": query,
                    "asr": query,
                },
                "weights": {"kis": 0.5, "ocr": 0.3, "asr": 0.2},
                "strategy": "local_fallback",
            }

        # Merge & normalize weights
        if custom_weights:
            analysis["weights"] = _normalize_weights(custom_weights)
        else:
            analysis["weights"] = _normalize_weights(analysis.get("weights", {}))

        components = analysis["components"]
        weights = analysis["weights"]

        # Thêm vào sau dòng: weights = analysis["weights"]

        print(f"\n🚀 Search Execution Plan:")
        print(f"📝 Query: {query}")
        enabled_components = []
        if enable_kis and components.get("kis") and weights.get("kis", 0) > 0:
            enabled_components.append(f"KIS({weights['kis']:.2f})")
        if enable_ocr and components.get("ocr") and weights.get("ocr", 0) > 0:
            enabled_components.append(f"OCR({weights['ocr']:.2f})")
        if enable_asr and components.get("asr") and weights.get("asr", 0) > 0:
            enabled_components.append(f"ASR({weights['asr']:.2f})")
        print(f"🎯 Enabled: {' + '.join(enabled_components)}")

        # 2. Chuẩn bị các search tasks
        kis_results, ocr_results, asr_results = [], [], []

        with ThreadPoolExecutor(max_workers=3) as ex:
            futures = {}

            if enable_kis and components.get("kis") and weights.get("kis", 0) > 0:
                futures["kis"] = ex.submit(
                    process_kis_search,
                    components["kis"],
                    None,  # image
                    min(num_results * 3, 100),
                    score_threshold,
                    1.0,   # query_weight khi chỉ text
                    True,  # auto_expand
                    3,     # num_expansions
                    True,  # use_rerank
                    "itm", # rerank_score_type
                    50,    # rerank_top_k
                    rrf_k,
                    rrf_weights
                )

            if enable_ocr and components.get("ocr") and weights.get("ocr", 0) > 0:
                futures["ocr"] = ex.submit(
                    resources.es_ocr.search,
                    components["ocr"],
                    min(num_results * 2, 50),
                    score_threshold,
                    {"content_type": "image"}
                )

            if enable_asr and components.get("asr") and weights.get("asr", 0) > 0:
                futures["asr"] = ex.submit(
                    resources.es_audio.search,
                    components["asr"],
                    min(num_results * 2, 50),
                    score_threshold,
                    {"content_type": "audio"}
                )

            # === DEBUG SAU KHI LẤY KẾT QUẢ TỪ 3 NGUỒN ===
            print("\n[DBG] === RAW COMPONENT COUNTS ===")
            print(f"[DBG] KIS: {len(kis_results)} | OCR: {len(ocr_results)} | ASR: {len(asr_results)}")

            # đảm bảo OCR có payload.video
            if ocr_results:
                print(f"[DBG] OCR top-5:")
                for r in ocr_results[:5]:
                    if not ((r.get("payload") or {}).get("video") or {}).get("name"):
                        meta = parse_meta_from_path(r.get("file_path", ""))
                        if meta:
                            r["payload"] = {**r.get("payload", {}) , **meta}
                    vid = (((r.get("payload") or {}).get("video") or {}).get("name")) or "unknown"
                    print(f"  [OCR] file={os.path.basename(r.get('file_path',''))}  score={r.get('score')}  video={vid}")

            # ASR cũng in video name & số frame enrich
            if asr_results:
                print(f"[DBG] ASR top-3:")
                for r in asr_results[:3]:
                    vid = (((r.get("payload") or {}).get("video") or {}).get("name")) or (r.get("video_name") or "unknown")
                    fr = len(r.get("frames") or [])
                    print(f"  [ASR] {r.get('file_name','?')}  [{r.get('start_time')} - {r.get('end_time')}]  video={vid}  frames={fr}")

            # Collect
            try:
                kis_results = futures["kis"].result() if "kis" in futures else []
            except Exception as e:
                logger.warning(f"KIS search failed: {e}")

            try:
                ocr_results = futures["ocr"].result() or [] if "ocr" in futures else []
            except Exception as e:
                logger.warning(f"OCR search failed: {e}")

            try:
                asr_results = futures["asr"].result() or [] if "asr" in futures else []
                if asr_results:
                    asr_results = enrich_asr_with_frames_batched(
                        asr_results,
                        context_seconds=2.0,
                        max_frames_per_hit=6,
                        merge_slack_seconds=1.0,
                        max_frames_per_merged_range=300
                    )
            except Exception as e:
                logger.warning(f"ASR search failed: {e}")

            if enable_ocr and ocr_results:
                for r in ocr_results:
                    # nếu ES chưa có payload.video.name thì parse từ đường dẫn ảnh OCR
                    if not ((r.get("payload") or {}).get("video") or {}).get("name"):
                        meta = parse_meta_from_path(r.get("file_path", ""))
                        if meta:
                            # giữ payload cũ nếu có rồi update thêm meta
                            r["payload"] = {**(r.get("payload") or {}), **meta}


        # 6. Kết hợp kết quả
        combined_results = combine_multimodal_results(
            kis_results=kis_results,
            ocr_results=ocr_results,
            asr_results=asr_results,
            weights=weights,
            same_video_only=same_video_only
        )

        # 7. Giới hạn số lượng kết quả cuối
        final_results = combined_results[:num_results]

        print(f"\n📊 Search Results Summary:")
        print(f"   KIS: {len(kis_results)} results")
        print(f"   OCR: {len(ocr_results)} results")
        print(f"   ASR: {len(asr_results)} results")
        print(f"   Combined: {len(combined_results)} results")
        print(f"   Final: {len(final_results)} results")
        print("=" * 60)

        if combined_results:
            final_results = combined_results[:num_results]
            result = {
                "success": True,
                "results": final_results,
                "analysis": analysis,
                "component_counts": {
                    "kis": len(kis_results),
                    "ocr": len(ocr_results),
                    "asr": len(asr_results)
                },
                "total_combined": len(combined_results)
            }
        else:
            result = {
                "success": True,   # ✅ vẫn thành công, chỉ là không có kết quả
                "results": [],
                "analysis": analysis,
                "component_counts": {
                    "kis": len(kis_results),
                    "ocr": len(ocr_results),
                    "asr": len(asr_results)
                },
                "total_combined": 0
            }

        return result

    except Exception as e:
        logger.error(f"Intelligent search failed: {e}")
        return {
            "success": False,
            "results": [],
            "message": str(e),
            "analysis": None,
            "component_counts": {"kis": 0, "ocr": 0, "asr": 0},
            "total_combined": 0
        }
