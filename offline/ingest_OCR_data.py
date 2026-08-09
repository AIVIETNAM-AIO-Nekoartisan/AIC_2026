# scripts/ingest_OCR_data.py
# -*- coding: utf-8 -*-
"""
Tự động 2 pha (resume-friendly, per-image):
1) OCR+Caption (MultiLLMOCR, nhiều API key) -> ghi JSONL (<input>/ocr_results.jsonl).
2) Đọc JSONL -> ES upsert nếu có text, Qdrant upsert nếu có caption.
Resume: chỉ coi là "good" khi đã có CAPTION (text có/không không quan trọng).
Chỉ SKIP hoàn toàn khi thiếu cả text & caption trong file kết quả lỗi.

- Qdrant point id = md5(file_path) để idempotent (chạy lại không nhân bản).
- filename/video.name = tiền tố "my_video_h264" tách từ tên file gốc.
- Qdrant dùng collection RIÊNG cho caption (mặc định: keyframe-captions), vector vô danh (size=1024, COSINE).
"""

import os, sys, re, json, uuid, argparse, hashlib, time
from pathlib import Path
from typing import Optional, Dict, Any, List, Iterable
import numpy as np

# project root
sys.path.append(str(Path(__file__).parent.parent))

# OCR đa khoá (nhiều API key, batch-per-request)
from core.models.MultiLLM_OCR import MultiLLMOCR

# ES + Qdrant
from core.services.elasticsearch_service import ElasticsearchService
from core.services.vector_store_optimized import get_qdrant_service
from qdrant_client.http.models import PointStruct

# BGE-M3
try:
    from FlagEmbedding import BGEM3FlagModel
except Exception as e:
    print(f"❌ Không import được FlagEmbedding.BGEM3FlagModel: {e}")
    sys.exit(1)


FRAME_RE = re.compile(r"__frame_(\d+)")
VIDEO_NAME_RE = re.compile(r"^(.*?)(?:__shot_\d+|__pos_|__frame_|__ts_)", re.IGNORECASE)

def _frame_id(fp: str) -> Optional[int]:
    m = FRAME_RE.search(os.path.basename(fp)); return int(m.group(1)) if m else None

def _video_name(fp: str) -> str:
    base = os.path.splitext(os.path.basename(fp))[0]
    m = VIDEO_NAME_RE.match(base)
    return m.group(1) if m else base.split("__", 1)[0]

def _stable_point_id(fp: str) -> str:
    return hashlib.md5(fp.encode("utf-8")).hexdigest()

def _is_jsonl(path: str) -> bool:
    return path.lower().endswith(".jsonl")

def _iter_results_from_file(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        if _is_jsonl(path):
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue
        else:
            data = json.load(f)
            if isinstance(data, dict) and "results" in data:
                for r in data["results"]:
                    if isinstance(r, dict):
                        yield r
            elif isinstance(data, list):
                for r in data:
                    if isinstance(r, dict):
                        yield r

def _resume_buckets(jsonl_path: str):
    """
    Phân loại theo kết quả đã có trong JSON/JSONL (ưu tiên CAPTION):
      - good:      ĐÃ có caption (bất kể text)
      - empty:     thiếu caption (kể cả có text) hoặc cả 2 đều rỗng → OCR lại
      - errors:    status != success
    """
    good, empty, errors = set(), set(), set()
    if not jsonl_path or not os.path.isfile(jsonl_path):
        return good, empty, errors
    for r in _iter_results_from_file(jsonl_path):
        fp = r.get("file_path")
        if not fp:
            continue
        if r.get("status") != "success":
            errors.add(fp)
            continue
        pair = (r.get("ocr_result") or {})
        has_cap = bool((pair.get("caption") or "").strip())
        if has_cap:
            good.add(fp)
        else:
            empty.add(fp)
    return good, empty, errors

def _es_doc(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if result.get("status") != "success":
        return None
    text = ((result.get("ocr_result") or {}).get("text") or "").strip()
    if not text:
        return None
    return {"file_path": result["file_path"], "content": text, "content_type": "image"}

def run_ocr_and_write_jsonl(args, out_path: str) -> List[str]:
    """Chạy OCR+caption (song song, nhiều API key) rồi ghi JSONL theo từng ảnh. Trả về danh sách file đã xử lý."""
    # Init MultiLLMOCR (mở rộng tham số; fallback nếu class cũ)
    try:
        ocr = MultiLLMOCR(
            num_llms=args.num_llms,
            batch_size=args.batch_size,
            max_workers=args.ocr_workers,
            model_name=args.model,
            temperature=args.temp,
            timeout=args.timeout,
            retries=args.retries,
            backoff=args.backoff,
            caption_min_words=args.caption_min_words,
            caption_max_words=args.caption_max_words,
            max_output_tokens=args.max_output_tokens,
        )
    except TypeError:
        ocr = MultiLLMOCR(
            num_llms=args.num_llms,
            batch_size=args.batch_size,
            max_workers=args.ocr_workers
        )

    files_all = ocr.get_image_files(args.input_folder)
    print(f"🔎 Tổng ảnh: {len(files_all)}")

    # Resume
    pending_files = files_all
    if not args.no_resume and os.path.isfile(out_path) and not args.redo_all:
        good, empty, errs = _resume_buckets(out_path)
        if args.redo_errors:
            targets = empty | errs
            pending_files = [p for p in files_all if p in targets]
        else:
            # mặc định: OCR lại mọi file chưa "good" (tức empty + errors)
            pending_files = [p for p in files_all if p not in good]
        print(f"⏩ Resume: good={len(good)}, empty={len(empty)}, errors={len(errs)}")
        print(f"   → sẽ OCR lại: {len(pending_files)} ảnh")

    if not pending_files:
        print("✅ Không còn ảnh cần OCR; dùng lại JSONL cũ.")
        return files_all

    # Ghi JSONL dòng-một
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fp_out = open(out_path, "a", encoding="utf-8") if os.path.isfile(out_path) else open(out_path, "w", encoding="utf-8")
    buf = []; FLUSH = 100

    # trick: chỉ cho MultiLLMOCR thấy pending_files
    orig_get = ocr.get_image_files
    ocr.get_image_files = lambda _dir: pending_files

    total = succ = err = 0
    t0 = time.time()
    try:
        for r in ocr.process_directory_iter(args.input_folder):
            total += 1
            succ += int(r.get("status") == "success")
            err  += int(r.get("status") != "success")
            buf.append(json.dumps(r, ensure_ascii=False) + "\n")
            if len(buf) >= FLUSH:
                fp_out.writelines(buf); fp_out.flush(); buf = []
            if total % 50 == 0:
                print(f"  → OCR {total}/{len(pending_files)} (succ={succ}, err={err})")
    finally:
        if buf:
            fp_out.writelines(buf)
        fp_out.close()
        ocr.get_image_files = orig_get

    dt = time.time() - t0
    print(f"✅ PHASE OCR DONE → {out_path} | processed={total}, success={succ}, error={err}, time={dt:.1f}s")
    return files_all


def upsert_from_json(args, json_path: str, files_all: List[str]) -> None:
    """Đọc JSON/JSONL và upsert ES (text nếu có), Qdrant (caption nếu có)."""
    # ES
    es = ElasticsearchService(host=args.host, port=args.port, index_name=args.index_name)
    es.setup_index()

    # Qdrant + BGE (collection riêng cho caption, vector vô danh 1024, COSINE)
    bge_dim = 1024
    qdrant = get_qdrant_service(
        api_key=args.qdrant_api_key,
        collection_name=args.qdrant_collection,  # mặc định: keyframe-captions
        vector_dim=bge_dim,
        qdrant_url=args.qdrant_url,
    )

    bge = BGEM3FlagModel(args.embed_model_id, use_fp16=(not args.no_embed_fp16))

    total = 0
    es_docs = 0
    qdrant_pts = 0
    skipped_none: List[str] = []      # thiếu cả text & caption
    es_errors: List[Dict[str, Any]] = []
    qdrant_errors: List[Dict[str, Any]] = []

    for r in _iter_results_from_file(json_path):
        fp = r.get("file_path"); total += 1
        if r.get("status") != "success":
            skipped_none.append(fp or "(unknown)")
            continue

        pair = (r.get("ocr_result") or {})
        text = (pair.get("text") or "").strip()
        cap  = (pair.get("caption") or "").strip()

        # ES: nếu có text
        if text:
            try:
                es.upsert_documents([{"file_path": fp, "content": text, "content_type": "image"}])
                es_docs += 1
            except Exception as e:
                es_errors.append({"file_path": fp, "error": str(e)})

        # Qdrant: nếu có caption
        if cap:
            try:
                enc = bge.encode(
                    [cap], batch_size=1, max_length=int(args.embed_max_length),
                    return_dense=True, return_sparse=False, return_colbert_vecs=False,
                )
                dense = enc.get("dense_vecs", None)
                if dense is None:
                    raise RuntimeError("BGE-M3 không trả dense_vecs")
                # L2-normalize để COSINE ổn định
                vec_np = np.asarray(dense[0], dtype=np.float32)
                norm = np.linalg.norm(vec_np) + 1e-12
                vec_np = vec_np / norm
                vec = vec_np.tolist()

                vid = _video_name(fp or "")
                payload = {
                    "file_path": fp,
                    "filename": vid,
                    "video": {"name": vid},
                    "caption": cap,
                }
                fid = _frame_id(fp or "")
                if fid is not None:
                    payload["frame"] = {"index": fid}

                # Upsert vào collection RIÊNG: vector vô danh
                qdrant.upsert_points([
                    PointStruct(
                        id=_stable_point_id(fp or str(uuid.uuid4())),
                        vector=vec,
                        payload=payload
                    )
                ])
                qdrant_pts += 1
            except Exception as e:
                qdrant_errors.append({"file_path": fp, "error": str(e)})

        # Nếu thiếu cả 2 -> skip
        if (not text) and (not cap):
            skipped_none.append(fp or "(unknown)")

        if total % 100 == 0:
            print(f"  → UPSERT {total} items (ES={es_docs}, Qdrant={qdrant_pts}, skipped_none={len(skipped_none)})")

    print("\n✅ PHASE UPSERT DONE")
    print(f"• Tổng kết quả đọc: {total}")
    print(f"• ES upsert (text): {es_docs}")
    print(f"• Qdrant upsert (captions): {qdrant_pts}")
    print(f"• Skip (thiếu cả text & caption): {len(skipped_none)}")
    if es_errors or qdrant_errors:
        print(f"• Lỗi ES: {len(es_errors)} | Lỗi Qdrant/embed: {len(qdrant_errors)}")

    # Lưu log lỗi/skip
    fail_log = {
        "skipped_none": skipped_none,
        "es_errors": es_errors,
        "qdrant_or_embed_errors": qdrant_errors,
    }
    try:
        out_fail = os.path.join(args.input_folder, "ingest_failures.json")
        with open(out_fail, "w", encoding="utf-8") as f:
            json.dump(fail_log, f, ensure_ascii=False, indent=2)
        print(f"📝 Đã lưu lỗi/skip vào: {out_fail}")
    except Exception:
        print("⚠️ Không ghi được ingest_failures.json")

def main():
    ap = argparse.ArgumentParser(description="OCR→JSONL rồi UPSERT ES/Qdrant (caption riêng, skip chỉ khi thiếu cả text & caption).")

    # Input
    ap.add_argument("-i","--input-folder", required=True, help="Thư mục chứa ảnh")
    ap.add_argument("--out", default=None, help="Đường dẫn JSON/JSONL OCR cache (mặc định: <input>/ocr_results.jsonl)")
    ap.add_argument("--no-resume", action="store_true", help="Không resume pha OCR từ JSONL cũ")
    ap.add_argument("--redo-errors", action="store_true", help="Chỉ OCR lại các ảnh empty/error đã có trong JSONL")
    ap.add_argument("--redo-all", action="store_true", help="OCR lại tất cả ảnh, bỏ qua resume")

    # Elasticsearch
    ap.add_argument("-n","--index-name", default="documents")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=9200)

    # Qdrant + BGE-M3
    ap.add_argument("--qdrant-collection", default="keyframe-captions")  # collection riêng cho caption
    ap.add_argument("--qdrant-url", default=None)
    ap.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY"))
    ap.add_argument("--embed-model-id", default="BAAI/bge-m3")
    ap.add_argument("--embed-max-length", type=int, default=128)
    ap.add_argument("--no-embed-fp16", action="store_true")

    # Multi LLM OCR
    ap.add_argument("--num-llms", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=3)
    ap.add_argument("--ocr-workers", type=int, default=None)
    ap.add_argument("--model", default="gemini-2.0-flash")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--backoff", type=float, default=1.5)
    ap.add_argument("--caption-min-words", type=int, default=22)
    ap.add_argument("--caption-max-words", type=int, default=42)
    ap.add_argument("--max-output-tokens", type=int, default=700)

    args = ap.parse_args()

    if not os.path.isdir(args.input_folder):
        print(f"❌ Thư mục không tồn tại: {args.input_folder}")
        sys.exit(1)

    jsonl_path = args.out or os.path.join(args.input_folder, "ocr_results.jsonl")

    # Pha 1: OCR+Caption -> JSONL
    files_all = run_ocr_and_write_jsonl(args, jsonl_path)

    # Pha 2: Đọc JSONL -> upsert ES/Qdrant
    upsert_from_json(args, jsonl_path, files_all)

if __name__ == "__main__":
    main()

# python -m offline.ingest_OCR_data -i results/test_64 -n documents_new7 --model gemini-2.0-flash --temp 0 --retries 1 --timeout 120 --ocr-workers 6 --num-llms 6 --batch-size 1 --qdrant-collection my_test_videos2 --embed-model-id BAAI/bge-m3 --embed-max-length 128 --max-output-tokens 1700 --out results/test_64/ocr_results.jsonl