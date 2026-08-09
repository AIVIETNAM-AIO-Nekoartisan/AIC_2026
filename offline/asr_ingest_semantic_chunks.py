# scripts/asr_ingest_semantic_chunks.py
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Tuple, Optional, Dict, Any, List

try:
    from elasticsearch import Elasticsearch, helpers  # pip install elasticsearch
except Exception as e:
    print("❗ Cần thư viện 'elasticsearch'. Cài: pip install elasticsearch", file=sys.stderr)
    raise

# -----------------------------
# Helpers: manifest (skip files)
# -----------------------------

def load_manifest(path: Path) -> Dict[str, Tuple[int, float]]:
    if path.is_file():
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            # {abs_path: [size, mtime]}
            return {k: (int(v[0]), float(v[1])) for k, v in data.items()}
        except Exception:
            return {}
    return {}


def save_manifest(path: Path, db: Dict[str, Tuple[int, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: [int(v[0]), float(v[1])] for k, v in db.items()}
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def file_fingerprint(p: Path) -> Tuple[int, float]:
    stat = p.stat()
    return stat.st_size, stat.st_mtime


# -----------------------------
# Misc
# -----------------------------

def now_iso() -> str:
    # ISO UTC ví dụ "2025-09-06T18:11:47.000Z"
    return time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())

def round1(x: Optional[float]) -> Optional[float]:
    try:
        return round(float(x), 3) if x is not None else None
    except Exception:
        return None

def fmt_timestamp(start: Optional[float], end: Optional[float]) -> Optional[str]:
    if start is None or end is None:
        return None
    def _fmt(x: float) -> str:
        v = round(float(x), 1)
        return str(int(v)) if abs(v - int(v)) < 1e-9 else f"{v:.1f}"
    return f"[{_fmt(start)} - {_fmt(end)}]"

def word_count(s: str) -> int:
    s = (s or "").strip()
    if not s:
        return 0
    return len([w for w in s.split() if w])


# -----------------------------
# ES utils
# -----------------------------

def ensure_index(es: Elasticsearch, index: str) -> None:
    if es.indices.exists(index=index):
        print(f"Index '{index}' đã tồn tại.")
        return

    # Mapping đủ dùng cho app (content + content.exact, thời gian, meta)
    body = {
        "mappings": {
            "dynamic": True,
            "properties": {
                "id": {"type": "keyword"},
                "chunk_index": {"type": "integer"},
                "content": {
                    "type": "text",
                    "fields": {
                        "exact": {"type": "keyword", "ignore_above": 32766}
                    }
                },
                "content_type": {"type": "keyword"},
                "language": {"type": "keyword"},
                "start_time": {"type": "double"},
                "end_time": {"type": "double"},
                "duration": {"type": "double"},
                "timestamp": {"type": "keyword"},
                "word_count": {"type": "integer"},
                "file_name": {"type": "keyword"},
                "file_path": {"type": "keyword"},
                "source_file": {"type": "keyword"},
                "payload": {"type": "object", "enabled": True},
                "created_at": {"type": "date"},
                "processed_at": {"type": "date"}
            }
        }
    }
    es.indices.create(index=index, body=body)
    print(f"Tạo index '{index}' với mapping chuẩn hoá.")


def extract_chunks(payload: Any) -> Iterable[Dict[str, Any]]:
    """
    Cố gắng trích các 'chunks' từ nhiều định dạng JSON phổ biến.
    Trả về iterator các dict có ít nhất key 'text'.
    """
    seq = None
    if isinstance(payload, list):
        seq = payload
    elif isinstance(payload, dict):
        for key in ("chunks", "segments", "items", "data", "results"):
            if key in payload and isinstance(payload[key], list):
                seq = payload[key]
                break
        if seq is None:
            # Giả định cả file là 1 item
            seq = [payload]
    else:
        seq = []

    for it in seq:
        if not isinstance(it, dict):
            continue
        text = it.get("text") or it.get("content") or it.get("transcript") or it.get("sentence") or ""
        if not text:
            continue
        start = it.get("start") or it.get("t0") or it.get("start_time")
        end = it.get("end") or it.get("t1") or it.get("end_time")
        yield {
            "text": text,
            "start": start,
            "end": end,
            **{k: v for k, v in it.items() if k not in ("text", "content", "transcript", "sentence")}
        }


def index_file_to_es(
    es: Elasticsearch,
    index: str,
    file_path: Path,
    base_root: Path,
) -> int:
    """
    Đọc 1 JSON và upsert từng chunk vào ES.
    - CHEAT: ép file_name/file_path -> <video_id>.mp4
    - Chuẩn hoá: content, content_type='audio', start_time/end_time, payload.video.name
    _id = f"{video_id}::{chunk_index}" để idempotent.
    """
    with file_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    rel_json = str(file_path.relative_to(base_root)) if file_path.is_relative_to(base_root) else str(file_path)
    video_id = file_path.stem  # ví dụ: K01_V001

    # --- CHEAT: dựng đường dẫn .mp4 bên cạnh file json ---
    mp4_path = file_path.with_suffix(".mp4")
    mp4_rel = str(mp4_path.relative_to(base_root)) if mp4_path.is_relative_to(base_root) else str(mp4_path)
    mp4_name = mp4_path.name

    created_at = now_iso()
    processed_at = created_at

    actions = []
    for i, ch in enumerate(extract_chunks(payload)):
        # Text/content
        text_content = ch.get("text") or ""
        # Time
        s_val = ch.get("start")
        e_val = ch.get("end")
        try:
            start_time = float(s_val) if s_val is not None else None
        except Exception:
            start_time = None
        try:
            end_time = float(e_val) if e_val is not None else None
        except Exception:
            end_time = None
        duration = (end_time - start_time) if (start_time is not None and end_time is not None) else None

        # Doc theo schema app
        doc: Dict[str, Any] = {
            "id": f"{video_id}::{i}",
            "video_id": video_id,
            "chunk_index": i,

            # 🔑 App search theo 'content'
            "content": text_content,
            "content_type": "audio",
            "language": "vi",

            # 🔑 Thời gian app đang dùng
            "start_time": round1(start_time),
            "end_time": round1(end_time),
            "duration": round1(duration),
            "timestamp": fmt_timestamp(start_time, end_time),
            "word_count": word_count(text_content),

            # 🔑 Meta + cheat .mp4
            "file_name": mp4_name,
            "file_path": mp4_rel,
            "source_file": rel_json,

            # 🔑 Để same_video_only/group theo video
            "payload": {"video": {"name": video_id}},

            "created_at": created_at,
            "processed_at": processed_at,
        }

        # (Tuỳ chọn) giữ 'text' cho tương thích ngược
        doc["text"] = text_content

        # Gộp các field còn lại (không đè key đã chuẩn hoá)
        extra = {k: v for k, v in ch.items()
                 if k not in {"text", "content", "transcript", "sentence", "start", "t0", "start_time", "end", "t1", "end_time"}}
        for k, v in extra.items():
            if k not in doc:
                doc[k] = v

        actions.append({
            "_op_type": "update",
            "_index": index,
            "_id": f"{video_id}::{i}",
            "doc": doc,
            "doc_as_upsert": True
        })

    if not actions:
        return 0

    helpers.bulk(es, actions, request_timeout=120)
    return len(actions)


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser("ASR ingest (recursive, skip already ingested) — normalized for app")
    parser.add_argument("--index", required=True, help="Tên index Elasticsearch")
    parser.add_argument("--path", required=True, help="Thư mục gốc (ví dụ: .../data_ASR)")
    parser.add_argument("--es", default=os.getenv("ES_URL", "http://localhost:9200"), help="ES URL (mặc định: http://localhost:9200)")
    parser.add_argument("--pattern", default="*.json", help="Mẫu file (mặc định: *.json)")
    parser.add_argument("--manifest", default=None, help="Đường dẫn file manifest (mặc định: <path>/.asr_ingested_manifest.json)")
    parser.add_argument("--force", action="store_true", help="Bỏ qua manifest, ingest lại tất cả")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ liệt kê file sẽ chạy, không ingest")
    args = parser.parse_args()

    root = Path(args.path).expanduser().resolve()
    if not root.exists():
        print(f"❗ Không tìm thấy: {root}", file=sys.stderr)
        sys.exit(1)

    manifest_path = Path(args.manifest) if args.manifest else (root / ".asr_ingested_manifest.json")
    manifest = {} if args.force else load_manifest(manifest_path)

    # Thu thập file đệ quy
    files = sorted(p for p in root.rglob(args.pattern) if p.is_file())
    print(f"Found {len(files)} JSON file(s). Index: '{args.index}'")

    if args.dry_run:
        for p in files:
            fp = file_fingerprint(p)
            mark = "SKIP" if (str(p) in manifest and manifest[str(p)] == fp) else "RUN"
            print(f"[{mark}] {p}")
        return

    es = Elasticsearch(args.es)
    ensure_index(es, args.index)

    ok = 0
    err = 0
    total = len(files)
    t0 = time.time()

    for idx, p in enumerate(files, 1):
        try:
            fp = file_fingerprint(p)
            if not args.force and str(p) in manifest and manifest[str(p)] == fp:
                print(f"[{idx}/{total}] SKIP | {p.name}")
                continue

            n = index_file_to_es(es, args.index, p, base_root=root)
            print(f"Upserted {n} documents")
            print(f"[{idx}/{total}] OK    | {p.name} | indexed={n} | deleted=0")
            manifest[str(p)] = fp
            ok += 1
        except Exception as e:
            print(f"[{idx}/{total}] FAIL  | {p.name} | {e}", file=sys.stderr)
            err += 1

    save_manifest(manifest_path, manifest)

    dt = time.time() - t0
    print("\n===== SUMMARY =====")
    print(f"Total files : {total}")
    print(f"Success     : {ok}")
    print(f"Errors      : {err}")
    print(f"Time        : {dt:.1f}s")
    print("===================")


if __name__ == "__main__":
    main()

# Ví dụ:
# python -m offline.asr_ingest_semantic_chunks \
#   --index asr_data \
#   --path /root/AIC_2025/z/drive-download-20250906T174531Z-1-001 \
#   --force
