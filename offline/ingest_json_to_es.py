# scripts/ingest_json_to_es.py
import argparse
import os
import json
from pathlib import Path
from typing import Iterator, Dict, List

from core.services.elasticsearch_service import ElasticsearchService

def iter_docs_from_path(p: Path) -> Iterator[Dict]:
    """
    Đọc .jsonl (mỗi dòng 1 doc) hoặc .json (mảng doc).
    Trả về từng document (dict).
    """
    if p.suffix.lower() == ".jsonl":
        with open(p, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    if isinstance(doc, dict):
                        yield doc
                except Exception:
                    continue
    elif p.suffix.lower() == ".json":
        try:
            data = json.loads(Path(p).read_text(encoding="utf-8"))
            if isinstance(data, list):
                for doc in data:
                    if isinstance(doc, dict):
                        yield doc
            elif isinstance(data, dict):
                # nếu là dict meta → bỏ qua
                pass
        except Exception:
            pass

def load_all_docs(folder: str) -> Iterator[Dict]:
    base = Path(folder)
    for p in sorted(base.glob("**/*.jsonl")):
        yield from iter_docs_from_path(p)
    for p in sorted(base.glob("**/*.json")):
        # tránh nạp trùng nếu file .json là meta
        if p.name.endswith(".meta.json"):
            continue
        yield from iter_docs_from_path(p)

def chunked(it: Iterator[Dict], size: int) -> Iterator[List[Dict]]:
    buf=[]
    for x in it:
        buf.append(x)
        if len(buf)>=size:
            yield buf
            buf=[]
    if buf:
        yield buf

def main():
    ap = argparse.ArgumentParser(description="Ingest exported ASR JSON into Elasticsearch")
    ap.add_argument("--json-dir", "-j", required=True, help="Folder chứa *.jsonl / *.json")
    ap.add_argument("--index-name", "-n", default="video_transcripts")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=9200)
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--timeout", type=int, default=60, help="Request timeout (seconds)")
    args = ap.parse_args()

    # Tạo ES client với timeout lớn hơn
    es = ElasticsearchService(host=args.host, port=args.port, index_name=args.index_name)
    # Nếu bạn đã thêm request_timeout trong constructor thì bỏ qua khúc dưới.
    try:
        # Đảm bảo index tồn tại
        es.setup_index()
        # chờ index sẵn sàng (không fail nếu chưa sẵn sàng)
        try:
            es.client.cluster.health(index=args.index_name, wait_for_status="yellow",
                                     request_timeout=args.timeout)
        except Exception:
            pass
    except Exception as e:
        print(f"⚠️ setup_index cảnh báo: {e}")

    # Nạp & upsert
    total=0; sent=0
    for batch in chunked(load_all_docs(args.json_dir), args.batch_size):
        # ép bổ sung 'created_at' trong upsert_documents đã có sẵn
        try:
            # override timeout chỉ cho bulk nếu cần:
            es.client.options(request_timeout=args.timeout)
            es.upsert_documents(batch)
            sent += len(batch)
        except Exception as e:
            print(f"❌ Lỗi upsert batch: {e}")
        total += len(batch)

    # Stats
    try:
        stats = es.get_stats()
        print(f"✅ Ingest xong. ES docs: {stats.get('document_count','?')} (index ~{stats.get('index_size_mb','?')} MB)")
    except Exception:
        print(f"✅ Ingest xong. (không đọc được stats)")
    print(f"• Tổng doc đọc từ JSON: {total}")
    print(f"• Tổng doc upsert: {sent}")

if __name__ == "__main__":
    main()

# python3 scripts/ingest_json_to_es.py  -j /path/to/asr_json  -n video_transcripts  --host localhost --port 9200 --batch-size 1000 --timeout 60
