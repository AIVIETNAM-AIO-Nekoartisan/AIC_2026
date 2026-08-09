# offline/ingest_object_detection.py
"""Chạy YOLO (Open Images V7, 600 class) trên toàn bộ keyframe ĐÃ CÓ trong Qdrant
và ghi kết quả vào payload `object_detection` (set_payload — không đụng vector,
không cần re-embed).

Payload ghi vào mỗi điểm:
    object_detection: {
        has_objects: bool,
        object_count: int,                  # tổng số object
        class_counts: {"person": 2, ...},   # đếm theo class (key lowercase)
        unique_classes: ["person", ...],
        max_confidence: float,
    }

Chạy từ repo root:
    python -m offline.ingest_object_detection --collection AIC_data
"""
import os
import sys
import time
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description="Nạp object detection vào payload Qdrant")
    ap.add_argument("--collection", default="AIC_data")
    ap.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    ap.add_argument("--model", default="model_weights/yolov8x-oiv7.pt",
                    help="Thiếu file thì ultralytics tự tải về đúng đường dẫn này")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument(
        "--force",
        action="store_true",
        help="re-detect points that already have object_detection payload",
    )
    args = ap.parse_args()

    client = QdrantClient(url=args.qdrant_url, timeout=60)

    # 1) Lấy toàn bộ (id, file_path) — không kéo vector
    points, offset = [], None
    while True:
        batch, offset = client.scroll(
            collection_name=args.collection,
            with_payload=["file_path", "object_detection"], with_vectors=False,
            limit=1000, offset=offset,
        )
        points.extend(
            (
                p.id,
                (p.payload or {}).get("file_path"),
                (p.payload or {}).get("object_detection"),
            )
            for p in batch
        )
        if offset is None:
            break
    points = [
        (pid, fp)
        for pid, fp, existing in points
        if fp and os.path.exists(fp) and (args.force or existing is None)
    ]
    print(f"[OD] {len(points)} keyframe cần detect", flush=True)

    # 2) Load YOLO (tự tải weights nếu thiếu)
    os.makedirs(os.path.dirname(args.model) or ".", exist_ok=True)
    model = YOLO(args.model if os.path.exists(args.model) else os.path.basename(args.model))
    print(f"[OD] model: {args.model} | conf={args.conf} | batch={args.batch_size}", flush=True)

    # 3) Detect theo batch + set_payload từng điểm
    t0 = time.time()
    done = with_objects = total_objects = errors = 0
    for i in range(0, len(points), args.batch_size):
        chunk = points[i:i + args.batch_size]
        paths = [fp for _, fp in chunk]
        try:
            results = model(paths, conf=args.conf, verbose=False)
        except Exception as e:
            print(f"[OD] LỖI batch tại {i}: {e}", flush=True)
            errors += len(chunk)
            continue

        for (pid, _fp), res in zip(chunk, results):
            class_counts, max_conf = {}, 0.0
            for box in res.boxes:
                cls = model.names[int(box.cls)].lower()
                class_counts[cls] = class_counts.get(cls, 0) + 1
                max_conf = max(max_conf, float(box.conf))
            n = int(sum(class_counts.values()))
            od = {
                "has_objects": n > 0,
                "object_count": n,
                "class_counts": class_counts,
                "unique_classes": sorted(class_counts),
                "max_confidence": round(max_conf, 4),
            }
            try:
                client.set_payload(collection_name=args.collection,
                                   payload={"object_detection": od}, points=[pid])
                done += 1
                with_objects += bool(n)
                total_objects += n
            except Exception as e:
                errors += 1
                print(f"[OD] LỖI set_payload {pid}: {e}", flush=True)

        if (i // args.batch_size) % 20 == 0:
            rate = done / max(time.time() - t0, 1)
            print(f"[OD] {done}/{len(points)} ({rate:.0f} ảnh/s)", flush=True)

    # 4) Index integer cho object_count để filter range nhanh
    try:
        client.create_payload_index(collection_name=args.collection,
                                    field_name="object_detection.object_count",
                                    field_schema=PayloadSchemaType.INTEGER)
        print("[OD] đã tạo payload index cho object_detection.object_count", flush=True)
    except Exception as e:
        print(f"[OD] tạo index lỗi (bỏ qua): {e}", flush=True)

    print(f"\n[OD] ✅ XONG sau {time.time()-t0:.0f}s: {done} điểm | "
          f"{with_objects} có object | tổng {total_objects} object | lỗi {errors}", flush=True)


if __name__ == "__main__":
    main()
