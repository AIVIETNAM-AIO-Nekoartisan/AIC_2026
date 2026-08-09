# offline/ingest_captions.py
"""Sinh caption cho keyframe bằng BLIP2 rồi embed bằng BGE-M3 vào collection
`keyframe-captions` — kênh retrieval thứ 3 (đã được nối sẵn vào SRRF fusion
với tên "caption" trong online/backend/search_kis.py).

Chạy 2 pha tách rời để không phải giữ 2 model trên GPU cùng lúc (máy 8GB VRAM):

  # pha 1: BLIP2 sinh caption -> JSONL (resume được, Ctrl+C thoải mái)
  python -m offline.ingest_captions caption --out results/captions.jsonl

  # pha 2: BGE-M3 embed JSONL -> Qdrant (+ ghi caption vào payload AIC_data để UI hiện)
  python -m offline.ingest_captions embed --inp results/captions.jsonl

Backend nên tắt trước khi chạy (cả hai pha đều cần GPU).
"""
import os
import sys
import json
import time
import argparse
from typing import Dict, List

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv(os.path.join(project_root, ".env"))


def _scroll_points(client, collection, with_payload=("file_path",)):
    """Lấy (id, payload) toàn bộ collection. with_payload=True để lấy đủ payload."""
    payload_sel = with_payload if isinstance(with_payload, bool) else list(with_payload)
    out, offset = [], None
    while True:
        batch, offset = client.scroll(collection_name=collection, with_payload=payload_sel,
                                      with_vectors=False, limit=1000, offset=offset)
        out.extend((p.id, p.payload or {}) for p in batch)
        if offset is None:
            return out


# ------------------------------------------------------------------ pha 1
def cmd_caption(args):
    import torch
    from PIL import Image

    client = QdrantClient(url=args.qdrant_url, timeout=60)
    raw = _scroll_points(client, args.collection, with_payload=["file_path", "shot_id", "shot"])
    points = [(pid, pl.get("file_path")) for pid, pl in raw if pl.get("file_path")]
    points = [(pid, fp) for pid, fp in points if os.path.exists(fp)]

    # 3 keyframe start/middle/end của cùng 1 shot gần như giống hệt nhau -> chỉ
    # caption 1 frame đại diện (ưu tiên middle) rồi nhân bản cho cả shot: nhanh gấp ~3.
    siblings: Dict[str, List[str]] = {}
    if args.per_shot:
        by_shot: Dict[str, List[tuple]] = {}
        for pid, pl in raw:
            fp = pl.get("file_path")
            if not fp or not os.path.exists(fp):
                continue
            key = pl.get("shot_id") or fp
            by_shot.setdefault(key, []).append((pid, fp, (pl.get("shot") or {}).get("position", "")))
        reps, siblings = [], {}
        for key, members in by_shot.items():
            members.sort(key=lambda m: {"middle": 0, "start": 1, "end": 2}.get(m[2], 3))
            rep_id, rep_fp, _ = members[0]
            reps.append((rep_id, rep_fp))
            siblings[rep_id] = [(m[0], m[1]) for m in members[1:]]
        print(f"[CAP] gộp {len(points)} keyframe -> {len(reps)} shot đại diện", flush=True)
        points = reps

    done = set()
    if os.path.exists(args.out):  # resume: bỏ qua ảnh đã caption
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
    todo = [(pid, fp) for pid, fp in points if pid not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"[CAP] {len(points)} keyframe, đã có {len(done)}, cần caption {len(todo)}", flush=True)
    if not todo:
        return

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[CAP] load {args.model} ({dtype}) trên {device}...", flush=True)

    if "blip2" in args.model.lower():
        from transformers import Blip2Processor, Blip2ForConditionalGeneration as CapModel
        processor = Blip2Processor.from_pretrained(args.model)
    else:  # BLIP-1 captioning (nhẹ hơn nhiều, dùng khi BLIP2 không đủ VRAM)
        from transformers import BlipProcessor as Blip2Processor, BlipForConditionalGeneration as CapModel
        processor = Blip2Processor.from_pretrained(args.model)

    # BLIP2-opt-2.7b fp16 chiếm ~7.1GB — không đủ chỗ cho activation trên card 8GB.
    # NF4 (~2.2GB) vừa nhẹ vừa NHANH HƠN int8 vì nhân ma trận vẫn chạy fp16.
    if args.quant in ("4bit", "8bit") and device == "cuda":
        from transformers import BitsAndBytesConfig
        qcfg = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.float16,
                                   bnb_4bit_use_double_quant=True)
                if args.quant == "4bit" else BitsAndBytesConfig(load_in_8bit=True))
        model = CapModel.from_pretrained(
            args.model, quantization_config=qcfg, device_map={"": 0}, low_cpu_mem_usage=True,
        ).eval()
    else:
        model = CapModel.from_pretrained(args.model, torch_dtype=dtype, low_cpu_mem_usage=True).to(device).eval()
    print(f"[CAP] VRAM sau khi load: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    t0, n = time.time(), 0
    with open(args.out, "a", encoding="utf-8") as fout:
        for i in range(0, len(todo), args.batch_size):
            chunk = todo[i:i + args.batch_size]
            images, ids, paths = [], [], []
            for pid, fp in chunk:
                try:
                    images.append(Image.open(fp).convert("RGB")); ids.append(pid); paths.append(fp)
                except Exception as e:
                    print(f"[CAP] bỏ qua {fp}: {e}", flush=True)
            if not images:
                continue
            try:
                # Prompt + min_length ép model tả dài; thiếu 2 thứ này BLIP2 chỉ ra
                # một câu ~8 từ ("a man standing in a room").
                if args.prompt:
                    inputs = processor(images=images, text=[args.prompt] * len(images),
                                       return_tensors="pt").to(device, dtype)
                else:
                    inputs = processor(images=images, return_tensors="pt").to(device, dtype)
                with torch.inference_mode():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=args.max_tokens,
                        min_new_tokens=args.min_tokens,
                        num_beams=args.num_beams,
                        length_penalty=args.length_penalty,   # >1 ưu tiên câu dài
                        repetition_penalty=1.5,               # tránh lặp vòng khi câu dài
                        no_repeat_ngram_size=3,
                    )
                caps = []
                for c in processor.batch_decode(out, skip_special_tokens=True):
                    c = c.strip()
                    if args.prompt and c.lower().startswith(args.prompt.lower()):
                        c = c[len(args.prompt):].strip()   # bỏ phần prompt bị echo lại
                    caps.append(c)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"[CAP] OOM ở batch {i}, giảm batch_size và chạy lại", flush=True)
                continue
            for pid, fp, cap in zip(ids, paths, caps):
                fout.write(json.dumps({"id": pid, "file_path": fp, "caption": cap}, ensure_ascii=False) + "\n")
                # gán cùng caption cho các keyframe còn lại trong shot
                for sib_id, sib_fp in siblings.get(pid, []):
                    fout.write(json.dumps({"id": sib_id, "file_path": sib_fp, "caption": cap},
                                          ensure_ascii=False) + "\n")
            fout.flush()
            n += len(ids)
            if (i // args.batch_size) % 10 == 0:
                rate = n / max(time.time() - t0, 1)
                eta = (len(todo) - n) / max(rate, 1e-6) / 60
                print(f"[CAP] {n}/{len(todo)} ({rate:.1f} ảnh/s, còn ~{eta:.0f} phút) | vd: {caps[0][:70]}", flush=True)

    print(f"[CAP] ✅ XONG {n} caption sau {(time.time()-t0)/60:.1f} phút -> {args.out}", flush=True)


# ------------------------------------------------------------------ pha 2
def cmd_embed(args):
    import numpy as np
    from FlagEmbedding import BGEM3FlagModel

    rows = []
    with open(args.inp, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("caption"):
                    rows.append(r)
            except Exception:
                pass
    print(f"[EMB] {len(rows)} caption cần embed", flush=True)
    if not rows:
        return

    client = QdrantClient(url=args.qdrant_url, timeout=120)
    # payload gốc (file_path, video, shot, frame...) để fusion & UI dùng lại được
    base_payloads = dict(_scroll_points(client, args.collection, with_payload=True))

    try:
        client.get_collection(args.caption_collection)
        print(f"[EMB] collection '{args.caption_collection}' đã tồn tại", flush=True)
    except Exception:
        client.create_collection(
            collection_name=args.caption_collection,
            vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
        )
        print(f"[EMB] tạo collection '{args.caption_collection}' (1024, COSINE)", flush=True)

    # Resume at point granularity: caption point IDs are stable and an upsert is
    # committed before the next batch starts.
    existing_ids = {
        pid
        for pid, _payload in _scroll_points(
            client,
            args.caption_collection,
            with_payload=False,
        )
    }
    if existing_ids:
        rows = [row for row in rows if row["id"] not in existing_ids]
        print(
            f"[EMB] resume: bỏ qua {len(existing_ids)} vector đã có, còn {len(rows)}",
            flush=True,
        )
    if not rows:
        print("[EMB] ✅ collection caption đã đầy đủ", flush=True)
        return

    bge_device = os.getenv("BGE_DEVICE", "cuda")
    model = BGEM3FlagModel(
        os.getenv("BGE_MODEL", "BAAI/bge-m3"),
        use_fp16=bge_device.startswith("cuda"),
        devices=bge_device,
    )
    t0 = time.time()
    for i in range(0, len(rows), args.batch_size):
        chunk = rows[i:i + args.batch_size]
        enc = model.encode([r["caption"] for r in chunk], batch_size=len(chunk) + 1,
                           max_length=128, return_dense=True, return_sparse=False, return_colbert_vecs=False)
        vecs = enc["dense_vecs"]

        points, payload_updates = [], []
        for r, v in zip(chunk, vecs):
            v = np.asarray(v, dtype=np.float32)
            v /= (np.linalg.norm(v) + 1e-12)
            payload = dict(base_payloads.get(r["id"]) or {"file_path": r["file_path"]})
            payload["caption"] = r["caption"]
            points.append(models.PointStruct(id=r["id"], vector=v.tolist(), payload=payload))
            payload_updates.append(r)

        client.upsert(collection_name=args.caption_collection, points=points)
        # ghi caption vào collection chính để UI hiện được ngay trên kết quả KIS
        for r in payload_updates:
            try:
                client.set_payload(collection_name=args.collection,
                                   payload={"caption": r["caption"]}, points=[r["id"]])
            except Exception as e:
                print(f"[EMB] set_payload {r['id']} lỗi: {e}", flush=True)

        if (i // args.batch_size) % 20 == 0:
            print(f"[EMB] {min(i+len(chunk), len(rows))}/{len(rows)}", flush=True)

    print(f"[EMB] ✅ XONG sau {(time.time()-t0)/60:.1f} phút", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Sinh caption (BLIP2) + embed (BGE-M3) cho keyframe")
    ap.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    ap.add_argument("--collection", default="AIC_data")
    ap.add_argument("--caption-collection", default="keyframe-captions")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("caption", help="pha 1: BLIP2 sinh caption -> JSONL")
    c.add_argument("--out", default="results/captions.jsonl")
    c.add_argument("--model", default="Salesforce/blip2-opt-2.7b",
                   help="đổi sang Salesforce/blip-image-captioning-large nếu thiếu VRAM")
    c.add_argument("--batch-size", type=int, default=8)
    c.add_argument("--max-tokens", type=int, default=90)
    c.add_argument("--min-tokens", type=int, default=25)
    c.add_argument("--num-beams", type=int, default=3)
    c.add_argument("--length-penalty", type=float, default=1.4)
    # Prompt phải TRUNG LẬP: nếu liệt kê "people" trước tiên, BLIP2 sẽ tả người ở
    # mọi khung hình và bịa ra người cả khi ảnh không có ai.
    c.add_argument("--prompt", default=("Question: What is shown in this video frame? Describe the scene in "
                                        "detail: the location and setting, the main objects and their colours, "
                                        "what is happening, and any text or graphics on screen. Answer:"),
                   help='đặt "" để tắt prompt (caption ngắn kiểu mặc định)')
    c.add_argument("--limit", type=int, default=0, help="chỉ chạy N ảnh đầu (để benchmark)")
    c.add_argument("--quant", choices=["4bit", "8bit", "none"], default="4bit",
                   help="4bit = nhanh nhất & nhẹ nhất (mặc định); none = fp16, cần >8GB VRAM")
    c.add_argument("--all-frames", dest="per_shot", action="store_false",
                   help="caption từng keyframe thay vì 1 frame đại diện mỗi shot (chậm gấp ~3)")
    c.set_defaults(per_shot=True)
    c.set_defaults(func=cmd_caption)

    e = sub.add_parser("embed", help="pha 2: BGE-M3 embed JSONL -> Qdrant")
    e.add_argument("--inp", default="results/captions.jsonl")
    e.add_argument("--batch-size", type=int, default=64)
    e.set_defaults(func=cmd_embed)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
