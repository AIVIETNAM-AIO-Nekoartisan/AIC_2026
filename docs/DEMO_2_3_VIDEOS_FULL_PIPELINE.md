# Chạy đầy đủ pipeline tìm kiếm với 2–3 video trên Linux

Tài liệu này dùng **cách 1**:

- Chạy map-keyframes, SigLIP, Jina, object detection, caption, OCR và ASR trực tiếp trên GPU Linux.
- Chạy Qdrant và Elasticsearch bằng Docker.
- Nạp đủ dữ liệu để giao diện có thể search theo hình ảnh/nội dung, caption, OCR và ASR.
- Chỉ dùng 2–3 video để kiểm tra pipeline trước khi chạy toàn bộ dataset.

Các lệnh dưới đây giả định terminal đang ở thư mục gốc của repo.

## 1. Kiến trúc và tên dữ liệu demo

| Loại dữ liệu | Nơi lưu | Tên dùng trong demo |
|---|---|---|
| SigLIP + Jina + payload keyframe | Qdrant | `AIC_demo_visual` |
| Caption BLIP2 + BGE-M3 | Qdrant | `AIC_demo_caption` |
| Caption phụ sinh cùng OCR | Qdrant | `AIC_demo_ocr_aux` |
| OCR text | Elasticsearch | `aic_demo_ocr` |
| ASR transcript | Elasticsearch | `aic_demo_asr` |
| Keyframe WebP đã giải nén | filesystem | `demo/imported-keyframes/` |
| Video gốc | filesystem | `demo/videos/` |

Không dùng lại tên collection/index của dữ liệu cũ. Nhờ vậy demo không ghi đè bộ 10k frame hoặc dữ liệu production.

## 2. Yêu cầu máy

- Linux có NVIDIA GPU và driver hoạt động.
- Docker và Docker Compose plugin.
- Python 3.10 hoặc 3.11.
- Node.js 18+ và npm nếu muốn chạy giao diện web.
- GitHub CLI `gh`, hoặc Git đã có quyền đọc private repo.
- `ffmpeg`, `curl`, `zip`, `unzip`.
- Khoảng 40–60 GB trống cho môi trường, model cache và dữ liệu trung gian. Với 2–3 video thực tế thường dùng ít hơn, nhưng nên để dư.

Cài gói hệ thống trên Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y ffmpeg curl zip unzip python3-venv
```

Kiểm tra GPU và Docker:

```bash
nvidia-smi
docker version
docker compose version
```

## 3. Lấy đúng branch từ private GitHub

Đăng nhập GitHub CLI nếu máy chưa đăng nhập:

```bash
gh auth login
```

Clone đúng branch có bundle map-keyframes:

```bash
gh repo clone AIVIETNAM-AIO-Nekoartisan/AIC_2025 AIC_2025-demo -- \
  --branch codex/colab-bundle \
  --single-branch
cd AIC_2025-demo
```

Nếu đã clone repo trước đó:

```bash
git fetch origin codex/colab-bundle
git switch codex/colab-bundle
git pull --ff-only
```

Kiểm tra đang ở đúng branch:

```bash
git branch --show-current
test -f colab/aic_colab/__main__.py
```

Kết quả branch phải là `codex/colab-bundle`.

## 4. Tạo môi trường Python

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -r colab/requirements-colab.txt
python -m pip install bitsandbytes psutil tqdm
```

Kiểm tra PyTorch nhìn thấy GPU:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
PY
```

Nếu `cuda available: False`, dừng tại đây và cài bản PyTorch CUDA phù hợp với driver theo hướng dẫn chính thức của PyTorch. Không nên chạy encode trên CPU.

## 5. Chuẩn bị `.env`

Tạo file `.env` ở thư mục gốc repo:

```dotenv
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=AIC_demo_visual
QDRANT_CAPTION_COLLECTION=AIC_demo_caption

ES_HOST=localhost
ES_PORT=9200
ES_ASR_INDEX=aic_demo_asr
ES_OCR_INDEX=aic_demo_ocr

ENABLE_SIGLIP=1
ENABLE_JINA=1
ENABLE_BGE=1
ENABLE_BEIT3=0
ENABLE_BLIP2=0
ENABLE_GPT_EXPANSION=0

BGE_DEVICE=cuda
HF_HOME=./demo/hf-cache

DATA_ROOT=.
VIDEO_DIRS=demo/videos

# Cần cho bước OCR. Chỉ điền một trong các dòng dưới.
GOOGLE_API_KEY=THAY_BANG_GEMINI_API_KEY
# GOOGLE_API_KEY_1=THAY_BANG_GEMINI_API_KEY
```

Không commit `.env` hoặc API key lên GitHub. `ENABLE_BLIP2=0` ở đây chỉ ngăn backend web tự nạp BLIP2; bước tạo caption riêng vẫn chạy BLIP2 bằng lệnh bên dưới.

## 6. Khởi động Qdrant và Elasticsearch

Chỉ bật hai datastore trước khi ingest để nhường VRAM cho các model:

```bash
docker compose up -d qdrant elasticsearch
docker compose ps
```

Đợi đến khi cả hai dịch vụ trả lời:

```bash
curl -fsS http://localhost:6333/healthz
curl -fsS http://localhost:9200
```

Nếu Elasticsearch chưa sẵn sàng, xem log:

```bash
docker compose logs --tail=100 elasticsearch
```

## 7. Chuẩn bị 2–3 video và map-keyframes

Tạo cấu trúc demo:

```bash
mkdir -p colab/assets demo/videos demo/work demo/captions demo/asr-output
```

### 7.1. Map-keyframes

File map không nằm trong GitHub vì là dữ liệu dataset. Copy file đã có trên máy nguồn vào:

```bash
cp /media/nekoartisan/Lexar/AIC/videos/map-keyframes-aic25-b1.zip \
  colab/assets/map-keyframes-aic25-b1.zip
```

Nếu chạy trên máy khác, hãy chép file `map-keyframes-aic25-b1.zip` sang máy đó rồi đặt đúng tại:

```text
colab/assets/map-keyframes-aic25-b1.zip
```

### 7.2. Chọn video

Copy 2–3 video vào `demo/videos/`. Tên file không được đổi vì pipeline ghép video với CSV bằng tên stem, ví dụ:

```text
demo/videos/L21_V001.mp4
demo/videos/L21_V002.mp4
demo/videos/L21_V003.mp4
```

Kiểm tra mỗi video có CSV tương ứng trong map ZIP:

```bash
python - <<'PY'
from pathlib import Path
from zipfile import ZipFile

videos = sorted(Path("demo/videos").glob("*"))
with ZipFile("colab/assets/map-keyframes-aic25-b1.zip") as zf:
    map_stems = {Path(name).stem for name in zf.namelist() if name.lower().endswith(".csv")}

missing = [video.name for video in videos if video.is_file() and video.stem not in map_stems]
print("videos:", [video.name for video in videos if video.is_file()])
print("missing maps:", missing)
raise SystemExit(1 if missing else 0)
PY
```

Đóng gói video cho runner:

```bash
cd demo/videos
zip -0 ../Videos_demo.zip ./*
cd ../..
```

### 7.3. Config demo

Tạo `demo/demo.json`:

```json
{
  "dataset_id": "aic-demo-3videos",
  "expected_videos": 873,
  "expected_keyframes": 177321,
  "archives": ["Videos_demo.zip"],
  "siglip_model": "google/siglip2-so400m-patch14-384",
  "siglip_dimension": 1152,
  "jina_model": "jinaai/jina-clip-v2",
  "jina_dimension": 1024,
  "shard_size": 256,
  "webp_quality": 90,
  "deadline_seconds": 21600
}
```

Không đổi `expected_videos` và `expected_keyframes` theo số video demo. Preflight dùng hai trường này để xác minh **toàn bộ map archive gốc**, trong khi `Videos_demo.zip` chỉ chứa 2–3 video sẽ thật sự được xử lý.

## 8. Chạy map-keyframes và visual encoding

Chạy preflight trước:

```bash
PYTHONPATH="$PWD/colab" python -m aic_colab preflight \
  --drive-root "$PWD/demo" \
  --bundle-root "$PWD/colab" \
  --work-root "$PWD/demo/work" \
  --config "$PWD/demo/demo.json"
```

Lần đầu preflight có thể tải SigLIP và Jina nên sẽ mất thời gian và trông như đứng yên. Theo dõi GPU/mạng ở terminal khác bằng `nvidia-smi` và `du -sh demo/work/hf-cache`.

Sau khi preflight trả về `"ready": true`, chạy toàn bộ keyframe thuộc 2–3 video:

```bash
PYTHONPATH="$PWD/colab" python -m aic_colab pilot \
  --limit 1000000 \
  --drive-root "$PWD/demo" \
  --bundle-root "$PWD/colab" \
  --work-root "$PWD/demo/work" \
  --config "$PWD/demo/demo.json"
```

`--limit` là giới hạn số keyframe, không phải số video. Một triệu ở đây là giới hạn an toàn để runner lấy hết keyframe của 2–3 video demo.

Kết quả phải xuất hiện tại:

```text
demo/AIC25-Colab/pilot/aic-demo-3videos/
demo/AIC25-Colab/pilot/report.json
demo/AIC25-Colab/runtime/pilot-state.json
```

Kiểm tra report:

```bash
python -m json.tool demo/AIC25-Colab/pilot/report.json
```

Chỉ tiếp tục khi `status` là `passed`.

## 9. Import visual embedding và keyframe vào Qdrant

```bash
python offline/import_colab_visual.py \
  demo/AIC25-Colab/pilot/aic-demo-3videos \
  demo/imported-keyframes \
  --qdrant-url http://localhost:6333 \
  --collection AIC_demo_visual
```

Nếu lệnh import bị ngắt sau khi collection đã được tạo, chạy lại với `--resume-existing`:

```bash
python offline/import_colab_visual.py \
  demo/AIC25-Colab/pilot/aic-demo-3videos \
  demo/imported-keyframes \
  --qdrant-url http://localhost:6333 \
  --collection AIC_demo_visual \
  --resume-existing
```

### 9.1. Bắt buộc chuẩn hóa payload `file_path`

Importer lưu đường dẫn trong `frame.path`, nhưng caption, object detection và UI đang đọc khóa top-level `file_path`. Chạy đoạn chuyển đổi này đúng một lần sau import:

```bash
python - <<'PY'
from pathlib import Path
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333", timeout=120)
collection = "AIC_demo_visual"
offset = None
updated = 0

while True:
    points, offset = client.scroll(
        collection_name=collection,
        limit=256,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )
    for point in points:
        payload = point.payload or {}
        frame = payload.get("frame") or {}
        file_path = frame.get("path")
        if not file_path:
            raise RuntimeError(f"Point {point.id} không có frame.path")
        client.set_payload(
            collection_name=collection,
            points=[point.id],
            payload={
                "file_path": file_path,
                "filename": Path(file_path).name,
                "source_type": "video_keyframe",
            },
            wait=True,
        )
        updated += 1
    if offset is None:
        break

print({"normalized": updated})
PY
```

Kiểm tra số point và `file_path`:

```bash
python - <<'PY'
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
collection = "AIC_demo_visual"
offset = None
seen = 0
with_path = 0
while True:
    points, offset = client.scroll(
        collection_name=collection,
        limit=256,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )
    seen += len(points)
    with_path += sum(bool((point.payload or {}).get("file_path")) for point in points)
    if offset is None:
        break
total = client.count(collection_name=collection, exact=True).count
print({"total": total, "scrolled": seen, "with_file_path": with_path})
assert total == seen == with_path and total > 0
PY
```

Ba số phải bằng nhau và lớn hơn 0.

## 10. Object detection

Đặt weight có sẵn vào repo:

```bash
mkdir -p model_weights
cp /mnt/data/AIC_2025-main/model_weights/yolov8x-oiv7.pt \
  model_weights/yolov8x-oiv7.pt
```

Trên máy khác, chép `yolov8x-oiv7.pt` vào `model_weights/`. Nếu file chưa có, Ultralytics có thể thử tải theo tên model, nhưng copy weight đã kiểm chứng sẽ ổn định hơn.

Chạy detection:

```bash
python -m offline.ingest_object_detection \
  --collection AIC_demo_visual \
  --qdrant-url http://localhost:6333 \
  --model model_weights/yolov8x-oiv7.pt \
  --batch-size 8
```

Script ghi `object_detection` vào payload của chính point visual, không tạo embedding mới. Lệnh có resume: point đã có object detection sẽ được bỏ qua. Chỉ thêm `--force` khi thật sự muốn chạy lại toàn bộ.

## 11. Caption BLIP2 + BGE-M3

Pha 1 sinh caption vào JSONL. Với GPU khoảng 12 GB, dùng 4-bit và batch nhỏ:

```bash
python -m offline.ingest_captions \
  --qdrant-url http://localhost:6333 \
  --collection AIC_demo_visual \
  --caption-collection AIC_demo_caption \
  caption \
  --out demo/captions/captions.jsonl \
  --model Salesforce/blip2-opt-2.7b \
  --batch-size 2 \
  --quant 4bit \
  --all-frames
```

Pha 2 embed caption bằng BGE-M3 và nạp vào Qdrant:

```bash
BGE_DEVICE=cuda python -m offline.ingest_captions \
  --qdrant-url http://localhost:6333 \
  --collection AIC_demo_visual \
  --caption-collection AIC_demo_caption \
  embed \
  --inp demo/captions/captions.jsonl \
  --batch-size 16
```

Hai pha đều resume được:

- Pha caption bỏ qua ID đã có trong JSONL.
- Pha embed bỏ qua ID đã có trong `AIC_demo_caption`.
- Nếu thiếu VRAM, giảm caption batch xuống `1`; giảm embed batch xuống `8`.

## 12. OCR bằng Gemini và nạp Elasticsearch

OCR hiện dùng `MultiLLMOCR` gọi Gemini API. Repo đang có default cũ `gemini-2.0-flash`; model Gemini 2.0 đã ngừng hoạt động từ ngày 1/6/2026, vì vậy lệnh demo phải override rõ thành `gemini-2.5-flash`. Xem [Gemini API changelog](https://ai.google.dev/gemini-api/docs/changelog) và [Gemini 2.5 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash).

Script OCR chỉ quét ảnh trực tiếp trong một folder, còn keyframe import được chia theo folder video. Vì vậy chạy lần lượt từng folder video:

```bash
find "$PWD/demo/imported-keyframes" \
  -mindepth 2 -maxdepth 2 -type d -print0 |
while IFS= read -r -d '' video_dir; do
  echo "OCR: $video_dir"
  python -m offline.ingest_OCR_data \
    --input-folder "$video_dir" \
    --out "$video_dir/ocr_results.jsonl" \
    --index-name aic_demo_ocr \
    --host localhost \
    --port 9200 \
    --qdrant-url http://localhost:6333 \
    --qdrant-collection AIC_demo_ocr_aux \
    --num-llms 1 \
    --ocr-workers 1 \
    --batch-size 2 \
    --model gemini-2.5-flash \
    --embed-model-id BAAI/bge-m3 \
    --embed-max-length 128
done
```

`AIC_demo_ocr_aux` được tách riêng vì OCR script vừa sinh OCR text vừa sinh một caption phụ. Search OCR chính dùng index Elasticsearch `aic_demo_ocr`; caption BLIP2 chính vẫn dùng `AIC_demo_caption`.

Resume và retry:

- Chạy lại đúng lệnh trên để tiếp tục từ JSONL.
- Thêm `--redo-errors` để gọi lại những ảnh empty/error.
- Không dùng `--redo-all` trừ khi muốn trả phí API và chạy lại tất cả.
- Nếu dùng CPU cho BGE, thêm `--no-embed-fp16` và đổi `BGE_DEVICE=cpu`.

## 13. ASR cho video và nạp Elasticsearch

Dùng `offline.Enhanced_ASR_ingest`, không dùng `offline.ingest_ASR_data` vì script cũ đang hard-code một đường dẫn model trên máy người phát triển.

```bash
python -m offline.Enhanced_ASR_ingest \
  --input-dir demo/videos \
  --output-dir demo/asr-output \
  --include-video \
  --use-vad \
  --device-id 0 \
  --model-name vinai/PhoWhisper-large \
  --language vi \
  --batch-size 2 \
  --index-name aic_demo_asr \
  --es-host localhost \
  --es-port 9200 \
  --checkpoint-file demo/asr-output/checkpoint.json \
  --save-checkpoint-every 1
```

Script lưu JSON ra `demo/asr-output/`, đồng thời nạp transcript vào Elasticsearch.

- Chạy lại lệnh để resume từ checkpoint.
- Thêm `--force-reprocess` chỉ khi muốn chạy lại toàn bộ video.
- Nếu thiếu VRAM, giảm `--batch-size 1`.
- Không bật `--flash` trừ khi máy đã cài và kiểm tra được Flash Attention 2.

## 14. Kiểm tra toàn bộ dữ liệu đã được nạp

### 14.1. Qdrant

```bash
curl -fsS http://localhost:6333/collections/AIC_demo_visual | python -m json.tool
curl -fsS http://localhost:6333/collections/AIC_demo_caption | python -m json.tool
curl -fsS http://localhost:6333/collections/AIC_demo_ocr_aux | python -m json.tool
```

Kiểm tra payload mẫu:

```bash
python - <<'PY'
from qdrant_client import QdrantClient
c = QdrantClient(url="http://localhost:6333")
points, _ = c.scroll("AIC_demo_visual", limit=3, with_payload=True, with_vectors=False)
for point in points:
    payload = point.payload or {}
    print({
        "id": point.id,
        "file_path": payload.get("file_path"),
        "caption": bool(payload.get("caption")),
        "object_detection": bool(payload.get("object_detection")),
    })
PY
```

Mỗi point mẫu phải có `file_path`; sau khi chạy đủ bước phải có thêm `caption` và `object_detection`.

### 14.2. Elasticsearch

```bash
curl -fsS http://localhost:9200/_cat/indices/aic_demo_asr,aic_demo_ocr?v
curl -fsS http://localhost:9200/aic_demo_asr/_count | python -m json.tool
curl -fsS http://localhost:9200/aic_demo_ocr/_count | python -m json.tool
```

Count ASR và OCR phải lớn hơn 0. Count không nhất thiết bằng số keyframe vì ASR lưu theo đoạn transcript và OCR có thể bỏ qua frame không có chữ.

## 15. Chạy backend và giao diện search

Backend cần nạp SigLIP, Jina và BGE. Đợi các bước ingest GPU kết thúc rồi mới bật backend để tránh tranh VRAM.

Terminal 1 (thay đường dẫn bằng nơi vừa clone repo):

```bash
cd /duong/dan/toi/AIC_2025-demo
source .venv/bin/activate
python -m online.backend.api_server
```

Terminal 2:

```bash
cd /duong/dan/toi/AIC_2025-demo
npm ci --prefix online/frontend
API_URL=http://localhost:8000 \
DATA_ROOT="$PWD" \
VIDEO_DIRS=demo/videos \
npm --prefix online/frontend start
```

Mở:

- Backend health: `http://localhost:8000/health`
- Giao diện: `http://localhost:3000`

Kiểm tra health bằng terminal:

```bash
curl -fsS http://localhost:8000/health | python -m json.tool
```

Đợi trạng thái model SigLIP, Jina và BGE thành `ready`. Các mode search cần kiểm tra:

1. KIS bằng mô tả hình ảnh.
2. Caption bằng mô tả cảnh.
3. OCR bằng chữ xuất hiện trong frame.
4. ASR bằng câu thoại trong video.
5. Lọc object nếu UI/backend đang bật bộ lọc object detection.

## 16. Thứ tự chạy khuyến nghị

Để giảm peak VRAM và dễ khoanh lỗi, chạy tuần tự:

1. Qdrant + Elasticsearch.
2. Preflight.
3. Map-keyframes + SigLIP + Jina.
4. Import và chuẩn hóa `file_path`.
5. Object detection.
6. Caption BLIP2.
7. Caption BGE-M3.
8. OCR.
9. ASR.
10. Backend + frontend và test search.

Không chạy BLIP2, YOLO, PhoWhisper và backend model cùng lúc trên GPU 12–16 GB.

## 17. Lỗi thường gặp

### `missing map CSV for ...`

Tên stem của video không trùng tên CSV trong map ZIP. Không rename video; chọn đúng video AIC hoặc đúng map archive.

### `collection already exists`

Import bị chạy lại. Dùng `--resume-existing`. Nếu đây là một lần demo hoàn toàn mới, hãy đổi namespace thay vì xóa nhầm collection cũ.

### Caption/object detection báo không tìm thấy ảnh

Kiểm tra `file_path` trong Qdrant và kiểm tra file thật:

```bash
python - <<'PY'
from pathlib import Path
from qdrant_client import QdrantClient
c = QdrantClient(url="http://localhost:6333")
points, _ = c.scroll("AIC_demo_visual", limit=20, with_payload=True, with_vectors=False)
missing = [p.payload.get("file_path") for p in points if not Path(p.payload.get("file_path", "")).is_file()]
print("missing:", missing)
PY
```

Nếu đã di chuyển repo sau import, các path tuyệt đối cũ sẽ sai. Import lại vào namespace mới hoặc cập nhật toàn bộ `file_path`/`frame.path` theo root mới.

### CUDA out of memory

Tắt backend nếu đang chạy, kiểm tra `nvidia-smi`, rồi giảm batch của bước hiện tại xuống `1`. Mỗi bước đã có cache/checkpoint nên có thể chạy lại.

### OCR báo model không tồn tại

Đảm bảo lệnh có `--model gemini-2.5-flash` và API key hợp lệ trong `.env`. Không dùng default `gemini-2.0-flash`.

### Elasticsearch `connection refused`

```bash
docker compose ps elasticsearch
docker compose logs --tail=100 elasticsearch
curl -fsS http://localhost:9200
```

### Dependency conflict với Gradio/Hugging Face Hub

Pipeline này không cần Gradio. Dùng venv riêng như phần 4; không hạ `huggingface-hub` chỉ để làm hết warning từ một Gradio đã cài ở môi trường khác.

## 18. Xóa riêng dữ liệu demo nếu cần

Các lệnh dưới đây xóa đúng các namespace demo trong database. Chúng không xóa dữ liệu 10k/production, nhưng không thể hoàn tác nếu không có backup:

```bash
curl -X DELETE http://localhost:6333/collections/AIC_demo_visual
curl -X DELETE http://localhost:6333/collections/AIC_demo_caption
curl -X DELETE http://localhost:6333/collections/AIC_demo_ocr_aux

curl -X DELETE http://localhost:9200/aic_demo_asr
curl -X DELETE http://localhost:9200/aic_demo_ocr
```

Không dùng `docker compose down -v` nếu trong Docker volume còn database cần giữ.

## 19. Checklist hoàn thành

- [ ] Đúng branch `codex/colab-bundle`.
- [ ] Có 2–3 video và đủ map CSV tương ứng.
- [ ] Pilot report là `passed`.
- [ ] `AIC_demo_visual` có cả vector `siglip` và `jina`.
- [ ] Mọi visual point có `file_path` hợp lệ.
- [ ] Payload visual có `object_detection`.
- [ ] Payload visual có `caption`.
- [ ] `AIC_demo_caption` có point.
- [ ] Elasticsearch `aic_demo_ocr` có document.
- [ ] Elasticsearch `aic_demo_asr` có document.
- [ ] Backend health không báo lỗi datastore.
- [ ] Search KIS, caption, OCR và ASR đều trả kết quả mở được ảnh/video.
