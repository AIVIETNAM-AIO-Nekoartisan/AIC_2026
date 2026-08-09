# AIC — Pipeline & Workflow

Tài liệu này mô tả luồng xử lý chung cho mọi dataset: chuẩn bị video, xây dựng index đa
phương thức, chạy hệ thống tìm kiếm và kiểm tra trạng thái dịch vụ. Tên collection/index
được cấu hình qua `.env`, không gắn với kích thước hay phiên bản dataset.

## 1. Kiến trúc tổng thể

```text
                         OFFLINE INDEXING
video ─> TransNetV2 ─> keyframe + GIF ─> SigLIP2/Jina/BEiT-3 ─> Qdrant
                                   ├──> caption + BGE-M3       ─> Qdrant
                                   └──> OCR                    ─> Elasticsearch
audio ─> Whisper/VAD ─> transcript                           ─> Elasticsearch

                          ONLINE SEARCH
browser ─> Express :3000 ─> FastAPI :8000 ─> Qdrant :6333
             │                    └─────────> Elasticsearch :9200
             ├──> keyframe/video files
             └──> DRES evaluation server
```

Qdrant lưu vector hình ảnh và caption. Elasticsearch lưu transcript ASR và text OCR.
Backend trộn kết quả nhiều nguồn, frontend hiển thị keyframe và gửi đáp án lên DRES.

## 2. Cấu trúc repository

| Đường dẫn | Trách nhiệm |
|---|---|
| `core/models/` | Wrapper cho SigLIP2, Jina, BEiT-3, BGE/BLIP2, Whisper, OCR và YOLO |
| `core/services/` | Client Qdrant, Elasticsearch và helper truy vấn |
| `offline/` | Tách shot, sinh keyframe/GIF, embedding, caption, ASR, OCR và object detection |
| `online/backend/` | FastAPI, resource lifecycle, KIS, temporal, intelligent search và agent |
| `online/frontend/` | Express proxy và giao diện vanilla JavaScript |
| `tests/` | Unit test và integration smoke test |
| `data/` | Dataset đầu vào local, không commit |
| `results/` | Keyframe, GIF và output sinh ra khi ingest, không commit |
| `qdrant_data/` | Qdrant local storage, không commit |
| `model_weights/` | Checkpoint model local, không commit |

## 3. Data stores

Tên mặc định nằm trong `.env.example` và có thể đổi độc lập cho từng dataset.

| Store | Biến cấu hình | Mặc định | Nội dung |
|---|---|---|---|
| Qdrant | `QDRANT_COLLECTION` | `AIC_data` | Vector `siglip`, `jina`, tùy chọn `beit3`, cùng payload keyframe |
| Qdrant | `QDRANT_CAPTION_COLLECTION` | `keyframe-captions` | Caption vector BGE-M3 |
| Elasticsearch | `ES_ASR_INDEX` | `asr_data` | Transcript theo video và timestamp |
| Elasticsearch | `ES_OCR_INDEX` | `ocr_data` | Text trên khung hình |

Payload keyframe cần giữ tối thiểu tên video, đường dẫn file, shot, frame index và timestamp.
Frontend đọc `file_path`/`gif_preview.path`; vì vậy `DATA_ROOT` phải trỏ tới thư mục chứa các
đường dẫn đó.

## 4. Chuẩn bị môi trường

```bash
cp .env.example .env
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd online/frontend && npm ci && cd ../..
```

Trong `.env`, cấu hình API key, vị trí model, đường dẫn dataset và thiết bị chạy model.
Không commit `.env`, dataset, output hoặc checkpoint.

Các biến quan trọng:

```dotenv
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=AIC_data
QDRANT_CAPTION_COLLECTION=keyframe-captions
ES_HOST=localhost
ES_PORT=9200
ES_ASR_INDEX=asr_data
ES_OCR_INDEX=ocr_data
DATA_ROOT=.
VIDEO_DIRS=data/video,data/videos,videos
```

## 5. Xây dựng index offline

### Video, keyframe và vector ảnh

```bash
python -m offline.ingest_data \
  --input-folder data/video \
  --collection-name AIC_data \
  --disable-object-detection
```

`ingest_data.py` quét video trong input folder, tách shot, lưu keyframe/GIF vào `results/`,
tạo vector ảnh và upsert vào Qdrant. Khi nạp bổ sung, dùng một input folder chỉ chứa batch
mới để tránh xử lý lại video đã hoàn thành.

### Caption

```bash
python -m offline.ingest_captions --help
```

Pipeline caption chọn một frame đại diện mỗi shot, sinh caption, embed bằng BGE-M3 và ghi
vào collection caption riêng.

### ASR

```bash
python -m offline.ingest_ASR_data \
  --input-folder data/video \
  --index-name asr_data
```

Các biến thể `ingest_ASR_parallel.py`, `Enhanced_ASR_ingest.py` và
`asr_ingest_semantic_chunks.py` dùng cho workload cần batch hoặc semantic chunking.

### OCR và object detection

```bash
python -m offline.ingest_OCR_data --help
python -m offline.ingest_object_detection --collection AIC_data
```

OCR ghi text vào Elasticsearch. Object detection cập nhật `payload.object_detection` trực
tiếp trong Qdrant và không cần tạo lại vector.

## 6. Luồng tìm kiếm online

- **KIS:** text/ảnh → query expansion tùy chọn → SigLIP2/Jina/BEiT-3/BGE → Qdrant →
  SRRF → BLIP2 rerank tùy chọn.
- **ASR/OCR:** full-text search trong Elasticsearch, sau đó enrich bằng keyframe gần nhất.
- **Temporal/TRAKE:** tách query thành các event có thứ tự, lấy candidate từng event rồi
  beam search theo video và timestamp.
- **Intelligent:** LLM phân rã query thành nhánh KIS/OCR/ASR và trộn kết quả theo trọng số.
- **Agent:** vòng lặp stateless gọi các công cụ search, tóm tắt candidate và hỏi lại khi cần.

Các endpoint chính:

| Endpoint | Chức năng |
|---|---|
| `POST /search/kis` | Text/image KIS và fusion nhiều encoder |
| `POST /search/audio` | Tìm transcript ASR |
| `POST /search/ocr` | Tìm text OCR |
| `POST /search/temporal/enhanced` | Temporal search từ mô tả tự nhiên |
| `POST /search/temporal/combinate` | Temporal search với field/trọng số tường minh |
| `POST /search/intelligent` | Phân rã và phối hợp đa phương thức |
| `POST /agent/chat` | Search agent nhiều vòng |
| `POST /shot/detail` | Các frame trong một shot |
| `GET /frame/info` | Payload frame cùng ASR/OCR lân cận |
| `GET /health` | Trạng thái model và data store |

## 7. Chạy hệ thống

### Docker Compose

```bash
docker compose up --build
```

Các cổng mặc định: frontend `3000`, backend `8000`, Qdrant `6333`, Elasticsearch `9200`.
Backend GPU trong Docker cần NVIDIA Container Toolkit.

### Chạy thủ công

```bash
# Terminal 1: Qdrant và Elasticsearch
docker compose up qdrant elasticsearch

# Terminal 2: backend, chạy từ repository root
python -m online.backend.api_server

# Terminal 3: frontend
cd online/frontend && npm start
```

## 8. Kiểm tra

```bash
python3 -m unittest discover -s tests -v
node --check online/frontend/server.js
node --check online/frontend/public/app.js
docker compose config -q
```

Integration test tự skip nếu backend, Qdrant hoặc Elasticsearch chưa chạy. Sau khi đã nạp
dataset, kiểm tra `GET /health`, số lượng point/document và thử một query đại diện cho từng
modality trước khi chạy production.

## 9. Nguyên tắc vận hành

- Mỗi dataset hoặc lần rebuild nên dùng namespace collection/index rõ ràng trong `.env`.
- Không dùng chung output folder cho hai job ingest đang chạy đồng thời.
- Giữ `results/` và Qdrant payload đồng bộ vì payload tham chiếu trực tiếp tới file keyframe.
- Backup data store trước khi rebuild; code và runtime data được quản lý tách biệt.
- `transformers` phải giữ dưới major version 5 để tương thích Jina remote code.
- Các model được khởi tạo theo kiểu degrade graceful: model lỗi sẽ tắt tính năng liên quan,
  nhưng không thay thế cho việc kiểm tra `/health` trước khi sử dụng.
