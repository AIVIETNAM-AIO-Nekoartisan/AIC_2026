# AIC — Hệ thống truy vấn video đa phương thức

Hệ thống truy vấn sự kiện từ kho video lớn (Event Retrieval from Large Video Collection) cho cuộc thi **HCMC AI Challenge** — định dạng tương tự Video Browser Showdown (VBS): tìm chính xác một đoạn video theo mô tả văn bản/ảnh (KIS), theo lời nói (ASR), theo chữ trên màn hình (OCR), theo chuỗi sự kiện thời gian (TRAKE), rồi nộp đáp án `(video, khoảng thời gian)` lên server chấm điểm **DRES** trong thời gian thực.

Thiết kế pipeline, data store và quy trình ingest tổng quát được mô tả trong
[`PIPELINE.md`](PIPELINE.md).

## Kiến trúc tổng thể

```
                    ┌─────────────── OFFLINE (trước cuộc thi) ───────────────┐
 video .mp4 ──► TransNetV2 tách shot ──► keyframe + GIF ──► SigLIP2 + Jina + BEiT-3 ──► Qdrant
 audio      ──► Whisper (VAD) ──► transcript ──► Elasticsearch (asr_data)
 keyframe   ──► LLM OCR ──► text trên hình ──► Elasticsearch (ocr_data)
 keyframe   ──► BLIP caption ──► BGE-M3 embed ──► Qdrant (caption collection)

                    ┌─────────────── ONLINE (khi thi) ────────────────────────┐
 Trình duyệt ──► Frontend Node.js :3000 ──► Backend FastAPI :8000 ──► Qdrant :6333
              (proxy /api, /dres, /files)   (embed query, fusion,     Elasticsearch :9200
                                             rerank, GPT expand)
                                        └──► DRES server (nộp đáp án)
```

**Luồng một lần search KIS**: query tiếng Việt → (tuỳ chọn) GPT-4o dịch + sinh N query mở rộng → embed bằng SigLIP2, Jina-CLIP-v2, BEiT-3 và BGE-M3 → search các vector/caption collection trong Qdrant → trộn kết quả bằng RRF có trọng số → (tuỳ chọn) BLIP2 rerank top-k → trả về grid keyframe có GIF preview → click nộp DRES.

## Cấu trúc thư mục

```
core/                       # Code DÙNG CHUNG cho cả offline lẫn online
├── config.py               #   đọc .env (OPENAI_API_KEY, QDRANT_URL, ...)
├── models/                 #   mỗi file 1 wrapper model
│   ├── SigLIP_embedding.py       # SigLIP2 so400m-384 (vector "siglip", 1152d)
│   │                             #   - bắt buộc lowercase + pad đủ 64 token (transformers#43054)
│   │                             #   - query >64 token: chia cửa sổ chồng lấp rồi average
│   ├── JinaCLIP_embedding.py     # Jina-CLIP-v2 (vector "jina", 1024d, đa ngôn ngữ, text 8192 token)
│   ├── BLIP2_rerank.py           # BLIP2 ITM/ITC rerank top-k (ENABLE_BLIP2)
│   ├── GPT4o_service.py          # dịch + mở rộng query, tách event, phân tích intelligent search
│   ├── Whisper_VAD.py            # ASR tiếng Việt (dùng khi ingest)
│   ├── MultiLLM_OCR.py           # OCR keyframe bằng LLM vision (dùng khi ingest)
│   ├── Beit3_embedding_optimized.py  # tuỳ chọn, cần weights trong model_weights/
│   └── transnetv2_pytorch.py     # model tách shot
├── services/
│   ├── vector_store_optimized.py # QdrantService: named vectors, cache, filter theo video
│   └── elasticsearch_service.py  # index/search ASR + OCR
└── utils/parsing.py         #   parse timestamp hh:mm:ss, rút frame-id từ payload/tên file

offline/                    # Pipeline INDEXING — chạy xong trước khi thi
├── ingest_data.py           #   video → shot → keyframe/GIF → embed 2 model → Qdrant
│                            #   python -m offline.ingest_data --input-folder data/video \
│                            #     --collection-name AIC_data --disable-object-detection
├── video_trans_detection.py #   TransNetV2 + lưu keyframe (start/middle/end mỗi shot) + GIF
├── database_processing.py   #   đọc keyframe từ results/ → upsert Qdrant (siglip + jina [+ beit3])
├── ingest_ASR_data.py       #   Whisper → transcript → ES asr_data (các biến thể *_parallel, Enhanced_*)
├── ingest_OCR_data.py       #   LLM OCR → ES ocr_data (cần GOOGLE_API_KEY/OPENAI_API_KEY)
└── ...                      #   asr_export_json, ingest_json_to_es, save_images_and_gifs, av1_to_h264

online/                     # Những gì chạy TRỰC TIẾP khi thi
├── backend/                 # FastAPI (python -m online.backend.api_server, chạy từ repo root)
│   ├── api_server.py        #   entrypoint: app + CORS + include router
│   ├── schemas.py           #   Pydantic request/response
│   ├── resources.py         #   AppResources + lifespan: init toàn bộ model/DB,
│   │                        #   mỗi init bọc try/except → thiếu gì tắt tính năng đó, không crash
│   ├── routes.py            #   mọi endpoint (APIRouter)
│   ├── search_kis.py        #   pipeline KIS: expand → embed → Qdrant → SRRF → BLIP2
│   ├── search_temporal.py   #   temporal enhanced + combinate (beam search theo thời gian)
│   ├── search_intelligent.py#   GPT phân tích query → KIS/OCR/ASR song song → RRF đa modal
│   ├── fusion.py            #   RRF / SRRF có trọng số
│   └── utils.py             #   helper frame/timestamp, enrich ASR bằng frames, Qdrant lookup
└── frontend/                # Node.js/Express + vanilla JS (không cần build)
    ├── server.js            #   proxy: /api/* → backend, /dres/* → DRES (né CORS),
    │                        #   /files?p= → ảnh keyframe dưới DATA_ROOT (chặn path traversal)
    └── public/              #   SPA: index.html + app.js + style.css

tests/                      # unittest chuẩn thư viện (không cần pytest)
data/                       # video nguồn (gitignore)
results/                    # keyframe + GIF sinh ra khi ingest, payload Qdrant trỏ vào đây
model_weights/              # checkpoint BEiT-3, TransNetV2 (gitignore)
```

## Model sử dụng

| Model | Vai trò | Ghi chú |
|---|---|---|
| SigLIP2 `so400m-patch14-384` | Embed ảnh/text chính (vector `siglip`, 1152d) | Text tối đa 64 token (~60 từ tiếng Việt); code tự lowercase + pad 64 + windowing câu dài |
| Jina-CLIP-v2 | Embed ảnh/text song song (vector `jina`, 1024d) | Đa ngôn ngữ (89 thứ tiếng, có tiếng Việt), text tới 8192 token |
| BLIP2 `blip2-itm-vit-g` | Rerank top-k (ITM chính xác / ITC nhanh) | bật bằng `ENABLE_BLIP2=1` |
| GPT-4o | Dịch + mở rộng query, tách event temporal, intelligent search | Cần `OPENAI_API_KEY` trong `.env` |
| BGE-M3 | Embed text → search caption (collection `keyframe-captions`) | |
| Whisper / PhoWhisper | ASR tiếng Việt (offline ingest) | |
| TransNetV2 | Tách shot khi ingest | weights trong `model_weights/` |
| BEiT-3 | Vector ảnh/text thứ 3 (`beit3`, 1024d) | cần checkpoint retrieval và `beit3.spm`; có thể đặt trên CPU |

Kết quả nhiều model được trộn bằng **RRF (Reciprocal Rank Fusion)** có trọng số — UI cho chọn dùng SigLIP2 / Jina / cả hai và chỉnh trọng số từng model.

## Chạy hệ thống

### Docker (đủ bộ)

```bash
cp .env.example .env        # điền OPENAI_API_KEY
docker compose up --build   # Qdrant :6333, ES :9200, backend :8000 (cần NVIDIA Container Toolkit), web :3000
```

### Thủ công (3 tiến trình, theo thứ tự)

```bash
# 1. Hạ tầng — container tự sống lại sau reboot nhờ --restart
docker run -d --name aic_qdrant --restart unless-stopped -p 6333:6333 \
    -v aic_qdrant_data:/qdrant/storage qdrant/qdrant
docker run -d --name aic_es --restart unless-stopped -p 9200:9200 \
    -e discovery.type=single-node -e xpack.security.enabled=false \
    -e ES_JAVA_OPTS="-Xms1g -Xmx1g" \
    -v aic_es_data:/usr/share/elasticsearch/data \
    docker.elastic.co/elasticsearch/elasticsearch:8.14.3

# 2. Backend (chạy từ repo root; load model lúc khởi động, cần CUDA)
export HF_HOME=$(pwd)/temp_cache        # giữ cache HuggingFace trong repo
python -m online.backend.api_server     # :8000
# Điều khiển model bằng ENABLE_SIGLIP/JINA/BGE/BEIT3/BLIP2 trong .env

# 3. Frontend
cd online/frontend && npm install
DATA_ROOT=/đường/dẫn/repo npm start     # :3000 — DATA_ROOT để serve ảnh keyframe
```

### Nạp dữ liệu (offline, trước khi thi)

```bash
# Video → keyframe → embedding → Qdrant (collection AIC_data)
python -m offline.ingest_data --input-folder data/video \
    --collection-name AIC_data --disable-object-detection

# Audio → transcript → Elasticsearch
python -m offline.ingest_ASR_data --input-folder data/video --index-name asr_data

# Keyframe → OCR → Elasticsearch (cần API key LLM vision)
python -m offline.ingest_OCR_data --help
```

Lưu ý: `ingest_data.py` quét **mọi** video trong folder input — muốn nạp thêm video mới mà không trùng thì tạo folder riêng (symlink được).

## API chính (backend :8000)

| Endpoint | Chức năng |
|---|---|
| `POST /search/kis` | Search ảnh/text; option: `use_siglip`, `use_jina`, `use_beit3`, `use_bge`, `rrf_weights`, `use_rerank` (BLIP2), `auto_expand` + `num_expansions` (GPT), `video_filter`, `score_threshold` |
| `POST /search/audio` · `/search/ocr` | Full-text ES trên transcript / chữ trên hình |
| `POST /search/temporal/enhanced` | Mô tả chuỗi sự kiện → GPT tách event → beam search theo thời gian trong cùng video |
| `POST /search/temporal/combinate` | Như trên nhưng mỗi event tự khai KIS/OCR/ASR + trọng số |
| `POST /search/intelligent` | GPT phân tích query rồi tự phối KIS + OCR + ASR |
| `POST /shot/detail` | Mọi frame của một shot |
| `GET /health` · `/stats/{service}` | Trạng thái từng dịch vụ (UI đọc để hiện pill bật/tắt) |

## Web UI (:3000)

- 8 tab: KIS · ASR · OCR · Temporal · T.Combinate · Intelligent · Combinate · Browse
- Sidebar phải kiểu Streamlit chứa toàn bộ option của tab đang mở
- Thanh trạng thái: Qdrant / ES·ASR / ES·OCR / GPT-4o / BLIP2 / Jina ✔✕ (tự refresh 15s)
- Hover ảnh → GIF preview cả shot; click → chi tiết + các frame lân cận
- **Bookmark** (localStorage) để gom frame làm đáp án TRAKE
- **DRES**: login, chọn evaluation, xem điểm, nộp 3 định dạng: KIS `{mediaItemName,start,end}` (ms) · QA `"QA-<đáp án>-<video>-<ms>"` · TRAKE `"TR-<video>-<f1>,<f2>,..."`

## Test

```bash
python3 -m unittest discover -s tests -v
```

Unit test chạy độc lập; integration test tự skip khi backend/Qdrant/ES chưa bật.

## Ghi chú quan trọng

- **`transformers` phải <5** (đã ghim trong requirements): code remote của jina-clip-v2 vỡ trên 5.x; SigLIP2 cần ≥4.49.
- **SigLIP2 text**: bắt buộc lowercase + pad đúng 64 token, nếu không recall giảm mạnh — wrapper đã xử lý, đừng gọi tokenizer trực tiếp.
- Backend **degrade mềm**: thiếu model/API key nào thì tính năng đó tắt (hiện ✕ trên UI), server vẫn chạy.
- Máy GPU 12 GB: giữ SigLIP/Jina/BLIP2 trên CUDA và đặt BGE/BEiT-3 trên CPU để tránh OOM; xem `.env.example`.
- `.env` (gitignore) chứa `OPENAI_API_KEY`, `QDRANT_URL`, `ES_HOST`, `ES_PORT` — **không** commit key vào `.env.example`.
