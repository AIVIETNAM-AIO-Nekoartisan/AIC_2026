# Thiết kế pipeline ingest toàn bộ AIC dataset

**Ngày:** 2026-08-09

**Trạng thái:** Chờ người dùng review

**Dataset ID:** `aic25-b1`
**Nguồn:** `/media/nekoartisan/Lexar/AIC/videos`

## 1. Mục tiêu

Xây dựng pipeline có thể resume để, sau khi toàn bộ archive video tải xong:

1. Giải nén và xác minh đủ 873 video.
2. Trích đúng 177.321 keyframe theo `map-keyframes`, không chạy lại TransNetV2.
3. Encode lại mọi keyframe bằng SigLIP2 và Jina-CLIP-v2.
4. Import media metadata và object detection có sẵn.
5. Sinh visual caption bằng BLIP2 và embed bằng BGE-M3.
6. Chạy OCR trên keyframe và index vào Elasticsearch.
7. Lấy subtitle YouTube trước; video không có subtitle phù hợp sẽ chạy Whisper local.
8. Chỉ chuyển backend sang dataset mới sau khi validation đạt đầy đủ các gate.

Pipeline phải chạy theo batch, có checkpoint ở cấp video/keyframe/stage, không tạo bản ghi
trùng khi retry và cho phép bổ sung batch mới về sau.

## 2. Các quyết định đã chốt

- Dùng `map-keyframes` làm nguồn keyframe chuẩn.
- Không dùng vector CLIP 512 chiều có sẵn làm retrieval vector chính; toàn bộ keyframe được
  encode lại bằng SigLIP2 1152 chiều và Jina 1024 chiều.
- Không chạy lại TransNetV2 và không chạy lại YOLO.
- Object metadata có sẵn được lọc và chuyển thành payload gọn trước khi import.
- Caption ảnh được sinh local bằng BLIP2; BGE-M3 tạo caption vector 1024 chiều.
- OCR chạy riêng, không dùng OCR API để sinh caption lần thứ hai.
- ASR ưu tiên subtitle YouTube theo thứ tự: `vi` manual, `vi` automatic, `en` manual,
  `en` automatic; nếu không có dữ liệu dùng được thì chạy Whisper trên video local.
- Mỗi thời điểm chỉ có một model GPU nặng hoạt động. Các stage SigLIP, Jina, BLIP2 và
  Whisper không chạy đồng thời.
- BEiT-3 và vector CLIP 512 chiều không thuộc lần build đầu tiên.

## 3. Inventory và quan hệ dữ liệu

Các package đã kiểm tra:

| Package | Số video | Số item |
|---|---:|---:|
| `clip-features-32-aic25-b1/clip-features-32` | 873 | 177.321 vector `(512,) float16` |
| `map-keyframes-aic25-b1/map-keyframes` | 873 | 177.321 dòng CSV |
| `media-info-aic25-b1/media-info` | 873 | 873 JSON |
| `objects-aic25-b1/objects` | 873 | 177.321 JSON |

Với mỗi video, ba số lượng sau phải bằng nhau:

```text
npy.shape[0] == số dòng CSV == số file object JSON
```

Quan hệ ordinal:

```text
map row n=1  <-> feature row 0  <-> objects/<video>/001.json
map row n=N  <-> feature row N-1 <-> objects/<video>/<N:03d>.json
```

Feature CLIP có sẵn chỉ dùng làm audit số lượng và có thể dùng cho benchmark hậu kiểm; nó
không được upsert vào collection production.

## 4. Bố trí dữ liệu và runtime state

### 4.1 Dữ liệu lớn trên Lexar

```text
/media/nekoartisan/Lexar/AIC/
├── videos/                         # zip và metadata package tải về
├── video/                          # 873 MP4 sau giải nén
└── results/aic25-b1/
    ├── keyframes/<group>/<video>/  # WebP trích theo map
    ├── manifests/                  # canonical JSONL + báo cáo inventory
    ├── captions/                   # caption JSONL theo batch
    ├── ocr/                        # OCR JSONL theo batch
    ├── asr/                        # subtitle/Whisper JSONL theo batch
    └── reports/                    # validation và throughput report
```

`DATA_ROOT=/media/nekoartisan/Lexar/AIC` cho phép frontend phục vụ file nằm dưới
`results/aic25-b1/`. `VIDEO_DIRS` phải chứa `/media/nekoartisan/Lexar/AIC/video`.

### 4.2 Checkpoint nhỏ trên ext4

```text
/mnt/data/AIC_2025-main/.runtime/aic25-b1/state.sqlite3
```

SQLite nằm trên ext4 thay vì exFAT để tránh rủi ro lock và giảm write amplification.
`.runtime/` phải được thêm vào `.gitignore`.

## 5. Đơn vị batch

Mỗi archive video là một batch độc lập:

```text
L21_a, L22_a, L23_a, L24_a, L25_a,
L26_a, L26_b, L26_c, L26_d, L26_e,
L27_a, L28_a, L29_a, L30_a
```

Batch chỉ được mở khi archive tương ứng:

- Không còn file `.part` liên quan.
- Có kích thước lớn hơn 0.
- `unzip -t` thành công.
- Danh sách MP4 trong archive không trùng video của batch khác.

Pipeline chạy một batch đến hết visual index trước khi chuyển sang batch kế tiếp. Caption,
OCR và ASR cũng ghi output riêng theo batch, vì vậy có thể retry mà không quét lại toàn dataset.

## 6. Canonical manifest

Preflight sinh file:

```text
/media/nekoartisan/Lexar/AIC/results/aic25-b1/manifests/keyframes.jsonl
```

Mỗi dòng có schema cố định:

```json
{
  "dataset_id": "aic25-b1",
  "batch_id": "L21_a",
  "video_id": "L21_V001",
  "ordinal": 1,
  "timestamp_seconds": 0.0,
  "fps": 30.0,
  "frame_index": 0,
  "video_path": "/media/nekoartisan/Lexar/AIC/video/L21_V001.mp4",
  "keyframe_path": "/media/nekoartisan/Lexar/AIC/results/aic25-b1/keyframes/L21/L21_V001/001.webp",
  "object_path": "/media/nekoartisan/Lexar/AIC/videos/objects-aic25-b1/objects/L21_V001/001.json",
  "media_info_path": "/media/nekoartisan/Lexar/AIC/videos/media-info-aic25-b1/media-info/L21_V001.json",
  "point_id": "9c6e7432-1908-54ea-8c96-8bbff30a1fc2"
}
```

`point_id` chỉ phụ thuộc `dataset_id`, `video_id` và `ordinal`; việc đổi đường dẫn file không
tạo point mới. Manifest được sinh lại phải byte-equivalent sau khi sort theo
`(video_id, ordinal)`.

Công thức chuẩn:

```python
uuid.uuid5(uuid.NAMESPACE_URL, f"aic://{dataset_id}/{video_id}/{ordinal}")
```

UUID trong ví dụ manifest là kết quả cho `aic25-b1/L21_V001/1`.

## 7. Runtime state model

SQLite có hai bảng:

```text
video_stage(dataset_id, batch_id, video_id, stage, status, attempts, error, updated_at)
frame_stage(dataset_id, batch_id, video_id, ordinal, point_id, stage, status, attempts, error, updated_at)
```

Giá trị `stage`:

```text
extract, siglip, jina, metadata, caption, caption_embed, ocr, asr, validate
```

Giá trị `status`:

```text
pending, running, done, error
```

Khi process chết, bản ghi `running` cũ được chuyển lại thành `pending` ở lần resume tiếp theo.
Một stage chỉ ghi `done` sau khi output đã được flush hoặc datastore xác nhận upsert.

## 8. Preflight và giải nén

Preflight thực hiện toàn bộ gate trước khi dùng GPU:

1. Xác minh đủ 14 archive, không còn `.part`, mọi ZIP pass integrity test.
2. Liệt kê MP4 trong archive và xác nhận đúng 873 video ID duy nhất.
3. Xác minh bốn metadata package đều có cùng 873 video ID.
4. Xác minh tổng 177.321 map row, feature row và object file, không có ordinal bị hổng.
5. Đọc tổng uncompressed size từ ZIP central directory.
6. Ước lượng keyframe WebP bằng benchmark 1.000 frame đầu.
7. Chỉ tiếp tục nếu free space trên Lexar lớn hơn:

   ```text
   total_uncompressed_video_bytes
   + estimated_keyframe_bytes
   + 20% safety margin
   ```

Giải nén dùng chế độ không ghi đè file đã tồn tại. Sau giải nén, mỗi video được đối chiếu
size lớn hơn 0, có video stream đọc được và duration sai lệch không quá 2 giây so với
`media-info.length`.

ZIP và `.part` không bị xóa tự động. Việc dọn archive chỉ diễn ra bằng thao tác riêng sau khi
toàn bộ validation production đạt.

## 9. Trích keyframe theo map

Mỗi video được decode một lần theo thứ tự frame. Extractor đọc danh sách `frame_idx` đã sort,
ghi ảnh khi decoder đi qua frame cần lấy và dùng `pts_time` làm kiểm tra chéo.

Định dạng output:

```text
WebP, quality=90, giữ nguyên aspect ratio
```

Một frame đạt khi:

- File WebP mở được và có width/height lớn hơn 0.
- Decoded frame index bằng `frame_idx` trong map.
- Timestamp decoder lệch `pts_time` không quá `max(0.10 giây, 2/fps)`.

Nếu timestamp vượt tolerance, frame được ghi vào error report và video không được mở stage
visual cho đến khi retry bằng timestamp seek. Không silently chấp nhận frame sai.

## 10. Qdrant visual index

Collection production mới:

```text
AIC_data_aic25_b1
```

Collection được tạo một lần với schema đầy đủ trước khi encode:

```text
siglip: 1152, cosine
jina:   1024, cosine
```

Payload index:

```text
video.name                 keyword
dataset_id                 keyword
batch_id                   keyword
frame.timestamp_seconds    float
object_detection.unique_classes keyword
object_detection.object_count   integer
```

Mỗi official keyframe được xem là một retrieval unit độc lập. Payload tương thích backend:

```json
{
  "dataset_id": "aic25-b1",
  "batch_id": "L21_a",
  "file_path": "/media/nekoartisan/Lexar/AIC/results/aic25-b1/keyframes/L21/L21_V001/001.webp",
  "video": {"name": "L21_V001", "filename": "L21_V001.mp4"},
  "frame": {"index": 0, "timestamp_seconds": 0.0},
  "keyframe": {"ordinal": 1},
  "shot": {"number": 1, "position": "middle"},
  "shot_id": "L21_V001_keyframe_000001"
}
```

SigLIP và Jina chạy thành hai stage GPU riêng. SigLIP tạo point cùng payload; Jina dùng
Qdrant `update_vectors` để thêm named vector vào point hiện có, không upsert một point thiếu
SigLIP. Encoder xử lý batch ảnh và checkpoint sau mỗi Qdrant batch được xác nhận.

## 11. Media và object metadata

Media payload chỉ lặp các field nhỏ trên mỗi keyframe:

```text
title, author, publish_date, watch_url, duration_seconds
```

Description và keywords đầy đủ không được lặp 177.321 lần trong Qdrant; chúng được lưu một
document/video trong Elasticsearch index `media_data_aic25_b1`.

Object JSON có top 100 candidate, bao gồm nhiều score rất thấp. Importer chỉ giữ detection
có `score >= 0.25` và tổng hợp thành payload backend đang sử dụng:

```json
{
  "has_objects": true,
  "object_count": 3,
  "unique_classes": ["person", "car"],
  "class_counts": {"person": 2, "car": 1},
  "max_confidence": 0.94
}
```

Tên class lấy từ `detection_class_entities`, chuẩn hóa lowercase. Raw boxes và top-100 list
không được copy vào Qdrant; source JSON tiếp tục là nguồn audit.

## 12. Caption và BGE

Caption chạy bằng `Salesforce/blip2-opt-2.7b`, quantization 4-bit, trên từng keyframe vì
official map không cung cấp nhóm start/middle/end của cùng shot. Output append-only:

```text
results/aic25-b1/captions/<batch_id>.jsonl
```

Mỗi record dùng `point_id` từ manifest. Resume bỏ qua point đã có caption hợp lệ. Caption
được ghi trở lại payload collection chính để UI hiển thị.

Collection caption:

```text
keyframe-captions_aic25_b1
vector: BGE-M3 1024, cosine
point ID: giống collection chính
```

BGE stage chỉ chạy sau khi BLIP2 của batch đã đóng file JSONL thành công. BGE point payload
sao chép payload cần thiết từ collection chính và thêm field `caption`.

## 13. OCR

OCR chạy ở chế độ OCR-only để không trả tiền và thời gian cho caption trùng. Mỗi batch ghi:

```text
results/aic25-b1/ocr/<batch_id>.jsonl
```

Mỗi record có `point_id`, `video_id`, `ordinal`, `text`, `provider`, `model`, `status` và
`error`. Text rỗng vẫn được đánh dấu `done` để resume không gọi API vô hạn.

Elasticsearch index:

```text
ocr_data_aic25_b1
document ID: point_id
```

OCR retry tối đa ba lần với exponential backoff. Lỗi rate-limit giữ status `error` và được
retry ở lượt chạy sau; lỗi ảnh hỏng chặn video tại validation thay vì gọi lại vô hạn.

## 14. ASR: YouTube subtitle và Whisper fallback

ASR hoạt động ở cấp video:

1. Đọc `watch_url` từ `media-info`.
2. Dùng `yt-dlp` lấy subtitle theo thứ tự ưu tiên đã chốt.
3. Chuẩn hóa VTT/SRT thành đoạn có `start_time`, `end_time`, `text`, `language`, `source`.
4. Gộp câu thành semantic chunk khoảng 150 từ, tối đa 90 giây.
5. Nếu không có subtitle, subtitle rỗng hoặc lỗi truy cập không thể khôi phục, enqueue video
   cho Whisper local.
6. Whisper dùng video đã giải nén và cùng thuật toán semantic chunking.

Output:

```text
results/aic25-b1/asr/<batch_id>.jsonl
```

Elasticsearch index:

```text
asr_data_aic25_b1
document ID: ba9ffb7b-f084-5ed5-b100-d3a1f5d4bbb2
```

Mỗi document lưu `video_id`, `content`, timestamp, language và source `youtube_manual`,
`youtube_auto` hoặc `whisper`. Retry không tạo document trùng.

Công thức document ID chuẩn:

```python
uuid.uuid5(
    uuid.NAMESPACE_URL,
    f"aic-asr://{dataset_id}/{video_id}/{start_ms}/{end_ms}/{source}",
)
```

UUID ví dụ tương ứng với `aic25-b1/L21_V001/0/90000/youtube_manual`.

## 15. Lập lịch tài nguyên

Thứ tự mặc định cho mỗi batch:

```text
extract
-> siglip
-> jina
-> metadata
-> caption
-> caption_embed
-> ocr
-> asr
-> validate
```

Quy tắc:

- Không chạy hai GPU stage đồng thời.
- BGE chạy CPU theo `.env` mặc định; nếu chuyển BGE sang GPU thì nó trở thành GPU stage độc quyền.
- OCR API có thể chạy khi Whisper dùng GPU, nhưng mặc định vẫn chạy tuần tự để log và retry rõ ràng.
- Backend production dừng trong lúc GPU ingest để không tranh VRAM; Qdrant và Elasticsearch
  tiếp tục chạy.
- Mỗi batch ghi throughput, peak VRAM, số success/error và ETA dựa trên median ba batch gần nhất.

## 16. Validation gates

### Gate cấp batch

Một batch chỉ đạt khi:

- Số WebP hợp lệ bằng số map row.
- Mỗi main point có cả vector `siglip` và `jina`.
- Mỗi main point có media payload và object payload, kể cả object count bằng 0.
- Số caption point bằng số main point.
- Mọi keyframe có OCR status `done` hoặc error report được người vận hành chấp nhận rõ ràng.
- Mọi video có ít nhất một ASR document hoặc có lý do `no_speech` đã xác minh.
- Sample 20 point ngẫu nhiên trả đúng file ảnh, video ID và timestamp.

### Gate toàn dataset

```text
main Qdrant points       = 177321
caption Qdrant points    = 177321
videos represented       = 873
missing siglip vectors   = 0
missing jina vectors     = 0
broken keyframe paths    = 0
manifest mismatches      = 0
```

OCR document count có thể nhỏ hơn 177.321 vì frame không có chữ. ASR document count phụ
thuộc số chunk, nhưng mọi video phải có terminal status.

Sau gate toàn dataset, chạy smoke test cho KIS tiếng Việt, KIS tiếng Anh, object filter,
caption search, OCR search, ASR search và temporal search. Mỗi modality phải trả ít nhất một
result có file tồn tại trước khi đổi `.env` backend.

## 17. Cutover và rollback

Cutover chỉ thay bốn biến:

```dotenv
QDRANT_COLLECTION=AIC_data_aic25_b1
QDRANT_CAPTION_COLLECTION=keyframe-captions_aic25_b1
ES_OCR_INDEX=ocr_data_aic25_b1
ES_ASR_INDEX=asr_data_aic25_b1
```

Sau restart, `/health` và smoke test được chạy lại. Nếu lỗi, rollback bằng cách trả `.env` về
namespace trước đó; collection/index mới không bị xóa. Không có stage nào tự động xóa datastore.

## 18. Bổ sung dữ liệu về sau (incremental ingestion)

Batch mới phải có `batch_id` mới và video ID không trùng. Preflight nối record mới vào manifest,
pipeline chỉ tạo state `pending` cho record mới và dùng cùng collection/index production.

Nếu một video ID đã tồn tại nhưng nội dung thay đổi, pipeline dừng và yêu cầu thao tác replace
riêng: xóa toàn bộ point/document của video cũ theo filter, tạo lại manifest record rồi chạy lại
các stage. Không tự động ghi đè một phần video.

## 19. Xử lý lỗi và tính idempotent

- File output ghi vào đường dẫn tạm cùng filesystem, `fsync`, sau đó atomic rename.
- Qdrant và Elasticsearch dùng ID ổn định, nên retry là upsert chứ không append duplicate.
- Error report luôn chứa stage, video, ordinal, exception class và message.
- Process trả exit code khác 0 nếu còn lỗi chưa đạt gate của batch.
- `--resume` là mặc định; `--force-stage <stage>` chỉ reset state của stage được chỉ định.
- Không có lệnh `--force-all` trong workflow production để tránh vô tình chạy lại 177.321 frame.

## 20. Phạm vi không bao gồm

- Không dùng precomputed CLIP vector làm kênh retrieval production.
- Không chạy BEiT-3 trong lần build đầu.
- Không chạy lại shot detection hoặc object detection.
- Không xóa ZIP, `.part`, raw object JSON hoặc video sau ingest.
- Không tự động cutover nếu validation chưa đạt.

## 21. Tiêu chí hoàn thành

Thiết kế được triển khai thành công khi một lệnh orchestrator có thể chạy hoặc resume từng
batch, toàn bộ gate đạt, backend dùng namespace `aic25-b1`, và một batch dữ liệu mới có thể
được thêm mà không reprocess hay làm thay đổi point của 873 video ban đầu.
