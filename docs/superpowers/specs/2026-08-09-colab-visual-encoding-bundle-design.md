# Colab Visual Encoding Bundle Design

**Ngày:** 2026-08-09

**Trạng thái:** Chờ người dùng review

**Dataset ID:** `aic25-b1`

## 1. Mục tiêu

Tạo một bundle tự chứa trên Google Drive để Colab cá nhân có thể trích đúng 177.321
keyframe từ 873 video theo `map-keyframes`, encode toàn bộ bằng SigLIP2 và Jina-CLIP-v2,
checkpoint theo shard, tự resume sau khi runtime bị thu hồi và xuất artifact có thể import vào
Qdrant local.

Bundle này không chạy caption, OCR, ASR, object detection, Qdrant hoặc Elasticsearch trên
Colab. Video ZIP không bị notebook tự động xóa.

## 2. Nguồn dữ liệu Drive

Folder đầu vào đã xác định:

```text
Video AIC
Drive folder ID: 1fNKiWkZ1iKe9pqwuEGpjglFQOsz1IgFR
```

Notebook yêu cầu đúng 14 archive:

```text
Videos_L21_a.zip
Videos_L22_a.zip
Videos_L23_a.zip
Videos_L24_a.zip
Videos_L25_a.zip
Videos_L26_a.zip
Videos_L26_b.zip
Videos_L26_c.zip
Videos_L26_d.zip
Videos_L26_e.zip
Videos_L27_a.zip
Videos_L28_a.zip
Videos_L29_a.zip
Videos_L30_a.zip
```

Tại thời điểm viết spec, `L22_a`, `L26_a`, `L26_c` và `L26_d` còn đang upload.
Preflight không cho chạy full cho đến khi đủ 14 file, nhưng pilot được phép chạy bằng một
archive hoàn chỉnh.

## 3. Cấu trúc Drive

Bundle được upload dưới folder input hiện tại:

```text
Video AIC/
├── Videos_*.zip
└── AIC25-Colab/
    ├── aic25-colab-bundle-v1.zip
    ├── models/
    │   ├── huggingface-cache.tar.zst
    │   ├── huggingface-cache.tar.zst.sha256
    │   └── model-cache-manifest.json
    ├── runtime/
    │   ├── state.json
    │   ├── heartbeat.json
    │   ├── events.jsonl
    │   └── errors.jsonl
    └── results/aic25-b1/
        ├── keyframes/<batch_id>/<shard_id>.tar
        ├── embeddings/<batch_id>/<shard_id>.siglip.npz
        ├── embeddings/<batch_id>/<shard_id>.jina.npz
        ├── manifests/<batch_id>.jsonl
        ├── reports/<batch_id>.json
        └── checksums/<relative_output_path>.sha256
```

Notebook ghi file output vào `/content` trước, gọi `fsync`, tính SHA-256, copy lên Drive bằng
tên `.partial`, kiểm tra lại kích thước/checksum rồi mới rename thành tên cuối. File `.partial`
không được coi là checkpoint hoàn chỉnh.

## 4. Nội dung bundle

```text
aic25-colab-bundle/
├── notebooks/
│   ├── 00_Setup_Preflight.ipynb
│   ├── 01_Pilot_1000_Frames.ipynb
│   └── 02_Encode_Full_Dataset.ipynb
├── aic_colab/
│   ├── config.py
│   ├── inventory.py
│   ├── map_reader.py
│   ├── ids.py
│   ├── state.py
│   ├── extraction.py
│   ├── sharding.py
│   ├── encoders.py
│   ├── persistence.py
│   ├── validation.py
│   └── runner.py
├── assets/
│   └── map-keyframes-aic25-b1.zip
├── config/
│   └── aic25-b1.json
├── tests/
├── requirements-colab.txt
├── README_COLAB.md
└── VERSION
```

Notebook chỉ gọi các module Python trong bundle. Logic pipeline không được nhúng thành các cell
lớn để có thể unit test local và sửa lỗi mà không chỉnh tay ba notebook.

## 5. Vai trò ba notebook

### 5.1 Setup và preflight

`00_Setup_Preflight.ipynb`:

1. Mount Google Drive bằng flow chính thức của Colab; data root mặc định là
   `/content/drive/MyDrive/Video AIC` và có thể đổi bằng một cell cấu hình duy nhất.
2. Xác định folder `Video AIC` và tạo các folder output còn thiếu.
3. Copy bundle xuống `/content` và cài dependency đã pin.
4. Kiểm tra GPU, CUDA, VRAM, dung lượng `/content` và CU/giờ do Colab hiển thị cho người dùng.
5. Inventory 14 ZIP theo tên và kích thước; chạy `ZipFile.testzip()` trước khi dùng archive.
6. Kiểm tra map có đúng 873 CSV và 177.321 row.
7. Nếu cache chưa có, tải model từ Hugging Face vào `/content/hf-cache`, đóng thành
   `huggingface-cache.tar.zst`, persist lên Drive và ghi manifest gồm repo ID, revision và file
   checksum. Phiên sau verify rồi giải nén cache về `/content`; model không được đọc trực tiếp
   từ hàng nghìn file nhỏ trên Drive. Không lưu access token vào bundle hoặc log.

### 5.2 Pilot 1.000 frame

`01_Pilot_1000_Frames.ipynb` chạy 1.000 mapped keyframe đầu của `L21_a`:

- Thực thi toàn bộ extraction, tar shard, SigLIP, Jina, persistence và validation.
- Ghi throughput, peak VRAM, Drive upload rate, kích thước keyframe và CU dự báo.
- Dùng namespace `pilot/`; không trộn artifact pilot với full dataset.
- Full notebook chỉ chạy khi `pilot/report.json` có `status=passed` và cùng bundle version.

### 5.3 Full dataset

`02_Encode_Full_Dataset.ipynb`:

- Đọc `state.json`, tự chọn batch/stage/shard chưa hoàn thành.
- Xử lý theo thứ tự archive nhưng cho phép cấu hình allow-list batch.
- Dừng nhận shard mới sau 8 giờ 45 phút và thoát an toàn trước mốc 9 giờ.
- Nếu runtime bị thu hồi sớm, phiên sau bỏ qua artifact đã có checksum hợp lệ.
- Không tự động xóa ZIP video trên Drive; report chỉ ghi `safe_to_delete=true` sau khi toàn bộ
  batch đạt validation.

## 6. Canonical ID và manifest

Mỗi mapped keyframe dùng UUID ổn định:

```python
uuid.uuid5(uuid.NAMESPACE_URL, f"aic://{dataset_id}/{video_id}/{ordinal}")
```

Manifest JSONL được sort theo `(video_id, ordinal)` và chứa:

```json
{
  "dataset_id": "aic25-b1",
  "batch_id": "L21_a",
  "video_id": "L21_V001",
  "ordinal": 1,
  "timestamp_seconds": 0.0,
  "fps": 30.0,
  "frame_index": 0,
  "point_id": "9c6e7432-1908-54ea-8c96-8bbff30a1fc2",
  "keyframe_shard": "keyframes/L21_a/000000.tar",
  "keyframe_member": "L21/L21_V001/001.webp"
}
```

Đường dẫn Drive tuyệt đối và `/content` không xuất hiện trong manifest. Local importer có thể
giải nén keyframe sang bất kỳ data root nào mà không làm đổi `point_id`.

## 7. Trích keyframe

Mỗi video được giải mã tuần tự một lần. Map reader dùng `frame_idx` làm giá trị chính và
`pts_time` làm kiểm tra chéo. Tại mỗi mapped frame:

- Sai lệch timestamp cho phép tối đa `max(1/fps, 0.05)` giây.
- Frame được lưu WebP quality 90, giữ nguyên aspect ratio.
- Tên member dùng ordinal ba chữ số tối thiểu, nhưng không giới hạn khi ordinal vượt 999.
- Video thiếu map, map không tăng dần, frame index ngoài video hoặc decode thất bại được đánh
  dấu lỗi theo video; runner tiếp tục video kế tiếp nhưng batch không thể đạt validation.

Keyframe được đóng thành tar shard tối đa 1.024 frame. Tar shard được persist trước khi encoder
xử lý để lần chạy Jina hoặc retry không cần giải mã lại video.

## 8. Encode và định dạng vector

Hai model chạy ở stage riêng để giảm VRAM:

```text
google/siglip2-so400m-patch14-384 -> float32, shape (N, 1152)
jinaai/jina-clip-v2               -> float32, shape (N, 1024)
```

Mỗi row được L2-normalize. File NPZ chứa chính xác:

```text
point_ids: unicode array shape (N,)
vectors:   float32 array shape (N, dim)
```

Batch size khởi tạo từ cấu hình theo GPU. Khi CUDA OOM, encoder giải phóng cache, giảm batch
size một nửa và retry shard; batch size đã giảm được ghi vào state cho các shard sau. Encoder
không được ghi vector từng frame lên Drive.

## 9. Checkpoint và resume

`runtime/state.json` chứa:

```text
schema_version, bundle_version, dataset_id, updated_at
batches[batch_id].stages[extract|siglip|jina|validate]
batches[batch_id].shards[shard_id].artifacts/checksums/status/error
```

Status hợp lệ: `pending`, `running`, `done`, `error`. Khi bắt đầu phiên mới, mọi status
`running` cũ được chuyển về `pending`; artifact cuối chỉ được dùng nếu file cuối, sidecar SHA-256
và metadata trong state khớp nhau.

`heartbeat.json` được cập nhật tối đa mỗi 60 giây. `events.jsonl` không chứa secret hoặc đường
dẫn token. Error record chứa batch, video, ordinal, stage, exception class, message và attempts.

## 10. Error handling

- CUDA OOM: giảm batch size và retry tối đa ba lần.
- Drive I/O/network: exponential backoff, tối đa năm lần; giữ local artifact đến khi upload được
  xác nhận.
- Runtime deadline: hoàn tất artifact đang ghi, checkpoint rồi exit code 0 với trạng thái
  `paused`.
- Archive corrupt: dừng batch trước extraction và ghi lỗi không thể retry tự động.
- Video corrupt hoặc map mismatch: tiếp tục batch, nhưng report `failed` và liệt kê video lỗi.
- Model revision/checksum đổi: dừng trước encode để tránh trộn vector từ hai revision.
- State JSON corrupt: khôi phục từ `state.json.backup`; nếu cả hai hỏng thì rebuild state từ
  checksum sidecar, không encode lại artifact hợp lệ.

## 11. Validation gates

Mỗi shard phải đạt:

- Tar mở được và member count bằng row manifest tương ứng.
- `point_ids` của SigLIP và Jina bằng nhau và bằng manifest.
- SigLIP shape `(N, 1152)`, Jina shape `(N, 1024)`, dtype float32.
- Không NaN/Inf; norm từng vector nằm trong `[0.999, 1.001]`.
- SHA-256 sidecar khớp file Drive.

Toàn dataset phải đạt:

```text
videos             = 873
manifest rows       = 177321
keyframe members    = 177321
SigLIP vectors      = 177321
Jina vectors        = 177321
duplicate point IDs = 0
missing artifacts   = 0
```

## 12. Handoff về máy local

Artifact Drive là source of truth có thể rebuild database. Sau khi batch hoàn thành:

1. Dùng Google Drive Desktop hoặc `rclone copy` tải `results/aic25-b1` về:

```text
/media/nekoartisan/Lexar/AIC/results/aic25-b1
```

2. Local importer kiểm tra checksum, giải nén keyframe tar vào:

```text
/media/nekoartisan/Lexar/AIC/results/aic25-b1/keyframes
```

3. Importer tạo collection Qdrant mới:

```text
AIC_data_aic25_b1
siglip: 1152 cosine
jina:   1024 cosine
```

4. Mỗi point dùng `point_id` và payload compact từ manifest. Import là idempotent và có thể
resume theo shard. Collection production cũ không bị ghi đè.
5. Media info và object metadata được enrich local ở stage riêng. OCR/ASR sau này ghi vào các
Elasticsearch index riêng; Colab visual import không sửa Elasticsearch.

Backend chỉ chuyển `QDRANT_COLLECTION=AIC_data_aic25_b1` sau khi point count, vector count và
smoke search đạt gate.

## 13. Lưu và backup database

Qdrant đang dùng bind mount:

```text
/mnt/data/AIC_2025-main/qdrant_data -> /qdrant/storage
```

Elasticsearch đang dùng Docker named volume `es_data`. Không copy trực tiếp các folder storage
khi service đang chạy.

Sau visual import:

1. Tạo Qdrant collection snapshot qua API.
2. Tải snapshot về:

```text
/media/nekoartisan/Lexar/AIC/backups/qdrant/AIC_data_aic25_b1/<timestamp>/
```

3. Ghi `backup-manifest.json` gồm collection, point count, Qdrant version, snapshot SHA-256 và
bundle version.
4. Giữ artifact NPZ/manifest/keyframe trên Drive và Lexar. Đây là lớp backup độc lập, cho phép
rebuild collection ngay cả khi Docker volume hoặc snapshot lỗi.

Elasticsearch chưa cần snapshot ở phase này vì visual encode không thay đổi ES. Khi OCR/ASR
được chạy sau, mỗi index phải có bulk NDJSON export hoặc snapshot repository trước cutover.

## 14. Bảo mật

- Bundle không chứa `.env`, Google API key, Hugging Face token, GitHub PAT hoặc Qdrant API key.
- Drive folder giữ trạng thái private hiện tại.
- Notebook không in environment hoặc mount token.
- Model public được tải không cần Hugging Face token; nếu provider yêu cầu xác thực về sau, token
  phải được nhập qua Colab Secrets và không persist.

## 15. Kiểm thử

Unit test local dùng video nhỏ sinh bằng FFmpeg, CSV map giả lập và fake encoder để kiểm tra:

- Inventory và mapping archive-to-batch.
- UUID ổn định.
- Map validation và extraction đúng ordinal/frame.
- Tar shard member names/count.
- Atomic persistence và recovery từ `.partial`/backup state.
- Resume bỏ qua shard hợp lệ, chạy lại shard checksum sai.
- OOM làm giảm batch size.
- Vector schema/norm/NaN validation.
- Notebook JSON hợp lệ và import đúng module runner.

Integration test local chạy extraction thật nhưng không tải model lớn. Pilot Colab 1.000 frame là
integration gate cho model, CUDA, Drive và CU thực tế.

## 16. Tiêu chí hoàn thành bundle

- Ba notebook mở được trong Colab và không chứa cell cần sửa code thủ công.
- Bundle ZIP tái tạo được bằng script build và có SHA-256.
- Tất cả unit/integration test local pass.
- README ghi rõ thứ tự chạy, cách resume, cách đọc lỗi và cách tải artifact về local.
- Bundle được upload vào `Video AIC/AIC25-Colab/` và được verify bằng Drive metadata readback.
- Không file video, database hoặc secret hiện có nào bị xóa hoặc thay đổi.
