# AIC25 Colab visual encoding bundle v1

Bundle này xử lý đúng 873 video / 177.321 mapped keyframe của dataset `aic25-b1`.
Colab chỉ làm keyframe extraction, SigLIP2 và Jina embedding. Caption, OCR, ASR,
object detection, Qdrant và Elasticsearch không chạy trong bundle này.

## 1. Chuẩn bị trên Drive

Giữ nguyên 14 file `Videos_*.zip` trong `MyDrive/Video AIC`. Giải nén bundle vào:

```text
MyDrive/Video AIC/AIC25-Colab/bundle-v1/
```

Không xóa video ZIP. Bundle không tự xóa input và không ghi đè database local.

## 2. Thứ tự chạy bắt buộc

1. Mở `notebooks/00_Setup_Preflight.ipynb`, chọn GPU runtime và Run all.
2. Chỉ khi `preflight.json` báo `ready: true`, mở `01_Pilot_1000_Frames.ipynb`.
3. Chỉ khi `pilot/report.json` báo `status: passed`, mở
   `02_Encode_Full_Dataset.ipynb`.
4. Nếu Colab ngắt phiên, mount lại Drive, chạy notebook 00 rồi chạy lại notebook
   02. Checksum hợp lệ sẽ được bỏ qua; trạng thái `running` cũ trở về `pending`.

Full runner không nhận shard mới sau 8 giờ 45 phút. Exit `0` với trạng thái
`paused` là checkpoint an toàn, không phải lỗi. Chạy lại notebook 02 để tiếp tục.

## 3. Kiểm tra và phục hồi

State nằm ở `AIC25-Colab/runtime/state.json`; bản trước ở
`state.json.backup`. Nếu state chính hỏng JSON, runner tự đọc backup. Có thể xem
trạng thái bằng:

```bash
PYTHONPATH=/path/to/bundle-v1 python -m aic_colab status \
  --drive-root "/path/to/Video AIC" \
  --bundle-root /path/to/bundle-v1
```

Không xóa file `.sha256`. Artifact chỉ được resume khi file và SHA-256 sidecar cùng
hợp lệ. File `.partial` không phải kết quả hoàn chỉnh.

## 4. Chép kết quả về Lexar và import database

Sau khi full validation đạt 873 video / 177.321 keyframe, chép nguyên cây kết quả:

```bash
rclone copy "gdrive:Video AIC/AIC25-Colab/results/aic25-b1" \
  /media/nekoartisan/Lexar/AIC/colab-results/aic25-b1 \
  --checksum --progress
```

Sau đó chạy công cụ trong `local_tools/` để verify checksum, giải nén keyframe và
import vào collection mới `AIC_data_aic25_b1`. Công cụ từ chối collection đang tồn
tại nếu không dùng resume với đúng schema. Elasticsearch không bị sửa.

Chỉ cân nhắc xóa video ZIP sau khi cả validation trên Drive, copy local, Qdrant point
count và snapshot backup đều đạt. Các report có trường `safe_to_delete`; trước thời
điểm đó giá trị phải được hiểu là `false`.

## 5. Lỗi thường gặp

- Exit `2`: thiếu/corrupt ZIP, map count sai hoặc pilot gate chưa đạt.
- Exit `3`: artifact/NPZ/checksum không hợp lệ.
- Exit `4`: model, CUDA hoặc runtime lỗi; xem `runtime/errors.jsonl`.
- CUDA OOM: runner giảm batch size tối đa ba lần. Nếu batch 1 vẫn OOM, đổi GPU runtime.
- Drive disconnect: không xóa `/content/aic25-work`; reconnect Drive rồi chạy lại.
