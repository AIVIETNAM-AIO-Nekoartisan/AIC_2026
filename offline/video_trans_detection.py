# utils/video_trans_detection.py (fast streamer)
import torch
import cv2
import numpy as np
import os
import re
from glob import glob
from PIL import Image
import io

# Try PyAV for much faster decode; fallback to OpenCV
try:
    import av  # type: ignore
    _HAS_PYAV = True
except Exception:
    av = None  # type: ignore
    _HAS_PYAV = False

# Thêm thư mục gốc của dự án vào đường dẫn tìm kiếm của Python
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
import sys
sys.path.insert(0, project_root)

# Import các module cần thiết từ project
from core.models.transnetv2_pytorch import TransNetV2
from concurrent.futures import ThreadPoolExecutor


def create_new_output_folder(base_path="results"):
    """
    Tạo một thư mục mới theo định dạng 'test_N' trong một thư mục cơ sở.
    Hàm sẽ tự động tìm số 'N' lớn nhất đã tồn tại và tạo thư mục tiếp theo (N+1).
    """
    os.makedirs(base_path, exist_ok=True)
    latest_run_number = 0
    dir_pattern = re.compile(r"^test_(\d+)$")
    for item_name in os.listdir(base_path):
        item_path = os.path.join(base_path, item_name)
        if os.path.isdir(item_path):
            match = dir_pattern.match(item_name)
            if match:
                run_number = int(match.group(1))
                if run_number > latest_run_number:
                    latest_run_number = run_number
    new_run_number = latest_run_number + 1
    new_folder_path = os.path.join(base_path, f"test_{new_run_number}")
    os.makedirs(new_folder_path)
    print(f"[INFO] Đã tạo thư mục lưu kết quả tại: '{new_folder_path}'")
    return new_folder_path


def save_frame_as_webp(frame, save_path, quality=85):
    """
    Lưu frame dưới dạng WebP với chất lượng tùy chỉnh.
    Args:
        frame: numpy array của frame (BGR hoặc RGB đều được – chỉ resize/ghi)
        save_path: đường dẫn lưu file
        quality: chất lượng WebP (0-100)
    """
    # Nếu frame đang là BGR (OpenCV), chuyển sang RGB cho PIL
    if frame.shape[-1] == 3:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        frame_rgb = frame
    pil_image = Image.fromarray(frame_rgb)
    pil_image.save(save_path, 'WEBP', quality=quality, method=6)


def create_shot_gif(frames, output_path, duration=500, resize_factor=0.5):
    """Tạo GIF từ list các frames với kích thước tối ưu cho web"""
    images = []
    for frame in frames:
        # Chuyển BGR -> RGB nếu cần
        if frame.shape[-1] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
        height, width = frame_rgb.shape[:2]
        new_width = int(width * resize_factor)
        new_height = int(height * resize_factor)
        frame_resized = cv2.resize(frame_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
        images.append(Image.fromarray(frame_resized))

    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
        optimize=True,
        quality=85,
    )
    return output_path


def save_transitions_as_images(
    video_path,
    predictions,
    output_folder,
    frame_skip,
    threshold=0.5,
    use_webp=True,
    yoloe_detector=None,
    create_gif_preview=True,
    frames_per_gif=5,
    io_threads=2,
):
    """
    Xác định các phân cảnh, lưu ảnh đại diện, tạo GIF preview (đa luồng an toàn cho I/O).
    """
    try:
        transition_probabilities = predictions[0, :, 0]
    except IndexError:
        print("[LỖI] Cấu trúc tensor dự đoán không đúng.")
        return {}

    transition_indices = torch.where(transition_probabilities > threshold)[0]
    if len(transition_indices) == 0:
        print("[INFO] Không tìm thấy chuyển cảnh nào để chia phân cảnh.")
        return {}

    num_processed_frames = predictions.shape[1]
    shot_boundaries = sorted(list(set([0] + transition_indices.tolist() + [num_processed_frames])))

    print(f"[INFO] Đã chia video thành {len(shot_boundaries) - 1} phân cảnh. Bắt đầu lưu ảnh...")

    keyframe_info = {
        'video_path': video_path,
        'total_shots': len(shot_boundaries) - 1,
        'keyframes': [],
        'shot_previews': []
    }

    try:
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"Lỗi: Không thể mở video gốc tại {video_path}.")
            return keyframe_info

        fps = cap.get(cv2.CAP_PROP_FPS) or 1
        video_filename = os.path.basename(video_path)
        video_filename_base = os.path.splitext(video_filename)[0]
        image_extension = ".webp" if use_webp else ".jpg"

        if create_gif_preview:
            gif_folder = os.path.join(output_folder, "gif_previews")
            os.makedirs(gif_folder, exist_ok=True)

        # ThreadPool chỉ cho I/O
        with ThreadPoolExecutor(max_workers=io_threads) as pool:
            futures = []
            for i in range(len(shot_boundaries) - 1):
                shot_number = i + 1
                start_idx, end_idx = shot_boundaries[i], shot_boundaries[i + 1]
                if (end_idx - start_idx) < 3:
                    continue

                shot_id = f"{video_filename_base}_shot_{shot_number:03d}"

                # Keyframes: start / middle / end
                indices_to_save = [
                    start_idx + 1,
                    start_idx + (end_idx - start_idx) // 2,
                    end_idx - 1,
                ]
                positions = ["start", "middle", "end"]

                for pos, frame_idx in zip(positions, indices_to_save):
                    orig_idx = frame_idx * frame_skip
                    cap.set(cv2.CAP_PROP_POS_FRAMES, orig_idx)
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    ts = round(orig_idx / fps, 2)

                    image_name = (
                        f"{video_filename_base}__shot_{shot_number:03d}__pos_{pos}__"
                        f"frame_{orig_idx:06d}__ts_{ts:.2f}{image_extension}"
                    )
                    save_path = os.path.join(output_folder, image_name)

                    # Copy để tránh giữ buffer
                    frame_cpu = frame.copy()
                    if use_webp:
                        futures.append(pool.submit(save_frame_as_webp, frame_cpu, save_path, 85))
                    else:
                        futures.append(pool.submit(cv2.imwrite, save_path, frame_cpu))

                    keyframe_info['keyframes'].append({
                        'image_name': image_name,
                        'image_path': save_path,
                        'shot_id': shot_id,
                        'shot_number': shot_number,
                        'position_in_shot': pos,
                        'frame_index': orig_idx,
                        'timestamp_seconds': ts,
                    })

                # GIF frames
                if create_gif_preview:
                    gif_idx = np.linspace(start_idx + 1, end_idx - 1, frames_per_gif, dtype=int)
                    gif_frames = []
                    for gi in gif_idx:
                        orig_idx = gi * frame_skip
                        cap.set(cv2.CAP_PROP_POS_FRAMES, orig_idx)
                        ret, frame = cap.read()
                        if ret:
                            gif_frames.append(frame.copy())
                    if gif_frames:
                        gif_filename = f"{shot_id}_preview.gif"
                        gif_path = os.path.join(gif_folder, gif_filename)
                        futures.append(pool.submit(create_shot_gif, gif_frames, gif_path, 400, 0.5))
                        keyframe_info['shot_previews'].append({
                            'shot_id': shot_id,
                            'shot_number': shot_number,
                            'gif_path': gif_path,
                            'gif_filename': gif_filename,
                            'frame_count': len(gif_frames),
                            'start_timestamp': round((start_idx * frame_skip) / fps, 2),
                            'end_timestamp': round((end_idx * frame_skip) / fps, 2),
                        })

            for f in futures:
                try:
                    f.result(timeout=60)
                except Exception as e:
                    print("[WARN] Thread I/O fail:", e)

    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()

    print(f"[INFO] Đã lưu {len(keyframe_info['keyframes'])} keyframes (threaded)")
    if create_gif_preview:
        print(f"[INFO] Đã tạo {len(keyframe_info['shot_previews'])} GIF previews (threaded)")

    return keyframe_info


class VideoIngestDatabase:
    def __init__(
        self,
        batch_size=1000,
        frame_skip=3,
        yoloe_model_path=None,
        yoloe_conf_threshold=0.25,
        # New fast-streamer knobs
        decode_threads: int | None = None,
        prefer_pyav: bool = True,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Sử dụng thiết bị: {self.device}")

        # Model weights
        weights_path = os.path.join(project_root, "model_weights/transnetv2-pytorch-weights.pth")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Không tìm thấy file trọng số model tại: {weights_path}")

        self.model = TransNetV2()
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval().to(self.device)

        self.batch_size = batch_size
        self.frame_skip = frame_skip

        # Fast streaming config
        self.decode_threads = decode_threads
        self.prefer_pyav = prefer_pyav and _HAS_PYAV

        # Detection (not used in save-only path)
        self.yoloe_detector = None
        if yoloe_model_path:
            print(f"[INFO] YOLOE sẽ được chạy trong database processing phase: {yoloe_model_path}")

    def _stream_video_frames(self, video_path: str, frame_skip: int):
        """
        Fast frame streamer with PyAV (FFmpeg) and multi-threaded decode.
        - Yields RGB resized frames of size (27, 48, 3) – same as before.
        - Keeps exact modulo-based skipping to avoid frame drift.
        """
        # Preferred: PyAV
        if self.prefer_pyav:
            try:
                container = av.open(video_path)  # type: ignore
                vstream = container.streams.video[0]
                # Enable threaded decoding
                if self.decode_threads and self.decode_threads > 0:
                    vstream.thread_type = "AUTO"  # let FFmpeg decide slice/frame threading
                    vstream.thread_count = self.decode_threads

                i = 0
                for frame in container.decode(vstream):
                    # modulo-based skip to match old behavior
                    if i % frame_skip == 0:
                        # Convert to RGB and resize to (48,27)
                        rgb = frame.to_ndarray(format="rgb24")
                        resized = cv2.resize(rgb, (48, 27), interpolation=cv2.INTER_AREA)
                        yield resized
                    i += 1
                container.close()
                return
            except Exception as e:
                print(f"[WARN] PyAV decode lỗi ({e}), fallback OpenCV...")
                # fall through to OpenCV

        # Fallback: OpenCV (can be slower)
        try:
            cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                print(f"Lỗi: Không thể mở file video tại {video_path}")
                return

            frame_index = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_index % frame_skip == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    resized_frame = cv2.resize(frame_rgb, (48, 27), interpolation=cv2.INTER_AREA)
                    yield resized_frame
                frame_index += 1
        finally:
            if 'cap' in locals() and cap.isOpened():
                cap.release()

    def process_video(self, video_path: str):
        """
        Xử lý một file video duy nhất để phát hiện chuyển cảnh.
        Returns tensor có shape (1, T, 1)
        """
        if not os.path.exists(video_path):
            print(f"Lỗi: Video không tồn tại tại '{video_path}'")
            return None

        print(f"\n[INFO] Bắt đầu xử lý video: {video_path}")

        all_predictions = []
        frames_batch = []
        frame_generator = self._stream_video_frames(video_path, self.frame_skip)

        for i, resized_frame in enumerate(frame_generator):
            frames_batch.append(resized_frame)
            if len(frames_batch) >= self.batch_size:
                print(f"  -> Đang xử lý batch {len(all_predictions) + 1}...")
                batch_tensor = torch.from_numpy(np.array(frames_batch)).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    single_frame_pred, _ = self.model(batch_tensor.byte())
                    all_predictions.append(single_frame_pred.cpu())
                frames_batch = []

        if frames_batch:
            print(f"  -> Đang xử lý batch cuối cùng với {len(frames_batch)} frames...")
            batch_tensor = torch.from_numpy(np.array(frames_batch)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                single_frame_pred, _ = self.model(batch_tensor.byte())
                all_predictions.append(single_frame_pred.cpu())

        if not all_predictions:
            print(f"[WARNING] Không xử lý được frame nào từ video: {video_path}")
            return None

        final_predictions = torch.cat(all_predictions, dim=1)
        print(f"✅ [SUCCESS] Xử lý hoàn tất. Tensor kết quả có shape: {final_predictions.shape}")
        return final_predictions
