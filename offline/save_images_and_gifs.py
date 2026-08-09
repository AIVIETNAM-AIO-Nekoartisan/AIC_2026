# scripts/save_images_and_gifs.py
"""
Extract shot transitions, save keyframes as images and optional GIF previews.
No object detection. No database upsert. Naming is delegated to
`offline.video_trans_detection.save_transitions_as_images`, ensuring the
same file-naming scheme as your existing pipeline.

Usage examples:

# Basic (WebP images + GIF previews)
python3 -m offline.save_images_and_gifs \
  --input-folder /path/to/videos

# Use JPEG instead of WebP
python3 -m offline.save_images_and_gifs \
  --input-folder /path/to/videos --use-jpeg

# Disable GIFs; custom threshold & frame-skip
python3 -m offline.save_images_and_gifs \
  --input-folder /path/to/videos \
  --threshold 0.45 --frame-skip 2 --disable-gif-preview

# Choose output root directory and enable recursive search
python3 -m offline.save_images_and_gifs \
  --input-folder /path/to/videos --output-root results \
  --recursive
"""

import os
import glob
import argparse
import logging
from datetime import datetime
import json
from typing import Dict, List, Optional

from offline.video_trans_detection import (
    VideoIngestDatabase,
    create_new_output_folder,
    save_transitions_as_images,
)

# --------------------------- Logging ---------------------------------------
LOG_NAME = f'saveonly_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_NAME),
        logging.StreamHandler()
    ]
)

# ----------------------- Helper: metadata dump -----------------------------

def save_processing_metadata(
    output_folder: str,
    video_files: List[str],
    processing_params: Dict,
    keyframes_info: Optional[Dict[str, Dict]] = None,
) -> str:
    """Write a processing_metadata.json into the output folder."""
    metadata = {
        "processing_timestamp": datetime.now().isoformat(),
        "total_videos": len(video_files),
        "videos": [os.path.basename(v) for v in video_files],
        "output_folder": output_folder,
        "parameters": processing_params,
    }

    if keyframes_info:
        total_keyframes = sum(len(info.get('keyframes', [])) for info in keyframes_info.values())
        total_gif_previews = sum(len(info.get('shot_previews', [])) for info in keyframes_info.values())
        metadata["keyframe_summary"] = {
            "total_keyframes": total_keyframes,
            "total_gif_previews": total_gif_previews,
            "gif_folder": os.path.join(output_folder, "gif_previews"),
        }

    metadata_path = os.path.join(output_folder, "processing_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logging.info(f"Đã lưu metadata xử lý tại: {metadata_path}")
    return metadata_path

# ----------------------------- Core logic ----------------------------------

def collect_videos(input_folder: str, recursive: bool) -> List[str]:
    patterns = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm"]
    videos: List[str] = []
    if recursive:
        for pat in patterns:
            videos.extend(glob.glob(os.path.join(input_folder, "**", pat), recursive=True))
    else:
        for pat in patterns:
            videos.extend(glob.glob(os.path.join(input_folder, pat)))
    # Stable order for reproducibility
    videos.sort()
    return videos


def process_videos_save_only(
    input_folder: str,
    output_root: str = "results",
    batch_size: int = 1000,
    frame_skip: int = 2,
    threshold: float = 0.45,
    use_webp: bool = True,
    create_gif_preview: bool = True,
    frames_per_gif: int = 5,
    recursive: bool = False,
):
    """Process videos: detect transitions, save keyframes & GIFs (no detection, no upsert)."""

    # Record params (handy for reproducibility)
    processing_params = {
        "batch_size": batch_size,
        "frame_skip": frame_skip,
        "threshold": threshold,
        "image_format": "webp" if use_webp else "jpeg",
        "gif_preview_enabled": create_gif_preview,
        "frames_per_gif": frames_per_gif,
        "recursive": recursive,
    }

    # TransNetV2-based ingestor (detection of shot transitions only)
    ingestor = VideoIngestDatabase(
        batch_size=batch_size,
        frame_skip=frame_skip,
        yoloe_model_path=None,          # Explicitly unused here
        yoloe_conf_threshold=0.0,       # Irrelevant; kept to satisfy constructor
    )

    video_files = collect_videos(input_folder, recursive=recursive)
    if not video_files:
        logging.warning(f"Không tìm thấy video nào trong thư mục: {input_folder}")
        return

    logging.info(f"Tìm thấy {len(video_files)} video để xử lý.")

    # Output session folder (e.g., results/2025-09-04_11-23-59)
    output_base_folder = create_new_output_folder(output_root)
    logging.info(f"Ảnh keyframe & GIF sẽ được lưu tại: {output_base_folder}")

    successful_videos = 0
    failed_videos = 0
    all_keyframes_info: Dict[str, Dict] = {}

    for video_path in video_files:
        vname = os.path.basename(video_path)
        try:
            logging.info("\n" + "="*50)
            logging.info(f"Đang xử lý video: {vname} ...")

            # 1) Shot boundary detection
            logging.info("--> Bước 1: Phát hiện chuyển cảnh (TransNetV2)...")
            predictions = ingestor.process_video(video_path)

            # 2) Save frames + optional GIFs (naming handled by utility)
            if predictions is not None and len(predictions) > 0:
                logging.info("--> Bước 2: Lưu keyframe + GIF Preview (không chạy object detection)...")
                kinfo = save_transitions_as_images(
                    video_path=video_path,
                    predictions=predictions,
                    output_folder=output_base_folder,
                    frame_skip=ingestor.frame_skip,
                    threshold=threshold,
                    use_webp=use_webp,
                    yoloe_detector=None,             # Absolutely no detection
                    create_gif_preview=create_gif_preview,
                    frames_per_gif=frames_per_gif,
                )

                if kinfo:
                    all_keyframes_info[vname] = kinfo
                    successful_videos += 1
                    logging.info(f"   Keyframes saved: {len(kinfo.get('keyframes', []))}")
                    if create_gif_preview:
                        logging.info(f"   GIF previews created: {len(kinfo.get('shot_previews', []))}")
                else:
                    failed_videos += 1
                    logging.warning(f"Không tạo được keyframe/GIF cho: {vname}")
            else:
                failed_videos += 1
                logging.warning(f"Không phát hiện chuyển cảnh rõ ràng trong: {vname}")

        except Exception as e:
            failed_videos += 1
            logging.exception(f"Lỗi khi xử lý video {vname}: {e}")

    # 3) Persist metadata only (no database upsert)
    save_processing_metadata(output_base_folder, video_files, processing_params, all_keyframes_info)

    # Summary
    logging.info("\n" + "="*50)
    logging.info("📊 TỔNG KẾT:")
    logging.info(f"  - Tổng video xử lý: {len(video_files)}")
    logging.info(f"  - Thành công: {successful_videos}")
    logging.info(f"  - Thất bại: {failed_videos}")
    logging.info(f"  - Thư mục output: {output_base_folder}")
    logging.info(f"  - GIF preview: {'Enabled' if create_gif_preview else 'Disabled'}")

    if all_keyframes_info:
        total_keyframes = sum(len(info.get('keyframes', [])) for info in all_keyframes_info.values())
        total_gif_previews = sum(len(info.get('shot_previews', [])) for info in all_keyframes_info.values())
        logging.info("\n📊 THỐNG KÊ CHI TIẾT:")
        logging.info(f"  - Total keyframes: {total_keyframes}")
        logging.info(f"  - Total GIF previews: {total_gif_previews}")


# ----------------------------- CLI entry -----------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chỉ phát hiện chuyển cảnh và lưu ảnh/GIF (không detection, không upsert)."
    )
    parser.add_argument(
        "--input-folder",
        type=str,
        required=True,
        help="Thư mục chứa video cần xử lý",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="results",
        help="Thư mục gốc để tạo phiên xuất (mặc định: results)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size cho TransNetV2 (mặc định: 1000)",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=2,
        help="Số frame bỏ qua giữa các lần suy luận (mặc định: 2)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.45,
        help="Ngưỡng lọc chuyển cảnh (mặc định: 0.45)",
    )
    parser.add_argument(
        "--use-jpeg",
        action="store_true",
        help="Sử dụng JPEG (mặc định xuất WebP)",
    )
    parser.add_argument(
        "--disable-gif-preview",
        action="store_true",
        help="Tắt tạo GIF preview",
    )
    parser.add_argument(
        "--frames-per-gif",
        type=int,
        default=5,
        help="Số frame trong mỗi GIF preview (mặc định: 5)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Tìm video đệ quy trong các thư mục con",
    )

    args = parser.parse_args()

    process_videos_save_only(
        input_folder=args.input_folder,
        output_root=args.output_root,
        batch_size=args.batch_size,
        frame_skip=args.frame_skip,
        threshold=args.threshold,
        use_webp=not args.use_jpeg,
        create_gif_preview=not args.disable_gif_preview,
        frames_per_gif=args.frames_per_gif,
        recursive=args.recursive,
    )
