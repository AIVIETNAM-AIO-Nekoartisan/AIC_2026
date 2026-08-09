# scripts/ingest_data.py
import os
import glob
import argparse
import logging
from datetime import datetime
import json

from offline.video_trans_detection import (
    VideoIngestDatabase,
    create_new_output_folder,
    save_transitions_as_images,
)
from offline.database_processing import ingest_images_from_folder

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'ingest_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

def save_processing_metadata(output_folder: str, video_files: list, processing_params: dict,
                           keyframes_info: dict = None):
    """
    Lưu metadata về quá trình xử lý.
    """
    metadata = {
        "processing_timestamp": datetime.now().isoformat(),
        "total_videos": len(video_files),
        "videos": [os.path.basename(v) for v in video_files],
        "output_folder": output_folder,
        "parameters": processing_params
    }

    # Thêm thống kê keyframes và GIF nếu có
    if keyframes_info:
        total_keyframes = sum(len(info.get('keyframes', [])) for info in keyframes_info.values())
        total_gif_previews = sum(
            len(info.get('shot_previews', []))
            for info in keyframes_info.values()
        )

        metadata["keyframe_summary"] = {
            "total_keyframes": total_keyframes,
            "total_gif_previews": total_gif_previews,
            "gif_folder": os.path.join(output_folder, "gif_previews")
        }

    metadata_path = os.path.join(output_folder, "processing_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logging.info(f"Đã lưu metadata xử lý tại: {metadata_path}")

def process_videos_in_folder(
    input_folder: str,
    collection_name: str = "test_OpenCLIP",
    batch_size: int = 1000,
    frame_skip: int = 2,
    threshold: float = 0.25,
    use_webp: bool = True,
    use_object_detection: bool = True,
    yoloe_model_path: str = None,
    create_gif_preview: bool = True,
    frames_per_gif: int = 5
):
    """
    Xử lý tất cả video trong một thư mục: phát hiện chuyển cảnh,
    lưu keyframe, tạo GIF preview, và nạp embedding vào database.
    Object detection chỉ chạy 1 lần duy nhất ở database processing phase.

    Args:
        input_folder: Thư mục chứa video
        collection_name: Tên collection Qdrant
        batch_size: Batch size cho TransNetV2
        frame_skip: Frame skip cho TransNetV2
        threshold: Ngưỡng phát hiện chuyển cảnh
        use_webp: Sử dụng WebP thay vì JPEG
        use_object_detection: Có chạy object detection không
        yoloe_model_path: Đường dẫn YOLOE model
        create_gif_preview: Có tạo GIF preview không
        frames_per_gif: Số frames cho mỗi GIF
    """
    # Lưu các tham số xử lý
    processing_params = {
        "batch_size": batch_size,
        "frame_skip": frame_skip,
        "threshold": threshold,
        "image_format": "webp" if use_webp else "jpeg",
        "collection_name": collection_name,
        "object_detection_enabled": use_object_detection,
        "yoloe_model_path": yoloe_model_path,
        "gif_preview_enabled": create_gif_preview,
        "frames_per_gif": frames_per_gif
    }

    # Khởi tạo VideoIngestDatabase (KHÔNG có object detection)
    ingestor = VideoIngestDatabase(
        batch_size=batch_size,
        frame_skip=frame_skip,
        yoloe_model_path=yoloe_model_path,  # Chỉ để log thông tin
        yoloe_conf_threshold=0.25
    )

    # Tìm tất cả các file video trong thư mục
    video_extensions = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm"]
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(input_folder, ext)))

    if not video_files:
        logging.warning(f"Không tìm thấy video nào trong thư mục: {input_folder}")
        return

    logging.info(f"Tìm thấy {len(video_files)} video để xử lý.")

    # Tạo một thư mục output chung cho tất cả keyframe
    output_base_folder = create_new_output_folder("results")
    logging.info(f"Các keyframe sẽ được lưu tại: {output_base_folder}")

    # Thống kê xử lý
    successful_videos = 0
    failed_videos = 0
    all_keyframes_info = {}

    for video_path in video_files:
        try:
            logging.info(f"\n{'='*50}")
            logging.info(f"Đang xử lý video: {os.path.basename(video_path)}...")

            # 1. Phát hiện chuyển cảnh
            logging.info("--> Bước 1: Phát hiện chuyển cảnh...")
            predictions = ingestor.process_video(video_path)

            if predictions is not None and len(predictions) > 0:
                # 2. Lưu các keyframe và tạo GIF (KHÔNG có object detection)
                logging.info(f"--> Bước 2: Lưu keyframe + GIF Preview...")
                keyframes_info = save_transitions_as_images(
                    video_path=video_path,
                    predictions=predictions,
                    output_folder=output_base_folder,
                    frame_skip=ingestor.frame_skip,
                    threshold=threshold,
                    use_webp=use_webp,
                    yoloe_detector=None,  # KHÔNG truyền detector
                    create_gif_preview=create_gif_preview,
                    frames_per_gif=frames_per_gif
                )

                if keyframes_info:
                    all_keyframes_info[os.path.basename(video_path)] = keyframes_info
                    successful_videos += 1

                    # Log thống kê cho video này
                    logging.info(f"   Keyframes saved: {len(keyframes_info.get('keyframes', []))}")

                    if create_gif_preview:
                        logging.info(f"   GIF previews created: {len(keyframes_info.get('shot_previews', []))}")
                else:
                    failed_videos += 1
            else:
                logging.warning(f"Không phát hiện cảnh rõ ràng trong {os.path.basename(video_path)}")
                failed_videos += 1

        except Exception as e:
            failed_videos += 1
            logging.exception(f"Lỗi nghiêm trọng khi xử lý video {os.path.basename(video_path)}: {e}")

    # Lưu metadata về quá trình xử lý
    save_processing_metadata(output_base_folder, video_files, processing_params, all_keyframes_info)

    # 3. Nạp tất cả các ảnh đã được tạo vào database (Object detection chạy ở đây)
    if os.path.exists(output_base_folder) and len(os.listdir(output_base_folder)) > 1:  # >1 vì có file metadata
        logging.info(f"\n{'='*50}")
        logging.info(f"--> Bước 3: Nạp embedding + metadata vào Qdrant collection '{collection_name}'...")
        if use_object_detection:
            logging.info(f"   Object detection sẽ chạy trong phase này với model: {yoloe_model_path}")

        ingest_images_from_folder(
            folder_path=output_base_folder,
            collection_name=collection_name,
            use_object_detection=use_object_detection,
            yoloe_model_path=yoloe_model_path
        )
        logging.info("✔️ Hoàn tất nạp dữ liệu vào database!")
    else:
        logging.info("Không có keyframe mới nào để nạp vào database.")

    # Báo cáo kết quả tổng quan
    logging.info(f"\n{'='*50}")
    logging.info("📊 TỔNG KẾT:")
    logging.info(f"  - Tổng video xử lý: {len(video_files)}")
    logging.info(f"  - Thành công: {successful_videos}")
    logging.info(f"  - Thất bại: {failed_videos}")
    logging.info(f"  - Thư mục output: {output_base_folder}")
    logging.info(f"  - Collection Qdrant: {collection_name}")
    logging.info(f"  - Object detection: {'Enabled (chạy trong database phase)' if use_object_detection else 'Disabled'}")
    logging.info(f"  - GIF preview: {'Enabled' if create_gif_preview else 'Disabled'}")

    # Thống kê tổng
    if all_keyframes_info:
        total_keyframes = sum(len(info.get('keyframes', [])) for info in all_keyframes_info.values())
        total_gif_previews = sum(len(info.get('shot_previews', [])) for info in all_keyframes_info.values())

        logging.info(f"\n📊 THỐNG KÊ CHI TIẾT:")
        logging.info(f"  - Total keyframes: {total_keyframes}")
        logging.info(f"  - Total GIF previews: {total_gif_previews}")
        logging.info(f"  - Object detection sẽ chạy cho {total_keyframes} ảnh trong database phase")


if __name__ == "__main__":
    # Thiết lập để nhận tham số từ dòng lệnh
    parser = argparse.ArgumentParser(description="Xử lý video và nạp vào Qdrant database với GIF preview và object detection.")
    parser.add_argument(
        "--input-folder",
        type=str,
        required=True,
        help="Đường dẫn đến thư mục chứa các file video cần xử lý."
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="test_OpenCLIP",
        help="Tên collection trong Qdrant (default: test_OpenCLIP)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size cho xử lý video (default: 1000)"
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=2,
        help="Số frame bỏ qua giữa các frame xử lý (default: 2)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.45,
        help="Ngưỡng phát hiện chuyển cảnh (default: 0.45)"
    )
    parser.add_argument(
        "--use-jpeg",
        action="store_true",
        help="Sử dụng JPEG thay vì WebP (mặc định là WebP)"
    )
    parser.add_argument(
        "--disable-object-detection",
        action="store_true",
        help="Tắt object detection"
    )
    parser.add_argument(
        "--yoloe-model-path",
        type=str,
        default="model_weights/yolov8x-oiv7.pt",
        help="Đường dẫn đến YOLOE model (default: model_weights/yolov8x-oiv7.pt)"
    )
    parser.add_argument(
        "--disable-gif-preview",
        action="store_true",
        help="Tắt tạo GIF preview"
    )
    parser.add_argument(
        "--frames-per-gif",
        type=int,
        default=5,
        help="Số frames cho mỗi GIF preview (default: 5)"
    )

    args = parser.parse_args()

    process_videos_in_folder(
        input_folder=args.input_folder,
        collection_name=args.collection_name,
        batch_size=args.batch_size,
        frame_skip=args.frame_skip,
        threshold=args.threshold,
        use_webp=not args.use_jpeg,
        use_object_detection=not args.disable_object_detection,
        yoloe_model_path=args.yoloe_model_path,
        create_gif_preview=not args.disable_gif_preview,
        frames_per_gif=args.frames_per_gif
    )


# Ví dụ sử dụng:
# python scripts/ingest_data.py --input-folder /path/to/videos
# python scripts/ingest_data.py --input-folder /path/to/videos --collection-name my_videos --use-jpeg
# python scripts/ingest_data.py --input-folder /path/to/videos --disable-object-detection
# python scripts/ingest_data.py --input-folder /path/to/videos --disable-gif-preview
# python scripts/ingest_data.py --input-folder /path/to/videos --frames-per-gif 7

# python3 -m offline.ingest_data --input-folder /media/tan/F/AI_Challenge_2025/data/video --collection-name my_videos --frames-per-gif 7