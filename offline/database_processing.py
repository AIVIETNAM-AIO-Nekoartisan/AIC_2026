# utils/database_processing.py
import sys
import os
import re
import uuid
from datetime import datetime
import hashlib

# Thêm thư mục gốc của dự án vào đường dẫn tìm kiếm của Python
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from qdrant_client import models

# Import các module cần thiết từ project
# from core.models.OpenCLIP_embedding_optimized import embedding_model
from core.models.Beit3_embedding_optimized import get_embedding_model as get_beit3_model
from core.models.SigLIP_embedding import embedding_model
from core.services.vector_store_optimized import QdrantService
from core.models.YOLOE_object_detection import YOLOEDetection


try:
    from core.models.JinaCLIP_embedding import get_jina_model
    _JINA = get_jina_model()
    _JINA_DIM = _JINA.DIM
except Exception as e:
    print(f"[WARNING] Jina-CLIP-v2 không khả dụng, ingest chỉ với SigLIP. Lý do: {e}")
    _JINA = None
    _JINA_DIM = None

try:
    if not os.getenv("BEIT3_CKPT"):
        # không có checkpoint thì model chỉ là random weights -> vô dụng, bỏ qua
        raise RuntimeError("BEIT3_CKPT chưa đặt")
    _BEIT3 = get_beit3_model(
        checkpoint_path=os.getenv("BEIT3_CKPT"),
        tokenizer_path=None,  # ingest ảnh nên không cần tokenizer
        fp16=True,
        input_size=384,
        warmup=False,
        aggressive_optimization=False
    )
    _BEIT3_DIM = int(_BEIT3.model.args.encoder_embed_dim)
except Exception as e:
    # BEiT-3 optional (giống api_server): thiếu weights thì chỉ ingest vector siglip
    print(f"[WARNING] BEiT-3 không khả dụng, ingest chỉ với SigLIP. Lý do: {e}")
    _BEIT3 = None
    _BEIT3_DIM = None


def generate_deterministic_id(video_name: str, shot_number: int, position: str):
    """
    Tạo ID xác định duy nhất cho mỗi frame dựa trên thông tin video.
    """
    id_string = f"{video_name}_{shot_number}_{position}"
    return hashlib.md5(id_string.encode()).hexdigest()

def parse_keyframe_filename(filename: str):
    """
    Phân tích tên file keyframe để trích xuất metadata.

    Returns:
        dict: Dictionary chứa metadata hoặc None nếu không phân tích được
    """
    # Pattern cho format mới với position
    pattern = re.compile(
        r"(.+?)__shot_(\d+)__pos_(start|middle|end)__frame_(\d+)__ts_([\d\.]+)\.(jpg|webp)",
        re.IGNORECASE
    )

    match = pattern.match(filename)
    if not match:
        return None

    return {
        "video_name": match.group(1),
        "shot_number": int(match.group(2)),
        "position_in_shot": match.group(3),
        "frame_index": int(match.group(4)),
        "timestamp_seconds": float(match.group(5)),
        "file_extension": match.group(6)
    }

def find_gif_preview(video_name: str, shot_number: int, gif_folder: str):
    """
    Tìm file GIF preview cho một shot cụ thể

    Args:
        video_name: Tên video
        shot_number: Số shot
        gif_folder: Thư mục chứa GIF

    Returns:
        str: Đường dẫn đến GIF hoặc None
    """
    gif_filename = f"{video_name}_shot_{shot_number:03d}_preview.gif"
    gif_path = os.path.join(gif_folder, gif_filename)

    if os.path.exists(gif_path):
        return gif_path

    return None

def ingest_images_from_folder(folder_path: str, collection_name: str = "test_OpenCLIP",
                            use_object_detection: bool = True, yoloe_model_path: str = None):
    """
    Quét một thư mục, tìm tất cả các file ảnh, tạo embeddings,
    trích xuất metadata từ tên file, chạy object detection và lưu vào Qdrant theo lô (batch).
    Bao gồm cả thông tin GIF preview nếu có.

    Args:
        folder_path: Đường dẫn thư mục chứa ảnh
        collection_name: Tên collection Qdrant
        use_object_detection: Có chạy object detection không
        yoloe_model_path: Đường dẫn đến YOLOE model
    """
    BATCH_SIZE = 32

    # Khởi tạo Qdrant service với collection name có thể tùy chỉnh
    # qdrant_service = QdrantService(collection_name=collection_name, vector_dim=1024) # for OpenCLIP
    # # qdrant_service = QdrantService(collection_name=collection_name, vector_dim=1152)  # for SigLIP
    # qdrant_service.setup_collection()

    named_vectors = {"siglip": 1152}
    if _JINA_DIM:
        named_vectors["jina"] = _JINA_DIM
    if _BEIT3_DIM:
        named_vectors["beit3"] = _BEIT3_DIM
    qdrant_service = QdrantService(
    collection_name=collection_name,
    named_vectors=named_vectors,
    )
    qdrant_service.setup_collection()
    # Kiểm tra thư mục GIF previews
    gif_folder = os.path.join(folder_path, "gif_previews")
    has_gif_previews = os.path.exists(gif_folder)
    if has_gif_previews:
        print(f"[INFO] Tìm thấy thư mục GIF previews: {gif_folder}")

    # Khởi tạo YOLOE detector nếu cần
    yoloe_detector = None
    if use_object_detection:
        if yoloe_model_path and os.path.exists(yoloe_model_path):
            print(f"[INFO] Khởi tạo YOLOE detector cho batch processing...")
            yoloe_detector = YOLOEDetection(
                model_path=yoloe_model_path,
                conf_threshold=0.25,
                batch_size=2
            )
        else:
            print(f"[WARNING] YOLOE model không tìm thấy, bỏ qua object detection")
            use_object_detection = False

    try:
        filenames = os.listdir(folder_path)
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy thư mục '{folder_path}'. Vui lòng kiểm tra lại đường dẫn.")
        return

    # Hỗ trợ cả JPG và WebP
    image_extensions = ('.png', '.jpg', '.jpeg', '.webp')
    image_files = [f for f in filenames if f.lower().endswith(image_extensions)]

    if not image_files:
        print(f"Không tìm thấy file ảnh nào trong thư mục '{folder_path}'.")
        return

    print(f"Tìm thấy {len(image_files)} ảnh. Bắt đầu xử lý...")
    if use_object_detection:
        print(f"Object detection: ENABLED")
    else:
        print(f"Object detection: DISABLED")

    # Chạy object detection batch trước nếu cần
    object_detection_results = {}
    if use_object_detection and yoloe_detector:
        print(f"Chạy object detection cho {len(image_files)} ảnh...")
        image_paths = [os.path.join(folder_path, f) for f in image_files]
        detection_results = yoloe_detector.detect_batch_images(image_paths)

        # Tạo mapping từ filename to detection results
        for result in detection_results:
            filename = os.path.basename(result['image_path'])
            object_detection_results[filename] = {
                'has_objects': result.get('has_objects', False),
                'object_count': result.get('detection_count', 0),
                'unique_classes': [d['class'] for d in result.get('detections', [])],
                'class_counts': {},
                'avg_confidence': 0.0,
                'max_confidence': 0.0
            }

            # Tính toán thống kê
            if result.get('detections'):
                confidences = [d['confidence'] for d in result['detections']]
                class_counts = {}
                for d in result['detections']:
                    class_name = d['class']
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1

                object_detection_results[filename].update({
                    'class_counts': class_counts,
                    'avg_confidence': sum(confidences) / len(confidences),
                    'max_confidence': max(confidences)
                })

    points_to_upsert = []
    processed_count = 0
    error_count = 0
    gif_preview_count = 0

    print(f"Tạo embeddings và chuẩn bị dữ liệu cho Qdrant...")
    for i, filename in enumerate(image_files):
        image_path = os.path.join(folder_path, filename)

        try:
            # Tạo embedding cho ảnh
            # image_vector = embedding_model.get_image_embedding(image_path)
            image_vector_siglip = embedding_model.get_image_embedding(image_path)
            image_vector_jina = _JINA.get_image_embedding(image_path) if _JINA else None
            image_vector_beit3 = _BEIT3.get_image_embedding(image_path) if _BEIT3 else None
            # Phân tích metadata từ tên file
            metadata = parse_keyframe_filename(filename)

            # Chuẩn bị payload với cấu trúc tốt hơn
            payload = {
                "file_path": image_path,
                "filename": filename,
                "ingested_at": datetime.now().isoformat(),
                "file_size_bytes": os.path.getsize(image_path),
            }

            # Thêm object detection results nếu có
            if use_object_detection and filename in object_detection_results:
                payload["object_detection"] = object_detection_results[filename]

            if metadata:
                # Đây là keyframe từ video
                shot_id = f"{metadata['video_name']}_shot_{metadata['shot_number']:03d}"

                payload.update({
                    "source_type": "video_keyframe",
                    "shot_id": shot_id,  # Thêm shot_id để group frames
                    "video": {
                        "name": metadata["video_name"],
                        "filename": f"{metadata['video_name']}.mp4",  # Giả sử là mp4
                    },
                    "shot": {
                        "number": metadata["shot_number"],
                        "position": metadata["position_in_shot"],
                    },
                    "frame": {
                        "index": metadata["frame_index"],
                        "timestamp_seconds": metadata["timestamp_seconds"],
                        "timestamp_formatted": f"{int(metadata['timestamp_seconds']//60):02d}:{metadata['timestamp_seconds']%60:05.2f}"
                    }
                })

                # Kiểm tra và thêm thông tin GIF preview nếu có
                if has_gif_previews:
                    gif_path = find_gif_preview(metadata["video_name"], metadata["shot_number"], gif_folder)
                    if gif_path:
                        payload["gif_preview"] = {
                            "path": gif_path,
                            "filename": os.path.basename(gif_path),
                            "exists": True
                        }
                        gif_preview_count += 1
                    else:
                        payload["gif_preview"] = {
                            "path": None,
                            "filename": None,
                            "exists": False
                        }

                # Tạo ID xác định cho frame
                point_id = generate_deterministic_id(
                    metadata["video_name"],
                    metadata["shot_number"],
                    metadata["position_in_shot"]
                )
            else:
                # Ảnh thường không phải từ video
                payload.update({
                    "source_type": "standalone_image",
                })
                point_id = str(uuid.uuid4())

            # Tạo point để upsert
            # point = models.PointStruct(
            #     id=point_id,
            #     vector=image_vector.tolist(),
            #     payload=payload
            # )
            # points_to_upsert.append(point)

            vectors = {"siglip": image_vector_siglip.tolist()}
            if image_vector_jina is not None:
                vectors["jina"] = image_vector_jina.tolist()
            if image_vector_beit3 is not None:
                vectors["beit3"] = image_vector_beit3.tolist()
            point = models.PointStruct(
                id=point_id,
                vector=vectors,
                payload=payload
            )
            points_to_upsert.append(point)

            # Upsert theo batch
            if len(points_to_upsert) >= BATCH_SIZE:
                qdrant_service.upsert_points(points_to_upsert)
                processed_count += len(points_to_upsert)
                print(f"  -> Đã nạp lô {i // BATCH_SIZE + 1} ({len(points_to_upsert)} điểm). Tổng: {processed_count}/{len(image_files)}")
                points_to_upsert = []

        except Exception as e:
            error_count += 1
            print(f"  - LỖI khi xử lý file {filename}: {e}")

    # Upsert batch cuối cùng
    if points_to_upsert:
        qdrant_service.upsert_points(points_to_upsert)
        processed_count += len(points_to_upsert)
        print(f"  -> Đã nạp lô cuối cùng ({len(points_to_upsert)} điểm).")

    # Thống kê object detection
    if use_object_detection and object_detection_results:
        images_with_objects = sum(1 for r in object_detection_results.values() if r['has_objects'])
        total_objects = sum(r['object_count'] for r in object_detection_results.values())
        all_classes = set()
        for r in object_detection_results.values():
            all_classes.update(r['unique_classes'])

        print(f"\n📊 OBJECT DETECTION SUMMARY:")
        print(f"  - Images with objects: {images_with_objects}/{len(image_files)}")
        print(f"  - Total objects detected: {total_objects}")
        print(f"  - Unique classes found: {len(all_classes)}")
        if all_classes:
            print(f"  - Classes: {', '.join(sorted(all_classes))}")

    # Thống kê kết quả tổng
    print(f"\n📊 KẾT QUẢ:")
    print(f"  - Tổng số ảnh: {len(image_files)}")
    print(f"  - Xử lý thành công: {processed_count}")
    print(f"  - Lỗi: {error_count}")
    print(f"  - Collection: {collection_name}")
    print(f"  - Object detection: {'Enabled' if use_object_detection else 'Disabled'}")
    if has_gif_previews:
        print(f"  - GIF previews found: {gif_preview_count}")