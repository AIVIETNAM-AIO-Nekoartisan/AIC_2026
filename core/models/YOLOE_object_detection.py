# model/YOLOE_object_detection.py
import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
from typing import List, Dict, Optional, Union

class YOLOEDetection:
    def __init__(self, model_path='yolov8x-oiv7.pt', conf_threshold=0.25, batch_size=32):
        """
        Khởi tạo YOLOE Detection cho pipeline video processing

        Args:
            model_path: Đường dẫn đến model YOLO
            conf_threshold: Ngưỡng confidence
            batch_size: Kích thước batch
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.batch_size = batch_size
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

    def detect_single_image(self, image_path: str) -> Optional[Dict]:
        """
        Detect objects trong một hình ảnh duy nhất

        Args:
            image_path: Đường dẫn đến hình ảnh

        Returns:
            Dict với detections hoặc None nếu không có detection
        """
        try:
            results = self.model(image_path, conf=self.conf_threshold, verbose=False)
            result = results[0]

            if len(result.boxes) > 0:
                detections = []
                for box in result.boxes:
                    detections.append({
                        'class': self.model.names[int(box.cls)],
                        'confidence': float(box.conf),
                        'bbox': box.xyxy.tolist()[0]
                    })

                return {
                    'detections': detections,
                    'detection_count': len(detections),
                    'annotated_image': result.plot()
                }
        except Exception as e:
            print(f"Error detecting {image_path}: {e}")

        return None

    def detect_frame_array(self, frame: np.ndarray) -> Dict:
        """
        Detect objects trong frame array (từ cv2) - chỉ lấy metadata, không bbox

        Args:
            frame: Frame dưới dạng numpy array (BGR format)

        Returns:
            Dict với object metadata
        """
        try:
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            result = results[0]

            if len(result.boxes) > 0:
                class_counts = {}
                confidences = []

                for box in result.boxes:
                    class_name = self.model.names[int(box.cls)]
                    confidence = float(box.conf)

                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    confidences.append(confidence)

                return {
                    'has_objects': True,
                    'object_count': len(result.boxes),
                    'unique_classes': list(class_counts.keys()),
                    'class_counts': class_counts,
                    'avg_confidence': sum(confidences) / len(confidences),
                    'max_confidence': max(confidences),
                    'min_confidence': min(confidences)
                }
        except Exception as e:
            print(f"Error detecting frame: {e}")

        return {
            'has_objects': False,
            'object_count': 0,
            'unique_classes': [],
            'class_counts': {},
            'avg_confidence': 0.0,
            'max_confidence': 0.0,
            'min_confidence': 0.0
        }

    def find_images(self, input_folder: Union[str, Path]) -> List[Path]:
        """Tìm tất cả hình ảnh trong thư mục"""
        input_path = Path(input_folder)
        images = []
        for file in input_path.rglob('*'):
            if file.suffix.lower() in self.supported_formats:
                images.append(file)
        return sorted(images)

    def detect_batch_images(self, image_paths: List[Union[str, Path]]) -> List[Dict]:
        """
        Batch detection cho list image paths

        Args:
            image_paths: List đường dẫn hình ảnh (str hoặc Path)

        Returns:
            List kết quả detection
        """
        results = []

        for i in tqdm(range(0, len(image_paths), self.batch_size), desc="YOLOE Detection"):
            batch_paths = image_paths[i:i + self.batch_size]

            try:
                # Convert to string paths for YOLO
                batch_str_paths = [str(p) for p in batch_paths]
                batch_results = self.model(batch_str_paths, conf=self.conf_threshold, verbose=False)

                for result, img_path in zip(batch_results, batch_paths):
                    # Convert to Path object if it's string
                    img_path = Path(img_path) if isinstance(img_path, str) else img_path

                    detection_data = {
                        'image_path': str(img_path),
                        'image_name': img_path.name,
                        'has_objects': len(result.boxes) > 0,
                        'detection_count': len(result.boxes),
                        'detections': []
                    }

                    if len(result.boxes) > 0:
                        for box in result.boxes:
                            detection_data['detections'].append({
                                'class': self.model.names[int(box.cls)],
                                'confidence': float(box.conf),
                                'bbox': box.xyxy.tolist()[0]
                            })

                    results.append(detection_data)

            except Exception as e:
                print(f"Error batch {i//self.batch_size + 1}: {e}")
                # Thêm empty results cho batch bị lỗi
                for img_path in batch_paths:
                    img_path = Path(img_path) if isinstance(img_path, str) else img_path
                    results.append({
                        'image_path': str(img_path),
                        'image_name': img_path.name,
                        'has_objects': False,
                        'detection_count': 0,
                        'detections': [],
                        'error': str(e)
                    })

        return results

    def process_folder(self, input_folder: Union[str, Path], save_annotated: bool = False,
                      output_folder: Optional[Union[str, Path]] = None) -> List[Dict]:
        """
        Xử lý tất cả hình ảnh trong thư mục

        Args:
            input_folder: Thư mục chứa hình ảnh
            save_annotated: Có lưu ảnh đã được annotate không
            output_folder: Thư mục lưu ảnh annotated

        Returns:
            List kết quả detection
        """
        # Tìm hình ảnh
        image_files = self.find_images(input_folder)
        if not image_files:
            print("No images found!")
            return []

        print(f"Processing {len(image_files)} images with YOLOE...")

        # Detect
        results = self.detect_batch_images(image_files)

        # Lưu ảnh annotated nếu cần
        if save_annotated and output_folder:
            output_path = Path(output_folder)
            output_path.mkdir(exist_ok=True)

            for result in tqdm(results, desc="Saving annotated", leave=False):
                if result['has_objects'] and 'error' not in result:
                    try:
                        # Chạy lại detection để lấy annotated image
                        img_result = self.model(result['image_path'], conf=self.conf_threshold, verbose=False)[0]
                        annotated = img_result.plot()

                        save_path = output_path / f"{Path(result['image_path']).stem}_detected.jpg"
                        cv2.imwrite(str(save_path), annotated)
                    except Exception as e:
                        print(f"Error saving annotated image: {e}")

        # Statistics
        total_detections = sum(r['detection_count'] for r in results)
        images_with_objects = sum(1 for r in results if r['has_objects'])

        print(f"YOLOE Detection completed:")
        print(f"  - Images processed: {len(results)}")
        print(f"  - Images with objects: {images_with_objects}")
        print(f"  - Total objects detected: {total_detections}")

        return results

# Utility functions cho integration
def get_detection_summary(detections: List[Dict]) -> Dict:
    """
    Tạo summary từ danh sách detections

    Args:
        detections: List detection results

    Returns:
        Dict summary
    """
    if not detections:
        return {
            'total_objects': 0,
            'unique_classes': [],
            'class_counts': {},
            'avg_confidence': 0.0,
            'has_objects': False
        }

    all_objects = []
    for det in detections:
        all_objects.extend(det.get('detections', []))

    if not all_objects:
        return {
            'total_objects': 0,
            'unique_classes': [],
            'class_counts': {},
            'avg_confidence': 0.0,
            'has_objects': False
        }

    # Thống kê
    class_counts = {}
    confidences = []

    for obj in all_objects:
        class_name = obj['class']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        confidences.append(obj['confidence'])

    return {
        'total_objects': len(all_objects),
        'unique_classes': list(class_counts.keys()),
        'class_counts': class_counts,
        'avg_confidence': sum(confidences) / len(confidences) if confidences else 0.0,
        'has_objects': True
    }