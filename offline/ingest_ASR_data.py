# scripts/ingest_ASR_data.py
import os
os.environ["HF_HUB_DISABLE_SAFETENSORS_CONVERSION"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
import json
import time
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import gc
import psutil
import numpy as np
from tqdm import tqdm
import torch
import argparse
from concurrent.futures import ThreadPoolExecutor

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.models.Whisper_VAD import WhisperTranscription
from core.services.elasticsearch_service import ElasticsearchService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_processing_es.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    input_folder: str
    output_folder: str = "./processing_output"
    checkpoint_file: str = "./processing_checkpoint_es.json"
    index_name: str = "video_transcripts"
    es_host: str = "localhost"
    es_port: int = 9200
    whisper_model: str = "/home/tan/.cache/huggingface/hub/models--vinai--PhoWhisper-medium/snapshots/55a7e3eb6c906de891f8f06a107754427dd3be79"
    whisper_device: str = "cuda:0"
    whisper_batch_size: int = 1
    speed_mode: bool = True
    chunk_duration: float = 20.0
    overlap: float = 5.0
    use_vad: bool = True
    target_words: int = 150
    similarity_threshold: float = 0.65
    max_chunk_duration: float = 90
    min_chunk_duration: float = 20
    batch_upload_size: int = 500  # NEW: tăng mạnh bulk size
    memory_threshold_gb: float = 0.85  # NEW: nới ngưỡng
    resume: bool = True
    save_checkpoint_every: int = 3
    force_reprocess: bool = False

class VideoProcessor:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.processed_files = {}
        self.failed_files = {}
        self.stats = {
            'total_videos': 0,
            'processed_videos': 0,
            'failed_videos': 0,
            'total_chunks': 0,
            'total_duration': 0,
            'start_time': None,
            'total_size_gb': 0,
            'reprocessed_videos': 0
        }
        Path(self.config.output_folder).mkdir(parents=True, exist_ok=True)
        self.whisper = None
        self.es_service = None
        self.executor = ThreadPoolExecutor(max_workers=2)  # NEW: ES upsert song song
        if self.config.resume and not self.config.force_reprocess:
            self.load_checkpoint()
        elif self.config.force_reprocess:
            logger.info("Force reprocess mode - ignoring checkpoint")
            self.clear_checkpoint()

    def clear_checkpoint(self):
        self.processed_files = {}
        self.failed_files = {}
        if Path(self.config.checkpoint_file).exists():
            backup_file = f"{self.config.checkpoint_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            Path(self.config.checkpoint_file).rename(backup_file)
            logger.info(f"Backed up old checkpoint to {backup_file}")

    def initialize_models(self):
        logger.info("Initializing models...")
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.9)
            torch.cuda.empty_cache()
        if self.whisper is None:
            self.whisper = WhisperTranscription(
                                model_name=self.config.whisper_model,
                                device=self.config.whisper_device,
                                batch_size=self.config.whisper_batch_size,
                                embedding_model="sentence-transformers/distiluse-base-multilingual-cased-v2",
                                speed_mode=self.config.speed_mode,
                                offload_folder=".whisper_offload",
                                force_pipe_bs=self.config.whisper_batch_size if self.config.whisper_batch_size > 0 else None  # NEW
                            )
        if self.es_service is None:
            self.es_service = ElasticsearchService(
                host=self.config.es_host,
                port=self.config.es_port,
                index_name=self.config.index_name
            )
            self.es_service.setup_index()
        logger.info("Models initialized successfully")

    def get_file_hash(self, file_path: Path) -> str:
        stat = file_path.stat()
        hash_input = f"{file_path}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    def should_process_file(self, file_path: Path) -> bool:
        file_hash = self.get_file_hash(file_path)
        if self.config.force_reprocess:
            return True
        if file_hash in self.processed_files:
            file_info = self.processed_files[file_hash]
            if isinstance(file_info, dict):
                if file_info.get('status') == 'success' and file_info.get('chunks', 0) > 0:
                    logger.info(f"Skipping {file_path.name} - already processed with {file_info['chunks']} chunks")
                    return False
                elif file_info.get('status') == 'failed':
                    logger.info(f"Retrying previously failed file: {file_path.name}")
                    return True
            else:
                logger.info(f"File in old checkpoint format, will reprocess: {file_path.name}")
                return True
        return True

    def find_video_files(self) -> List[Path]:
        input_path = Path(self.config.input_folder)
        video_extensions = [
            '.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.mp3', '.wav', '.m4a'
        ]
        all_video_files = []
        for ext in video_extensions:
            all_video_files.extend(input_path.glob(f"**/*{ext}"))
            all_video_files.extend(input_path.glob(f"**/*{ext.upper()}"))
        filtered_files = []
        skipped_count = 0
        for file in sorted(all_video_files):
            if self.should_process_file(file):
                filtered_files.append(file)
            else:
                skipped_count += 1
        if skipped_count > 0:
            logger.info(f"Skipping {skipped_count} already processed files")
        return filtered_files

    def check_memory(self) -> bool:
        memory = psutil.virtual_memory()
        usage_ratio = memory.percent / 100
        if usage_ratio > self.config.memory_threshold_gb:
            logger.warning(f"Memory usage high: {usage_ratio:.1%}")
            gc.collect()
            return False
        return True

    def _upload_batches(self, documents: List[Dict]) -> bool:
        if not documents:
            return False
        success = True
        for i in range(0, len(documents), self.config.batch_upload_size):
            batch = documents[i:i + self.config.batch_upload_size]
            try:
                self.es_service.upsert_documents(batch)
                logger.info(f"✓ Uploaded batch {i//self.config.batch_upload_size + 1}: {len(batch)} chunks")
            except Exception as e:
                logger.error(f"✗ Failed to upload batch: {e}")
                success = False
        return success

    def process_single_video(self, video_path: Path) -> Dict:
        file_hash = self.get_file_hash(video_path)
        try:
            logger.info(f"Processing: {video_path.name}")
            is_reprocess = file_hash in self.processed_files
            if is_reprocess:
                logger.info(f"Reprocessing file: {video_path.name}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # chỉ dọn 1 lần trước video

            # ==== TRANSCRIBE (GPU batched) ====
            result = self.whisper.transcribe_file_with_sliding_window(
                video_path,
                language=None,
                chunk_duration=self.config.chunk_duration,
                overlap=self.config.overlap,
                use_vad=self.config.use_vad,
                vad_min_silence_ms=250,
                vad_speech_pad_ms=200,
                vad_threshold=0.5,
                pad_silence_tail=0.3,
                batch_infer=True  # giữ API
            )
            if not result['success']:
                raise Exception(result.get('error', 'Unknown error'))

            semantic_chunks = result.get('semantic_chunks', [])
            if not semantic_chunks and result.get('segments'):
                logger.info("Creating semantic chunks from segments...")
                semantic_chunks = self.create_semantic_chunks(result['segments'])

            logger.info(f"Got {len(semantic_chunks)} semantic chunks")

            # ==== BUILD DOCS ====
            documents = []
            for i, chunk in enumerate(semantic_chunks):
                text = (chunk.get('text', '') or '').strip()
                if not text:
                    continue
                doc = {
                    'id': f"{file_hash}_{i:04d}",
                    'file_path': str(video_path),
                    'file_name': video_path.name,
                    'file_hash': file_hash,
                    'chunk_index': i,
                    'content': text,
                    'content_type': 'audio',
                    'start_time': float(chunk.get('start_time', 0)),
                    'end_time': float(chunk.get('end_time', 0)),
                    'duration': float(chunk.get('duration', 0)),
                    'word_count': chunk.get('word_count', len(text.split())),
                    'timestamp': chunk.get('timestamp', ''),
                    'language': result.get('language', 'unknown'),
                    'total_duration': result.get('total_duration', 0),
                    'processed_at': datetime.now().isoformat(),
                }
                documents.append(doc)

            # ==== ASYNC ES INGEST ====
            upload_success = True
            if documents:
                fut = self.executor.submit(self._upload_batches, documents)
                upload_success = fut.result()  # nếu muốn overlap với video kế tiếp, có thể KHÔNG .result()
            else:
                logger.warning("No chunks to upload!")
                upload_success = False

            # ==== SAVE LOCAL JSON ====
            output_file = Path(self.config.output_folder) / f"{video_path.stem}_result.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'file_path': str(video_path),
                    'file_hash': file_hash,
                    'chunks_count': len(documents),
                    'upload_success': upload_success,
                    'processing_time': result.get('processing_time', 0),
                    'total_duration': result.get('total_duration', 0),
                    'language': result.get('language', 'unknown'),
                    'full_text': result.get('full_text', ''),
                    'semantic_chunks': semantic_chunks
                }, f, indent=2, ensure_ascii=False)

            if not is_reprocess:
                self.stats['processed_videos'] += 1
            self.stats['total_chunks'] += len(documents)
            self.stats['total_duration'] += result.get('total_duration', 0)

            self.processed_files[file_hash] = {
                'status': 'success' if upload_success else 'partial',
                'chunks': len(documents),
                'timestamp': datetime.now().isoformat()
            }

            total_duration = result.get('total_duration', 0)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return {
                'success': True,
                'file_hash': file_hash,
                'chunks_uploaded': len(documents),
                'duration': total_duration
            }
        except Exception as e:
            logger.error(f"Failed to process {video_path}: {e}")
            self.failed_files[file_hash] = str(e)
            self.stats['failed_videos'] += 1
            self.processed_files[file_hash] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return {'success': False, 'file_hash': file_hash, 'error': str(e)}

    # giữ nguyên thuật toán từ file gốc
    def create_semantic_chunks(self, segments: List[Dict]) -> List[Dict]:
        if not segments:
            return []
        chunks = []
        current_chunk = {'segments': [], 'text': [], 'word_count': 0}
        for seg in segments:
            text = (seg.get('text', '') or '').strip()
            if not text:
                continue
            words = len(text.split())
            if current_chunk['word_count'] + words > self.config.target_words:
                if current_chunk['segments']:
                    chunk_text = ' '.join(current_chunk['text'])
                    chunks.append({
                        'text': chunk_text,
                        'start_time': current_chunk['segments'][0]['start'],
                        'end_time': current_chunk['segments'][-1]['end'],
                        'duration': current_chunk['segments'][-1]['end'] - current_chunk['segments'][0]['start'],
                        'word_count': current_chunk['word_count'],
                        'timestamp': f"[{current_chunk['segments'][0]['start']:.1f} - {current_chunk['segments'][-1]['end']:.1f}]"
                    })
                current_chunk = {'segments': [], 'text': [], 'word_count': 0}
            current_chunk['segments'].append(seg)
            current_chunk['text'].append(text)
            current_chunk['word_count'] += words
        if current_chunk['segments']:
            chunk_text = ' '.join(current_chunk['text'])
            chunks.append({
                'text': chunk_text,
                'start_time': current_chunk['segments'][0]['start'],
                'end_time': current_chunk['segments'][-1]['end'],
                'duration': current_chunk['segments'][-1]['end'] - current_chunk['segments'][0]['start'],
                'word_count': current_chunk['word_count'],
                'timestamp': f"[{current_chunk['segments'][0]['start']:.1f} - {current_chunk['segments'][-1]['end']:.1f}]"
            })
        logger.info(f"Created {len(chunks)} semantic chunks from {len(segments)} segments")
        return chunks

    def save_checkpoint(self):
        checkpoint = {
            'processed_files': self.processed_files,
            'failed_files': self.failed_files,
            'stats': self.stats,
            'config': asdict(self.config),
            'timestamp': datetime.now().isoformat(),
            'version': '2.1'
        }
        with open(self.config.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        success_count = sum(1 for f in self.processed_files.values() if isinstance(f, dict) and f.get('status') == 'success')
        logger.info(f"Checkpoint saved: {success_count} successfully processed")

    def load_checkpoint(self):
        if Path(self.config.checkpoint_file).exists():
            try:
                with open(self.config.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                processed = checkpoint.get('processed_files', {})
                if isinstance(processed, list):
                    self.processed_files = {f: {'status': 'unknown', 'chunks': 0} for f in processed}
                else:
                    self.processed_files = processed
                self.failed_files = checkpoint.get('failed_files', {})
                self.stats.update(checkpoint.get('stats', {}))
                success_count = sum(1 for f in self.processed_files.values() if isinstance(f, dict) and f.get('status') == 'success')
                logger.info(f"Checkpoint loaded: {success_count} successfully processed")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")

    def run(self):
        logger.info("="*60)
        logger.info("VIDEO PROCESSING PIPELINE TO ELASTICSEARCH")
        logger.info("="*60)
        self.stats['start_time'] = datetime.now().isoformat()
        self.initialize_models()
        logger.info("Scanning for video files...")
        video_files = self.find_video_files()
        if not video_files:
            logger.info("No new videos to process")
            if self.processed_files:
                success_count = sum(1 for f in self.processed_files.values() if isinstance(f, dict) and f.get('status') == 'success')
                logger.info(f"Already processed {success_count} files successfully")
            return
        self.stats['total_videos'] = len(video_files)
        total_size = sum(f.stat().st_size for f in video_files)
        self.stats['total_size_gb'] = total_size / (1024**3)
        logger.info(f"Found {len(video_files)} videos to process ({self.stats['total_size_gb']:.2f} GB)")
        with tqdm(total=len(video_files), desc="Processing videos") as pbar:
            for i, video_path in enumerate(video_files):
                result = self.process_single_video(video_path)
                pbar.update(1)
                if torch.cuda.is_available():
                    mem_used = torch.cuda.memory_allocated() / 1024**3
                    mem_reserved = torch.cuda.memory_reserved() / 1024**3
                    pbar.set_postfix({
                        'file': video_path.name[:25],
                        'success': result['success'],
                        'chunks': result.get('chunks_uploaded', 0),
                        'VRAM': f"{mem_used:.1f}/{mem_reserved:.1f}GB"
                    })
                if (i + 1) % self.config.save_checkpoint_every == 0:
                    self.save_checkpoint()
        self.save_checkpoint()
        self.print_summary()

    def print_summary(self):
        logger.info("="*60)
        logger.info("PROCESSING SUMMARY")
        logger.info("="*60)
        logger.info(f"Total videos found: {self.stats['total_videos']}")
        logger.info(f"Newly processed: {self.stats['processed_videos']}")
        logger.info(f"Reprocessed: {self.stats.get('reprocessed_videos', 0)}")
        logger.info(f"Failed: {self.stats['failed_videos']}")
        logger.info(f"Total chunks created: {self.stats['total_chunks']}")
        logger.info(f"Total duration: {self.stats['total_duration']/3600:.2f} hours")
        logger.info(f"Total size: {self.stats['total_size_gb']:.2f} GB")

        # Count successful processes
        if self.processed_files:
            success_count = sum(1 for f in self.processed_files.values() if isinstance(f, dict) and f.get('status') == 'success')
            total_chunks = sum(f.get('chunks', 0) for f in self.processed_files.values() if isinstance(f, dict))
            logger.info(f"Total successful files: {success_count}")
            logger.info(f"Total chunks in Elasticsearch: {total_chunks}")
        if self.stats['start_time']:
            start = datetime.fromisoformat(self.stats['start_time'])
            duration = datetime.now() - start
            logger.info(f"Processing time: {duration}")
        if self.es_service:
            try:
                stats = self.es_service.get_stats()
                logger.info(f"Elasticsearch index: {self.config.index_name} - {stats['document_count']} documents")
            except Exception as e:
                logger.warning(f"Could not get ES stats: {e}")


def main():
    parser = argparse.ArgumentParser(description="Process videos for Elasticsearch")
    parser.add_argument("--input-folder", "-i", required=True, help="Input folder containing videos")
    parser.add_argument("--output", default="./processing_output", help="Output folder")
    parser.add_argument("--index-name", default="video_transcripts", help="Elasticsearch index name")
    parser.add_argument("--es-host", default="localhost", help="Elasticsearch host")
    parser.add_argument("--es-port", type=int, default=9200, help="Elasticsearch port")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from checkpoint")
    parser.add_argument("--force-reprocess", action="store_true", help="Force reprocess all files")
    parser.add_argument("--check-status", action="store_true", help="Just check processing status")
    parser.add_argument("--batch-size", type=int, default=500, help="ES batch upload size")
    parser.add_argument("--gpu-batch-size", type=int, default=0, help="Force Whisper pipeline GPU batch size (0 = auto)")
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9)
        torch.backends.cudnn.benchmark = True
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

    config = ProcessingConfig(
        input_folder=args.input_folder,
        output_folder=args.output,
        index_name=args.index_name,
        es_host=args.es_host,
        es_port=args.es_port,
        resume=not args.no_resume,
        force_reprocess=args.force_reprocess,
        batch_upload_size=args.batch_size,
        speed_mode=True,
        whisper_batch_size=args.gpu_batch_size if args.gpu_batch_size > 0 else 1,
    )

    processor = VideoProcessor(config)

    if args.check_status:
        processor.load_checkpoint()
        processor.print_summary()
        return

    processor.run()


if __name__ == "__main__":
    main()


# # Example usage:
# # Basic usage
# python3 scripts/ingest_video_data.py --input-folder /path/to/videos

# # Advanced với custom ES settings
# python3 -m offline.ingest_ASR_data --input-folder /media/tan/F/AI_Challenge_2025/data/video --index-name my_video_transcripts --es-host localhost --es-port 9200 --batch-size 100 --gpu-batch-size 6
# python3 scripts/ingest_video_data.py \
#   --input-folder /path/to/videos \
#   --force-reprocess

# # Check processing status
# python3 scripts/ingest_video_data.py \
#   --input-folder /path/to/videos \
#   --check-status