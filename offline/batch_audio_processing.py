#!/usr/bin/env python3
"""
Fixed Video Processing Pipeline
- Fixed reprocessing issue
- Better checkpoint management
- Force reprocess option
"""

import os
import sys
import json
import time
import uuid
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, asdict
import gc
import psutil
import numpy as np
from tqdm import tqdm
import torch

# Import modules
from core.models.Whisper_VAD import WhisperTranscription
from core.services.vector_store_optimized import get_qdrant_service
from qdrant_client.http.models import PointStruct

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuration for processing pipeline"""
    input_folder: str
    output_folder: str = "./processing_output"
    checkpoint_file: str = "./processing_checkpoint.json"
    collection_name: str = "video_semantic_chunks"

    # Whisper settings
    whisper_model: str = "openai/whisper-medium"  # For 8GB GPU
    whisper_device: str = "cuda:0"
    whisper_batch_size: int = 1
    speed_mode: bool = True

    # Processing settings
    chunk_duration: float = 20.0
    overlap: float = 5.0
    use_vad: bool = True

    # Semantic chunking
    target_words: int = 150
    similarity_threshold: float = 0.65
    max_chunk_duration: float = 90
    min_chunk_duration: float = 20

    # Embedding settings
    embedding_model: str = "sentence-transformers/distiluse-base-multilingual-cased-v2"
    embedding_dim: int = 512

    # Performance settings
    max_workers: int = 1
    batch_upload_size: int = 30
    memory_threshold_gb: float = 0.75

    # Resume settings
    resume: bool = True
    save_checkpoint_every: int = 3
    force_reprocess: bool = False  # New option to force reprocess

class VideoProcessor:
    """Main video processing pipeline with fixes"""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.processed_files = {}  # Changed to dict: hash -> {'status': 'success/failed', 'chunks': N}
        self.failed_files = {}
        self.id_mapping = {}
        self.stats = {
            'total_videos': 0,
            'processed_videos': 0,
            'failed_videos': 0,
            'total_chunks': 0,
            'total_duration': 0,
            'start_time': None,
            'total_size_gb': 0,
            'reprocessed_videos': 0  # Track reprocessed
        }

        # Create output folder
        Path(self.config.output_folder).mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.whisper = None
        self.embedding_model = None
        self.qdrant_service = None

        # Load checkpoint if exists
        if self.config.resume and not self.config.force_reprocess:
            self.load_checkpoint()
        elif self.config.force_reprocess:
            logger.info("Force reprocess mode - ignoring checkpoint")
            self.clear_checkpoint()

    def clear_checkpoint(self):
        """Clear checkpoint to force reprocess"""
        self.processed_files = {}
        self.failed_files = {}
        self.id_mapping = {}
        if Path(self.config.checkpoint_file).exists():
            # Backup old checkpoint
            backup_file = f"{self.config.checkpoint_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            Path(self.config.checkpoint_file).rename(backup_file)
            logger.info(f"Backed up old checkpoint to {backup_file}")

    def initialize_models(self):
        """Initialize models with memory management"""
        logger.info("Initializing models...")

        # Set memory fraction for 8GB GPU
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.8)
            torch.cuda.empty_cache()

        # Initialize Whisper
        if self.whisper is None:
            self.whisper = WhisperTranscription(
                model_name=self.config.whisper_model,
                device=self.config.whisper_device,
                batch_size=1,  # Force batch_size=1 for 8GB
                embedding_model=self.config.embedding_model,
                speed_mode=True,  # Force speed mode
                offload_folder=".whisper_offload"
            )

        # Initialize embedding model
        if self.embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(self.config.embedding_model)
            self.embedding_model.max_seq_length = 256

        # Initialize Qdrant
        if self.qdrant_service is None:
            self.qdrant_service = get_qdrant_service(
                collection_name=self.config.collection_name,
                vector_dim=self.config.embedding_dim
            )
            self.qdrant_service.setup_collection()

        logger.info("Models initialized successfully")

    def get_file_hash(self, file_path: Path) -> str:
        """Get unique hash for file"""
        stat = file_path.stat()
        hash_input = f"{file_path}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    def generate_point_id(self, file_hash: str, chunk_index: int) -> str:
        """Generate valid UUID for Qdrant point"""
        namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')
        name = f"{file_hash}_{chunk_index:04d}"
        return str(uuid.uuid5(namespace, name))

    def should_process_file(self, file_path: Path) -> bool:
        """Check if file should be processed"""
        file_hash = self.get_file_hash(file_path)

        # Force reprocess mode
        if self.config.force_reprocess:
            return True

        # Check if already processed successfully
        if file_hash in self.processed_files:
            file_info = self.processed_files[file_hash]
            if isinstance(file_info, dict):
                # New format with status
                if file_info.get('status') == 'success' and file_info.get('chunks', 0) > 0:
                    logger.info(f"Skipping {file_path.name} - already processed with {file_info['chunks']} chunks")
                    return False
                elif file_info.get('status') == 'failed':
                    logger.info(f"Retrying previously failed file: {file_path.name}")
                    return True
            else:
                # Old format - check if has chunks
                logger.info(f"File in old checkpoint format, will reprocess: {file_path.name}")
                return True

        return True

    def find_video_files(self) -> List[Path]:
        """Find all video files in input folder"""
        input_path = Path(self.config.input_folder)
        video_extensions = [
            '.mp4', '.mkv', '.avi', '.mov', '.wmv',
            '.flv', '.webm', '.m4v', '.mp3', '.wav', '.m4a'
        ]

        all_video_files = []
        for ext in video_extensions:
            all_video_files.extend(input_path.glob(f"**/*{ext}"))
            all_video_files.extend(input_path.glob(f"**/*{ext.upper()}"))

        # Filter based on processing status
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
        """Check if memory usage is within threshold"""
        memory = psutil.virtual_memory()
        usage_ratio = memory.percent / 100

        if usage_ratio > self.config.memory_threshold_gb:
            logger.warning(f"Memory usage high: {usage_ratio:.1%}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False
        return True

    def delete_old_chunks(self, file_hash: str):
        """Delete old chunks from Qdrant before reprocessing"""
        if file_hash in self.id_mapping:
            old_ids = self.id_mapping[file_hash]
            if old_ids:
                try:
                    logger.info(f"Deleting {len(old_ids)} old chunks from Qdrant")
                    self.qdrant_service.delete_points(old_ids)
                except Exception as e:
                    logger.warning(f"Failed to delete old chunks: {e}")

    def process_single_video(self, video_path: Path) -> Dict:
        """Process single video file with fixes"""
        file_hash = self.get_file_hash(video_path)

        try:
            logger.info(f"Processing: {video_path.name}")

            # Check if reprocessing
            is_reprocess = file_hash in self.processed_files
            if is_reprocess:
                logger.info(f"Reprocessing file: {video_path.name}")
                self.stats['reprocessed_videos'] += 1
                # Delete old chunks from Qdrant
                self.delete_old_chunks(file_hash)

            # Check memory before processing
            if not self.check_memory():
                time.sleep(5)

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Transcribe with Whisper
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
                batch_infer=False  # Disable batch for 8GB GPU
            )

            if not result['success']:
                raise Exception(result.get('error', 'Unknown error'))

            # Process semantic chunks
            semantic_chunks = result.get('semantic_chunks', [])
            if not semantic_chunks and result.get('segments'):
                logger.info("Creating semantic chunks from segments...")
                semantic_chunks = self.create_semantic_chunks(result['segments'])

            logger.info(f"Got {len(semantic_chunks)} semantic chunks")

            # Prepare points for Qdrant
            points = []
            chunk_ids = []

            for i, chunk in enumerate(semantic_chunks):
                # Skip empty chunks
                text = chunk.get('text', '').strip()
                if not text:
                    continue

                # Get or compute embedding
                if 'embedding' in chunk and chunk['embedding']:
                    embedding = chunk['embedding']
                else:
                    # Compute embedding
                    text_truncated = text[:500]
                    embedding = self.embedding_model.encode(text_truncated).tolist()

                # Generate UUID for this point
                point_id = self.generate_point_id(file_hash, i)
                chunk_ids.append(point_id)

                # Prepare metadata
                metadata = {
                    'file_path': str(video_path),
                    'file_name': video_path.name,
                    'file_hash': file_hash,
                    'chunk_index': i,
                    'text': text[:1000],
                    'start_time': float(chunk.get('start_time', 0)),
                    'end_time': float(chunk.get('end_time', 0)),
                    'duration': float(chunk.get('duration', 0)),
                    'word_count': chunk.get('word_count', len(text.split())),
                    'timestamp': chunk.get('timestamp', ''),
                    'processed_at': datetime.now().isoformat(),
                }

                # Create point
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=metadata
                )
                points.append(point)

            # Upload to Qdrant
            upload_success = True
            if points:
                logger.info(f"Uploading {len(points)} chunks to Qdrant...")
                for i in range(0, len(points), self.config.batch_upload_size):
                    batch = points[i:i + self.config.batch_upload_size]
                    try:
                        self.qdrant_service.upsert_points(batch)
                        logger.info(f"✓ Uploaded batch {i//self.config.batch_upload_size + 1}: {len(batch)} chunks")
                    except Exception as e:
                        logger.error(f"✗ Failed to upload batch: {e}")
                        upload_success = False
            else:
                logger.warning("No chunks to upload!")
                upload_success = False

            # Store ID mapping
            self.id_mapping[file_hash] = chunk_ids

            # Save transcription locally
            output_file = Path(self.config.output_folder) / f"{video_path.stem}_result.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'file_path': str(video_path),
                    'file_hash': file_hash,
                    'chunk_ids': chunk_ids,
                    'chunks_count': len(points),
                    'upload_success': upload_success,
                    'processing_time': result.get('processing_time', 0),
                    'total_duration': result.get('total_duration', 0),
                }, f, indent=2, ensure_ascii=False)

            # Update stats
            if not is_reprocess:
                self.stats['processed_videos'] += 1
            self.stats['total_chunks'] += len(points)
            self.stats['total_duration'] += result.get('total_duration', 0)

            # Mark as processed with chunk count
            self.processed_files[file_hash] = {
                'status': 'success' if upload_success else 'partial',
                'chunks': len(points),
                'timestamp': datetime.now().isoformat()
            }

            # Clear memory
            del result
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return {
                'success': True,
                'file_hash': file_hash,
                'chunks_uploaded': len(points),
                'duration': semantic_chunks[0].get('duration', 0) if semantic_chunks else 0
            }

        except Exception as e:
            logger.error(f"Failed to process {video_path}: {e}")
            self.failed_files[file_hash] = str(e)
            self.stats['failed_videos'] += 1

            # Mark as failed in processed_files
            self.processed_files[file_hash] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

            return {
                'success': False,
                'file_hash': file_hash,
                'error': str(e)
            }

    def create_semantic_chunks(self, segments: List[Dict]) -> List[Dict]:
        """Create semantic chunks from segments"""
        if not segments:
            return []

        chunks = []
        current_chunk = {
            'segments': [],
            'text': [],
            'start_time': None,
            'word_count': 0
        }

        for seg in segments:
            text = (seg.get('text', '') or '').strip()
            if not text:
                continue

            words = len(text.split())

            # Check if we should start new chunk
            if current_chunk['word_count'] + words > self.config.target_words:
                if current_chunk['segments']:
                    # Finalize current chunk
                    chunk_text = ' '.join(current_chunk['text'])
                    chunks.append({
                        'text': chunk_text,
                        'start_time': current_chunk['segments'][0]['start'],
                        'end_time': current_chunk['segments'][-1]['end'],
                        'duration': current_chunk['segments'][-1]['end'] - current_chunk['segments'][0]['start'],
                        'word_count': current_chunk['word_count'],
                        'timestamp': f"[{current_chunk['segments'][0]['start']:.1f} - {current_chunk['segments'][-1]['end']:.1f}]"
                    })

                # Start new chunk
                current_chunk = {
                    'segments': [],
                    'text': [],
                    'start_time': None,
                    'word_count': 0
                }

            current_chunk['segments'].append(seg)
            current_chunk['text'].append(text)
            current_chunk['word_count'] += words

        # Add final chunk
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
        """Save processing checkpoint"""
        checkpoint = {
            'processed_files': self.processed_files,  # Now saves as dict with status
            'failed_files': self.failed_files,
            'id_mapping': self.id_mapping,
            'stats': self.stats,
            'config': asdict(self.config),
            'timestamp': datetime.now().isoformat(),
            'version': '2.0'  # Version for checkpoint format
        }

        with open(self.config.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        success_count = sum(1 for f in self.processed_files.values()
                          if isinstance(f, dict) and f.get('status') == 'success')
        logger.info(f"Checkpoint saved: {success_count} successfully processed")

    def load_checkpoint(self):
        """Load processing checkpoint"""
        if Path(self.config.checkpoint_file).exists():
            try:
                with open(self.config.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)

                # Handle both old and new checkpoint formats
                processed = checkpoint.get('processed_files', {})
                if isinstance(processed, list):
                    # Old format - convert to new
                    self.processed_files = {f: {'status': 'unknown', 'chunks': 0} for f in processed}
                else:
                    self.processed_files = processed

                self.failed_files = checkpoint.get('failed_files', {})
                self.id_mapping = checkpoint.get('id_mapping', {})
                self.stats.update(checkpoint.get('stats', {}))

                success_count = sum(1 for f in self.processed_files.values()
                                  if isinstance(f, dict) and f.get('status') == 'success')
                logger.info(f"Checkpoint loaded: {success_count} successfully processed")

            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")

    def run(self):
        """Main processing pipeline"""
        logger.info("="*60)
        logger.info("VIDEO PROCESSING PIPELINE (8GB GPU Optimized)")
        logger.info("="*60)

        self.stats['start_time'] = datetime.now().isoformat()

        # Initialize models
        self.initialize_models()

        # Find video files
        logger.info("Scanning for video files...")
        video_files = self.find_video_files()

        if not video_files:
            logger.info("No new videos to process")
            # Check if we have processed files
            if self.processed_files:
                success_count = sum(1 for f in self.processed_files.values()
                                  if isinstance(f, dict) and f.get('status') == 'success')
                logger.info(f"Already processed {success_count} files successfully")
            return

        self.stats['total_videos'] = len(video_files)

        # Calculate total size
        total_size = sum(f.stat().st_size for f in video_files)
        self.stats['total_size_gb'] = total_size / (1024**3)

        logger.info(f"Found {len(video_files)} videos to process ({self.stats['total_size_gb']:.2f} GB)")

        # Process videos sequentially
        with tqdm(total=len(video_files), desc="Processing videos") as pbar:
            for i, video_path in enumerate(video_files):
                # Memory check
                if torch.cuda.is_available():
                    mem_reserved = torch.cuda.memory_reserved() / 1024**3
                    if mem_reserved > 6.0:
                        logger.info("High VRAM usage, cleaning up...")
                        torch.cuda.empty_cache()
                        gc.collect()
                        time.sleep(3)

                # Process video
                result = self.process_single_video(video_path)

                # Update progress
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

                # Save checkpoint periodically
                if (i + 1) % self.config.save_checkpoint_every == 0:
                    self.save_checkpoint()

        # Final checkpoint
        self.save_checkpoint()

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print processing summary"""
        logger.info("="*60)
        logger.info("PROCESSING SUMMARY")
        logger.info("="*60)
        logger.info(f"Total videos found: {self.stats['total_videos']}")
        logger.info(f"Newly processed: {self.stats['processed_videos']}")
        logger.info(f"Reprocessed: {self.stats['reprocessed_videos']}")
        logger.info(f"Failed: {self.stats['failed_videos']}")
        logger.info(f"Total chunks created: {self.stats['total_chunks']}")
        logger.info(f"Total duration: {self.stats['total_duration']/3600:.2f} hours")
        logger.info(f"Total size: {self.stats['total_size_gb']:.2f} GB")

        # Count successful processes
        if self.processed_files:
            success_count = sum(1 for f in self.processed_files.values()
                              if isinstance(f, dict) and f.get('status') == 'success')
            total_chunks = sum(f.get('chunks', 0) for f in self.processed_files.values()
                             if isinstance(f, dict))
            logger.info(f"Total successful files: {success_count}")
            logger.info(f"Total chunks in database: {total_chunks}")

        if self.stats['start_time']:
            start = datetime.fromisoformat(self.stats['start_time'])
            duration = datetime.now() - start
            logger.info(f"Processing time: {duration}")

        # Check Qdrant
        if self.qdrant_service:
            try:
                info = self.qdrant_service.get_collection_info()
                logger.info(f"Qdrant collection: {info.get('name')} - {info.get('points_count', 0)} points")
            except:
                pass


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Process videos for Qdrant (8GB GPU)")
    parser.add_argument("input_folder", help="Input folder containing videos")
    parser.add_argument("--output", default="./processing_output", help="Output folder")
    parser.add_argument("--collection", default="video_semantic_chunks", help="Qdrant collection")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from checkpoint")
    parser.add_argument("--force-reprocess", action="store_true", help="Force reprocess all files")
    parser.add_argument("--check-status", action="store_true", help="Just check processing status")

    args = parser.parse_args()

    # Setup for 8GB GPU
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.8)
        torch.backends.cudnn.benchmark = True
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

    # Create config
    config = ProcessingConfig(
        input_folder=args.input_folder,
        output_folder=args.output,
        collection_name=args.collection,
        resume=not args.no_resume,
        force_reprocess=args.force_reprocess
    )

    # Create processor
    processor = VideoProcessor(config)

    # Check status only
    if args.check_status:
        processor.load_checkpoint()
        processor.print_summary()
        return

    # Run processing
    processor.run()


if __name__ == "__main__":
    main()