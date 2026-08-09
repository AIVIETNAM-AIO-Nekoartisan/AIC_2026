# Enhanced_ASR_ingest.py
import os
import sys
import json
import argparse
import subprocess
import hashlib
import logging
import gc
import psutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict

import torch
from transformers import pipeline
from tqdm.auto import tqdm

# Try importing VAD and Whisper_VAD
try:
    sys.path.append(str(Path(__file__).parent.parent))
    from core.models.Whisper_VAD import WhisperTranscription
    HAS_WHISPER_VAD = True
except ImportError:
    HAS_WHISPER_VAD = False
    print("[WARNING] Whisper_VAD không khả dụng - sẽ dùng pipeline thường")

# utils.result chưa từng tồn tại trong repo -> luôn dùng đường fallback
USE_BUILD_RESULT = False

# Try importing Elasticsearch service
try:
    from core.services.elasticsearch_service import ElasticsearchService
    HAS_ELASTICSEARCH = True
except ImportError:
    HAS_ELASTICSEARCH = False
    print("[WARNING] ElasticsearchService không khả dụng - chỉ lưu JSON")

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_asr_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    input_dir: str
    output_dir: str
    wav_dir: Optional[str] = None
    recursive: bool = True

    # Model settings
    device_id: str = "0"
    model_name: str = "openai/whisper-large-v3"
    task: str = "transcribe"
    language: str = "None"
    batch_size: int = 24
    flash: bool = False
    timestamp: str = "chunk"

    # VAD settings
    use_vad: bool = True
    vad_min_silence_ms: int = 250
    vad_speech_pad_ms: int = 200
    vad_threshold: float = 0.5
    pad_silence_tail: float = 0.3

    # Chunking settings
    chunk_duration: float = 30.0
    overlap: float = 5.0
    target_words: int = 150
    max_chunk_duration: float = 90
    min_chunk_duration: float = 20

    # Elasticsearch settings
    index_name: str = "video_transcripts"
    es_host: str = "localhost"
    es_port: int = 9200
    batch_upload_size: int = 500
    enable_elasticsearch: bool = True

    # Processing settings
    checkpoint_file: str = "./enhanced_asr_checkpoint.json"
    resume: bool = True
    force_reprocess: bool = False
    save_checkpoint_every: int = 10
    memory_threshold: float = 0.85

    # File filters
    include_audio: bool = False
    include_video: bool = False


class EnhancedASRProcessor:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.processed_files = {}
        self.failed_files = {}
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'total_chunks': 0,
            'total_duration': 0,
            'start_time': None,
            'total_size_gb': 0
        }

        # Setup directories
        self.out_dir = Path(self.config.output_dir).expanduser().resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.wav_root = (Path(self.config.wav_dir).expanduser().resolve()
                        if self.config.wav_dir
                        else self.out_dir / ".wavs")
        self.wav_root.mkdir(parents=True, exist_ok=True)

        # Models
        self.whisper = None
        self.pipe = None
        self.es_service = None

        # Load checkpoint if resuming
        if self.config.resume and not self.config.force_reprocess:
            self.load_checkpoint()
        elif self.config.force_reprocess:
            logger.info("Force reprocess mode - ignoring checkpoint")
            self.clear_checkpoint()

    def clear_checkpoint(self):
        """Clear existing checkpoint"""
        self.processed_files = {}
        self.failed_files = {}
        if Path(self.config.checkpoint_file).exists():
            backup_file = f"{self.config.checkpoint_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            Path(self.config.checkpoint_file).rename(backup_file)
            logger.info(f"Backed up old checkpoint to {backup_file}")

    def get_file_hash(self, file_path: Path) -> str:
        """Generate file hash based on path, size and mtime"""
        stat = file_path.stat()
        hash_input = f"{file_path}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    def should_process_file(self, file_path: Path) -> bool:
        """Check if file should be processed"""
        file_hash = self.get_file_hash(file_path)
        if self.config.force_reprocess:
            return True

        if file_hash in self.processed_files:
            file_info = self.processed_files[file_hash]
            if isinstance(file_info, dict) and file_info.get('status') == 'success':
                logger.debug(f"Skipping {file_path.name} - already processed")
                return False

        return True

    def have_ffmpeg(self) -> bool:
        """Check if ffmpeg is available"""
        try:
            subprocess.run(["ffmpeg", "-version"],
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL,
                         check=True)
            return True
        except Exception:
            return False

    def ffmpeg_extract_wav(self, src_path: Path, dst_path: Path, sr: int = 16000, mono: bool = True) -> None:
        """Extract WAV using ffmpeg"""
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(src_path),
            "-vn", "-sn", "-dn",
            "-acodec", "pcm_s16le",
            "-ar", str(sr),
            "-ac", "1" if mono else "2",
            str(dst_path),
        ]
        subprocess.run(cmd, check=True)

    def find_media_files(self) -> List[Path]:
        """Find all media files in input directory"""
        in_dir = Path(self.config.input_dir).expanduser().resolve()
        files = []

        iterable = in_dir.rglob("*") if self.config.recursive else in_dir.glob("*")
        for p in iterable:
            if p.is_file() and p.suffix.lower() in (AUDIO_EXTS | VIDEO_EXTS):
                files.append(p)

        # Apply filters
        if self.config.include_audio and not self.config.include_video:
            files = [p for p in files if p.suffix.lower() in AUDIO_EXTS]
        elif self.config.include_video and not self.config.include_audio:
            files = [p for p in files if p.suffix.lower() in VIDEO_EXTS]

        # Filter already processed
        filtered_files = []
        skipped_count = 0
        for file in sorted(files):
            if self.should_process_file(file):
                filtered_files.append(file)
            else:
                skipped_count += 1

        if skipped_count > 0:
            logger.info(f"Skipping {skipped_count} already processed files")

        return filtered_files

    def check_memory(self) -> bool:
        """Check if memory usage is acceptable"""
        memory = psutil.virtual_memory()
        usage_ratio = memory.percent / 100
        if usage_ratio > self.config.memory_threshold:
            logger.warning(f"High memory usage: {usage_ratio:.1%}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False
        return True

    def initialize_models(self):
        """Initialize ASR models"""
        logger.info("Initializing ASR models...")

        # Setup device
        if self.config.device_id == "mps":
            device = "mps"
        else:
            device = f"cuda:{self.config.device_id}"

        # Memory optimization
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.9)
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True

        # Choose between Whisper_VAD or regular pipeline
        if HAS_WHISPER_VAD and self.config.use_vad:
            logger.info("Using Whisper_VAD with VAD capabilities")
            # Format device properly for Whisper_VAD
            whisper_device = device if self.config.device_id == "mps" else f"cuda:{self.config.device_id}"
            self.whisper = WhisperTranscription(
                model_name=self.config.model_name,
                device=whisper_device,
                batch_size=self.config.batch_size,
                speed_mode=True,
                embedding_model="sentence-transformers/distiluse-base-multilingual-cased-v2",
                offload_folder=".whisper_offload",
                force_pipe_bs=self.config.batch_size
            )
        else:
            logger.info("Using standard transformers pipeline")
            model_kwargs = {"attn_implementation": "flash_attention_2"} if self.config.flash else {"attn_implementation": "sdpa"}

            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.config.model_name,
                torch_dtype=torch.float16,
                device=device,
                model_kwargs=model_kwargs,
            )

        # Initialize Elasticsearch if enabled
        if self.config.enable_elasticsearch and HAS_ELASTICSEARCH:
            logger.info("Initializing Elasticsearch service...")
            self.es_service = ElasticsearchService(
                host=self.config.es_host,
                port=self.config.es_port,
                index_name=self.config.index_name
            )
            self.es_service.setup_index()
            logger.info("Elasticsearch initialized successfully")
        elif self.config.enable_elasticsearch and not HAS_ELASTICSEARCH:
            logger.warning("Elasticsearch requested but service not available - only saving JSON")

        logger.info("Models initialized successfully")

    def _upload_to_elasticsearch(self, documents: List[Dict]) -> bool:
        """Upload documents to Elasticsearch in batches"""
        if not self.es_service or not documents:
            return False

        success = True
        try:
            for i in range(0, len(documents), self.config.batch_upload_size):
                batch = documents[i:i + self.config.batch_upload_size]
                self.es_service.upsert_documents(batch)
                logger.debug(f"✓ Uploaded ES batch: {len(batch)} chunks")
        except Exception as e:
            logger.error(f"✗ Failed to upload to Elasticsearch: {e}")
            success = False

        return success

    def create_semantic_chunks(self, segments: List[Dict]) -> List[Dict]:
        """Create semantic chunks from segments - giống logic trong ingest_ASR_data"""
        if not segments:
            return []

        chunks = []
        current_chunk = {'segments': [], 'text': [], 'word_count': 0}

        for seg in segments:
            text = (seg.get('text', '') or '').strip()
            if not text:
                continue

            words = len(text.split())

            # If adding this segment would exceed target words, finish current chunk
            if current_chunk['word_count'] + words > self.config.target_words:
                if current_chunk['segments']:
                    chunk_text = ' '.join(current_chunk['text'])
                    start_time = current_chunk['segments'][0].get('start', 0.0)
                    end_time = current_chunk['segments'][-1].get('end', start_time + 1.0)

                    chunks.append({
                        'text': chunk_text,
                        'start_time': float(start_time),
                        'end_time': float(end_time),
                        'duration': float(end_time - start_time),
                        'word_count': current_chunk['word_count'],
                        'timestamp': f"[{start_time:.1f} - {end_time:.1f}]"
                    })
                current_chunk = {'segments': [], 'text': [], 'word_count': 0}

            current_chunk['segments'].append(seg)
            current_chunk['text'].append(text)
            current_chunk['word_count'] += words

        # Handle last chunk
        if current_chunk['segments']:
            chunk_text = ' '.join(current_chunk['text'])
            start_time = current_chunk['segments'][0].get('start', 0.0)
            end_time = current_chunk['segments'][-1].get('end', start_time + 1.0)

            chunks.append({
                'text': chunk_text,
                'start_time': float(start_time),
                'end_time': float(end_time),
                'duration': float(end_time - start_time),
                'word_count': current_chunk['word_count'],
                'timestamp': f"[{start_time:.1f} - {end_time:.1f}]"
            })

        logger.info(f"Created {len(chunks)} semantic chunks from {len(segments)} segments")
        return chunks

    def make_output_json_path(self, input_path: Path, in_root: Path) -> Path:
        """Generate output JSON path"""
        rel = input_path.relative_to(in_root)
        return (self.out_dir / rel).with_suffix(".json")

    def process_single_file(self, src_path: Path, wav_path: Path, in_root: Path) -> Dict[str, Any]:
        """Process a single audio/video file"""
        file_hash = self.get_file_hash(src_path)

        try:
            logger.info(f"Processing: {src_path.name}")

            # Generate output path
            out_json = self.make_output_json_path(src_path, in_root)
            out_json.parent.mkdir(parents=True, exist_ok=True)

            # Check memory before processing
            if not self.check_memory():
                logger.warning("High memory usage detected")

            semantic_chunks = []

            # Process with VAD-enabled Whisper if available
            if self.whisper:
                result = self.whisper.transcribe_file_with_sliding_window(
                    wav_path,
                    language=None if self.config.language == "None" else self.config.language,
                    chunk_duration=self.config.chunk_duration,
                    overlap=self.config.overlap,
                    use_vad=self.config.use_vad,
                    vad_min_silence_ms=self.config.vad_min_silence_ms,
                    vad_speech_pad_ms=self.config.vad_speech_pad_ms,
                    vad_threshold=self.config.vad_threshold,
                    pad_silence_tail=self.config.pad_silence_tail,
                    batch_infer=True
                )

                if not result['success']:
                    raise Exception(result.get('error', 'Unknown error'))

                # Get semantic chunks from result or create from segments
                semantic_chunks = result.get('semantic_chunks', [])
                if not semantic_chunks:
                    segments = result.get('segments', [])
                    if segments:
                        semantic_chunks = self.create_semantic_chunks(segments)

                # Build final result
                final_result = {
                    "source": str(src_path),
                    "wav": str(wav_path),
                    "file_hash": file_hash,
                    "semantic_chunks": semantic_chunks,
                    "segments": result.get('segments', []),
                    "full_text": result.get('full_text', ''),
                    "processing_time": result.get('processing_time', 0),
                    "total_duration": result.get('total_duration', 0),
                    "language": result.get('language', 'unknown'),
                    "processed_at": datetime.now().isoformat()
                }

            else:
                # Use regular pipeline
                language = None if self.config.language == "None" else self.config.language
                generate_kwargs = {"task": self.config.task, "language": language}

                if self.config.model_name.split(".")[-1] == "en":
                    generate_kwargs.pop("task", None)

                ts = "word" if self.config.timestamp == "word" else True

                outputs = self.pipe(
                    str(wav_path),
                    chunk_length_s=30,
                    batch_size=self.config.batch_size,
                    generate_kwargs=generate_kwargs,
                    return_timestamps=ts,
                )

                # Create segments from pipeline output
                segments = []
                if 'chunks' in outputs:
                    for chunk in outputs['chunks']:
                        # Handle missing timestamps
                        timestamp = chunk.get('timestamp', [None, None])
                        start_time = timestamp[0] if timestamp[0] is not None else 0.0
                        end_time = timestamp[1] if timestamp[1] is not None else start_time + 1.0

                        segments.append({
                            'start': float(start_time),
                            'end': float(end_time),
                            'text': chunk['text']
                        })

                # Create semantic chunks from segments
                semantic_chunks = self.create_semantic_chunks(segments) if segments else []

                final_result = {
                    "source": str(src_path),
                    "wav": str(wav_path),
                    "output": outputs,
                    "segments": segments,
                    "semantic_chunks": semantic_chunks,
                    "file_hash": file_hash,
                    "processed_at": datetime.now().isoformat()
                }

            # ====== BUILD ELASTICSEARCH DOCUMENTS ======
            es_documents = []
            if self.es_service and semantic_chunks:
                for i, chunk in enumerate(semantic_chunks):
                    text = chunk.get('text', '').strip()
                    if not text:
                        continue

                    doc = {
                        'id': f"{file_hash}_{i:04d}",
                        'file_path': str(src_path),
                        'file_name': src_path.name,
                        'file_hash': file_hash,
                        'chunk_index': i,
                        'content': text,
                        'content_type': 'audio',
                        'start_time': float(chunk.get('start_time', 0)),
                        'end_time': float(chunk.get('end_time', 0)),
                        'duration': float(chunk.get('duration', 0)),
                        'word_count': chunk.get('word_count', len(text.split())),
                        'timestamp': chunk.get('timestamp', ''),
                        'language': final_result.get('language', 'unknown'),
                        'total_duration': final_result.get('total_duration', 0),
                        'processed_at': datetime.now().isoformat(),
                    }
                    es_documents.append(doc)

            # ====== UPLOAD TO ELASTICSEARCH ======
            upload_success = True
            if self.config.enable_elasticsearch and es_documents:
                upload_success = self._upload_to_elasticsearch(es_documents)
                if upload_success:
                    logger.info(f"✓ Uploaded {len(es_documents)} chunks to Elasticsearch")
                else:
                    logger.warning(f"✗ Failed to upload {len(es_documents)} chunks to Elasticsearch")

            # Save result to JSON
            with open(out_json, "w", encoding="utf8") as fp:
                json.dump(final_result, fp, ensure_ascii=False, indent=2)

            # Update stats
            self.stats['processed_files'] += 1
            self.stats['total_chunks'] += len(semantic_chunks)
            if 'total_duration' in final_result:
                self.stats['total_duration'] += final_result['total_duration']

            # Mark as successfully processed
            self.processed_files[file_hash] = {
                'status': 'success',
                'chunks': len(semantic_chunks),
                'es_uploaded': upload_success if self.config.enable_elasticsearch else None,
                'timestamp': datetime.now().isoformat(),
                'output_file': str(out_json)
            }

            # Memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return {
                'success': True,
                'file_hash': file_hash,
                'chunks': len(semantic_chunks),
                'es_uploaded': upload_success if self.config.enable_elasticsearch else None,
                'output_file': str(out_json)
            }

        except Exception as e:
            logger.error(f"Failed to process {src_path}: {e}")
            self.failed_files[file_hash] = str(e)
            self.stats['failed_files'] += 1
            self.processed_files[file_hash] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return {'success': False, 'file_hash': file_hash, 'error': str(e)}

    def prepare_wav_files(self, files: List[Path], in_root: Path) -> List[Tuple[Path, Path]]:
        """Convert files to WAV format if needed"""
        wav_files = []

        logger.info("🎵 Step 1/2: Preparing WAV files")
        for src in tqdm(files, desc="Converting to WAV", unit="file"):
            try:
                rel = src.relative_to(in_root)
                target_wav = (self.wav_root / rel).with_suffix(".wav")
                target_wav.parent.mkdir(parents=True, exist_ok=True)

                if src.suffix.lower() in VIDEO_EXTS:
                    # Video → need WAV conversion
                    if not target_wav.exists():
                        self.ffmpeg_extract_wav(src, target_wav, sr=16000, mono=True)
                    wav_files.append((src, target_wav))

                elif src.suffix.lower() in AUDIO_EXTS:
                    if src.suffix.lower() == ".wav":
                        # Already WAV, use original
                        wav_files.append((src, src))
                    else:
                        # Other audio → convert to WAV if needed
                        if not target_wav.exists():
                            self.ffmpeg_extract_wav(src, target_wav, sr=16000, mono=True)
                        wav_files.append((src, target_wav))

            except subprocess.CalledProcessError as e:
                logger.error(f"FFMPEG error for {src}: {e}")
            except Exception as e:
                logger.error(f"Error processing {src}: {e}")

        return wav_files

    def save_checkpoint(self):
        """Save processing checkpoint"""
        checkpoint = {
            'processed_files': self.processed_files,
            'failed_files': self.failed_files,
            'stats': self.stats,
            'config': asdict(self.config),
            'timestamp': datetime.now().isoformat(),
            'version': '3.0'
        }

        with open(self.config.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)

        success_count = sum(1 for f in self.processed_files.values()
                           if isinstance(f, dict) and f.get('status') == 'success')
        logger.info(f"Checkpoint saved: {success_count} files processed successfully")

    def load_checkpoint(self):
        """Load processing checkpoint"""
        if Path(self.config.checkpoint_file).exists():
            try:
                with open(self.config.checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)

                self.processed_files = checkpoint.get('processed_files', {})
                self.failed_files = checkpoint.get('failed_files', {})
                self.stats.update(checkpoint.get('stats', {}))

                success_count = sum(1 for f in self.processed_files.values()
                                   if isinstance(f, dict) and f.get('status') == 'success')
                logger.info(f"Checkpoint loaded: {success_count} files already processed")

            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")

    def print_summary(self):
        """Print processing summary"""
        logger.info("="*60)
        logger.info("PROCESSING SUMMARY")
        logger.info("="*60)
        logger.info(f"Total files found: {self.stats['total_files']}")
        logger.info(f"Successfully processed: {self.stats['processed_files']}")
        logger.info(f"Failed: {self.stats['failed_files']}")
        logger.info(f"Total chunks created: {self.stats['total_chunks']}")
        logger.info(f"Total duration: {self.stats['total_duration']/3600:.2f} hours")
        logger.info(f"Total size: {self.stats['total_size_gb']:.2f} GB")

        if self.stats['start_time']:
            start = datetime.fromisoformat(self.stats['start_time'])
            duration = datetime.now() - start
            logger.info(f"Processing time: {duration}")

        # Elasticsearch stats
        if self.es_service:
            try:
                stats = self.es_service.get_stats()
                logger.info(f"Elasticsearch index '{self.config.index_name}': {stats['document_count']} documents")
            except Exception as e:
                logger.warning(f"Could not get Elasticsearch stats: {e}")

    def run(self):
        """Main processing pipeline"""
        logger.info("="*60)
        logger.info("ENHANCED ASR PROCESSING PIPELINE")
        logger.info("="*60)

        self.stats['start_time'] = datetime.now().isoformat()

        # Check ffmpeg
        if not self.have_ffmpeg():
            logger.error("ffmpeg not available in PATH")
            sys.exit(1)

        # Find files
        logger.info("Scanning for media files...")
        in_root = Path(self.config.input_dir).expanduser().resolve()
        files = self.find_media_files()

        if not files:
            logger.info("No new files to process")
            return

        self.stats['total_files'] = len(files)
        total_size = sum(f.stat().st_size for f in files)
        self.stats['total_size_gb'] = total_size / (1024**3)

        logger.info(f"Found {len(files)} files to process ({self.stats['total_size_gb']:.2f} GB)")

        # Initialize models
        self.initialize_models()

        # Prepare WAV files
        wav_files = self.prepare_wav_files(files, in_root)

        if not wav_files:
            logger.info("No WAV files to process after conversion")
            return

        # Process files
        logger.info("🧠 Step 2/2: ASR Processing")
        with tqdm(total=len(wav_files), desc="Transcribing", unit="file") as pbar:
            for i, (src, wav_path) in enumerate(wav_files):
                result = self.process_single_file(src, wav_path, in_root)

                pbar.update(1)
                pbar.set_postfix({
                    'file': src.name[:20],
                    'success': '✓' if result['success'] else '✗',
                    'chunks': result.get('chunks', 0),
                    'ES': '✓' if result.get('es_uploaded') else '✗' if self.config.enable_elasticsearch else 'N/A'
                })

                # Save checkpoint periodically
                if (i + 1) % self.config.save_checkpoint_every == 0:
                    self.save_checkpoint()

        # Final checkpoint and summary
        self.save_checkpoint()
        self.print_summary()
        logger.info("🎉 Processing completed!")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Enhanced Batch ASR with VAD and semantic chunking")

    # I/O
    parser.add_argument("--input-dir", required=True, type=str, help="Input directory containing audio/video files")
    parser.add_argument("--output-dir", required=True, type=str, help="Output directory for JSON results")
    parser.add_argument("--recursive", action="store_true", help="Process subdirectories recursively")
    parser.add_argument("--wav-dir", type=str, default=None, help="Directory for WAV files (default: output-dir/.wavs)")

    # Model/device
    parser.add_argument("--device-id", default="0", type=str, help='GPU ID ("0","1",...) or "mps" for Apple Silicon')
    parser.add_argument("--model-name", default="openai/whisper-large-v3", type=str, help="ASR model checkpoint")
    parser.add_argument("--task", default="transcribe", choices=["transcribe", "translate"], help="ASR task")
    parser.add_argument("--language", default="None", type=str, help='Language code (e.g., "vi"), "None" for auto-detect')
    parser.add_argument("--batch-size", default=24, type=int, help="Pipeline batch size (reduce if VRAM limited)")
    parser.add_argument("--flash", action="store_true", help="Use Flash Attention 2 if available")
    parser.add_argument("--timestamp", default="chunk", choices=["chunk", "word"], help="Timestamp granularity")

    # VAD settings
    parser.add_argument("--use-vad", action="store_true", help="Enable Voice Activity Detection")
    parser.add_argument("--vad-threshold", default=0.5, type=float, help="VAD threshold")
    parser.add_argument("--vad-min-silence-ms", default=250, type=int, help="Minimum silence duration (ms)")
    parser.add_argument("--vad-speech-pad-ms", default=200, type=int, help="Speech padding (ms)")

    # Chunking
    parser.add_argument("--chunk-duration", default=30.0, type=float, help="Chunk duration for processing")
    parser.add_argument("--overlap", default=5.0, type=float, help="Overlap between chunks")
    parser.add_argument("--target-words", default=150, type=int, help="Target words per semantic chunk")

    # Processing
    parser.add_argument("--checkpoint-file", default="./enhanced_asr_checkpoint.json", type=str, help="Checkpoint file path")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from checkpoint")
    parser.add_argument("--force-reprocess", action="store_true", help="Force reprocess all files")
    parser.add_argument("--save-checkpoint-every", default=10, type=int, help="Save checkpoint every N files")

    # File filters
    parser.add_argument("--include-audio", action="store_true", help="Process only audio files")
    parser.add_argument("--include-video", action="store_true", help="Process only video files")

    # Elasticsearch
    parser.add_argument("--index-name", default="video_transcripts", help="Elasticsearch index name")
    parser.add_argument("--es-host", default="localhost", help="Elasticsearch host")
    parser.add_argument("--es-port", type=int, default=9200, help="Elasticsearch port")
    parser.add_argument("--batch-upload-size", type=int, default=500, help="ES batch upload size")
    parser.add_argument("--disable-elasticsearch", action="store_true", help="Disable Elasticsearch, only save JSON")

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # Show GPU info if available
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

    # Build config
    config = ProcessingConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        wav_dir=args.wav_dir,
        recursive=args.recursive,
        device_id=args.device_id,
        model_name=args.model_name,
        task=args.task,
        language=args.language,
        batch_size=args.batch_size,
        flash=args.flash,
        timestamp=args.timestamp,
        use_vad=args.use_vad,
        vad_threshold=args.vad_threshold,
        vad_min_silence_ms=args.vad_min_silence_ms,
        vad_speech_pad_ms=args.vad_speech_pad_ms,
        chunk_duration=args.chunk_duration,
        overlap=args.overlap,
        target_words=args.target_words,
        index_name=args.index_name,
        es_host=args.es_host,
        es_port=args.es_port,
        batch_upload_size=args.batch_upload_size,
        enable_elasticsearch=not args.disable_elasticsearch,
        checkpoint_file=args.checkpoint_file,
        resume=not args.no_resume,
        force_reprocess=args.force_reprocess,
        save_checkpoint_every=args.save_checkpoint_every,
        include_audio=args.include_audio,
        include_video=args.include_video,
    )

    # Run processor
    processor = EnhancedASRProcessor(config)
    processor.run()


if __name__ == "__main__":
    main()

# Example usage:

# With Elasticsearch (default)
# python3 -m offline.Enhanced_ASR_ingest --input-dir ./data/video --output-dir ./output --recursive --use-vad --device-id 0 --model-name vinai/PhoWhisper-large --language vi --index-name my_transcripts --es-host localhost --es-port 9200

# JSON only (no Elasticsearch)
# python3 -m offline.Enhanced_ASR_ingest --input-dir ./data/video --output-dir ./output --recursive --use-vad --disable-elasticsearch --target-words 200

# High performance mode
# python3 -m offline.Enhanced_ASR_ingest --input-dir ./data/video --output-dir ./output --recursive --use-vad --device-id 0 --model-name vinai/PhoWhisper-large --language vi --batch-size 4 --flash --chunk-duration 45.0 --overlap 10.0 --target-words 200

# TÂN RECOMMEND A THÔNG :))))))
#  python3 -m offline.Enhanced_ASR_ingest --input-dir ./data/video/ --output-dir ./output --recursive --use-vad --model-name vinai/PhoWhisper-large --language vi --batch-size 3 --index-name my_asr_2 --flash