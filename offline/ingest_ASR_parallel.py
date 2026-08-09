# scripts/ingest_ASR_data.py
import os
os.environ["HF_HUB_DISABLE_SAFETENSORS_CONVERSION"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
import json
import time
import hashlib
import logging
import subprocess
import threading
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
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.models.Whisper_VAD import WhisperTranscription
from core.services.elasticsearch_service import ElasticsearchService

SAFE_EPS = 1e-2  # 10ms

def _to_float_safe(v, default=0.0):
    try:
        if v is None:
            return float(default)
        return float(v)
    except Exception:
        return float(default)

def sanitize_time_pair(start, end, total=None, eps: float = SAFE_EPS, swap_ok: bool = True):
    s = _to_float_safe(start, 0.0)
    e = _to_float_safe(end, s)
    if e < s:
        if swap_ok and (s - e) < 2.0:
            s, e = e, s
        else:
            e = s + eps
    if total is not None:
        total = max(0.0, float(total))
        s = max(0.0, min(s, total))
        e = max(0.0, min(e, total))
    if e - s < eps:
        e = s + eps
    return s, e, e - s

def sanitize_segments(segments, total=None, min_duration=0.3, max_wps=6.0):
    """
    - Lọc rỗng
    - Ép kiểu + sửa start/end
    - Sanity check ratio words/sec
    """
    out = []
    for seg in segments or []:
        s, e, d = sanitize_time_pair(seg.get("start"), seg.get("end"), total=total)
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        words = len(text.split())

        # Nếu duration quá nhỏ so với số từ -> kéo dãn
        if d < min_duration and words > 3:
            d = max(min_duration, words / max_wps)   # ước lượng lại theo tốc độ đọc
            e = s + d

        out.append({"start": s, "end": e, "duration": d, "text": text, "word_count": words})
    out.sort(key=lambda x: x["start"])
    return out


def sanitize_semantic_chunks(chunks, total=None, eps: float = SAFE_EPS):
    fixed = []
    for ch in chunks or []:
        s, e, d = sanitize_time_pair(ch.get("start_time"), ch.get("end_time"), total=total, eps=eps)
        text = (ch.get("text") or "").strip()
        if not text:
            continue
        ch_fixed = dict(ch)
        ch_fixed["start_time"] = s
        ch_fixed["end_time"] = e
        ch_fixed["duration"] = d
        ch_fixed["timestamp"] = f"[{s:.1f} - {e:.1f}]"
        fixed.append(ch_fixed)
    fixed.sort(key=lambda x: x["start_time"])
    return fixed

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('video_processing_es.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ============================
# CONFIG
# ============================
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

    batch_upload_size: int = 500
    memory_threshold_gb: float = 0.85
    resume: bool = True
    save_checkpoint_every: int = 3
    force_reprocess: bool = False

    # === Audio preprocessing (ffmpeg) → WAV PCM 16k mono ===
    audio_tmp_dir: str = "./_audio_cache"
    audio_rate: int = 16000
    audio_mono: int = 1
    ffmpeg_workers: int = 4
    keep_audio_cache: bool = True  # False = xóa wav sau khi xong

# ============================
# UTIL: FFmpeg audio extraction (robust → WAV PCM)
# ============================
def _safe_stem(path: Path) -> str:
    return Path(path).stem

def extract_audio_ffmpeg(in_path: Path, cfg: ProcessingConfig) -> Optional[Path]:
    """
    Tách audio sang WAV PCM 16k mono, bỏ qua frame hỏng.
    Trả về path WAV hoặc None nếu không có audio stream.
    """
    try:
        Path(cfg.audio_tmp_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(cfg.audio_tmp_dir) / f"{_safe_stem(in_path)}.wav"
        if out_path.exists() and out_path.stat().st_size > 10_000:
            return out_path

        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning", "-stats",
            "-fflags", "+discardcorrupt", "-err_detect", "ignore_err",
            "-analyzeduration", "200M", "-probesize", "200M",
            "-i", str(in_path),
            "-vn", "-sn", "-dn",
            "-map", "a:0?",
            "-ac", str(cfg.audio_mono),
            "-ar", str(cfg.audio_rate),
            "-c:a", "pcm_s16le",
            str(out_path)
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if proc.returncode != 0:
            err = proc.stderr.decode(errors="ignore")
            # Không có audio stream
            if "Stream specifier 'a:0' in filtergraph description matches no streams" in err or "Stream map 'a:0' matches no streams" in err:
                logger.warning(f"No audio stream in: {in_path.name}")
                return None
            logger.error(f"ffmpeg failed for {in_path.name}: {err[:400]}")
            if out_path.exists():
                out_path.unlink(missing_ok=True)
            return None

        if out_path.exists() and out_path.stat().st_size > 10_000:
            return out_path
        return None
    except Exception as e:
        logger.error(f"extract_audio_ffmpeg error: {e}")
        return None

# ============================
# PROCESSOR
# ============================
class VideoProcessor:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.processed_files: Dict[str, Dict] = {}
        self.failed_files: Dict[str, str] = {}
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
        self.whisper: Optional[WhisperTranscription] = None
        self.es_service: Optional[ElasticsearchService] = None
        self.executor = ThreadPoolExecutor(max_workers=2)  # ES upsert song song
        self.gpu_lock = threading.Lock()                   # khóa GPU khi suy luận

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
                force_pipe_bs=self.config.whisper_batch_size if self.config.whisper_batch_size > 0 else None
            )
        if self.es_service is None:
            self.es_service = ElasticsearchService(
                host=self.config.es_host,
                port=self.config.es_port,
                index_name=self.config.index_name
            )
            self.es_service.setup_index()

        try:
            self.es_service.client.cluster.health(
                index=self.config.index_name,
                wait_for_status="yellow",
                request_timeout=30,
            )
            logger.info("Elasticsearch index is ready (yellow+).")
        except Exception as e:
            logger.warning(f"ES index not ready yet (continue anyway): {e}")
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
        video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.mp3', '.wav', '.m4a']
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

    # Safe chunker
    def create_semantic_chunks(self, segments: List[Dict]) -> List[Dict]:
        if not segments:
            return []
        segs = sanitize_segments(segments, total=None)
        chunks = []
        current = {'segments': [], 'text': [], 'word_count': 0}
        for seg in segs:
            text = seg["text"]
            if not text:
                continue
            words = len(text.split())
            if current['word_count'] + words > self.config.target_words and current['segments']:
                s0 = current['segments'][0]['start']
                e1 = current['segments'][-1]['end']
                s, e, d = sanitize_time_pair(s0, e1)
                chunks.append({
                    'text': ' '.join(current['text']),
                    'start_time': s,
                    'end_time': e,
                    'duration': d,
                    'word_count': current['word_count'],
                    'timestamp': f"[{s:.1f} - {e:.1f}]"
                })
                current = {'segments': [], 'text': [], 'word_count': 0}
            current['segments'].append(seg)
            current['text'].append(text)
            current['word_count'] += words
        if current['segments']:
            s0 = current['segments'][0]['start']
            e1 = current['segments'][-1]['end']
            s, e, d = sanitize_time_pair(s0, e1)
            chunks.append({
                'text': ' '.join(current['text']),
                'start_time': s,
                'end_time': e,
                'duration': d,
                'word_count': current['word_count'],
                'timestamp': f"[{s:.1f} - {e:.1f}]"
            })
        chunks = sanitize_semantic_chunks(chunks, total=None)
        logger.info(f"Created {len(chunks)} semantic chunks from {len(segs)} segments (sanitized)")
        return chunks

    def process_single_video(self, video_path: Path, audio_path: Optional[Path] = None) -> Dict:
        file_hash = self.get_file_hash(video_path)
        try:
            logger.info(f"Processing: {video_path.name}")
            is_reprocess = file_hash in self.processed_files
            if is_reprocess:
                logger.info(f"Reprocessing file: {video_path.name}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Ưu tiên WAV đã tách; nếu video không có audio → transcribe video_path (Whisper sẽ xử lý nếu có audio)
            media_input = audio_path if (audio_path and audio_path.exists()) else video_path

            with self.gpu_lock:
                result = self.whisper.transcribe_file_with_sliding_window(
                    media_input,
                    language=None,
                    chunk_duration=self.config.chunk_duration,
                    overlap=self.config.overlap,
                    use_vad=self.config.use_vad,
                    vad_min_silence_ms=250,
                    vad_speech_pad_ms=200,
                    vad_threshold=0.5,
                    pad_silence_tail=0.3,
                    batch_infer=True
                )

            if not result.get('success'):
                raise Exception(result.get('error', 'Unknown error'))

            total_dur = _to_float_safe(result.get('total_duration'), 0.0)

            semantic_chunks = result.get('semantic_chunks', [])
            if not semantic_chunks and result.get('segments'):
                logger.info("Creating semantic chunks from segments...")
                semantic_chunks = self.create_semantic_chunks(result['segments'])

            semantic_chunks = sanitize_semantic_chunks(semantic_chunks, total=total_dur)
            logger.info(f"Got {len(semantic_chunks)} semantic chunks (sanitized)")

            documents = []
            for i, chunk in enumerate(semantic_chunks):
                text = (chunk.get('text', '') or '').strip()
                if not text:
                    continue
                s, e, d = sanitize_time_pair(chunk.get('start_time', 0.0), chunk.get('end_time', 0.0), total=total_dur)
                doc = {
                    'id': f"{file_hash}_{i:04d}",
                    'file_path': str(video_path),
                    'file_name': video_path.name,
                    'file_hash': file_hash,
                    'chunk_index': i,
                    'content': text,
                    'content_type': 'audio',
                    'start_time': s,
                    'end_time': e,
                    'duration': d,
                    'word_count': chunk.get('word_count', len(text.split())),
                    'timestamp': f"[{s:.1f} - {e:.1f}]",
                    'language': result.get('language', 'unknown'),
                    'total_duration': total_dur,
                    'processed_at': datetime.now().isoformat(),
                }
                documents.append(doc)

            upload_success = True
            if documents:
                fut = self.executor.submit(self._upload_batches, documents)
                upload_success = fut.result()
            else:
                logger.warning("No chunks to upload!")
                upload_success = False

            output_file = Path(self.config.output_folder) / f"{video_path.stem}_result.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'file_path': str(video_path),
                    'file_hash': file_hash,
                    'chunks_count': len(documents),
                    'upload_success': upload_success,
                    'processing_time': result.get('processing_time', 0),
                    'total_duration': total_dur,
                    'language': result.get('language', 'unknown'),
                    'full_text': result.get('full_text', ''),
                    'semantic_chunks': semantic_chunks
                }, f, indent=2, ensure_ascii=False)

            if not is_reprocess:
                self.stats['processed_videos'] += 1
            self.stats['total_chunks'] += len(documents)
            self.stats['total_duration'] += total_dur

            self.processed_files[file_hash] = {
                'status': 'success' if upload_success else 'partial',
                'chunks': len(documents),
                'timestamp': datetime.now().isoformat()
            }

            if audio_path and audio_path.exists() and not self.config.keep_audio_cache:
                try:
                    audio_path.unlink(missing_ok=True)
                except Exception:
                    pass

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return {'success': True, 'file_hash': file_hash, 'chunks_uploaded': len(documents), 'duration': total_dur}
        except Exception as e:
            logger.error(f"Failed to process {video_path}: {e}")
            self.failed_files[file_hash] = str(e)
            self.stats['failed_videos'] += 1
            self.processed_files[file_hash] = {'status': 'failed', 'error': str(e), 'timestamp': datetime.now().isoformat()}
            if audio_path and audio_path.exists() and not self.config.keep_audio_cache:
                try:
                    audio_path.unlink(missing_ok=True)
                except Exception:
                    pass
            return {'success': False, 'file_hash': file_hash, 'error': str(e)}

    def save_checkpoint(self):
        checkpoint = {
            'processed_files': self.processed_files,
            'failed_files': self.failed_files,
            'stats': self.stats,
            'config': asdict(self.config),
            'timestamp': datetime.now().isoformat(),
            'version': '2.3-wav'
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

        # Pre-extract audio in parallel
        logger.info(f"Pre-extracting audio for {len(video_files)} files with {self.config.ffmpeg_workers} workers...")
        video_to_audio: Dict[Path, Optional[Path]] = {}
        try:
            self.es_service.upsert_documents([{
                "id": "__warmup__",
                "file_path": "__warmup__",
                "content": "warmup",
                "content_type": "meta",
            }])
            # xóa warmup cho sạch, bỏ qua nếu không tồn tại
            self.es_service.client.delete(index=self.config.index_name, id="__warmup__", ignore=[404])
        except Exception as e:
            logger.warning(f"ES warmup failed (safe to ignore): {e}")

        with ThreadPoolExecutor(max_workers=max(1, self.config.ffmpeg_workers)) as pool:
            futures = {pool.submit(extract_audio_ffmpeg, vp, self.config): vp for vp in video_files}
            for fut in as_completed(futures):
                vp = futures[fut]
                try:
                    ap = fut.result()
                except Exception as e:
                    logger.error(f"Audio extract failed on {vp.name}: {e}")
                    ap = None
                video_to_audio[vp] = ap

        # Process videos one by one (GPU-locked inference)
        with tqdm(total=len(video_files), desc="Processing videos") as pbar:
            for i, video_path in enumerate(video_files):
                result = self.process_single_video(video_path, audio_path=video_to_audio.get(video_path))
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

# ============================
# CLI
# ============================
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

    # ffmpeg / audio cache
    parser.add_argument("--ffmpeg-workers", type=int, default=4, help="Number of parallel ffmpeg processes")
    parser.add_argument("--audio-cache", default="./_audio_cache", help="Folder to store extracted audios")
    parser.add_argument("--purge-audio", action="store_true", help="Delete extracted audio after ingest")

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
        audio_tmp_dir=args.audio_cache,
        ffmpeg_workers=args.ffmpeg_workers,
        keep_audio_cache=not args.purge_audio,
    )

    processor = VideoProcessor(config)

    if args.check_status:
        processor.load_checkpoint()
        processor.print_summary()
        return

    processor.run()

if __name__ == "__main__":
    main()

# --- Example usage ---
# Basic:
# python3 scripts/ingest_ASR_data.py --input-folder /path/to/videos
#
# With parallel ffmpeg & purge audio cache after:
# python3 scripts/ingest_ASR_data.py -i /path/to/videos --ffmpeg-workers 6 --purge-audio
#
# Custom ES / bigger bulk / GPU batch for Whisper:
# python3 -m offline.ingest_ASR_data \
#   -i /media/tan/F/AI_Challenge_2025/data/video \
#   --index-name my_video_transcripts \
#   --es-host localhost --es-port 9200 \
#   --batch-size 1000 --gpu-batch-size 6
