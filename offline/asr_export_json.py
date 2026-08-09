#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["HF_HUB_DISABLE_SAFETENSORS_CONVERSION"] = "1"
os.environ["HF_HUB_OFFLINE"] = "0"

import sys
import json
import hashlib
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


import torch

SAFE_EPS = 1e-2  # 10ms

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
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
    out = []
    for seg in segments or []:
        s, e, d = sanitize_time_pair(seg.get("start"), seg.get("end"), total=total)
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        words = len(text.split())
        if d < min_duration and words > 3:
            d = max(min_duration, words / max_wps)
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

def human_bytes(n):
    for unit in ['B','KB','MB','GB','TB']:
        if abs(n) < 1024.0:
            return f"{n:3.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"

# -----------------------------------------------------------------------------
# Config / Logging
# -----------------------------------------------------------------------------
@dataclass
class ProcessingConfig:
    input_folder: str
    output_json_dir: str
    checkpoint_file: str
    audio_tmp_dir: str
    ffmpeg_workers: int
    whisper_device: str
    whisper_batch_size: int
    chunk_duration: float
    overlap: float
    use_vad: bool
    target_words: int
    resume: bool
    keep_audio_cache: bool
    pattern: str
    max_files: int

def setup_logging(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "asr_export_json.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger("ASR")
    logger.info(f"LOG FILE: {log_file}")
    return logger

# -----------------------------------------------------------------------------
# Audio extract
# -----------------------------------------------------------------------------
def extract_audio_ffmpeg(in_path: Path, out_wav: Path, rate=16000, mono=1) -> Optional[Path]:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    if out_wav.exists() and out_wav.stat().st_size > 10_000:
        return out_wav
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning", "-stats",
        "-fflags", "+discardcorrupt", "-err_detect", "ignore_err",
        "-analyzeduration", "200M", "-probesize", "200M",
        "-i", str(in_path),
        "-vn", "-sn", "-dn",
        "-map", "a:0?",
        "-ac", str(mono),
        "-ar", str(rate),
        "-c:a", "pcm_s16le",
        str(out_wav)
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        return None
    if out_wav.exists() and out_wav.stat().st_size > 10_000:
        return out_wav
    return None

# -----------------------------------------------------------------------------
# Whisper wrapper
# -----------------------------------------------------------------------------
def load_whisper(device: str, batch_size: int, logger: logging.Logger):
    try:
        sys.path.append(str(Path(__file__).parent.parent))  # project root
        from core.models.Whisper_VAD import WhisperTranscription
    except Exception as e:
        logger.error("Không import được WhisperTranscription (model/Whisper_VAD). Kiểm tra path & module.")
        raise

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9)
        torch.cuda.empty_cache()

    logger.info(f"Loading Whisper on device={device}, batch_size={batch_size}")
    wt = WhisperTranscription(
        model_name="vinai/PhoWhisper-large",
        device=device,
        batch_size=batch_size,
        embedding_model="sentence-transformers/distiluse-base-multilingual-cased-v2",
        speed_mode=False,
        offload_folder=".whisper_offload",
        force_pipe_bs=batch_size if batch_size > 0 else None
    )
    return wt

# -----------------------------------------------------------------------------
# Exporter
# -----------------------------------------------------------------------------
class ASRExporter:
    def __init__(self, cfg: ProcessingConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        out_dir = Path(cfg.output_json_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_path = Path(cfg.checkpoint_file)
        self.processed = {}
        if cfg.resume and self.ckpt_path.exists():
            try:
                self.processed = json.loads(self.ckpt_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        self.whisper = None

    def save_ckpt(self):
        try:
            self.ckpt_path.write_text(json.dumps(self.processed, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def find_media(self) -> List[Path]:
        base = Path(self.cfg.input_folder)
        if not base.exists():
            return []
        pats = self.cfg.pattern.split(",") if self.cfg.pattern else []
        exts = [p.strip().lower() for p in pats if p.strip()]
        default_exts = ['.mp4','.mkv','.avi','.mov','.wmv','.flv','.webm','.m4v','.mp3','.wav','.m4a']
        if not exts:
            exts = default_exts
        files = []
        for ext in exts:
            if not ext.startswith("."):  # allow bare 'mp4'
                ext = "." + ext
            files += list(base.glob(f"**/*{ext}"))
            files += list(base.glob(f"**/*{ext.upper()}"))
        files = sorted(set(files))
        if self.cfg.max_files > 0:
            files = files[: self.cfg.max_files]
        return files

    def file_hash(self, p: Path) -> str:
        st = p.stat()
        return hashlib.md5(f"{p}_{st.st_size}_{st.st_mtime}".encode()).hexdigest()

    def init_model(self):
        if self.whisper is None:
            self.whisper = load_whisper(self.cfg.whisper_device, self.cfg.whisper_batch_size, self.logger)

    def transcribe_one(self, media: Path, audio_wav: Optional[Path]) -> Dict:
        file_hash = self.file_hash(media)
        out_dir = Path(self.cfg.output_json_dir)
        out_jsonl = out_dir / f"{media.stem}.jsonl"
        meta_json  = out_dir / f"{media.stem}.meta.json"

        # Skip nếu đã có
        if out_jsonl.exists() and out_jsonl.stat().st_size > 0 and self.cfg.resume:
            self.logger.info(f"[SKIP] {out_jsonl.name} (exists)")
            return {"success": True, "file": str(media)}

        media_in = audio_wav if (audio_wav and audio_wav.exists()) else media

        result = self.whisper.transcribe_file_with_sliding_window(
            media_in,
            language=None,
            chunk_duration=self.cfg.chunk_duration,
            overlap=self.cfg.overlap,
            use_vad=self.cfg.use_vad,
            vad_min_silence_ms=250,
            vad_speech_pad_ms=200,
            vad_threshold=0.5,
            pad_silence_tail=0.3,
            batch_infer=True
        )
        if not result.get("success"):
            raise RuntimeError(result.get("error", "ASR error"))

        total_dur = _to_float_safe(result.get("total_duration"), 0.0)
        chunks = result.get("semantic_chunks") or []
        if not chunks and result.get("segments"):
            segs = sanitize_segments(result["segments"], total=None)
            # gộp theo ~ target_words
            cur, merged = {"segments": [], "text": [], "wc": 0}, []
            for seg in segs:
                tx = seg["text"]; w = len(tx.split())
                if cur["wc"] + w > self.cfg.target_words and cur["segments"]:
                    s0 = cur["segments"][0]["start"]; e1 = cur["segments"][-1]["end"]
                    s,e,_ = sanitize_time_pair(s0,e1,total=None)
                    merged.append({"text":" ".join(cur["text"]), "start_time":s, "end_time":e})
                    cur = {"segments": [], "text": [], "wc": 0}
                cur["segments"].append(seg); cur["text"].append(tx); cur["wc"] += w
            if cur["segments"]:
                s0 = cur["segments"][0]["start"]; e1 = cur["segments"][-1]["end"]
                s,e,_ = sanitize_time_pair(s0,e1,total=None)
                merged.append({"text":" ".join(cur["text"]), "start_time":s, "end_time":e})
            chunks = sanitize_semantic_chunks(merged, total=total_dur)

        # Ghi JSONL
        with open(out_jsonl, "w", encoding="utf-8") as fp:
            for i, ch in enumerate(chunks):
                text = (ch.get("text") or "").strip()
                if not text:
                    continue
                s,e,d = sanitize_time_pair(ch.get("start_time",0), ch.get("end_time",0), total=total_dur)
                doc = {
                    "id": f"{file_hash}_{i:04d}",
                    "file_path": str(media),
                    "file_name": media.name,
                    "file_hash": file_hash,
                    "chunk_index": i,
                    "content": text,
                    "content_type": "audio",
                    "start_time": s,
                    "end_time": e,
                    "duration": d,
                    "word_count": len(text.split()),
                    "timestamp": f"[{s:.1f} - {e:.1f}]",
                    "language": result.get("language","unknown"),
                    "total_duration": total_dur,
                    "processed_at": datetime.now().isoformat(),
                }
                fp.write(json.dumps(doc, ensure_ascii=False) + "\n")

        with open(meta_json, "w", encoding="utf-8") as fp:
            json.dump({
                "file_path": str(media),
                "file_hash": file_hash,
                "total_duration": total_dur,
                "chunks_count": len(chunks),
                "language": result.get("language","unknown"),
            }, fp, indent=2, ensure_ascii=False)


        print(f"[SUCCESS] {media.name}: {len(chunks)} chunks, duration {total_dur:.1f}s")
        return {"success": True, "file": str(media), "chunks": len(chunks)}

    def run(self, list_only: bool = False):
        out_dir = Path(self.cfg.output_json_dir)
        print(f"OUTPUT_DIR: {out_dir.resolve()}")
        print(f"CHECKPOINT: {self.ckpt_path.resolve()}")

        files = self.find_media()
        if not files:
            self.logger.warning("Không tìm thấy file input nào.")
            return

        # liệt kê trước
        self.logger.info(f"Found {len(files)} input file(s):")
        for p in files:
            try:
                sz = p.stat().st_size
                self.logger.info(f"  - {p.name}  ({human_bytes(sz)})")
            except Exception:
                self.logger.info(f"  - {p.name}")

        if list_only:
            self.logger.info("List-only mode: kết thúc.")
            return

        # Pre-extract audio song song
        self.logger.info(f"Pre-extract audio with ffmpeg (workers={self.cfg.ffmpeg_workers}) …")
        to_audio: Dict[Path, Optional[Path]] = {}
        with ThreadPoolExecutor(max_workers=max(1, self.cfg.ffmpeg_workers)) as pool:
            futs = {}
            for vp in files:
                out_wav = Path(self.cfg.audio_tmp_dir) / f"{vp.stem}.wav"
                futs[pool.submit(extract_audio_ffmpeg, vp, out_wav)] = (vp, out_wav)
            for fut in as_completed(futs):
                vp, out_wav = futs[fut]
                try:
                    ap = fut.result()
                except Exception:
                    ap = None
                to_audio[vp] = ap

        # Init model sau khi chắc có input
        self.init_model()

        ok, fail = 0, 0
        with tqdm(total=len(files), desc="ASR → JSONL", ncols=100) as pbar:
            for idx, vp in enumerate(files, 1):
                try:
                    res = self.transcribe_one(vp, to_audio.get(vp))
                    self.processed[str(vp)] = {"ok": True, "chunks": res.get("chunks", 0)}
                    ok += 1
                except Exception as e:
                    print(e)
                    self.logger.error(f"[FAIL] {vp.name}: {e}")
                    self.processed[str(vp)] = {"ok": False, "error": str(e)}
                    fail += 1

                if idx % 3 == 0:
                    self.save_ckpt()

                pbar.update(1)


        self.save_ckpt()
        self.logger.info(f"Done. Success={ok}, Fail={fail}. JSONL saved to {out_dir.resolve()}")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def clamp_batch_size(x: int) -> int:
    # Kaggle GPU hay OOM nếu bs lớn; kẹp trong [1, 8]
    if x <= 0: x = 1
    return max(1, min(x, 8))

def parse_args():
    ap = argparse.ArgumentParser(description="ASR → JSONL (Kaggle-friendly)")
    ap.add_argument("-i", "--input-folder", required=True, help="Thư mục chứa video/audio")
    ap.add_argument("--output-json-dir", default="./asr_json", help="Nơi ghi *.jsonl, *.meta.json, log & ckpt")
    ap.add_argument("--audio-cache", default="./_audio_cache", help="Cache WAV 16k mono")
    ap.add_argument("--ffmpeg-workers", type=int, default=4)
    ap.add_argument("--gpu-batch-size", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--chunk-duration", type=float, default=20.0)
    ap.add_argument("--overlap", type=float, default=5.0)
    ap.add_argument("--no-vad", action="store_true")
    ap.add_argument("--target-words", type=int, default=150)
    ap.add_argument("--resume", action="store_true", help="Bỏ qua file đã có output")
    ap.add_argument("--purge-audio", action="store_true", help="Xóa WAV cache sau khi xong (không khuyến nghị)")
    ap.add_argument("--list-only", action="store_true", help="Chỉ liệt kê file input rồi thoát")
    ap.add_argument("--pattern", default="", help="Lọc phần mở rộng, ví dụ: 'mp4,mp3,wav'")
    ap.add_argument("--max-files", type=int, default=0, help="Giới hạn số file để test")
    return ap.parse_args()

def main():
    args = parse_args()

    # Chuẩn bị đường dẫn tuyệt đối
    input_folder = str(Path(args.input_folder).resolve())
    output_dir = Path(args.output_json_dir).resolve()
    audio_cache = Path(args.audio_cache).resolve()
    checkpoint = output_dir / "asr_export_ckpt.json"

    # Logging nằm trong output_dir (đảm bảo thấy trên sidebar Kaggle)
    logger = setup_logging(output_dir)

    bs = clamp_batch_size(args.gpu_batch_size)
    if bs != args.gpu_batch_size:
        logger.info(f"gpu-batch-size={args.gpu_batch_size} → điều chỉnh thành {bs} để tránh OOM")

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9)
        torch.backends.cudnn.benchmark = True

    cfg = ProcessingConfig(
        input_folder=input_folder,
        output_json_dir=str(output_dir),
        checkpoint_file=str(checkpoint),
        audio_tmp_dir=str(audio_cache),
        ffmpeg_workers=max(1, args.ffmpeg_workers),
        whisper_device=args.device,
        whisper_batch_size=bs,
        chunk_duration=args.chunk_duration,
        overlap=args.overlap,
        use_vad=not args.no_vad,
        target_words=args.target_words,
        resume=bool(args.resume),
        keep_audio_cache=not args.purge_audio,
        pattern=args.pattern,
        max_files=args.max_files,
    )

    exporter = ASRExporter(cfg, logger)
    exporter.run(list_only=bool(args.list_only))

if __name__ == "__main__":
    main()
