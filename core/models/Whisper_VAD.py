# model/Whisper_VAD.py

import os
os.environ["HF_HUB_DISABLE_SAFETENSORS_CONVERSION"] = "1"

import time
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import librosa
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
warnings.filterwarnings("ignore")
load_dotenv()

# Giảm phân mảnh bộ nhớ CUDA (nếu user chưa set trong shell)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


class WhisperTranscription:
    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3",
        device: str = "auto",
        batch_size: int = 4,  # giữ nhỏ cho an toàn, nhưng pipeline sẽ chọn batch động
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        speed_mode: bool = False,
        offload_folder: str = ".whisper_offload",
        force_pipe_bs: int = None,  # NEW: force Whisper pipeline batch size
    ):
        # Device
        if device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.force_pipe_bs = force_pipe_bs

        # Dtype
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.speed_mode = speed_mode

        # Speed mode: hạ model
        if speed_mode and "large" in model_name:
            print("Speed mode: switching to medium model for faster processing")
            model_name = (
                model_name.replace("large-v3", "medium")
                .replace("large-v2", "medium")
                .replace("large", "medium")
            )

        # Ước lượng VRAM để chọn offload
        use_offload = False
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total_gb = props.total_memory / (1024**3)
            # Nếu GPU <= 8GB & model là large => bật offload
            if total_gb <= 8 and "large" in model_name:
                use_offload = True

        # Load model (ưu tiên offload cho GPU nhỏ)
        if use_offload:
            print(f"[INFO] Low VRAM ({total_gb:.1f}GB). Using device_map='auto' with offload.")
            max_mem = {0: "7.0GiB", "cpu": "48GiB"}
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=False,
                device_map="auto",
                max_memory=max_mem,
                offload_folder=offload_folder,

            )
        else:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=False,
            )
            # Đẩy toàn bộ lên device nếu không offload
            if "cuda" in str(self.device):
                self.model.to(self.device)

        try:
            if hasattr(self.model, "generation_config"):
                self.model.generation_config.forced_decoder_ids = None
            if hasattr(self.model, "config"):
                self.model.config.forced_decoder_ids = None
        except Exception:
            pass

        # GPU optim
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            try:
                if hasattr(self.model, "config"):
                    self.model.config.attn_implementation = "sdpa"
            except Exception:
                pass
            # Tuỳ chọn: compile để tăng throughput (PyTorch >= 2.0)
            try:
                self.model = torch.compile(self.model, mode="max-autotune")
            except Exception:
                pass

        self.processor = AutoProcessor.from_pretrained(model_name)
        try:
            self.processor.feature_extractor.return_attention_mask = True
        except Exception:
            pass

        # Pipeline: dùng generate_kwargs để cố định task => tránh cảnh báo
        chunk_length = 20 if speed_mode else 30
        pipe_bs = max(1, min(batch_size, 4))  # seed value; sẽ override theo VRAM khi chạy

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=128 if speed_mode else 256,
            chunk_length_s=chunk_length,
            batch_size=pipe_bs,
            return_timestamps=True,
            torch_dtype=self.torch_dtype,
            device=self.device if not use_offload else None,
            framework="pt",
            generate_kwargs={"task": "transcribe"},  # NEW
        )

        # Embedding model
        if speed_mode:
            embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.embedding_model = SentenceTransformer(embedding_model)

        # Silero VAD
        self.vad_ready = False
        try:
            self.vad_model, self.vad_utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
            )
            (
                self.get_speech_timestamps,
                self.save_audio,
                self.read_audio,
                self.VADIterator,
                self.collect_chunks,
            ) = self.vad_utils
            self.vad_ready = True
        except Exception as e:
            print(f"[WARN] Silero VAD load failed: {e}. VAD disabled.")

        self.model_name = model_name.split("/")[-1]
        self.batch_size = batch_size
        self.supported_formats = [
            ".mp4",".mkv",".avi",".mov",".wmv",".flv",".webm",".m4v",".mp3",".wav",".m4a",
        ]

    # -------------------- Utils --------------------

    def find_video_files(self, input_folder):
        input_path = Path(input_folder)
        video_files = []
        for ext in self.supported_formats:
            video_files.extend(input_path.glob(f"**/*{ext}"))
            video_files.extend(input_path.glob(f"**/*{ext.upper()}"))
        return sorted(video_files)

    def format_timestamp(self, seconds):
        if seconds is None:
            return "00:00:00"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def get_audio_duration(self, file_path):
        try:
            duration = librosa.get_duration(path=str(file_path))
            return duration
        except Exception:
            return None

    def load_audio_chunk(self, file_path, offset=None, duration=None):
        try:
            audio, _ = librosa.load(
                str(file_path), sr=16000, offset=offset, duration=duration, mono=True
            )
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            return audio
        except Exception:
            return None

    # -------------------- VAD --------------------

    def run_vad(self, file_path, min_silence_ms=200, speech_pad_ms=200, threshold=0.5):
        if not self.vad_ready:
            return []
        wav = self.read_audio(str(file_path), sampling_rate=16000)
        speech_ts = self.get_speech_timestamps(
            wav,
            self.vad_model,
            sampling_rate=16000,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
            threshold=threshold,
        )
        segments = []
        for ts in speech_ts:
            s = ts["start"] / 16000.0
            e = ts["end"] / 16000.0
            if e - s > 0.2:
                segments.append((s, e))
        return segments

    def _split_long_segment(self, start_sec, end_sec, chunk_duration, overlap):
        segs, total = [], (end_sec - start_sec)
        if total <= chunk_duration:
            return [(start_sec, end_sec)]
        step = max(chunk_duration - overlap, 0.001)
        t = start_sec
        while t < end_sec:
            t_end = min(t + chunk_duration, end_sec)
            segs.append((t, t_end))
            if t_end >= end_sec:
                break
            t = t + step
        return segs

    # -------------------- Semantic chunking --------------------

    def semantic_chunking(self, segments, target_words=200, similarity_threshold=0.6, max_duration=120, min_duration=30):
        if not segments:
            return []
        texts = [seg["text"] for seg in segments if (seg.get("text") or "").strip()]
        if not texts:
            return []
        embeddings = self.embedding_model.encode(texts, batch_size=128, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)  # NEW: batch encode + normalize
        out, cur = [], {"segments": [], "embeddings": [], "word_count": 0, "duration": 0}
        for seg, emb in zip(segments, embeddings):
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            words = len(text.split())
            dur = float(seg["end"]) - float(seg["start"])

            split = False
            if cur["word_count"] + words > target_words * 1.5:
                split = True
            elif cur["duration"] + dur > max_duration:
                split = True
            elif cur["embeddings"]:
                cemb = np.mean(cur["embeddings"], axis=0)
                sim = np.dot(cemb, emb)
                if sim < similarity_threshold:
                    split = True

            if split and cur["segments"] and (cur["word_count"] >= target_words * 0.5 or cur["duration"] >= min_duration):
                out.append(self._finalize_semantic_chunk(cur, len(out)))
                cur = {"segments": [], "embeddings": [], "word_count": 0, "duration": 0}

            cur["segments"].append(seg)
            cur["embeddings"].append(emb)
            cur["word_count"] += words
            cur["duration"] += dur
        if cur["segments"]:
            out.append(self._finalize_semantic_chunk(cur, len(out)))
        return out

    def _finalize_semantic_chunk(self, chunk_data, chunk_index):
        segs = chunk_data["segments"]
        s = segs[0]["start"]
        e = segs[-1]["end"]
        text = " ".join([(x.get("text") or "").strip() for x in segs])
        emb = np.mean(chunk_data["embeddings"], axis=0)
        return {
            "id": f"semantic_chunk_{chunk_index:04d}",
            "text": text,
            "start_time": s,
            "end_time": e,
            "duration": float(e) - float(s),
            "word_count": len(text.split()),
            "segments": segs,
            "segment_count": len(segs),
            "embedding": emb.tolist(),
            "timestamp": f"[{self.format_timestamp(s)} - {self.format_timestamp(e)}]",
        }

    # -------------------- Core batched inference --------------------

    def _choose_pipe_batch_size(self, fallback=8):
        if self.force_pipe_bs and self.force_pipe_bs > 0:
            return int(self.force_pipe_bs)
        if not torch.cuda.is_available():
            return max(1, fallback // 2)
        try:
            free, total = torch.cuda.mem_get_info()
            free_gb = free / (1024**3)
        except Exception:
            # mem_get_info not available (older drivers), fallback
            free_gb = 6
        if free_gb >= 12: return 16
        if free_gb >= 8:  return 8
        if free_gb >= 6:  return 6
        return 4

    def _infer_batched(self, windows, file_path, pad_silence_tail):
        all_audio, all_meta = [], []
        for (cs, ce) in windows:
            dur = ce - cs
            if dur < 0.5:
                continue
            audio = self.load_audio_chunk(file_path, offset=cs, duration=dur)
            if audio is None:
                continue
            if pad_silence_tail and pad_silence_tail > 0:
                pad = int(pad_silence_tail * 16000)
                audio = np.concatenate([audio, np.zeros(pad, dtype=audio.dtype)])
            all_audio.append(audio)
            all_meta.append((cs, ce))

        if not all_audio:
            return [], ""

        # >>> KHÁC BIỆT Ở ĐÂY: dùng list inputs thay vì Dataset
        inputs = [{"array": a, "sampling_rate": 16000} for a in all_audio]
        pipe_bs = self._choose_pipe_batch_size()

        # Log để kiểm tra batch thực tế
        try:
            import logging
            logging.getLogger(__name__).info(f"[ASR] Using GPU batch size: {pipe_bs}")
        except Exception:
            pass

        results_iter = self.pipe(inputs, batch_size=pipe_bs, return_timestamps=True)

        all_segments, full_text_parts = [], []
        for i, res in enumerate(results_iter):
            cs, ce = all_meta[i]
            text = (res.get("text") or "").strip()
            if text:
                full_text_parts.append(text)
            chunks = res.get("chunks", [])
            for ch in (chunks or []):
                ts = ch.get("timestamp", [None, None])
                st = ts[0] if ts and ts[0] is not None else 0.0
                en = ts[1] if ts and ts[1] is not None else (st + 0.6)
                sg = cs + float(st)
                eg = min(cs + float(en), ce)
                all_segments.append({"start": sg, "end": eg, "text": (ch.get("text") or "").strip()})
        return all_segments, " ".join(full_text_parts)


    # -------------------- Transcription --------------------

    def transcribe_file_with_sliding_window(
        self,
        file_path,
        language: str = None,
        chunk_duration: float = 30,
        overlap: float = 8,
        use_vad: bool = True,
        vad_min_silence_ms: int = 250,
        vad_speech_pad_ms: int = 200,
        vad_threshold: float = 0.5,
        pad_silence_tail: float = 0.5,
        batch_infer: bool = True,  # kept for API compatibility, ignored (we always batch)
    ):
        try:
            t0 = time.time()
            file_path = Path(file_path)

            total_dur = self.get_audio_duration(file_path)
            if total_dur is None:
                raise Exception("Cannot get audio duration")

            if self.speed_mode:
                chunk_duration = min(chunk_duration, 20)
                overlap = min(overlap, 2)

            # Windows theo VAD hoặc fixed
            windows = []
            if use_vad and self.vad_ready:
                vad_segments = self.run_vad(
                    file_path,
                    min_silence_ms=vad_min_silence_ms,
                    speech_pad_ms=vad_speech_pad_ms,
                    threshold=vad_threshold,
                )
                for (s, e) in vad_segments:
                    windows.extend(self._split_long_segment(s, e, chunk_duration, overlap))
            else:
                step = max(chunk_duration - overlap, 0.001)
                if total_dur <= chunk_duration:
                    windows = [(0.0, total_dur)]
                else:
                    num_chunks = int(np.ceil((total_dur - overlap) / step))
                    for i in range(num_chunks):
                        s = i * step
                        e = min(s + chunk_duration, total_dur)
                        if e - s >= 1.0:
                            windows.append((s, e))

            # Nếu speed_mode và không dùng VAD, có thể skip bớt để nhanh
            if self.speed_mode and total_dur > 300 and not (use_vad and self.vad_ready):
                windows = windows[::2]

            # ==== BATCHED INFERENCE ====
            all_segments, full_text = self._infer_batched(windows, file_path, pad_silence_tail)
            proc_time = time.time() - t0

            # Semantic chunking
            if self.speed_mode and total_dur > 600:
                sem_chunks = []
            else:
                sem_chunks = self.semantic_chunking(all_segments)

            formatted = self.create_formatted_text(all_segments)
            detected_language = language or "auto-detected"

            return {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "full_text": full_text,
                "formatted_text": formatted,
                "language": detected_language,
                "segments": all_segments,
                "semantic_chunks": sem_chunks,
                "processing_time": proc_time,
                "total_duration": total_dur,
                "num_chunks": len(windows),
                "chunk_duration": chunk_duration,
                "word_count": len(full_text.split()),
                "speed_mode": self.speed_mode,
                "success": True,
            }

        except Exception as e:
            return {
                "file_path": str(file_path) if isinstance(file_path, Path) else file_path,
                "file_name": Path(file_path).name if isinstance(file_path, (str, Path)) else "",
                "error": str(e),
                "success": False,
            }

    # -------------------- Formatting --------------------

    def create_formatted_text(self, segments):
        if not segments:
            return ""
        lines = []
        for seg in segments:
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            st = self.format_timestamp(seg["start"])
            en = self.format_timestamp(seg["end"])
            lines.append(f"[{st} - {en}] {text}")
        return "\n".join(lines)

    def create_srt_format(self, segments):
        if not segments:
            return ""
        srt_lines = []
        n = 1
        for seg in segments:
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            st = self.format_srt_timestamp(seg["start"])
            en = self.format_srt_timestamp(seg["end"])
            srt_lines.append(f"{n}\n{st} --> {en}\n{text}\n")
            n += 1
        return "\n".join(srt_lines)

    def format_srt_timestamp(self, seconds):
        if seconds is None:
            seconds = 0
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
