import os
import sys
import time
import json
import math
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import av
import numpy as np
from faster_whisper.audio import decode_audio
from backend.audio.subtitle_parser import seconds_to_timestamp
from backend.config import BASE_DIR, get_logger

logger = get_logger("groq_asr")

CONFIG_FILE = BASE_DIR / "data" / "groq_config.json"

def get_saved_groq_key() -> Optional[str]:
    """Retrieves saved Groq API key from environment or config file."""
    env_key = os.environ.get("GROQ_API_KEY", "").strip()
    if env_key:
        return env_key
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            k = data.get("api_key", "").strip()
            if k:
                return k
        except Exception:
            pass
    return None

def save_groq_key(api_key: str):
    """Saves Groq API key to persistent config."""
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps({"api_key": api_key.strip()}, indent=2), encoding="utf-8")

def encode_numpy_to_mp3(audio_arr: np.ndarray, output_path: str, sr: int = 16000, bit_rate: int = 32000):
    """
    Encodes float32 numpy audio array [-1.0, 1.0] to a compact MP3 file using PyAV.
    32kbps mono @ 16kHz produces ~240 KB per minute (~14.4 MB per hour).
    """
    container = av.open(output_path, mode="w", format="mp3")
    stream = container.add_stream("mp3", rate=sr)
    stream.bit_rate = bit_rate
    
    int_audio = (np.clip(audio_arr, -1.0, 1.0) * 32767).astype(np.int16)
    frame = av.AudioFrame.from_ndarray(int_audio.reshape(1, -1), format="s16", layout="mono")
    frame.rate = sr
    
    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

def transcribe_with_groq(
    video_path: str,
    api_key: Optional[str] = None,
    model: str = "whisper-large-v3",
    language: Optional[str] = None
) -> List[Dict]:
    """
    Transcribes audio from a video/audio file using Groq Cloud LPU Whisper.
    Xong trong 5-8 giây cho video dài 80-120 phút!
    """
    from groq import Groq

    key = api_key or get_saved_groq_key()
    if not key:
        raise ValueError("Chưa cấu hình Groq API Key. Vui lòng nhập API key của bạn.")

    client = Groq(api_key=key)

    tmp_dir = BASE_DIR / "data" / "cache" / "groq_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    temp_stem = f"groq_audio_{int(time.time())}"
    
    logger.info(f"Đang trích xuất audio 16kHz từ '{Path(video_path).name}'...")
    t0 = time.time()
    audio_data = decode_audio(video_path, sampling_rate=16000)
    total_samples = len(audio_data)
    total_duration_sec = total_samples / 16000.0
    logger.info(f"Trích xuất xong: {total_duration_sec:.1f} giây âm thanh ({time.time()-t0:.2f}s).")

    vn_chars = "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ"
    if not language:
        name_lower = Path(video_path).name.lower()
        if any(c in name_lower for c in vn_chars) or any(w in name_lower for w in ["viet", "tieng", "bai", "giang", "hoc", "trong", "tat", "ca"]):
            language = "vi"

    prompt_guide = "Bài giảng Machine Learning, Deep Learning, AI bằng tiếng Việt chuẩn có dấu, kèm thuật ngữ công nghệ." if language == "vi" else None

    # 15 minutes per slice = 900 seconds = 14,400,000 samples (~3.6 MB MP3)
    # This guarantees staying way below Groq's 25 MB limit for any lecture length!
    SLICE_SECONDS = 900
    SLICE_SAMPLES = SLICE_SECONDS * 16000

    num_slices = max(1, math.ceil(total_samples / SLICE_SAMPLES))
    logger.info(f"Chia audio thành {num_slices} phân đoạn để gửi lên Groq LPU (ngôn ngữ: {language or 'auto'})...")

    def process_slice(slice_idx: int) -> List[Dict]:
        start_samp = slice_idx * SLICE_SAMPLES
        end_samp = min(total_samples, (slice_idx + 1) * SLICE_SAMPLES)
        chunk_audio = audio_data[start_samp:end_samp]
        
        chunk_mp3_path = str(tmp_dir / f"{temp_stem}_slice_{slice_idx}.mp3")
        encode_numpy_to_mp3(chunk_audio, chunk_mp3_path, sr=16000, bit_rate=32000)
        
        offset_sec = slice_idx * SLICE_SECONDS
        slice_subs = []
        try:
            with open(chunk_mp3_path, "rb") as f:
                transcription = client.audio.transcriptions.create(
                    file=(Path(chunk_mp3_path).name, f),
                    model=model,
                    response_format="verbose_json",
                    language=language,
                    prompt=prompt_guide,
                    temperature=0.0
                )
            
            segments = getattr(transcription, "segments", None)
            if segments is None and isinstance(transcription, dict):
                segments = transcription.get("segments", [])

            if segments:
                for seg in segments:
                    seg_start = float(getattr(seg, "start", 0) if hasattr(seg, "start") else seg.get("start", 0)) + offset_sec
                    seg_end = float(getattr(seg, "end", 0) if hasattr(seg, "end") else seg.get("end", 0)) + offset_sec
                    seg_text = str(getattr(seg, "text", "") if hasattr(seg, "text") else seg.get("text", "")).strip()
                    if seg_text:
                        slice_subs.append({
                            "start": round(seg_start, 2),
                            "end": round(seg_end, 2),
                            "text": seg_text
                        })
        finally:
            if os.path.exists(chunk_mp3_path):
                try:
                    os.remove(chunk_mp3_path)
                except Exception:
                    pass
        return slice_subs

    all_raw_cues = []
    with ThreadPoolExecutor(max_workers=min(4, num_slices)) as executor:
        slice_results = list(executor.map(process_slice, range(num_slices)))

    for res in slice_results:
        all_raw_cues.extend(res)

    all_raw_cues.sort(key=lambda x: x["start"])
    final_subtitles = []
    for idx, cue in enumerate(all_raw_cues, 1):
        final_subtitles.append({
            "index": idx,
            "start": cue["start"],
            "end": cue["end"],
            "start_str": seconds_to_timestamp(cue["start"]),
            "end_str": seconds_to_timestamp(cue["end"]),
            "text": cue["text"]
        })

    elapsed = time.time() - t0
    logger.info(f"Hoàn tất nhận dạng: {len(final_subtitles)} câu trong {elapsed:.2f} giây! (Tốc độ: {total_duration_sec/max(0.1, elapsed):.1f}x real-time).")
    return final_subtitles
