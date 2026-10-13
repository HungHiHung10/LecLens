import sys
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import re
import os
from pathlib import Path
from typing import List, Dict, Optional
from backend.audio.subtitle_parser import seconds_to_timestamp

def extract_youtube_video_id(url_or_id: str) -> Optional[str]:
    """
    Extracts 11-character YouTube video ID from various URL formats or raw ID.
    Examples:
    - https://www.youtube.com/watch?v=dQw4w9WgXcQ
    - https://youtu.be/dQw4w9WgXcQ
    - https://www.youtube.com/embed/dQw4w9WgXcQ
    - dQw4w9WgXcQ
    """
    url_or_id = url_or_id.strip()
    if len(url_or_id) == 11 and re.match(r'^[0-9A-Za-z_-]{11}$', url_or_id):
        return url_or_id

    patterns = [
        r'(?:v=|\/embed\/|\/v\/|youtu\.be\/|\/shorts\/)([0-9A-Za-z_-]{11})',
        r'(?:watch\?.*v=)([0-9A-Za-z_-]{11})'
    ]
    for p in patterns:
        m = re.search(p, url_or_id)
        if m:
            return m.group(1)
    return None

def fetch_youtube_transcript(url_or_id: str) -> List[Dict]:
    """
    Fetches transcript directly from YouTube using youtube-transcript-api.
    No local GPU/CPU computation needed, returns in 1 second.
    """
    from youtube_transcript_api import YouTubeTranscriptApi

    video_id = extract_youtube_video_id(url_or_id)
    if not video_id:
        raise ValueError(f"Không thể nhận diện YouTube Video ID từ: '{url_or_id}'")

    print(f"[YouTubeTranscriber] Fetching transcript for YouTube ID: {video_id}...")
    yta = YouTubeTranscriptApi()

    # Try list transcripts to find vi or en or any available
    raw_snippets = None
    try:
        t_list = yta.list(video_id)
        try:
            t = t_list.find_transcript(['vi', 'en'])
        except Exception:
            try:
                t = t_list.find_generated_transcript(['vi', 'en'])
            except Exception:
                t = next(iter(t_list))
        raw_snippets = t.fetch()
    except Exception as e:
        # Fallback to direct fetch
        try:
            raw_snippets = yta.fetch(video_id, languages=['vi', 'en', 'auto'])
        except Exception as inner_e:
            raise RuntimeError(f"Video YouTube này không có phụ đề (hoặc phụ đề tự động bị vô hiệu hóa): {inner_e}")

    subtitles = []
    for idx, item in enumerate(raw_snippets):
        start = float(getattr(item, 'start', 0.0) if hasattr(item, 'start') else item.get('start', 0.0))
        duration = float(getattr(item, 'duration', 2.0) if hasattr(item, 'duration') else item.get('duration', 2.0))
        end = start + duration
        text = str(getattr(item, 'text', '') if hasattr(item, 'text') else item.get('text', '')).strip()
        
        if text:
            subtitles.append({
                "index": idx + 1,
                "start": round(start, 2),
                "end": round(end, 2),
                "start_str": seconds_to_timestamp(start),
                "end_str": seconds_to_timestamp(end),
                "text": text
            })

    print(f"[YouTubeTranscriber] Successfully fetched {len(subtitles)} cues from YouTube.")
    return subtitles

_whisper_model = None

def transcribe_local_video(
    video_path: str,
    model_size: str = "tiny",
    language: Optional[str] = None
) -> List[Dict]:
    """
    Transcribes audio directly from local video file (.mp4, .webm, .wav, .mp3)
    using faster-whisper with int8 quantization on CPU.
    """
    global _whisper_model
    from faster_whisper import WhisperModel

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    print(f"[WhisperASR] Loading faster-whisper '{model_size}' model on CPU (int8)...")
    if _whisper_model is None:
        _whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=4)

    print(f"[WhisperASR] Transcribing audio from '{video_path}'...")
    segments, info = _whisper_model.transcribe(
        video_path,
        beam_size=3,
        language=language,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )

    subtitles = []
    idx = 1
    for seg in segments:
        text = seg.text.strip()
        if text:
            subtitles.append({
                "index": idx,
                "start": round(float(seg.start), 2),
                "end": round(float(seg.end), 2),
                "start_str": seconds_to_timestamp(seg.start),
                "end_str": seconds_to_timestamp(seg.end),
                "text": text
            })
            idx += 1

    print(f"[WhisperASR] Transcription completed: {len(subtitles)} segments.")
    return subtitles
