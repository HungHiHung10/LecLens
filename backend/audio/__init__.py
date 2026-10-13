"""
backend.audio: Speech Recognition, Subtitle Parsing, and Temporal Chunker.
"""
from backend.audio.subtitle_parser import parse_srt_or_vtt, seconds_to_timestamp
from backend.audio.chunker import create_temporal_chunks
from backend.audio.audio_transcriber import fetch_youtube_transcript, transcribe_local_video, extract_youtube_video_id
from backend.audio.groq_transcriber import transcribe_with_groq, get_saved_groq_key, save_groq_key

__all__ = [
    "parse_srt_or_vtt",
    "seconds_to_timestamp",
    "create_temporal_chunks",
    "fetch_youtube_transcript",
    "transcribe_local_video",
    "extract_youtube_video_id",
    "transcribe_with_groq",
    "get_saved_groq_key",
    "save_groq_key",
]
