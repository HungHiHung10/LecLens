from typing import List, Dict
from backend.audio.subtitle_parser import seconds_to_timestamp

def create_temporal_chunks(
    subtitles: List[Dict],
    window_seconds: float = 30.0,
    stride_seconds: float = 15.0
) -> List[Dict]:
    """
    Groups subtitles into overlapping temporal windows (e.g., 30s window with 15s stride).
    Each chunk retains exact start and end timestamps and combined text.
    """
    if not subtitles:
        return []

    video_start = subtitles[0]["start"]
    video_end = subtitles[-1]["end"]

    # If the video is shorter than the window, return single chunk
    if video_end - video_start <= window_seconds:
        combined_text = " ".join([s["text"] for s in subtitles])
        return [{
            "chunk_id": "chunk_0",
            "start": round(video_start, 2),
            "end": round(video_end, 2),
            "start_str": seconds_to_timestamp(video_start),
            "end_str": seconds_to_timestamp(video_end),
            "duration": round(video_end - video_start, 2),
            "text": combined_text,
            "subtitle_indices": [s["index"] for s in subtitles]
        }]

    chunks = []
    current_start = video_start
    chunk_index = 0

    while current_start < video_end:
        current_end = current_start + window_seconds
        
        # Gather all subtitles that overlap with [current_start, current_end]
        overlapping_subs = []
        for s in subtitles:
            # Overlap condition: subtitle starts before window ends AND subtitle ends after window starts
            if s["start"] < current_end and s["end"] > current_start:
                overlapping_subs.append(s)

        if overlapping_subs:
            # Anchor start/end to actual speech boundaries for tighter temporal grounding
            actual_start = min(s["start"] for s in overlapping_subs)
            actual_end = max(s["end"] for s in overlapping_subs)
            
            combined_text = " ".join([s["text"] for s in overlapping_subs]).strip()
            
            # Avoid duplicate chunk text if subtitle coverage didn't change
            if not chunks or chunks[-1]["text"] != combined_text:
                chunks.append({
                    "chunk_id": f"chunk_{chunk_index}",
                    "start": round(actual_start, 2),
                    "end": round(actual_end, 2),
                    "start_str": seconds_to_timestamp(actual_start),
                    "end_str": seconds_to_timestamp(actual_end),
                    "duration": round(actual_end - actual_start, 2),
                    "text": combined_text,
                    "subtitle_indices": [s["index"] for s in overlapping_subs]
                })
                chunk_index += 1

        current_start += stride_seconds

    return chunks
