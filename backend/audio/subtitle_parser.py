import re
from pathlib import Path
from typing import List, Dict, Union

def timestamp_to_seconds(ts_str: str) -> float:
    """
    Converts timestamp string 'HH:MM:SS,mmm' or 'HH:MM:SS.mmm' or 'MM:SS.mmm' to seconds (float).
    """
    ts_str = ts_str.strip().replace(',', '.')
    parts = ts_str.split(':')
    if len(parts) == 3:
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        return hours * 3600.0 + minutes * 60.0 + seconds
    elif len(parts) == 2:
        minutes = float(parts[0])
        seconds = float(parts[1])
        return minutes * 60.0 + seconds
    else:
        return float(parts[0])

def seconds_to_timestamp(seconds: float, format_style: str = "compact") -> str:
    """
    Converts float seconds to formatted string.
    - 'compact': mm:ss (e.g. 03:45) or hh:mm:ss if > 1 hour
    - 'srt': 00:03:45,000
    """
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))

    if format_style == "srt":
        return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"
    
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"

def clean_subtitle_text(text: str) -> str:
    """
    Removes HTML tags, WebVTT cue tags, and normalizes whitespaces.
    """
    # Remove HTML tags like <i>, <b>, <font>, etc.
    cleaned = re.sub(r'<[^>]+>', '', text)
    # Remove WebVTT formatting e.g. <v Speaker>
    cleaned = re.sub(r'<\/?v[^>]*>', '', cleaned)
    # Replace newlines with spaces and condense repeated whitespaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def parse_srt_or_vtt(content_or_path: Union[str, Path]) -> List[Dict]:
    """
    Parses SRT or VTT content or file path into a list of subtitle dictionaries:
    [{'index': int, 'start': float, 'end': float, 'text': str, 'start_str': str, 'end_str': str}]
    """
    if isinstance(content_or_path, (str, Path)) and (str(content_or_path).endswith('.srt') or str(content_or_path).endswith('.vtt')) and Path(content_or_path).is_file():
        text = Path(content_or_path).read_text(encoding='utf-8', errors='replace')
    else:
        text = str(content_or_path)

    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Strip WEBVTT header if present
    if text.startswith('WEBVTT'):
        lines = text.split('\n')
        # Skip header lines until first empty line or first cue
        idx = 0
        while idx < len(lines) and (lines[idx].startswith('WEBVTT') or lines[idx].startswith('NOTE') or lines[idx].strip() == ''):
            idx += 1
        text = '\n'.join(lines[idx:])

    # Regex pattern to match timestamp line:
    # 00:00:10,000 --> 00:00:15,000 or 00:10.000 --> 00:15.000
    timestamp_pattern = re.compile(
        r'(\d{1,2}:\d{2}(?::\d{2})?(?:[,\.]\d{1,3})?)\s*-->\s*(\d{1,2}:\d{2}(?::\d{2})?(?:[,\.]\d{1,3})?)'
    )

    blocks = re.split(r'\n\s*\n', text.strip())
    subtitles = []
    item_index = 1

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.split('\n')
        
        ts_match = None
        ts_line_idx = -1
        for i, line in enumerate(lines):
            m = timestamp_pattern.search(line)
            if m:
                ts_match = m
                ts_line_idx = i
                break
        
        if ts_match:
            start_str, end_str = ts_match.group(1), ts_match.group(2)
            try:
                start_sec = timestamp_to_seconds(start_str)
                end_sec = timestamp_to_seconds(end_str)
            except Exception:
                continue

            # Text lines follow the timestamp line
            text_lines = lines[ts_line_idx + 1:]
            raw_text = " ".join(text_lines)
            cleaned = clean_subtitle_text(raw_text)

            if cleaned:
                subtitles.append({
                    "index": item_index,
                    "start": round(start_sec, 2),
                    "end": round(end_sec, 2),
                    "start_str": seconds_to_timestamp(start_sec),
                    "end_str": seconds_to_timestamp(end_sec),
                    "text": cleaned
                })
                item_index += 1

    return subtitles
