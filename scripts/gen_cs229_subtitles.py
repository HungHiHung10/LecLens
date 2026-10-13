import sys
from pathlib import Path

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from backend.audio.audio_transcriber import fetch_youtube_transcript

subs = fetch_youtube_transcript('rjbkWSTjHzM')

def format_srt_time(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int((sec - int(sec)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

srt_blocks = []
for i, cue in enumerate(subs, 1):
    start_fmt = format_srt_time(cue['start'])
    end_fmt = format_srt_time(cue['end'])
    text = cue['text']
    srt_blocks.append(f"{i}\n{start_fmt} --> {end_fmt}\n{text}")

content = "\n\n".join(srt_blocks) + "\n"

target_paths = [
    Path("data/samples/stanford_cs229_lecture_8.srt"),
    Path("data/samples/Lecture-8-Data-Splits-Models-Cross-Validation-Stanford-CS229-Machine-Learning-Au.srt"),
    Path("data/samples/Lecture-8-Data-Splits-Models-Cross-Validation-Stanford-CS229-Machine-Learning-Au_002_720p.srt"),
    Path("Lecture-8-Data-Splits-Models-Cross-Validation-Stanford-CS229-Machine-Learning-Au.srt")
]

for p in target_paths:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    print(f"Generated: {p} ({p.stat().st_size} bytes)")
