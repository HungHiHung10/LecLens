import sys
import time
from pathlib import Path

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from backend.audio.groq_transcriber import transcribe_with_groq, get_saved_groq_key

key = get_saved_groq_key()
video_path = r"a:\LecLens\Lecture-8-Data-Splits-Models-Cross-Validation-Stanford-CS229-Machine-Learning-Au.mp4"

print(f"Bắt đầu trích xuất âm thanh và gọi Groq Whisper API...")
print(f"File video: {video_path}")
print(f"Groq API Key: {key[:8]}...{key[-4:]}")

t0 = time.time()
subs = transcribe_with_groq(video_path, api_key=key, model="whisper-large-v3")
t_elapsed = time.time() - t0

print(f"\n==========================================")
print(f"THÀNH CÔNG: Đã trích xuất {len(subs)} câu phụ đề trong {t_elapsed:.2f} giây!")
print(f"Tốc độ: {5005.7 / max(0.1, t_elapsed):.1f}x thời gian thực!")
print(f"==========================================\n")

print("3 câu đầu tiên:")
for c in subs[:3]:
    print(f"  [{c['start_str']} -> {c['end_str']}]: {c['text']}")

print("\n3 câu cuối cùng:")
for c in subs[-3:]:
    print(f"  [{c['start_str']} -> {c['end_str']}]: {c['text']}")

# Lưu ra file SRT
srt_blocks = []
for cue in subs:
    srt_blocks.append(f"{cue['index']}\n{cue['start_str']} --> {cue['end_str']}\n{cue['text']}")

out_paths = [
    Path("data/samples/stanford_cs229_lecture_8.srt"),
    Path(video_path).with_suffix(".groq.srt"),
    Path(video_path).with_suffix(".srt")
]

for out_srt in out_paths:
    out_srt.parent.mkdir(parents=True, exist_ok=True)
    out_srt.write_text("\n\n".join(srt_blocks), encoding="utf-8")
    print(f"Đã lưu phụ đề vào: {out_srt} ({out_srt.stat().st_size} bytes)")
