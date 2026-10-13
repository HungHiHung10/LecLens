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
video_path = r"a:\LecLens\Tất cả những gì bạn cần biết về Transformer trong 35 phút.mp4"

print(f"Bắt đầu nhận dạng chính xác tiếng Việt cho: {Path(video_path).name}")
t0 = time.time()
subs = transcribe_with_groq(video_path, api_key=key, model="whisper-large-v3", language="vi")
elapsed = time.time() - t0

print(f"\nHoàn tất {len(subs)} câu phụ đề trong {elapsed:.2f}s!")

srt_blocks = []
for cue in subs:
    srt_blocks.append(f"{cue['index']}\n{cue['start_str']} --> {cue['end_str']}\n{cue['text']}")

content = "\n\n".join(srt_blocks)

out_paths = [
    Path("data/samples/transformer_35_phut_vi.srt"),
    Path(video_path).with_suffix(".srt"),
    Path("data/uploads/custom_1791384742.mp4.srt")
]

for p in out_paths:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    print(f"Đã lưu: {p} ({p.stat().st_size} bytes)")
