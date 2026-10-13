import sys
import os
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from backend.audio.groq_transcriber import get_saved_groq_key, encode_numpy_to_mp3
from faster_whisper.audio import decode_audio
from groq import Groq

key = get_saved_groq_key()
client = Groq(api_key=key)

video_path = r"a:\LecLens\Tất cả những gì bạn cần biết về Transformer trong 35 phút.mp4"
print("Decoding audio from minute 17 to 20...")
audio = decode_audio(video_path, sampling_rate=16000)

s_start = 17 * 60 * 16000
s_end = 20 * 60 * 16000
chunk_audio = audio[s_start:s_end]

tmp_mp3 = "a:/LecLens/test_slice1.mp3"
encode_numpy_to_mp3(chunk_audio, tmp_mp3, sr=16000, bit_rate=32000)

print("Sending to Groq with language='vi'...")
with open(tmp_mp3, "rb") as f:
    res = client.audio.transcriptions.create(
        file=("test_slice1.mp3", f),
        model="whisper-large-v3",
        response_format="verbose_json",
        language="vi",
        temperature=0.0
    )

segments = getattr(res, "segments", [])
print(f"Total segments returned: {len(segments)}")
for seg in segments[:8]:
    start = seg.get("start") if isinstance(seg, dict) else getattr(seg, "start", 0)
    text = seg.get("text") if isinstance(seg, dict) else getattr(seg, "text", "")
    print(f"[{start:.1f}s]: {text}")

if os.path.exists(tmp_mp3):
    os.remove(tmp_mp3)
