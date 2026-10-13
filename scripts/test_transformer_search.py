import sys
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from backend.audio.subtitle_parser import parse_srt_or_vtt
from backend.audio.chunker import create_temporal_chunks
from backend.core.searcher import LectureSearchEngine

subs = parse_srt_or_vtt("a:/LecLens/data/uploads/custom_1791384742.mp4.srt")
chunks = create_temporal_chunks(subs, window_seconds=45, stride_seconds=20)

query = "Cơ chế Self-Attention hoạt động như thế nào?"

for m in ["all-MiniLM-L6-v2", "vietnamese-sbert"]:
    engine = LectureSearchEngine(chunks, model_key=m)
    res = engine.search(query, top_k=2)
    print(f"\n=== Model: {m} ===")
    for r in res:
        print(f"[{r['start_str']} -> {r['end_str']}] (Score: {r['score']:.3f})")
        print(f"Text: {r['text'][:140]}...\n")
