import json
from pathlib import Path
from backend.vision.vietnamese_text_enhancer import VietnameseTextEnhancer

srt_path = "data/uploads/custom_1791461236.mp4.srt"
enhancer = VietnameseTextEnhancer(srt_path=srt_path)

meta_file = Path("data/keyframes/custom_1791461236/keyframes_metadata.json")
if meta_file.exists():
    meta = json.load(open(meta_file, "r", encoding="utf-8"))
    for idx, kf in enumerate(meta):
        lines = [l.strip() for l in kf.get("ocr_text", "").split("|") if l.strip()]
        ts_str = kf.get("timestamp_str", "00:00")
        default_title = f"Slide {idx + 1} ({ts_str})"
        kf["slide_title"] = enhancer.detect_title(lines, default_title)
        kw = enhancer.extract_keywords(lines, max_keywords=8)
        kf["ocr_keywords"] = ", ".join(kw) if kw else "Nội dung Slide"

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Updated metadata for {len(meta)} keyframes successfully!")
