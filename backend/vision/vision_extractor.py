import sys
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image
import av

from backend.config import (
    KEYFRAMES_DIR,
    DEFAULT_KEYFRAME_INTERVAL_SEC,
    DEFAULT_SCENE_DIFF_THRESHOLD,
    DEFAULT_MAX_KEYFRAMES,
    get_logger
)
from backend.audio.subtitle_parser import seconds_to_timestamp
from backend.vision.vietnamese_text_enhancer import VietnameseTextEnhancer

logger = get_logger("vision")

class VisionExtractor:
    """
    Lightweight, CPU-optimized Multimodal Vision & OCR Branch:
    1. Keyframe Extraction: Detects slide transitions via lightweight luminance diffs using PyAV.
    2. Slide OCR: High-precision text & diagram keyword extraction via RapidOCR (ONNX Runtime CPU).
    3. Multimodal Indexing: Fuses slide visual context, titles, and diagrams into temporal chunks.
    """
    def __init__(self):
        self._ocr_engine = None

    def _get_ocr(self):
        if self._ocr_engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
                self._ocr_engine = RapidOCR()
            except Exception as e:
                logger.warning(f"RapidOCR initialization notice: {e}")
                self._ocr_engine = False
        return self._ocr_engine

    def extract_slide_keyframes(
        self,
        video_path: str,
        video_id: str,
        sample_interval_sec: float = DEFAULT_KEYFRAME_INTERVAL_SEC,
        diff_threshold: float = DEFAULT_SCENE_DIFF_THRESHOLD,
        max_keyframes: int = DEFAULT_MAX_KEYFRAMES,
        force_recompute: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Extracts unique slide keyframes from video and performs OCR on each slide.
        Results are cached to disk so future loads are instantaneous (< 10ms).
        """
        video_dir = KEYFRAMES_DIR / video_id
        meta_file = video_dir / "keyframes_metadata.json"

        # 1. Return cached metadata if available
        if not force_recompute and meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    cached_keyframes = json.load(f)
                if cached_keyframes:
                    logger.info(f"Loaded {len(cached_keyframes)} cached keyframes for '{video_id}'")
                    return cached_keyframes
            except Exception as e:
                logger.warning(f"Cache read error for '{video_id}': {e}")

        if not os.path.exists(video_path):
            logger.warning(f"Video path does not exist: {video_path}")
            return []

        video_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        logger.info(f"Bắt đầu trích xuất Slide Keyframes từ: {Path(video_path).name}...")

        # 2. Fast Keyframe Sampling with PyAV
        container = av.open(video_path)
        stream = container.streams.video[0]
        time_base = stream.time_base
        duration = float(stream.duration * time_base) if stream.duration else 2300.0

        timestamps = list(np.arange(5.0, duration, sample_interval_sec))
        candidate_slides = []
        prev_thumb = None

        for sec in timestamps:
            try:
                pts = int(sec / time_base)
                container.seek(pts, stream=stream)
                for frame in container.decode(stream):
                    img = frame.to_image()
                    # 64x64 grayscale thumbnail for fast MSE diff
                    thumb = np.array(img.resize((64, 64)).convert("L"), dtype=np.float32)
                    if prev_thumb is None:
                        candidate_slides.append((sec, img))
                        prev_thumb = thumb
                    else:
                        diff = np.mean(np.abs(thumb - prev_thumb)) / 255.0
                        if diff >= diff_threshold:
                            candidate_slides.append((sec, img))
                            prev_thumb = thumb
                    break
            except Exception as seek_err:
                continue

        container.close()

        # Limit to max_keyframes to prevent memory/CPU bloat
        if len(candidate_slides) > max_keyframes:
            step = len(candidate_slides) / max_keyframes
            candidate_slides = [candidate_slides[int(i * step)] for i in range(max_keyframes)]

        logger.info(f"Đã phát hiện {len(candidate_slides)} slide keyframes trong {round(time.perf_counter() - t0, 2)}s.")

        # 3. Perform OCR & Save Slide Thumbnails
        ocr = self._get_ocr()
        keyframes = []

        # Find matching .srt subtitles if available for Vietnamese enhancement
        possible_srt = Path(video_path).with_suffix(Path(video_path).suffix + ".srt")
        srt_path = str(possible_srt) if possible_srt.exists() else None
        if not srt_path:
            alt_srt = Path(video_path).with_suffix(".srt")
            if alt_srt.exists():
                srt_path = str(alt_srt)
        enhancer = VietnameseTextEnhancer(srt_path=srt_path)

        for idx, (sec, img) in enumerate(candidate_slides):
            img_filename = f"slide_{idx:03d}_{int(sec)}s.jpg"
            img_save_path = video_dir / img_filename

            # Resize to web-friendly 960x540 if larger
            w, h = img.size
            if w > 960:
                new_h = int(h * (960.0 / w))
                save_img = img.resize((960, new_h), Image.Resampling.BILINEAR)
            else:
                save_img = img

            save_img.save(img_save_path, "JPEG", quality=82)

            # Run OCR on the keyframe
            ocr_text = ""
            slide_title = ""
            detected_keywords = []

            if ocr:
                try:
                    ocr_res, _ = ocr(str(img_save_path))
                    if ocr_res:
                        lines = [line[1].strip() for line in ocr_res if line[2] >= 0.4 and len(line[1].strip()) > 1]
                        ocr_text = " | ".join(lines)
                        
                        # Detect authentic Vietnamese slide title bypassing watermarks
                        default_fallback_title = f"Slide {idx + 1} ({seconds_to_timestamp(sec)})"
                        slide_title = enhancer.detect_title(lines, default_fallback_title)
                        
                        # Extract clean, spaced, accented Vietnamese keywords
                        detected_keywords = enhancer.extract_keywords(lines, max_keywords=8)
                except Exception as ocr_err:
                    logger.warning(f"OCR error on slide {idx}: {ocr_err}")

            if not slide_title:
                slide_title = f"Slide {idx + 1} ({seconds_to_timestamp(sec)})"

            kf_record = {
                "keyframe_id": f"{video_id}_kf_{idx}",
                "timestamp": round(float(sec), 2),
                "timestamp_str": seconds_to_timestamp(sec),
                "image_filename": img_filename,
                "image_url": f"/keyframes/{video_id}/{img_filename}",
                "slide_title": slide_title,
                "ocr_text": ocr_text,
                "ocr_keywords": ", ".join(detected_keywords[:10]) if detected_keywords else "Slide Content"
            }
            keyframes.append(kf_record)

        # 4. Save metadata to cache
        try:
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(keyframes, f, ensure_ascii=False, indent=2)
            logger.info(f"Đã lưu cache {len(keyframes)} keyframes vào: {meta_file.name}")
        except Exception as cache_err:
            logger.warning(f"Lưu cache keyframes thất bại: {cache_err}")

        total_time = round(time.perf_counter() - t0, 2)
        logger.info(f"Hoàn tất trích xuất Vision & OCR trong {total_time}s.")
        return keyframes

    def get_keyframe_for_timestamp(self, keyframes: List[Dict], timestamp_sec: float) -> Optional[Dict]:
        """
        Finds the active slide keyframe corresponding to a specific lecture timestamp.
        """
        if not keyframes:
            return None

        # Binary search or scan for latest keyframe <= timestamp_sec + 2.0s
        matched_kf = keyframes[0]
        for kf in keyframes:
            if kf["timestamp"] <= timestamp_sec + 2.0:
                matched_kf = kf
            else:
                break
        return matched_kf

    def enrich_chunks_with_vision(self, chunks: List[Dict], keyframes: List[Dict]) -> List[Dict]:
        """
        Fuses slide keyframes and OCR text into each temporal chunk.
        """
        if not keyframes:
            return chunks

        enriched = []
        for c in chunks:
            item = dict(c)
            # Find keyframe at midpoint of chunk
            mid_sec = (c["start"] + c["end"]) / 2.0
            kf = self.get_keyframe_for_timestamp(keyframes, mid_sec)

            if kf:
                item["keyframe_url"] = kf["image_url"]
                item["keyframe_timestamp"] = kf["timestamp"]
                item["keyframe_timestamp_str"] = kf["timestamp_str"]
                
                # If chunk didn't have an accurate slide title, use OCR title
                if not item.get("slide_title") or item.get("slide_title") == "Nội Dung Bài Giảng":
                    item["slide_title"] = kf["slide_title"]
                
                if not item.get("ocr_keywords") or item.get("ocr_keywords") == "Lecture Notes, Concept Discussion":
                    item["ocr_keywords"] = kf["ocr_keywords"]

                item["ocr_text"] = kf.get("ocr_text", "")
                
                # Augment semantic retrieval text with Slide OCR!
                # This ensures queries for diagrams/equations/slide bullets match instantly
                if item["ocr_text"]:
                    item["semantic_metadata_text"] = (
                        f"Slide: {item['slide_title']}. "
                        f"Nội dung hình ảnh & OCR: {item['ocr_text']}. "
                        f"Lời giảng viên: {item['text']}"
                    )
            enriched.append(item)

        return enriched

# Global singleton
vision_extractor = VisionExtractor()
