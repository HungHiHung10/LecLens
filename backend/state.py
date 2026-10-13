import os
from typing import List, Dict, Optional
from backend.config import (
    DEFAULT_MODEL_KEY, DEFAULT_WINDOW_SECONDS,
    DEFAULT_STRIDE_SECONDS, get_logger
)
from backend.audio.chunker import create_temporal_chunks
from backend.core.multimodal_indexer import enrich_chunks_with_multimodal_metadata
from backend.core.searcher import LectureSearchEngine
from backend.core.video_grounding_rag import VideoGroundingRAG
from backend.vision.vision_extractor import vision_extractor

logger = get_logger("state")

# Active session state in memory
current_state = {
    "video_id": "transformer_35_phut_vi",
    "video_title": "Tất Cả Về Transformer Trong 35 Phút (Tiếng Việt)",
    "video_path": None,
    "model_key": DEFAULT_MODEL_KEY,
    "window_size": DEFAULT_WINDOW_SECONDS,
    "stride_size": DEFAULT_STRIDE_SECONDS,
    "subtitles": [],
    "chunks": [],
    "keyframes": [],
    "search_engine": None,
    "rag_engine": None,
    "total_duration": 0.0
}

def initialize_engine(
    subtitles: List[Dict],
    window: float,
    stride: float,
    model_key: str,
    video_id: str,
    title: str,
    video_path: Optional[str] = None
) -> List[Dict]:
    """
    Initializes or updates the in-memory engine state with:
    1. Vision & Slide Keyframe OCR extraction.
    2. Temporal Sliding-Window chunking.
    3. Multimodal metadata binding (Speech + Slide Title + Slide Text).
    4. Hybrid Searcher (Dense SBERT + Sparse BM25) and Grounding RAG.
    """
    # 1. Vision & Slide OCR extraction (cached if already processed)
    keyframes = []
    if video_path and os.path.exists(video_path):
        try:
            keyframes = vision_extractor.extract_slide_keyframes(video_path, video_id=video_id)
        except Exception as kf_err:
            logger.warning(f"Vision extraction error: {kf_err}")

    raw_chunks = create_temporal_chunks(subtitles, window_seconds=window, stride_seconds=stride)
    chunks = enrich_chunks_with_multimodal_metadata(raw_chunks, keyframes=keyframes)
    engine = LectureSearchEngine(chunks, model_key=model_key)
    rag = VideoGroundingRAG(search_engine=engine, enriched_chunks=chunks)
    total_dur = subtitles[-1]["end"] if subtitles else 0.0

    current_state["video_id"] = video_id
    current_state["video_title"] = title
    current_state["video_path"] = video_path
    current_state["model_key"] = model_key
    current_state["window_size"] = window
    current_state["stride_size"] = stride
    current_state["subtitles"] = subtitles
    current_state["chunks"] = chunks
    current_state["keyframes"] = keyframes
    current_state["search_engine"] = engine
    current_state["rag_engine"] = rag
    current_state["total_duration"] = total_dur
    return chunks
