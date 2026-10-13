import os
import time
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse

from backend.config import (
    BASE_DIR, SAMPLES_DIR, UPLOADS_DIR, KEYFRAMES_DIR,
    DEFAULT_WINDOW_SECONDS, DEFAULT_STRIDE_SECONDS, get_logger
)
from backend.schemas import LoadSampleRequest, YouTubeLoadRequest
from backend.state import current_state, initialize_engine
from backend.audio import (
    parse_srt_or_vtt, seconds_to_timestamp,
    fetch_youtube_transcript, transcribe_local_video, extract_youtube_video_id,
    transcribe_with_groq, get_saved_groq_key
)

logger = get_logger("media")
router = APIRouter(tags=["Media & Video Ingestion"])

@router.post("/api/load-sample")
def load_sample(payload: LoadSampleRequest):
    """
    Loads pre-indexed lecture samples or custom processed lectures.
    """
    sample_key = payload.sample_key
    srt_path = SAMPLES_DIR / f"{sample_key}.srt"
    vid_path = None
    title = sample_key
    if sample_key.startswith("custom_"):
        upload_srt = UPLOADS_DIR / f"{sample_key}.mp4.srt"
        upload_mp4 = UPLOADS_DIR / f"{sample_key}.mp4"
        if upload_srt.exists():
            srt_path = upload_srt
            vid_path = str(upload_mp4) if upload_mp4.exists() else None
            title = "Bài giảng Địa lý - Ôn thi THPT (Thầy Trần Ngọc Anh)"

    if not srt_path.exists():
        raise HTTPException(status_code=404, detail="Sample lecture file not found.")

    subs = parse_srt_or_vtt(srt_path)
    model_k = payload.model_key or ("vietnamese-sbert" if "vi" in sample_key or sample_key.startswith("custom_") else "all-MiniLM-L6-v2")
    
    titles = {
        "transformer_35_phut_vi": "Tất Cả Về Transformer Trong 35 Phút (Tiếng Việt)",
        "stanford_cs229_lecture_8": "Stanford CS229: Lecture 8 — Data Splits, Models & Cross-Validation",
        "machine_learning_intro_vi": "Machine Learning: Overfitting & Kỹ Thuật Khắc Phục (Tiếng Việt)",
        "deep_learning_fundamentals_en": "Deep Learning Fundamentals & CNNs (English)"
    }
    title = titles.get(sample_key, sample_key)

    # Check if physical video exists locally on disk
    if not vid_path:
        if sample_key == "transformer_35_phut_vi":
            local_mp4 = BASE_DIR / "Tất cả những gì bạn cần biết về Transformer trong 35 phút.mp4"
            if local_mp4.exists():
                vid_path = str(local_mp4)
        elif sample_key == "stanford_cs229_lecture_8":
            local_mp4 = BASE_DIR / "Lecture-8-Data-Splits-Models-Cross-Validation-Stanford-CS229-Machine-Learning-Au.mp4"
            if local_mp4.exists():
                vid_path = str(local_mp4)
        elif (SAMPLES_DIR / f"{sample_key}.mp4").exists():
            vid_path = str(SAMPLES_DIR / f"{sample_key}.mp4")

    chunks = initialize_engine(
        subtitles=subs,
        window=payload.window_size or DEFAULT_WINDOW_SECONDS,
        stride=payload.stride_size or DEFAULT_STRIDE_SECONDS,
        model_key=model_k,
        video_id=sample_key,
        title=title,
        video_path=vid_path
    )

    return {
        "success": True,
        "video_id": sample_key,
        "title": title,
        "has_video": vid_path is not None,
        "num_subtitles": len(subs),
        "num_chunks": len(chunks),
        "total_duration": current_state["total_duration"],
        "formatted_duration": seconds_to_timestamp(current_state["total_duration"])
    }

@router.post("/api/youtube/load")
def load_youtube_video(req: YouTubeLoadRequest):
    """
    Fetches transcripts from YouTube video and indexes them directly into search memory.
    """
    try:
        subtitles = fetch_youtube_transcript(req.url)
        vid_id = extract_youtube_video_id(req.url) or "youtube_lecture"
        title = f"YouTube ({vid_id})"
        
        # Auto-detect language
        sample_str = " ".join([s["text"] for s in subtitles[:15]]).lower()
        vn_chars = "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ"
        is_vi = any(c in sample_str for c in vn_chars)
        model_k = req.model_key or ("vietnamese-sbert" if is_vi else "all-MiniLM-L6-v2")
        
        chunks = initialize_engine(
            subtitles=subtitles,
            window=DEFAULT_WINDOW_SECONDS,
            stride=DEFAULT_STRIDE_SECONDS,
            model_key=model_k,
            video_id=vid_id,
            title=title
        )
        current_state["youtube_id"] = vid_id
        return {
            "success": True,
            "video_id": vid_id,
            "title": title,
            "youtube_id": vid_id,
            "num_subtitles": len(subtitles),
            "num_chunks": len(chunks),
            "total_duration": current_state["total_duration"],
            "formatted_duration": seconds_to_timestamp(current_state["total_duration"])
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/api/upload")
async def upload_files(
    video_file: Optional[UploadFile] = File(None),
    subtitle_file: Optional[UploadFile] = File(None),
    title: Optional[str] = Form(None),
    model_key: Optional[str] = Form(None),
    groq_api_key: Optional[str] = Form(None),
    window_size: Optional[float] = Form(DEFAULT_WINDOW_SECONDS),
    stride_size: Optional[float] = Form(DEFAULT_STRIDE_SECONDS)
):
    """
    Accepts video MP4 file and/or subtitle file.
    Performs speech-to-text (Groq Whisper or CPU Faster-Whisper) and keyframe OCR.
    """
    if not subtitle_file and not video_file:
        raise HTTPException(status_code=400, detail="Vui lòng cung cấp ít nhất 1 file video (.mp4) hoặc phụ đề.")

    video_path = None
    video_id = f"custom_{int(time.time())}"
    raw_filename = video_file.filename if video_file and video_file.filename else ""
    stem_name = Path(raw_filename).stem if raw_filename else ""
    
    if not title:
        title = stem_name.replace("-", " ").replace("_", " ") if stem_name else "Custom Lecture Video"

    # 1. Save video file if uploaded
    if video_file:
        video_save_dir = BASE_DIR / "data" / "uploads"
        video_save_dir.mkdir(parents=True, exist_ok=True)
        ext = Path(video_file.filename).suffix or ".mp4"
        save_path = video_save_dir / f"{video_id}{ext}"
        with open(save_path, "wb") as f:
            v_bytes = await video_file.read()
            f.write(v_bytes)
        video_path = str(save_path)

    # 2. Extract Subtitles:
    subtitles = []
    if subtitle_file:
        content_bytes = await subtitle_file.read()
        sub_text = content_bytes.decode("utf-8", errors="replace")
        subtitles = parse_srt_or_vtt(sub_text)
    else:
        matching_srt = None
        candidates = [
            SAMPLES_DIR / f"{stem_name}.srt",
            BASE_DIR / f"{stem_name}.srt",
            BASE_DIR / "data" / "uploads" / f"{stem_name}.srt"
        ]
        if any(kw in stem_name.lower() for kw in ["lecture-8", "cs229", "cross-validation", "data-splits"]):
            candidates.insert(0, SAMPLES_DIR / "stanford_cs229_lecture_8.srt")
        elif any(kw in stem_name.lower() for kw in ["transformer", "35 phút", "35-phut", "35phut"]):
            candidates.insert(0, SAMPLES_DIR / "transformer_35_phut_vi.srt")

        for c in candidates:
            if c and c.exists():
                matching_srt = c
                break

        if matching_srt:
            logger.info(f"Found pre-indexed subtitle match: {matching_srt.name}")
            subtitles = parse_srt_or_vtt(matching_srt)
        elif video_path:
            effective_key = (groq_api_key or "").strip() or get_saved_groq_key()
            if effective_key:
                logger.info("Nhận diện Groq API Key! Khởi chạy Groq Cloud LPU Whisper (5s)...")
                try:
                    subtitles = transcribe_with_groq(video_path, api_key=effective_key)
                except Exception as groq_err:
                    logger.warning(f"Groq Whisper error ({groq_err}), chuyển sang Faster-Whisper CPU...")
                    subtitles = transcribe_local_video(video_path, model_size="tiny")
            else:
                logger.info("Không có Groq API key, chạy Faster-Whisper trên CPU...")
                subtitles = transcribe_local_video(video_path, model_size="tiny")

            if subtitles:
                try:
                    srt_blocks = []
                    for i, cue in enumerate(subtitles, 1):
                        srt_blocks.append(f"{i}\n{cue['start_str']} --> {cue['end_str']}\n{cue['text']}")
                    Path(f"{video_path}.srt").write_text("\n\n".join(srt_blocks), encoding="utf-8")
                except Exception as save_err:
                    print(f"[Upload] Cache srt warning: {save_err}")
        else:
            raise HTTPException(status_code=400, detail="Không có dữ liệu âm thanh hoặc video.")

    if not subtitles:
        raise HTTPException(status_code=400, detail="Không thể trích xuất phụ đề từ video. Vui lòng kiểm tra lại file.")

    if not model_key:
        sample_str = " ".join([s["text"] for s in subtitles[:15]]).lower()
        vn_chars = "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ"
        is_vi = any(c in sample_str for c in vn_chars)
        model_key = "vietnamese-sbert" if is_vi else "all-MiniLM-L6-v2"

    chunks = initialize_engine(
        subtitles=subtitles,
        window=window_size or DEFAULT_WINDOW_SECONDS,
        stride=stride_size or DEFAULT_STRIDE_SECONDS,
        model_key=model_key,
        video_id=video_id,
        title=title,
        video_path=video_path
    )

    return {
        "success": True,
        "video_id": video_id,
        "title": title,
        "has_video": video_path is not None,
        "num_subtitles": len(subtitles),
        "num_chunks": len(chunks),
        "total_duration": current_state["total_duration"],
        "formatted_duration": seconds_to_timestamp(current_state["total_duration"])
    }

@router.get("/api/video-stream/{video_id}")
def stream_video(video_id: str):
    """
    Video Stream endpoint with HTTP 206 Partial Content support for smooth HTML5 seeking.
    """
    if not current_state["video_path"] or not os.path.exists(current_state["video_path"]):
        raise HTTPException(status_code=404, detail="No physical video file uploaded for this session.")
    return FileResponse(current_state["video_path"], media_type="video/mp4")

@router.get("/api/keyframes")
def get_keyframes():
    """
    Returns all slide keyframes and OCR metadata for the active lecture.
    """
    return {
        "video_id": current_state["video_id"],
        "count": len(current_state.get("keyframes", [])),
        "keyframes": current_state.get("keyframes", [])
    }
