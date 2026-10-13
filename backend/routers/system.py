import os
import json
from pathlib import Path
from fastapi import APIRouter, HTTPException, Response
from backend.config import (
    AVAILABLE_MODELS, CACHE_DIR, SAMPLES_DIR,
    DEFAULT_WINDOW_SECONDS, DEFAULT_STRIDE_SECONDS
)
from backend.schemas import GroqKeyRequest, AnnotationRequest
from backend.state import current_state, initialize_engine
from backend.audio import parse_srt_or_vtt, seconds_to_timestamp, get_saved_groq_key, save_groq_key
from backend.evaluation import evaluate_grounding_dataset, annotation_manager

router = APIRouter(tags=["System & Administration"])

@router.get("/api/status")
def get_status():
    """
    Returns server health, hardware execution profile, active model, and video details.
    """
    cache_files = list(CACHE_DIR.glob("*.npz"))
    return {
        "status": "ready",
        "cpu_device": "Intel Core i5-1135G7 (CPU Optimized)",
        "active_model": current_state["model_key"],
        "available_models": AVAILABLE_MODELS,
        "current_video": {
            "id": current_state["video_id"],
            "title": current_state["video_title"],
            "has_video_file": current_state["video_path"] is not None and os.path.exists(current_state["video_path"]),
            "num_subtitles": len(current_state["subtitles"]),
            "num_chunks": len(current_state["chunks"]),
            "num_keyframes": len(current_state.get("keyframes", [])),
            "total_duration": current_state["total_duration"],
            "formatted_duration": seconds_to_timestamp(current_state["total_duration"]),
            "window_size": current_state["window_size"],
            "stride_size": current_state["stride_size"]
        },
        "cache_count": len(cache_files)
    }

@router.get("/api/samples")
def list_samples():
    """
    Returns the list of available curated benchmark lectures.
    """
    return [
        {
            "key": "transformer_35_phut_vi",
            "title": "Tất Cả Về Transformer Trong 35 Phút (Tiếng Việt)",
            "lang": "vi",
            "recommended_model": "vietnamese-sbert",
            "description": "Bài giảng chuyên sâu về kiến trúc Transformer: Token, Embedding, Self-Attention, Multi-Head Attention, Encoder & Decoder."
        },
        {
            "key": "stanford_cs229_lecture_8",
            "title": "Stanford CS229: Data Splits, Models & Cross-Validation (English)",
            "lang": "en",
            "recommended_model": "all-MiniLM-L6-v2",
            "description": "Lecture 8 by Andrew Ng covering Train/Dev/Test splits, Bias/Variance, Regularization, and Hold-out & K-Fold Cross-Validation."
        },
        {
            "key": "machine_learning_intro_vi",
            "title": "Machine Learning: Overfitting & Kỹ Thuật Khắc Phục (Tiếng Việt)",
            "lang": "vi",
            "recommended_model": "vietnamese-sbert",
            "description": "Bài giảng chuyên sâu về Overfitting, Regularization L1/L2, Dropout, Loss Functions, Gradient Descent và Dataset Splitting."
        },
        {
            "key": "deep_learning_fundamentals_en",
            "title": "Deep Learning Fundamentals & CNNs (English)",
            "lang": "en",
            "recommended_model": "all-MiniLM-L6-v2",
            "description": "Lecture on Perceptron, Activation Functions (ReLU), Backpropagation, Gradient Clipping, and CNN layers."
        }
    ]

@router.get("/api/groq/config")
def get_groq_config():
    """
    Checks if a Groq Cloud API key is currently saved.
    """
    key = get_saved_groq_key()
    return {
        "has_key": bool(key),
        "masked_key": f"{key[:6]}...{key[-4:]}" if key and len(key) > 10 else None
    }

@router.post("/api/groq/config")
def update_groq_config(req: GroqKeyRequest):
    """
    Saves a Groq Cloud API key for ultra-fast LPU Whisper transcription.
    """
    key = req.api_key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="API key không được để trống.")
    save_groq_key(key)
    return {
        "success": True,
        "has_key": True,
        "masked_key": f"{key[:6]}...{key[-4:]}"
    }

@router.post("/api/benchmark/run")
def run_benchmark(top_k: int = 3, alpha: float = 0.7):
    """
    Executes automated academic benchmark measuring mIoU, R@1, R@3, and latency.
    """
    gt_file = SAMPLES_DIR / "benchmark_ground_truth.json"
    if not gt_file.exists():
        raise HTTPException(status_code=404, detail="Benchmark ground truth file not found.")

    with open(gt_file, "r", encoding="utf-8") as f:
        benchmark_items = json.load(f)

    if current_state["video_id"] != "machine_learning_intro_vi":
        sample_file = SAMPLES_DIR / "machine_learning_intro_vi.srt"
        subs = parse_srt_or_vtt(sample_file)
        initialize_engine(
            subtitles=subs,
            window=DEFAULT_WINDOW_SECONDS,
            stride=DEFAULT_STRIDE_SECONDS,
            model_key="vietnamese-sbert",
            video_id="machine_learning_intro_vi",
            title="Machine Learning: Overfitting & Kỹ Thuật Khắc Phục (Tiếng Việt)"
        )

    metrics = evaluate_grounding_dataset(
        search_engine=current_state["search_engine"],
        benchmark_items=benchmark_items,
        top_k=top_k,
        alpha=alpha
    )
    return metrics

@router.get("/api/annotations")
def get_annotations():
    return annotation_manager.load_annotations()

@router.post("/api/annotations")
def add_annotation(req: AnnotationRequest):
    item = annotation_manager.add_annotation(
        video_id=req.video_id,
        query=req.query,
        start=req.start,
        end=req.end,
        rating=req.rating,
        notes=req.notes or "",
        predicted_start=req.predicted_start,
        predicted_end=req.predicted_end
    )
    return {"success": True, "annotation": item}

@router.delete("/api/annotations/{ann_id}")
def delete_annotation(ann_id: str):
    success = annotation_manager.delete_annotation(ann_id)
    return {"success": success}

@router.get("/api/annotations/export")
def export_annotations():
    csv_data = annotation_manager.export_csv()
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=lecturelens_grounding_annotations.csv"}
    )
