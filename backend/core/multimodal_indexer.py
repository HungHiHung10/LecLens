import re
from typing import List, Dict, Any, Optional
from backend.audio.subtitle_parser import seconds_to_timestamp

def infer_topic_and_slide_metadata(text: str) -> Dict[str, str]:
    """
    Infers topic header and slide bullet points from lecture segment text,
    simulating OCR and slide structure (as in AIC2026 metadata preprocessing).
    """
    lower = text.lower()
    
    if any(k in lower for k in ["overfitting", "quá khớp", "ghi nhớ nhiễu", "tập test"]):
        return {
            "slide_title": "Hiện Tượng Quá Khớp (Overfitting) & Nguyên Nhân",
            "ocr_keywords": "Overfitting, High Variance, Train Error vs Test Error, Noise",
            "concept_category": "Model Generalization"
        }
    elif any(k in lower for k in ["underfitting", "chưa khớp", "mô hình quá đơn giản"]):
        return {
            "slide_title": "Hiện Tượng Chưa Khớp (Underfitting)",
            "ocr_keywords": "Underfitting, High Bias, Linear Model, Non-linear data",
            "concept_category": "Model Generalization"
        }
    elif any(k in lower for k in ["bias", "variance", "độ chệch", "phương sai"]):
        return {
            "slide_title": "Sự Đánh Đổi Giữa Bias và Variance (Bias-Variance Tradeoff)",
            "ocr_keywords": "Bias vs Variance, Tradeoff, Total Error = Bias² + Variance + Irreducible",
            "concept_category": "Statistical Theory"
        }
    elif any(k in lower for k in ["l1", "lasso", "l2", "ridge", "weight decay", "regularization", "điều chuẩn"]):
        return {
            "slide_title": "Kỹ Thuật Điều Chuẩn (L1 Lasso & L2 Ridge Regularization)",
            "ocr_keywords": "Regularization, L1 Norm, L2 Weight Decay, Sparsity, Feature Selection",
            "concept_category": "Regularization"
        }
    elif any(k in lower for k in ["dropout", "tắt nơ-ron", "tắt ngẫu nhiên", "co-adaptation", "hinton"]):
        return {
            "slide_title": "Kỹ Thuật Dropout Trong Mạng Nơ-ron Sâu",
            "ocr_keywords": "Dropout, Probability p=0.5, Co-adaptation, Test Mode (1-p) scaling",
            "concept_category": "Deep Learning Architecture"
        }
    elif any(k in lower for k in ["early stopping", "dừng sớm", "patience", "validation loss"]):
        return {
            "slide_title": "Early Stopping — Dừng Huấn Luyện Sớm",
            "ocr_keywords": "Early Stopping, Val Loss Monitoring, Patience, Checkpoint Recovery",
            "concept_category": "Training Optimization"
        }
    elif any(k in lower for k in ["data augmentation", "tăng cường dữ liệu", "xoay ảnh", "lật ảnh"]):
        return {
            "slide_title": "Tăng Cường Dữ Liệu (Data Augmentation)",
            "ocr_keywords": "Data Augmentation, Image Flips, Rotation, Scaling, Brightness",
            "concept_category": "Data Engineering"
        }
    elif any(k in lower for k in ["loss", "mất mát", "mse", "mae", "cross-entropy", "softmax"]):
        return {
            "slide_title": "Hàm Mất Mát (Loss Functions) & Sai Số Dự Đoán",
            "ocr_keywords": "Loss Function, MSE, MAE (Regression), Cross-Entropy Loss, Softmax",
            "concept_category": "Loss Functions"
        }
    elif any(k in lower for k in ["gradient descent", "hạ độ dốc", "đạo hàm", "lan truyền ngược"]):
        return {
            "slide_title": "Thuật Toán Tối Ưu Hạ Độ Dốc (Gradient Descent)",
            "ocr_keywords": "Gradient Descent, Learning Rate alpha, Backpropagation, Gradients",
            "concept_category": "Optimization"
        }
    elif any(k in lower for k in ["learning rate", "tốc độ học", "adam", "rmsprop", "momentum"]):
        return {
            "slide_title": "Tốc Độ Học (Learning Rate) & Các Bộ Tối Ưu Hiện Đại",
            "ocr_keywords": "Learning Rate, Local Minima, Saddle Points, Adam, RMSprop, Momentum",
            "concept_category": "Optimization"
        }
    elif any(k in lower for k in ["batch normalization", "internal covariate shift", "chuẩn hóa"]):
        return {
            "slide_title": "Batch Normalization & Tăng Tốc Huấn Luyện",
            "ocr_keywords": "Batch Normalization, Mean 0, Variance 1, Internal Covariate Shift",
            "concept_category": "Neural Layer Normalization"
        }
    elif any(k in lower for k in ["train", "validation", "test", "data leakage", "rò rỉ dữ liệu"]):
        return {
            "slide_title": "Phân Chia Tập Dữ Liệu: Train, Validation & Test Set",
            "ocr_keywords": "Train Set, Validation Set, Test Set, Data Leakage Prevention",
            "concept_category": "Machine Learning Workflow"
        }
    else:
        return {
            "slide_title": "Nội Dung Bài Giảng",
            "ocr_keywords": "Lecture Notes, Concept Discussion",
            "concept_category": "General Topic"
        }

def enrich_chunks_with_multimodal_metadata(
    chunks: List[Dict],
    keyframes: Optional[List[Dict]] = None,
    hedge_padding_seconds: float = 4.0
) -> List[Dict]:
    """
    Enriches basic subtitle chunks with multimodal fields:
    - asr_context_text
    - slide_title (from real OCR or heuristic)
    - ocr_keywords (from real OCR or heuristic)
    - ocr_text (from slide image OCR)
    - keyframe_url (thumbnail of the actual lecture slide)
    - keyframe_timestamp / keyframe_timestamp_str
    - semantic_metadata_text (fused text for hybrid retrieval)
    - temporal_hedge (buffered boundaries for smooth playback)
    """
    enriched = []
    total_chunks = len(chunks)

    # Build quick lookup helper if keyframes provided
    def get_kf_for_time(ts: float):
        if not keyframes:
            return None
        matched = keyframes[0]
        for kf in keyframes:
            if kf["timestamp"] <= ts + 2.0:
                matched = kf
            else:
                break
        return matched

    for i, c in enumerate(chunks):
        item = dict(c)
        mid_sec = (c["start"] + c["end"]) / 2.0
        kf = get_kf_for_time(mid_sec)

        # 1. Fallback heuristic metadata
        meta = infer_topic_and_slide_metadata(c["text"])
        slide_title = meta["slide_title"]
        ocr_keywords = meta["ocr_keywords"]
        ocr_text = ""
        keyframe_url = None
        keyframe_ts = None
        keyframe_ts_str = None

        # 2. Real Keyframe & OCR override if available
        if kf and (not kf.get("slide_title") or kf["slide_title"].startswith("Slide ")):
            # Look within chunk window for an informative slide
            informative = [
                k for k in keyframes
                if (c["start"] - 5.0 <= k["timestamp"] <= c["end"] + 15.0)
                and k.get("slide_title") and not k.get("slide_title", "").startswith("Slide ")
            ]
            if informative:
                kf = informative[0]

        if kf:
            keyframe_url = kf.get("image_url")
            keyframe_ts = kf.get("timestamp")
            keyframe_ts_str = kf.get("timestamp_str")
            if kf.get("slide_title") and not kf["slide_title"].startswith("Slide "):
                slide_title = kf["slide_title"]
            if kf.get("ocr_keywords") and kf["ocr_keywords"] not in ("Slide Content", "Nội dung Slide"):
                ocr_keywords = ", ".join(kf["ocr_keywords"]) if isinstance(kf["ocr_keywords"], list) else kf["ocr_keywords"]
            ocr_text = kf.get("ocr_text", "")

        item["slide_title"] = slide_title
        item["ocr_keywords"] = ocr_keywords
        item["ocr_text"] = ocr_text
        item["keyframe_url"] = keyframe_url
        item["keyframe_timestamp"] = keyframe_ts
        item["keyframe_timestamp_str"] = keyframe_ts_str
        item["concept_category"] = meta["concept_category"]

        # Hedged playback boundaries
        hedged_start = max(0.0, round(c["start"] - hedge_padding_seconds, 2))
        hedged_end = round(c["end"] + hedge_padding_seconds, 2)
        item["hedged_start"] = hedged_start
        item["hedged_end"] = hedged_end
        item["hedged_start_str"] = seconds_to_timestamp(hedged_start)
        item["hedged_end_str"] = seconds_to_timestamp(hedged_end)

        # Combined semantic text (ASR speech + Slide Title + OCR text)
        if ocr_text:
            item["semantic_metadata_text"] = (
                f"Slide: {slide_title}. "
                f"Nội dung hình ảnh & OCR: {ocr_text}. "
                f"Lời giảng: {c['text']}"
            )
        else:
            item["semantic_metadata_text"] = (
                f"Chủ đề: {slide_title}. "
                f"Từ khóa trên slide: {ocr_keywords}. "
                f"Lời giảng: {c['text']}"
            )

        item["temporal_segment_id"] = i
        enriched.append(item)

    return enriched
