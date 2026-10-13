import re
from typing import Dict, List, Any

# Technical synonyms and Vietnamese <-> English mapping for lecture grounding
KEYWORD_SYNONYMS = {
    "overfitting": ["quá khớp", "overfitting", "high variance", "ghi nhớ dữ liệu"],
    "underfitting": ["chưa khớp", "underfitting", "high bias"],
    "regularization": ["điều chuẩn", "regularization", "chuẩn hóa trọng số", "phạt trọng số"],
    "l1": ["l1 regularization", "lasso", "l1", "mô hình thưa", "trọng số bằng 0"],
    "l2": ["l2 regularization", "ridge", "weight decay", "l2"],
    "dropout": ["dropout", "tắt nơ-ron", "tắt ngẫu nhiên", "co-adaptation"],
    "early stopping": ["dừng sớm", "early stopping", "patience", "validation loss"],
    "loss": ["hàm mất mát", "loss function", "sai số", "hàm loss", "chi phí"],
    "cross entropy": ["cross-entropy", "cross entropy", "phân loại", "softmax", "sigmoid"],
    "gradient descent": ["hạ độ dốc", "gradient descent", "đạo hàm", "lan truyền ngược", "backprop"],
    "learning rate": ["tốc độ học", "learning rate", "hệ số học", "alpha", "bước đi"],
    "batch norm": ["batch normalization", "chuẩn hóa theo batch", "internal covariate shift"],
    "train test": ["chia dữ liệu", "train test split", "validation set", "tập kiểm thử", "data leakage"],
    "data augmentation": ["tăng cường dữ liệu", "data augmentation", "xoay ảnh", "lật ảnh"]
}

VISUAL_GROUNDING_CUES = [
    "đồ thị", "biểu đồ", "hình vẽ", "slide", "công thức", "phương trình", "code", 
    "bảng", "trên bảng", "minh họa", "curve", "plot", "chart", "figure", "equation"
]

class MultimodalQueryParser:
    """
    Decomposes user query into structured retrieval components, inspired by
    AIC 2026 Textual KIS Query Retrieval notebook:
    - Core query
    - Intent type (explanation, definition, comparison, example, timestamp)
    - Grounding visual cues (diagrams, equations, slides)
    - Speech / ASR keywords
    - OCR / Slide keywords
    - Multi-query expansion variants
    """
    def __init__(self):
        pass

    def parse_query(self, raw_query: str) -> Dict[str, Any]:
        query_clean = raw_query.strip()
        lower_q = query_clean.lower()

        # 1. Detect Intent
        intent = "general_search"
        if any(w in lower_q for w in ["là gì", "định nghĩa", "khái niệm", "what is"]):
            intent = "definition"
        elif any(w in lower_q for w in ["tại sao", "vì sao", "nguyên nhân", "how", "why"]):
            intent = "explanation"
        elif any(w in lower_q for w in ["khác nhau", "so sánh", "so với", "difference", "compare"]):
            intent = "comparison"
        elif any(w in lower_q for w in ["ở đoạn nào", "khi nào", "thời gian nào", "timestamp", "where"]):
            intent = "timestamp_lookup"
        elif any(w in lower_q for w in ["ví dụ", "minh họa", "example"]):
            intent = "example"

        # 2. Detect Visual Grounding Cues
        detected_visual_cues = []
        for cue in VISUAL_GROUNDING_CUES:
            if cue in lower_q:
                detected_visual_cues.append(cue)

        # 3. Extract Technical Speech & Slide Terms
        matched_technical_terms = []
        expanded_keywords = set()
        
        for canonical_term, variants in KEYWORD_SYNONYMS.items():
            if any(v in lower_q for v in variants):
                matched_technical_terms.append(canonical_term)
                expanded_keywords.update(variants)

        # 4. Tokenization for lexical matching
        tokens = [t for t in re.sub(r'[^\w\s]', ' ', lower_q).split() if len(t) > 1]

        # 5. Build Multi-Query Variants (inspired by AIC 2026 multi_query_retrieve)
        query_variants = [query_clean]

        # Variant 1: Pure technical keyword string
        if matched_technical_terms:
            variant_tech = " ".join(matched_technical_terms)
            if variant_tech not in query_variants:
                query_variants.append(variant_tech)

        # Variant 2: Cleaned intent-stripped core query
        cleaned_core = re.sub(r'(thầy|giảng viên|giải thích|ở đoạn nào|cho em hỏi|là gì|như thế nào|tại sao|vì sao|đoạn nào)', '', lower_q).strip()
        cleaned_core = re.sub(r'\s+', ' ', cleaned_core)
        if cleaned_core and cleaned_core not in query_variants:
            query_variants.append(cleaned_core)

        # Variant 3: Expanded synonyms query
        if expanded_keywords:
            variant_expanded = " ".join(list(expanded_keywords)[:5])
            if variant_expanded not in query_variants:
                query_variants.append(variant_expanded)

        return {
            "raw_query": query_clean,
            "core_query": cleaned_core or query_clean,
            "intent": intent,
            "visual_grounding_cues": detected_visual_cues,
            "technical_terms": matched_technical_terms,
            "speech_terms": list(expanded_keywords) if expanded_keywords else tokens[:5],
            "ocr_terms": matched_technical_terms,
            "query_variants": query_variants[:4] # Top 4 variants
        }

query_parser = MultimodalQueryParser()

def parse_grounding_query(query: str) -> Dict[str, Any]:
    """Helper functional wrapper for MultimodalQueryParser.parse_query"""
    return query_parser.parse_query(query)
