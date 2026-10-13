import re
import numpy as np
from typing import List, Dict, Tuple, Optional
from rank_bm25 import BM25Okapi
from backend.core.embedder import embedder_manager
from backend.config import DEFAULT_HYBRID_ALPHA, DEFAULT_TOP_K, DEFAULT_MODEL_KEY

def tokenize_text(text: str) -> List[str]:
    """
    Tokenizes text for BM25 search. Handles both Vietnamese syllables and English words.
    """
    clean = re.sub(r'[^\w\s]', ' ', text.lower())
    tokens = [t for t in clean.split() if len(t) > 1]
    return tokens

class LectureSearchEngine:
    """
    Hybrid Search Engine combining Dense Semantic Vectors with Sparse BM25.
    """
    def __init__(self, chunks: List[Dict], model_key: str = DEFAULT_MODEL_KEY):
        self.chunks = chunks
        self.model_key = model_key
        self.embeddings: Optional[np.ndarray] = None
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_corpus: List[List[str]] = []
        
        self._initialize_index()

    def _initialize_index(self):
        if not self.chunks:
            return

        # 1. Initialize BM25 with multimodal text (speech + OCR keywords + slide title)
        self.tokenized_corpus = [tokenize_text(c.get("semantic_metadata_text", c["text"])) for c in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # 2. Initialize / load cached embeddings
        self.embeddings = embedder_manager.embed_chunks(self.chunks, model_key=self.model_key)

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        alpha: float = DEFAULT_HYBRID_ALPHA
    ) -> List[Dict]:
        """
        Executes hybrid search and returns top-k ranked video segments.
        alpha: Weight for dense semantic search (0.0 = pure BM25, 1.0 = pure Semantic).
        """
        if not self.chunks or not query.strip():
            return []

        num_chunks = len(self.chunks)
        
        # 1. Dense Semantic Scoring (Cosine similarity)
        query_vec = embedder_manager.embed_query(query, model_key=self.model_key)
        dense_scores = np.dot(self.embeddings, query_vec) # Since vectors are L2-normalized
        # Clip to [0, 1] for sanity
        dense_scores = np.clip(dense_scores, 0.0, 1.0)

        # 2. Sparse BM25 Scoring
        query_tokens = tokenize_text(query)
        if query_tokens and self.bm25:
            bm25_raw_scores = np.array(self.bm25.get_scores(query_tokens))
            max_bm25 = np.max(bm25_raw_scores)
            min_bm25 = np.min(bm25_raw_scores)
            if max_bm25 > min_bm25:
                bm25_norm_scores = (bm25_raw_scores - min_bm25) / (max_bm25 - min_bm25)
            elif max_bm25 > 0:
                bm25_norm_scores = np.ones_like(bm25_raw_scores) * 0.5
            else:
                bm25_norm_scores = np.zeros_like(bm25_raw_scores)
        else:
            bm25_norm_scores = np.zeros(num_chunks)

        # 3. Hybrid Combination Score
        hybrid_scores = (alpha * dense_scores) + ((1.0 - alpha) * bm25_norm_scores)

        # 4. Rank indices descending
        ranked_indices = np.argsort(-hybrid_scores)[:top_k * 2] # take candidate pool

        results = []
        for rank, idx in enumerate(ranked_indices):
            chunk = self.chunks[idx]
            
            # Extract key matching words for evidence explanation
            chunk_tokens = set(self.tokenized_corpus[idx])
            matched_words = [tok for tok in query_tokens if tok in chunk_tokens]
            
            # Evidence explanation
            evidence_summary = (
                f"Độ khớp ngữ nghĩa: {dense_scores[idx]*100:.1f}%, "
                f"Từ khóa trùng khớp: {len(matched_words)} ({', '.join(matched_words[:4]) if matched_words else 'ngữ cảnh gần'})"
            )

            results.append({
                "rank": rank + 1,
                "chunk_id": chunk["chunk_id"],
                "start": chunk["start"],
                "end": chunk["end"],
                "duration": chunk["duration"],
                "start_str": chunk["start_str"],
                "end_str": chunk["end_str"],
                "text": chunk["text"],
                "score": round(float(hybrid_scores[idx]), 4),
                "dense_score": round(float(dense_scores[idx]), 4),
                "bm25_score": round(float(bm25_norm_scores[idx]), 4),
                "matched_keywords": matched_words,
                "evidence": evidence_summary,
                "slide_title": chunk.get("slide_title", "Nội Dung Bài Giảng"),
                "ocr_keywords": chunk.get("ocr_keywords", ""),
                "ocr_text": chunk.get("ocr_text", ""),
                "keyframe_url": chunk.get("keyframe_url"),
                "keyframe_timestamp": chunk.get("keyframe_timestamp"),
                "keyframe_timestamp_str": chunk.get("keyframe_timestamp_str")
            })

        # Apply Non-Maximum Suppression (NMS) / Temporal Merging to avoid redundant overlapping fragments in top-K
        filtered_results = self._temporal_nms(results, iou_threshold=0.6, top_k=top_k)
        for i, res in enumerate(filtered_results):
            res["rank"] = i + 1

        return filtered_results

    def _temporal_nms(self, candidates: List[Dict], iou_threshold: float = 0.6, top_k: int = 3) -> List[Dict]:
        """
        Suppresses redundant heavily overlapping temporal chunks so top-K presents diverse lecture moments.
        """
        selected = []
        for cand in candidates:
            keep = True
            for sel in selected:
                # Calculate 1D temporal IoU
                inter_start = max(cand["start"], sel["start"])
                inter_end = min(cand["end"], sel["end"])
                inter = max(0.0, inter_end - inter_start)
                
                union = (cand["end"] - cand["start"]) + (sel["end"] - sel["start"]) - inter
                iou = inter / union if union > 0 else 0.0
                
                if iou >= iou_threshold:
                    keep = False
                    break
            
            if keep:
                selected.append(cand)
            if len(selected) >= top_k:
                break

        return selected
