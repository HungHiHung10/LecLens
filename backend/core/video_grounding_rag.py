import time
import json
from typing import Dict, List, Any, Optional
from collections import defaultdict
import numpy as np

from backend.config import KEYFRAMES_DIR
from backend.core.query_parser import query_parser
from backend.core.searcher import LectureSearchEngine
from backend.audio.subtitle_parser import seconds_to_timestamp

class VideoGroundingRAG:
    """
    Video Grounding RAG Pipeline inspired by AIC 2026 KIS Retrieval notebooks:
    1. Query Parsing & Decomposition (intent, speech terms, slide keywords, query variants)
    2. Multi-Query Retrieval & Multimodal Feature Fusion
    3. Temporal Segment Grouping & Contextual Hedging
    4. Grounded Synthesis & Timestamped Evidence Generation
    """
    def __init__(self, search_engine: LectureSearchEngine, enriched_chunks: List[Dict]):
        self.search_engine = search_engine
        self.chunks = enriched_chunks
        self.chunk_map = {c["chunk_id"]: c for c in enriched_chunks}

    def answer_query(
        self,
        query: str,
        alpha: float = 0.7,
        use_query_expansion: bool = True
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        
        # 1. Parse and decompose query
        parsed = query_parser.parse_query(query)
        variants = parsed["query_variants"] if use_query_expansion else [query]

        # 2. Multi-Query Retrieval (combines hits across query variants)
        # Weights: primary query gets 0.7, variants get 0.3 split
        variant_weights = [0.7] + [0.3 / max(1, len(variants) - 1)] * (len(variants) - 1)
        
        aggregated_scores = defaultdict(float)
        variant_hit_counts = defaultdict(int)
        raw_results_by_variant = {}

        for v_idx, v_text in enumerate(variants):
            w = variant_weights[v_idx]
            v_results = self.search_engine.search(v_text, top_k=6, alpha=alpha)
            raw_results_by_variant[v_text] = v_results
            for r in v_results:
                cid = r["chunk_id"]
                aggregated_scores[cid] += w * r["score"]
                variant_hit_counts[cid] += 1

        # 3. Add Multimodal Metadata & Term Alignment Bonuses (inspired by AIC2026 pre_rerank_fusion)
        for cid, score in list(aggregated_scores.items()):
            chunk = self.chunk_map.get(cid)
            if not chunk:
                continue
            
            bonus = 0.0
            chunk_text_lower = chunk["text"].lower()
            slide_title_lower = chunk.get("slide_title", "").lower()
            ocr_lower = chunk.get("ocr_keywords", "").lower()
            ocr_full_lower = chunk.get("ocr_text", "").lower()

            # Technical term matches in Speech OR Slide OCR
            for term in parsed["technical_terms"]:
                if term in chunk_text_lower or term in ocr_lower or term in slide_title_lower or term in ocr_full_lower:
                    bonus += 0.10

            # Slide title match
            for q_term in parsed["core_query"].split():
                if len(q_term) > 2 and (q_term in slide_title_lower or q_term in ocr_full_lower):
                    bonus += 0.06

            aggregated_scores[cid] += bonus

        # 4. Rank Candidates
        sorted_chunks = sorted(aggregated_scores.items(), key=lambda x: x[1], reverse=True)
        top_candidates = []
        for cid, agg_score in sorted_chunks[:5]:
            ch = self.chunk_map[cid]
            top_candidates.append({
                **ch,
                "rag_score": round(float(agg_score), 4),
                "variant_hits": variant_hit_counts[cid]
            })

        if not top_candidates:
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return {
                "query": query,
                "parsed_query": parsed,
                "answer": "Không tìm thấy đoạn nội dung nào trong bài giảng phù hợp với câu hỏi này.",
                "primary_grounding": None,
                "evidence_citations": [],
                "alternative_moments": [],
                "confidence_score": 0.0,
                "latency_ms": elapsed_ms
            }

        # 5. Temporal Consolidation for Primary Moment (Merge adjacent chunks)
        primary_chunk = top_candidates[0]
        grounded_start = primary_chunk["start"]
        grounded_end = primary_chunk["end"]
        grounded_text = primary_chunk["text"]
        slide_title = primary_chunk.get("slide_title", "Nội dung bài giảng")

        # Dynamically sync with enhanced keyframe metadata if available on disk
        kf_url = primary_chunk.get("keyframe_url", "")
        if kf_url and "/keyframes/" in kf_url:
            parts = kf_url.strip("/").split("/")
            if len(parts) >= 2:
                vid_folder = parts[1]
                meta_json_path = KEYFRAMES_DIR / vid_folder / "keyframes_metadata.json"
                if meta_json_path.exists():
                    try:
                        with open(meta_json_path, "r", encoding="utf-8") as f:
                            disk_kfs = json.load(f)
                        matched_meta = next((k for k in disk_kfs if k.get("image_url") == kf_url or k.get("keyframe_id") == primary_chunk.get("keyframe_id")), None)
                        if matched_meta:
                            if matched_meta.get("slide_title"):
                                slide_title = matched_meta["slide_title"]
                            if matched_meta.get("ocr_keywords"):
                                primary_chunk["ocr_keywords"] = matched_meta["ocr_keywords"]
                    except Exception:
                        pass

        # Check if rank 2 is immediately adjacent to rank 1 (within 10s gap)
        if len(top_candidates) > 1:
            second_chunk = top_candidates[1]
            if abs(second_chunk["start"] - grounded_end) <= 12.0 or abs(second_chunk["end"] - grounded_start) <= 12.0:
                grounded_start = min(grounded_start, second_chunk["start"])
                grounded_end = max(grounded_end, second_chunk["end"])
                # Combine speech without duplicating phrases
                if second_chunk["text"] not in grounded_text:
                    grounded_text = f"{grounded_text} {second_chunk['text']}"

        hedged_start = max(0.0, round(grounded_start - 3.0, 2))
        hedged_end = round(grounded_end + 3.0, 2)

        # 6. Extract Verbatim Citations with Timestamps
        citations = []
        sentences = [s.strip() for s in grounded_text.split('.') if len(s.strip()) > 10]
        # Distribute timestamps along the interval for each sentence
        num_sentences = max(1, len(sentences))
        step = (grounded_end - grounded_start) / num_sentences
        for idx, sentence in enumerate(sentences[:4]):
            s_start = grounded_start + idx * step
            s_end = min(grounded_end, s_start + step)
            citations.append({
                "timestamp_str": f"{seconds_to_timestamp(s_start)} - {seconds_to_timestamp(s_end)}",
                "start": round(s_start, 2),
                "end": round(s_end, 2),
                "quote": f"{sentence}.",
                "source": "Lời giảng viên (Speech ASR)"
            })

        # 7. Generate Pedagogical Grounded Answer (RAG Generator)
        confidence = min(0.98, max(0.65, primary_chunk["rag_score"]))
        conf_label = "Cao (High)" if confidence >= 0.8 else "Trung bình (Moderate)"

        answer_text = self._synthesize_answer(
            query=query,
            intent=parsed["intent"],
            grounded_text=grounded_text,
            start_str=seconds_to_timestamp(grounded_start),
            end_str=seconds_to_timestamp(grounded_end),
            slide_title=slide_title,
            sentences=sentences
        )

        # Alternative Moments (distinct in time)
        alternative_moments = []
        for alt in top_candidates[1:4]:
            if abs(alt["start"] - grounded_start) > 40.0: # Distinct temporal cluster
                alternative_moments.append({
                    "start": alt["start"],
                    "end": alt["end"],
                    "start_str": alt["start_str"],
                    "end_str": alt["end_str"],
                    "duration": alt["duration"],
                    "slide_title": alt.get("slide_title", ""),
                    "keyframe_url": alt.get("keyframe_url"),
                    "snippet": alt["text"][:120] + "...",
                    "score": alt["rag_score"]
                })

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return {
            "query": query,
            "parsed_query": parsed,
            "answer": answer_text,
            "primary_grounding": {
                "start": round(grounded_start, 2),
                "end": round(grounded_end, 2),
                "start_str": seconds_to_timestamp(grounded_start),
                "end_str": seconds_to_timestamp(grounded_end),
                "duration": round(grounded_end - grounded_start, 2),
                "hedged_start": hedged_start,
                "hedged_end": hedged_end,
                "slide_title": slide_title,
                "ocr_keywords": (
                    [k.strip() for k in primary_chunk.get("ocr_keywords", "").split(",") if k.strip()]
                    if isinstance(primary_chunk.get("ocr_keywords"), str)
                    else (primary_chunk.get("ocr_keywords") or [])
                ),
                "ocr_text": primary_chunk.get("ocr_text", ""),
                "keyframe_url": primary_chunk.get("keyframe_url"),
                "keyframe_timestamp": primary_chunk.get("keyframe_timestamp"),
                "keyframe_timestamp_str": primary_chunk.get("keyframe_timestamp_str"),
                "confidence": round(float(confidence), 3),
                "confidence_label": conf_label
            },
            "evidence_citations": citations,
            "alternative_moments": alternative_moments[:2],
            "latency_ms": elapsed_ms
        }

    def _synthesize_answer(
        self,
        query: str,
        intent: str,
        grounded_text: str,
        start_str: str,
        end_str: str,
        slide_title: str,
        sentences: List[str]
    ) -> str:
        """
        Synthesizes structured, precise grounded explanation answering user query.
        """
        core_summary = " ".join([s + "." for s in sentences[:3]])
        
        intro_phrases = {
            "explanation": f"Tại phân đoạn **[{start_str} - {end_str}]** (*Slide: {slide_title}*), giảng viên giải thích chi tiết vấn đề này như sau:",
            "definition": f"Khái niệm này được giảng viên định nghĩa và giải thích từ **[{start_str} - {end_str}]**:",
            "comparison": f"Giảng viên thực hiện so sánh và phân tích sự khác biệt tại phân đoạn **[{start_str} - {end_str}]**:",
            "timestamp_lookup": f"Nội dung bạn tìm kiếm nằm chính xác ở khoảng thời gian **[{start_str} - {end_str}]** (*{slide_title}*):",
            "general_search": f"Tại thời điểm **[{start_str} - {end_str}]**, bài giảng đề cập đến vấn đề này:"
        }

        intro = intro_phrases.get(intent, intro_phrases["general_search"])
        
        return f"{intro}\n\n> {core_summary}\n\n*Bạn có thể bấm vào khung thời gian **[{start_str} - {end_str}]** bên dưới để trình phát tự động nhảy và phát đúng đoạn giảng này.*"
