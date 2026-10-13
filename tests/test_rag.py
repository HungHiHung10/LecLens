import unittest
import time
from backend.audio.subtitle_parser import parse_srt_or_vtt
from backend.audio.chunker import create_temporal_chunks
from backend.core.multimodal_indexer import enrich_chunks_with_multimodal_metadata
from backend.core.searcher import LectureSearchEngine
from backend.core.video_grounding_rag import VideoGroundingRAG
from backend.core.query_parser import query_parser
from backend.config import SAMPLES_DIR

class TestVideoGroundingRAG(unittest.TestCase):
    def test_query_parser(self):
        q = "Thầy giải thích vì sao mô hình bị overfitting ở đoạn nào?"
        parsed = query_parser.parse_query(q)
        self.assertIn("overfitting", parsed["technical_terms"])
        self.assertGreaterEqual(len(parsed["query_variants"]), 2)
        self.assertIn(parsed["intent"], ["explanation", "timestamp_lookup"])

    def test_video_grounding_rag_pipeline(self):
        sample_file = SAMPLES_DIR / "machine_learning_intro_vi.srt"
        subs = parse_srt_or_vtt(sample_file)
        raw_chunks = create_temporal_chunks(subs, window_seconds=30.0, stride_seconds=15.0)
        enriched = enrich_chunks_with_multimodal_metadata(raw_chunks)
        
        engine = LectureSearchEngine(enriched, model_key="vietnamese-sbert")
        rag = VideoGroundingRAG(search_engine=engine, enriched_chunks=enriched)

        res = rag.answer_query("Thầy giải thích vì sao mô hình bị overfitting ở đoạn nào?", alpha=0.7)
        self.assertIsNotNone(res["primary_grounding"])
        self.assertGreater(len(res["evidence_citations"]), 0)
        self.assertGreater(res["confidence_score"] if "confidence_score" in res else res["primary_grounding"]["confidence"], 0.5)

if __name__ == "__main__":
    unittest.main()
