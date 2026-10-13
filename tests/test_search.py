import unittest
import time
from backend.audio.subtitle_parser import parse_srt_or_vtt
from backend.audio.chunker import create_temporal_chunks
from backend.core.searcher import LectureSearchEngine
from backend.config import SAMPLES_DIR

class TestSearchEngine(unittest.TestCase):
    def setUp(self):
        sample_file = SAMPLES_DIR / "machine_learning_intro_vi.srt"
        subs = parse_srt_or_vtt(sample_file)
        self.chunks = create_temporal_chunks(subs, window_seconds=30.0, stride_seconds=15.0)
        self.engine = LectureSearchEngine(self.chunks, model_key="vietnamese-sbert")

    def test_overfitting_search(self):
        res = self.engine.search("Mô hình bị overfitting ở đoạn nào?", top_k=3, alpha=0.7)
        self.assertGreater(len(res), 0)
        self.assertLess(res[0]["start"], 60.0)

    def test_dropout_search(self):
        res = self.engine.search("Kỹ thuật Dropout hoạt động như thế nào?", top_k=3, alpha=0.7)
        self.assertGreater(len(res), 0)
        self.assertTrue(any("Dropout" in r["text"] or "nơ-ron" in r["text"] for r in res))

if __name__ == "__main__":
    unittest.main()
