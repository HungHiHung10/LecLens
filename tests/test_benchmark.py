import json
import unittest
from backend.audio.subtitle_parser import parse_srt_or_vtt
from backend.audio.chunker import create_temporal_chunks
from backend.core.searcher import LectureSearchEngine
from backend.evaluation.grounding_eval import evaluate_grounding_dataset
from backend.config import SAMPLES_DIR

class TestGroundingBenchmark(unittest.TestCase):
    def test_grounding_benchmark(self):
        sample_file = SAMPLES_DIR / "machine_learning_intro_vi.srt"
        gt_file = SAMPLES_DIR / "benchmark_ground_truth.json"

        subs = parse_srt_or_vtt(sample_file)
        chunks = create_temporal_chunks(subs, window_seconds=30.0, stride_seconds=15.0)
        engine = LectureSearchEngine(chunks, model_key="vietnamese-sbert")

        with open(gt_file, "r", encoding="utf-8") as f:
            benchmark_items = json.load(f)

        metrics = evaluate_grounding_dataset(engine, benchmark_items, top_k=3, alpha=0.7)
        self.assertEqual(metrics["num_queries"], 30)
        self.assertGreaterEqual(metrics["R3_iou03"], 70.0)

if __name__ == "__main__":
    test_grounding_benchmark()
