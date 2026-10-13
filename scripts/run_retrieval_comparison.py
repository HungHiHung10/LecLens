import json
import time
import sys
import numpy as np
from pathlib import Path

# UTF-8 stdout for Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path
import os
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.audio.subtitle_parser import parse_srt_or_vtt
from backend.audio.chunker import create_temporal_chunks
from backend.core.searcher import LectureSearchEngine
from backend.core.multimodal_indexer import enrich_chunks_with_multimodal_metadata
from backend.evaluation.grounding_eval import evaluate_grounding_dataset, compute_temporal_iou
from backend.config import SAMPLES_DIR

def run_benchmarks():
    sample_file = SAMPLES_DIR / "machine_learning_intro_vi.srt"
    gt_file = SAMPLES_DIR / "benchmark_ground_truth.json"

    with open(gt_file, "r", encoding="utf-8") as f:
        benchmark_items = json.load(f)

    subs = parse_srt_or_vtt(sample_file)
    raw_chunks = create_temporal_chunks(subs, window_seconds=30.0, stride_seconds=15.0)
    multimodal_chunks = enrich_chunks_with_multimodal_metadata(raw_chunks)

    # 1. Speech Only Chunks (No Slide/OCR metadata)
    speech_only_chunks = []
    for c in raw_chunks:
        c_copy = dict(c)
        c_copy["semantic_metadata_text"] = c["text"]
        speech_only_chunks.append(c_copy)

    print(f"Loaded {len(benchmark_items)} benchmark queries across 29 temporal chunks.")

    configurations = [
        {"name": "BM25 Only (Sparse)", "chunks": speech_only_chunks, "model": "vietnamese-sbert", "alpha": 0.0},
        {"name": "Dense Only (MiniLM-L6-v2)", "chunks": speech_only_chunks, "model": "all-MiniLM-L6-v2", "alpha": 1.0},
        {"name": "Dense Only (Vietnamese SBERT)", "chunks": speech_only_chunks, "model": "vietnamese-sbert", "alpha": 1.0},
        {"name": "Hybrid Speech-Only (α=0.5)", "chunks": speech_only_chunks, "model": "vietnamese-sbert", "alpha": 0.5},
        {"name": "Hybrid Speech-Only (α=0.7)", "chunks": speech_only_chunks, "model": "vietnamese-sbert", "alpha": 0.7},
        {"name": "Multimodal Hybrid (Speech + Slide OCR, α=0.7)", "chunks": multimodal_chunks, "model": "vietnamese-sbert", "alpha": 0.7},
    ]

    results_table = []

    for cfg in configurations:
        print(f"\nEvaluating: {cfg['name']}...")
        engine = LectureSearchEngine(cfg["chunks"], model_key=cfg["model"])

        latencies = []
        # Warmup
        engine.search("test query", top_k=3, alpha=cfg["alpha"])

        for item in benchmark_items:
            t0 = time.perf_counter()
            engine.search(item["query"], top_k=3, alpha=cfg["alpha"])
            latencies.append((time.perf_counter() - t0) * 1000.0)

        metrics = evaluate_grounding_dataset(engine, benchmark_items, top_k=3, alpha=cfg["alpha"])
        
        p50 = float(np.percentile(latencies, 50))
        p90 = float(np.percentile(latencies, 90))

        summary = {
            "name": cfg["name"],
            "mIoU": metrics["mIoU"],
            "R1_03": metrics["R1_iou03"],
            "R1_05": metrics["R1_iou05"],
            "R3_03": metrics["R3_iou03"],
            "R3_05": metrics["R3_iou05"],
            "avg_ms": metrics["avg_latency_ms"],
            "p50_ms": round(p50, 2),
            "p90_ms": round(p90, 2)
        }
        results_table.append(summary)
        print(f" -> mIoU: {summary['mIoU']:.4f} | R@1(0.3): {summary['R1_03']}% | R@3(0.3): {summary['R3_03']}% | R@3(0.5): {summary['R3_05']}% | p50: {summary['p50_ms']}ms")

    # Output JSON report
    report_path = Path("a:/LecLens/data/benchmark_comparison_report.json")
    report_path.write_text(json.dumps(results_table, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved benchmark comparison report to {report_path}")

if __name__ == "__main__":
    run_benchmarks()
