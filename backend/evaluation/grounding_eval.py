import time
from typing import List, Dict, Tuple
import numpy as np

def compute_temporal_iou(pred_interval: Tuple[float, float], gt_interval: Tuple[float, float]) -> float:
    """
    Computes 1D temporal Intersection over Union (IoU) between prediction and ground truth.
    """
    p_s, p_e = pred_interval
    g_s, g_e = gt_interval

    inter_start = max(p_s, g_s)
    inter_end = min(p_e, g_e)
    inter = max(0.0, inter_end - inter_start)

    union = (p_e - p_s) + (g_e - g_s) - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)

def evaluate_grounding_dataset(
    search_engine,
    benchmark_items: List[Dict],
    top_k: int = 3,
    alpha: float = 0.7
) -> Dict:
    """
    Evaluates a benchmark dataset of queries with ground-truth intervals.
    Returns:
    - mIoU: Mean IoU of top-1 prediction
    - R@1 (IoU >= 0.3)
    - R@1 (IoU >= 0.5)
    - R@3 (IoU >= 0.3)
    - R@3 (IoU >= 0.5)
    - average_latency_ms: Average query execution latency on CPU
    - query_details: Individual breakdown for each query
    """
    if not benchmark_items:
        return {
            "num_queries": 0,
            "mIoU": 0.0,
            "R1_iou03": 0.0,
            "R1_iou05": 0.0,
            "R3_iou03": 0.0,
            "R3_iou05": 0.0,
            "avg_latency_ms": 0.0,
            "details": []
        }

    total_iou_top1 = 0.0
    r1_03_hits = 0
    r1_05_hits = 0
    r3_03_hits = 0
    r3_05_hits = 0
    total_time = 0.0

    details = []

    for item in benchmark_items:
        query = item["query"]
        gt_start = float(item["gt_start"])
        gt_end = float(item["gt_end"])
        gt_interval = (gt_start, gt_end)

        t0 = time.perf_counter()
        results = search_engine.search(query, top_k=top_k, alpha=alpha)
        lat = (time.perf_counter() - t0) * 1000.0
        total_time += lat

        if not results:
            details.append({
                "query": query,
                "gt_start": gt_start,
                "gt_end": gt_end,
                "pred_start": None,
                "pred_end": None,
                "top1_iou": 0.0,
                "max_topk_iou": 0.0,
                "latency_ms": round(lat, 2),
                "hit_iou03": False,
                "hit_iou05": False
            })
            continue

        # Top-1 IoU
        top1 = results[0]
        top1_interval = (top1["start"], top1["end"])
        top1_iou = compute_temporal_iou(top1_interval, gt_interval)
        total_iou_top1 += top1_iou

        if top1_iou >= 0.3:
            r1_03_hits += 1
        if top1_iou >= 0.5:
            r1_05_hits += 1

        # Top-K max IoU
        ious = [compute_temporal_iou((r["start"], r["end"]), gt_interval) for r in results]
        max_iou = max(ious)

        if any(iou >= 0.3 for iou in ious):
            r3_03_hits += 1
        if any(iou >= 0.5 for iou in ious):
            r3_05_hits += 1

        details.append({
            "query": query,
            "gt_start": gt_start,
            "gt_end": gt_end,
            "pred_start": top1["start"],
            "pred_end": top1["end"],
            "pred_text": top1["text"],
            "top1_iou": round(top1_iou, 4),
            "max_topk_iou": round(max_iou, 4),
            "latency_ms": round(lat, 2),
            "hit_iou03": top1_iou >= 0.3,
            "hit_iou05": top1_iou >= 0.5
        })

    n = len(benchmark_items)
    return {
        "num_queries": n,
        "mIoU": round(total_iou_top1 / n, 4),
        "R1_iou03": round((r1_03_hits / n) * 100.0, 2),
        "R1_iou05": round((r1_05_hits / n) * 100.0, 2),
        "R3_iou03": round((r3_03_hits / n) * 100.0, 2),
        "R3_iou05": round((r3_05_hits / n) * 100.0, 2),
        "avg_latency_ms": round(total_time / n, 2),
        "details": details
    }
