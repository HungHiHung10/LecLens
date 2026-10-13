"""
backend.evaluation: Grounding Evaluation Benchmark and Annotation Dataset Manager.
"""
from backend.evaluation.grounding_eval import evaluate_grounding_dataset, compute_temporal_iou
from backend.evaluation.annotation_manager import annotation_manager, AnnotationManager

__all__ = [
    "evaluate_grounding_dataset",
    "compute_temporal_iou",
    "annotation_manager",
    "AnnotationManager",
]
