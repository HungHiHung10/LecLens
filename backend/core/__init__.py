"""
backend.core: Core AI Retrieval, Embedding, and Multimodal Grounding Engine.
"""
from backend.core.embedder import embedder_manager, EmbedderManager
from backend.core.searcher import LectureSearchEngine
from backend.core.multimodal_indexer import enrich_chunks_with_multimodal_metadata, infer_topic_and_slide_metadata
from backend.core.video_grounding_rag import VideoGroundingRAG
from backend.core.query_parser import parse_grounding_query, query_parser, MultimodalQueryParser

__all__ = [
    "embedder_manager",
    "EmbedderManager",
    "LectureSearchEngine",
    "enrich_chunks_with_multimodal_metadata",
    "infer_topic_and_slide_metadata",
    "VideoGroundingRAG",
    "parse_grounding_query",
    "query_parser",
    "MultimodalQueryParser",
]
