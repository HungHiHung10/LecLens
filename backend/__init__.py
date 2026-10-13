"""
LectureLens Backend Package.
Enterprise-grade modular architecture:
- backend.core: Search, Embeddings, Multimodal Indexing, Grounding RAG
- backend.vision: PyAV Keyframe Extraction, RapidOCR, Vietnamese Diacritics Recovery
- backend.audio: Whisper Speech Recognition, Subtitle Parsing, Temporal Chunker
- backend.evaluation: Academic Benchmarking (mIoU, R@k), Annotation Dataset Manager
- backend.routers: Modular FastAPI Route Controllers
"""
# Re-exports for convenient access
from backend.core import (
    embedder_manager, LectureSearchEngine,
    enrich_chunks_with_multimodal_metadata,
    VideoGroundingRAG, parse_grounding_query
)
from backend.vision import vision_extractor, VietnameseTextEnhancer
from backend.audio import (
    parse_srt_or_vtt, seconds_to_timestamp, create_temporal_chunks,
    transcribe_with_groq, transcribe_local_video
)
from backend.evaluation import evaluate_grounding_dataset, annotation_manager

# Backward-compatibility module aliases
import sys
from backend import core, vision, audio, evaluation
sys.modules['backend.searcher'] = core.searcher
sys.modules['backend.embedder'] = core.embedder
sys.modules['backend.multimodal_indexer'] = core.multimodal_indexer
sys.modules['backend.video_grounding_rag'] = core.video_grounding_rag
sys.modules['backend.query_parser'] = core.query_parser
sys.modules['backend.vision_extractor'] = vision.vision_extractor
sys.modules['backend.vietnamese_text_enhancer'] = vision.vietnamese_text_enhancer
sys.modules['backend.chunker'] = audio.chunker
sys.modules['backend.subtitle_parser'] = audio.subtitle_parser
sys.modules['backend.audio_transcriber'] = audio.audio_transcriber
sys.modules['backend.groq_transcriber'] = audio.groq_transcriber
sys.modules['backend.grounding_eval'] = evaluation.grounding_eval
sys.modules['backend.annotation_manager'] = evaluation.annotation_manager
