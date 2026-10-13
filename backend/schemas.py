from typing import Optional
from pydantic import BaseModel
from backend.config import (
    DEFAULT_MODEL_KEY, DEFAULT_WINDOW_SECONDS,
    DEFAULT_STRIDE_SECONDS, DEFAULT_HYBRID_ALPHA, DEFAULT_TOP_K
)

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = DEFAULT_TOP_K
    alpha: Optional[float] = DEFAULT_HYBRID_ALPHA
    model_key: Optional[str] = DEFAULT_MODEL_KEY

class RagQueryRequest(BaseModel):
    query: str
    alpha: Optional[float] = DEFAULT_HYBRID_ALPHA
    use_query_expansion: Optional[bool] = True

class AnnotationRequest(BaseModel):
    video_id: str
    query: str
    start: float
    end: float
    rating: str
    notes: Optional[str] = ""
    predicted_start: Optional[float] = None
    predicted_end: Optional[float] = None

class LoadSampleRequest(BaseModel):
    sample_key: str
    model_key: Optional[str] = None
    window_size: Optional[float] = DEFAULT_WINDOW_SECONDS
    stride_size: Optional[float] = DEFAULT_STRIDE_SECONDS

class YouTubeLoadRequest(BaseModel):
    url: str
    model_key: Optional[str] = None

class GroqKeyRequest(BaseModel):
    api_key: str
