import time
from fastapi import APIRouter, HTTPException
from backend.config import DEFAULT_TOP_K, DEFAULT_HYBRID_ALPHA
from backend.schemas import SearchRequest, RagQueryRequest
from backend.state import current_state

router = APIRouter(tags=["Search & Grounding RAG"])

@router.post("/api/search")
def search_lecture(req: SearchRequest):
    """
    Executes hybrid search (Dense SBERT + Sparse BM25) with Temporal NMS across lecture moments.
    """
    if not current_state["search_engine"]:
        raise HTTPException(status_code=400, detail="Chưa có bài giảng nào được tải.")

    t0 = time.perf_counter()
    results = current_state["search_engine"].search(
        query=req.query,
        top_k=req.top_k or DEFAULT_TOP_K,
        alpha=req.alpha if req.alpha is not None else DEFAULT_HYBRID_ALPHA
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    return {
        "query": req.query,
        "top_k": req.top_k,
        "alpha": req.alpha,
        "model_key": current_state["model_key"],
        "video_id": current_state["video_id"],
        "latency_ms": elapsed_ms,
        "total_found": len(results),
        "results": results
    }

@router.post("/api/rag/ask")
def rag_ask_lecture(req: RagQueryRequest):
    """
    Video Grounding RAG: Answers natural language question, grounds precise video timestamp,
    and returns slide keyframe + speech evidence citations.
    """
    if not current_state["rag_engine"]:
        raise HTTPException(status_code=400, detail="Chưa có bài giảng nào được tải hoặc RAG engine chưa sẵn sàng.")

    rag_response = current_state["rag_engine"].answer_query(
        query=req.query,
        alpha=req.alpha if req.alpha is not None else DEFAULT_HYBRID_ALPHA,
        use_query_expansion=req.use_query_expansion if req.use_query_expansion is not None else True
    )
    return rag_response

@router.get("/api/subtitles")
def get_all_subtitles():
    """
    Returns all subtitle cues parsed for the active lecture.
    """
    return {
        "video_id": current_state["video_id"],
        "total": len(current_state["subtitles"]),
        "subtitles": current_state["subtitles"]
    }
