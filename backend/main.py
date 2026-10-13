import sys
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

from backend.config import (
    BASE_DIR, FRONTEND_DIR, SAMPLES_DIR, KEYFRAMES_DIR,
    DEFAULT_WINDOW_SECONDS, DEFAULT_STRIDE_SECONDS, get_logger
)
from backend.state import current_state, initialize_engine
from backend.audio.subtitle_parser import parse_srt_or_vtt
from backend.routers.search import router as search_router
from backend.routers.media import router as media_router
from backend.routers.system import router as system_router

logger = get_logger("main")

app = FastAPI(
    title="LectureLens — Video Search Assistant & Grounding Studio",
    description="Multimodal Video Search & Grounding Studio with Speech + Slide Vision, optimized for CPU",
    version="2.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Modular Routers
app.include_router(search_router)
app.include_router(media_router)
app.include_router(system_router)

# Mount Static File Endpoints
app.mount("/keyframes", StaticFiles(directory=str(KEYFRAMES_DIR)), name="keyframes")
app.mount("/logo", StaticFiles(directory=str(BASE_DIR / "logo")), name="logo")
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.on_event("startup")
def startup_event():
    """
    Preloads initial default lecture into memory upon application launch.
    """
    transformer_mp4 = BASE_DIR / "Tất cả những gì bạn cần biết về Transformer trong 35 phút.mp4"
    transformer_srt = SAMPLES_DIR / "transformer_35_phut_vi.srt"
    if transformer_mp4.exists() and transformer_srt.exists():
        subs = parse_srt_or_vtt(transformer_srt)
        initialize_engine(
            subtitles=subs,
            window=DEFAULT_WINDOW_SECONDS,
            stride=DEFAULT_STRIDE_SECONDS,
            model_key="vietnamese-sbert",
            video_id="transformer_35_phut_vi",
            title="Tất Cả Về Transformer Trong 35 Phút (Tiếng Việt)",
            video_path=str(transformer_mp4)
        )
        logger.info(f"Preloaded Transformer lecture with {len(subs)} cues and {len(current_state['keyframes'])} slide keyframes.")
    else:
        sample_file = SAMPLES_DIR / "machine_learning_intro_vi.srt"
        if sample_file.exists():
            subs = parse_srt_or_vtt(sample_file)
            initialize_engine(
                subtitles=subs,
                window=DEFAULT_WINDOW_SECONDS,
                stride=DEFAULT_STRIDE_SECONDS,
                model_key="vietnamese-sbert",
                video_id="machine_learning_intro_vi",
                title="Machine Learning: Overfitting & Kỹ Thuật Khắc Phục (Tiếng Việt)"
            )
            logger.info(f"Preloaded sample with {len(subs)} subtitle cues.")

@app.get("/favicon.ico", include_in_schema=False)
def serve_favicon():
    """Direct root favicon route for browser compatibility."""
    fav = FRONTEND_DIR / "favicon.ico"
    if fav.exists():
        return FileResponse(fav, media_type="image/x-icon")
    raise HTTPException(status_code=404, detail="Favicon not found")

@app.get("/", response_class=HTMLResponse)
def serve_index():
    """Serves the primary Single-Page Studio Application."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>LectureLens API running</h1>")
