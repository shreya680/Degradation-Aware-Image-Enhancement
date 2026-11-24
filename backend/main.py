"""
FastAPI Application Entry Point
================================
Degradation-Aware Image Enhancement - Backend Server
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUT_SUBDIRS = [
    OUTPUTS_DIR / "uploads",
    OUTPUTS_DIR / "enhanced",
    OUTPUTS_DIR / "annotated",
    OUTPUTS_DIR / "reports",
]


def _create_output_dirs() -> None:
    """Create all required output directories."""
    for d in OUTPUT_SUBDIRS:
        d.mkdir(parents=True, exist_ok=True)
    logger.info("Output directories ready.")


# ---------------------------------------------------------------------------
# Lifespan: load models at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan:
    - Creates output directories
    - Loads Zero-DCE model (once, at startup)
    - Loads YOLOv8 model (once, at startup)
    """
    logger.info("=" * 60)
    logger.info("Degradation-Aware Image Enhancement - Starting up")
    logger.info("=" * 60)

    # Create directories
    _create_output_dirs()

    # Load Zero-DCE
    logger.info("Loading Zero-DCE model...")
    try:
        from backend.enhancement.zero_dce import ZeroDCEEnhancer, _enhancer_instance
        import backend.enhancement.zero_dce as _zd_module
        if _zd_module._enhancer_instance is None:
            enhancer = ZeroDCEEnhancer()
            enhancer.load()
            _zd_module._enhancer_instance = enhancer
        logger.info(
            f"Zero-DCE loaded. Weights: {'YES' if _zd_module._enhancer_instance.weights_loaded else 'NO (untrained)'}"
        )
    except Exception as e:
        logger.error(f"Zero-DCE load failed: {e}")

    # Load YOLOv8
    logger.info("Loading YOLOv8 model...")
    try:
        from backend.detection.yolov8_detector import YOLOv8Detector
        import backend.detection.yolov8_detector as _yolo_module
        if _yolo_module._detector_instance is None:
            detector = YOLOv8Detector("yolov8n.pt")
            detector.load()
            _yolo_module._detector_instance = detector
        logger.info(
            f"YOLOv8 loaded: {'YES' if _yolo_module._detector_instance.loaded else 'NO'}"
        )
    except Exception as e:
        logger.error(f"YOLOv8 load failed: {e}")

    # Initialise history DB
    try:
        from backend.services.history_service import get_history_service
        get_history_service()
        logger.info("History database ready.")
    except Exception as e:
        logger.error(f"History DB init failed: {e}")

    logger.info("Startup complete. Server is ready.")
    logger.info("=" * 60)

    yield  # App runs here

    logger.info("Shutting down application...")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Degradation-Aware Image Enhancement API",
    description=(
        "AI/ML backend for low-light image enhancement with Zero-DCE, "
        "YOLOv8 object detection, and comprehensive image quality evaluation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Static files (serve images)
# ---------------------------------------------------------------------------
# Mount outputs/ at /images so images are accessible at /images/uploads/..., etc.
# Directory will exist after lifespan creates it on first startup.
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=str(OUTPUTS_DIR)), name="images")

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
from backend.api.routes import router  # noqa: E402

app.include_router(router)


# ---------------------------------------------------------------------------
# Root redirect / info
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "name": "Degradation-Aware Image Enhancement API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }


# ---------------------------------------------------------------------------
# Entry point for direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
