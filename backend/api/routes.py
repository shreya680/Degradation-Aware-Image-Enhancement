"""
API Routes
===========
All FastAPI route definitions for the Degradation-Aware Image Enhancement application.
"""
from __future__ import annotations

import io
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, Request
from fastapi.responses import FileResponse
from PIL import Image

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR = PROJECT_ROOT / "outputs" / "uploads"
ENHANCED_DIR = PROJECT_ROOT / "outputs" / "enhanced"
ANNOTATED_DIR = PROJECT_ROOT / "outputs" / "annotated"

router = APIRouter(prefix="/api")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _get_zero_dce():
    from backend.enhancement.zero_dce import get_enhancer
    return get_enhancer()


def _get_detector():
    from backend.detection.yolov8_detector import get_detector
    return get_detector()


def _get_history():
    from backend.services.history_service import get_history_service
    return get_history_service()


def _load_array(image_path: Path) -> np.ndarray:
    """Load image as uint8 RGB numpy array."""
    with Image.open(image_path) as img:
        return np.array(img.convert("RGB"))


def _save_pil_to_enhanced(pil_image: Image.Image, stem: str) -> tuple[Path, str]:
    """Save PIL image to enhanced dir, return (path, url)."""
    ENHANCED_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"enhanced_{stem}.png"
    path = ENHANCED_DIR / filename
    pil_image.save(str(path))
    return path, f"/images/enhanced/{filename}"


# ---------------------------------------------------------------------------
# Health / status
# ---------------------------------------------------------------------------

@router.get("/health")
async def health() -> dict[str, Any]:
    """Return service health and loaded model status."""
    zero_dce = _get_zero_dce()
    detector = _get_detector()
    zero_loaded = zero_dce.model is not None
    yolo_loaded = detector.loaded
    return {
        "status": "ok",
        "models_loaded": {
            "zero_dce": zero_loaded,
            "yolov8": yolo_loaded,
        },
        "all_models_loaded": zero_loaded and yolo_loaded,
        "device": "cpu",
        "zero_dce_loaded": zero_loaded,
        "yolov8_loaded": yolo_loaded,
    }


@router.get("/models/status")
async def models_status() -> dict[str, Any]:
    """Return detailed model loading status."""
    zero_dce = _get_zero_dce()
    detector = _get_detector()
    return {
        "zero_dce": zero_dce.status,
        "yolov8": detector.status,
    }


@router.get("/datasets/status")
async def datasets_status() -> dict[str, Any]:
    """Return dataset availability stats."""
    from backend.datasets.lol_dataset import LOLDataset
    from backend.datasets.exdark_dataset import ExDarkDataset

    lol = LOLDataset().get_stats()
    exdark = ExDarkDataset().get_stats()

    return {
        "lol": {
            "available": lol["available"],
            "path": lol["path"],
            "count": lol["train_count"] + lol["eval_count"],
            "train_count": lol["train_count"],
            "eval_count": lol["eval_count"],
        },
        "exdark": {
            "available": exdark["available"],
            "path": exdark["path"],
            "count": exdark["total_images"],
            "categories": exdark["categories"],
            "counts_per_category": exdark["counts_per_category"],
        },
    }


@router.get("/history")
async def get_history(limit: int = Query(default=20, ge=1, le=200)) -> dict[str, Any]:
    """Return the last N processing sessions from history."""
    history = _get_history().get_history(limit=limit)
    return {"sessions": history, "count": len(history)}


# ---------------------------------------------------------------------------
# Image serving
# ---------------------------------------------------------------------------

@router.get("/images/{subfolder}/{filename}")
async def serve_image(subfolder: str, filename: str):
    """Serve an image from the outputs directory."""
    allowed = {"uploads", "enhanced", "annotated", "reports"}
    if subfolder not in allowed:
        raise HTTPException(status_code=404, detail=f"Unknown image subfolder: {subfolder}")
    path = PROJECT_ROOT / "outputs" / subfolder / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {filename}")
    return FileResponse(str(path))


# ---------------------------------------------------------------------------
# POST /api/upload
# ---------------------------------------------------------------------------

@router.post("/upload")
async def upload_image(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Upload an image file.
    Returns filename, dimensions, file_size, preview_url.
    """
    from backend.services.image_service import validate_image, save_upload, ImageServiceError

    try:
        await validate_image(file)
    except ImageServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        result = await save_upload(file)
    except Exception as e:
        logger.error(f"Upload save error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save image: {e}")

    return {
        "filename": result["filename"],
        "original_filename": result.get("original_filename"),
        "dimensions": result["dimensions"],
        "file_size": result["file_size"],
        "preview_url": result["preview_url"],
    }


# ---------------------------------------------------------------------------
# POST /api/analyze
# ---------------------------------------------------------------------------

@router.post("/analyze")
async def analyze_image(
    request: Request,
    filename: Optional[str] = None,
) -> dict[str, Any]:
    """
    Run degradation analysis on an uploaded image or a previously saved filename.
    Returns degradation_type, severity, severity_level, confidence, details.
    """
    from backend.degradation.analyzer import analyze_image as _analyze
    from backend.services.image_service import validate_image, save_upload, ImageServiceError

    upload_filename = filename
    # Check JSON body if provided
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
            if isinstance(body, dict):
                upload_filename = body.get("filename") or upload_filename
        except Exception:
            pass
    elif "multipart/form-data" in content_type:
        try:
            form = await request.form()
            upload_filename = form.get("filename") or upload_filename
            if "file" in form and hasattr(form["file"], "file"):
                file_obj = form["file"]
                await validate_image(file_obj)
                res = await save_upload(file_obj)
                upload_filename = res["filename"]
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    if not upload_filename:
        raise HTTPException(status_code=400, detail="Provide 'filename' parameter or upload an image.")

    image_path = UPLOAD_DIR / upload_filename
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {upload_filename}")

    try:
        analysis = _analyze(image_path)
    except Exception as e:
        logger.error(f"Degradation analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    return analysis


# ---------------------------------------------------------------------------
# POST /api/enhance
# ---------------------------------------------------------------------------

@router.post("/enhance")
async def enhance_image(
    request: Request,
    filename: Optional[str] = None,
) -> dict[str, Any]:
    """
    Run Zero-DCE + adaptive enhancement controller.
    Returns enhanced image URL and analysis info.
    """
    from backend.degradation.analyzer import analyze_image as _analyze
    from backend.enhancement.adaptive import get_controller
    from backend.services.image_service import validate_image, save_upload, ImageServiceError

    upload_filename = filename
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
            if isinstance(body, dict):
                upload_filename = body.get("filename") or upload_filename
        except Exception:
            pass
    elif "multipart/form-data" in content_type:
        try:
            form = await request.form()
            upload_filename = form.get("filename") or upload_filename
            if "file" in form and hasattr(form["file"], "file"):
                file_obj = form["file"]
                await validate_image(file_obj)
                res = await save_upload(file_obj)
                upload_filename = res["filename"]
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    if not upload_filename:
        raise HTTPException(status_code=400, detail="Provide 'filename' parameter or upload an image.")

    image_path = UPLOAD_DIR / upload_filename
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {upload_filename}")

    try:
        analysis = _analyze(image_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    try:
        controller = get_controller(_get_zero_dce())
        enhancement_result = controller.enhance(image_path, analysis)
    except Exception as e:
        logger.error(f"Enhancement error: {e}")
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {e}")

    stem = Path(upload_filename or "image").stem
    enhanced_pil = enhancement_result["enhanced_image"]
    _, enhanced_url = _save_pil_to_enhanced(enhanced_pil, stem)
    enh_name = Path(enhanced_url).name

    return {
        "enhanced_url": enhanced_url,
        "enhanced_filename": enh_name,
        "original_filename": upload_filename,
        "dimensions": f"{enhanced_pil.width} × {enhanced_pil.height}",
        "degradation_type": enhancement_result["degradation_type"],
        "severity_level": enhancement_result["severity_level"],
        "strategy_chosen": enhancement_result["strategy_chosen"],
        "strategy": enhancement_result["strategy_chosen"],
        "model_used": enhancement_result["model_used"],
        "processing_time_ms": enhancement_result["processing_time_ms"],
        "analysis": analysis,
    }


# ---------------------------------------------------------------------------
# POST /api/detect
# ---------------------------------------------------------------------------

@router.post("/detect")
async def detect_objects(
    request: Request,
    filename: Optional[str] = None,
    enhanced_filename: Optional[str] = None,
    conf_threshold: float = 0.25,
) -> dict[str, Any]:
    """
    Run YOLOv8 detection on original and/or enhanced image.
    Returns bounding boxes, labels, scores.
    """
    from backend.services.image_service import validate_image, save_upload, ImageServiceError

    upload_filename = filename
    enh_fname = enhanced_filename
    thresh = conf_threshold

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
            if isinstance(body, dict):
                upload_filename = body.get("filename") or upload_filename
                enh_fname = body.get("enhanced_filename") or enh_fname
                if "conf_threshold" in body:
                    thresh = float(body["conf_threshold"])
        except Exception:
            pass
    elif "multipart/form-data" in content_type:
        try:
            form = await request.form()
            upload_filename = form.get("filename") or upload_filename
            enh_fname = form.get("enhanced_filename") or enh_fname
            if "conf_threshold" in form:
                thresh = float(form["conf_threshold"])
        except Exception:
            pass

    if not upload_filename:
        raise HTTPException(status_code=400, detail="Provide 'filename' parameter.")

    image_path = UPLOAD_DIR / upload_filename
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {upload_filename}")

    detector = _get_detector()

    try:
        orig_detections = detector.detect(image_path, conf_threshold=thresh)
    except Exception as e:
        logger.error(f"Detection error on original: {e}")
        orig_detections = []

    # Annotate original
    orig_annotated_url: Optional[str] = None
    try:
        ann_path = detector.annotate(image_path, orig_detections, f"orig_{upload_filename}")
        orig_annotated_url = f"/images/annotated/{ann_path.name}"
    except Exception as e:
        logger.warning(f"Annotation error: {e}")

    # Detection on enhanced image if provided
    enh_detections: list = []
    enh_annotated_url: Optional[str] = None
    if enh_fname:
        enh_path = ENHANCED_DIR / enh_fname
        if enh_path.exists():
            try:
                enh_detections = detector.detect(enh_path, conf_threshold=thresh)
                ann_enh = detector.annotate(enh_path, enh_detections, f"enh_{enh_fname}")
                enh_annotated_url = f"/images/annotated/{ann_enh.name}"
            except Exception as e:
                logger.warning(f"Detection/annotation on enhanced failed: {e}")

    return {
        "original": {
            "filename": upload_filename,
            "detections": orig_detections,
            "count": len(orig_detections),
            "annotated_url": orig_annotated_url,
        },
        "enhanced": {
            "filename": enh_fname,
            "detections": enh_detections,
            "count": len(enh_detections),
            "annotated_url": enh_annotated_url,
        },
    }


# ---------------------------------------------------------------------------
# POST /api/evaluate
# ---------------------------------------------------------------------------

@router.post("/evaluate")
async def evaluate_images(
    request: Request,
    original_filename: Optional[str] = None,
    filename: Optional[str] = None,
    enhanced_filename: Optional[str] = None,
    reference_filename: Optional[str] = None,
) -> dict[str, Any]:
    """
    Compute PSNR, SSIM, LPIPS (if reference available), and brightness metrics.
    """
    from backend.evaluation.metrics import (
        compute_psnr, compute_ssim, compute_lpips, compute_brightness
    )

    orig_fname = original_filename or filename
    enh_fname = enhanced_filename
    ref_fname = reference_filename

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
            if isinstance(body, dict):
                orig_fname = body.get("original_filename") or body.get("filename") or orig_fname
                enh_fname = body.get("enhanced_filename") or enh_fname
                ref_fname = body.get("reference_filename") or ref_fname
        except Exception:
            pass
    elif "multipart/form-data" in content_type:
        try:
            form = await request.form()
            orig_fname = form.get("original_filename") or form.get("filename") or orig_fname
            enh_fname = form.get("enhanced_filename") or enh_fname
            ref_fname = form.get("reference_filename") or ref_fname
        except Exception:
            pass

    if not orig_fname and not enh_fname:
        raise HTTPException(
            status_code=400,
            detail="Provide at least 'original_filename' or 'enhanced_filename'.",
        )

    orig_array: Optional[np.ndarray] = None
    enh_array: Optional[np.ndarray] = None
    ref_array: Optional[np.ndarray] = None

    if orig_fname:
        orig_path = UPLOAD_DIR / orig_fname
        if orig_path.exists():
            orig_array = _load_array(orig_path)
        else:
            raise HTTPException(status_code=404, detail=f"Original not found: {orig_fname}")

    if enh_fname:
        enh_path = ENHANCED_DIR / enh_fname
        if enh_path.exists():
            enh_array = _load_array(enh_path)
        else:
            raise HTTPException(status_code=404, detail=f"Enhanced not found: {enh_fname}")

    if ref_fname:
        ref_path = UPLOAD_DIR / ref_fname
        if ref_path.exists():
            ref_array = _load_array(ref_path)

    # Use reference if available, else compare original vs enhanced
    ref = ref_array if ref_array is not None else orig_array
    target = enh_array

    psnr = compute_psnr(ref, target) if ref is not None and target is not None else None
    ssim = compute_ssim(ref, target) if ref is not None and target is not None else None
    lpips = compute_lpips(ref, target) if ref is not None and target is not None else None
    brightness_orig = compute_brightness(orig_array) if orig_array is not None else None
    brightness_enh = compute_brightness(enh_array) if enh_array is not None else None

    return {
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips,
        "brightness_original": brightness_orig,
        "brightness_enhanced": brightness_enh,
        "reference_used": ref_fname or orig_fname,
    }


# ---------------------------------------------------------------------------
# POST /api/process  (all-in-one pipeline)
# ---------------------------------------------------------------------------

@router.post("/process")
async def process_image(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    All-in-one pipeline: upload → analyze → enhance → detect → evaluate.
    Returns full result JSON.
    """
    from backend.services.image_service import validate_image, save_upload, ImageServiceError
    from backend.degradation.analyzer import analyze_image as _analyze
    from backend.enhancement.adaptive import get_controller
    from backend.evaluation.metrics import (
        compute_psnr, compute_ssim, compute_lpips, compute_brightness
    )
    from backend.services.report_service import generate_report

    t_start = time.perf_counter()

    # 1. Upload
    try:
        await validate_image(file)
        upload_result = await save_upload(file)
    except ImageServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

    orig_path = Path(upload_result["filepath"])
    upload_filename = upload_result["filename"]

    # 2. Analyze
    try:
        analysis = _analyze(orig_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    # 3. Enhance
    try:
        controller = get_controller(_get_zero_dce())
        enhancement_result = controller.enhance(orig_path, analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {e}")

    stem = Path(upload_filename).stem
    enhanced_pil = enhancement_result["enhanced_image"]
    enhanced_path, enhanced_url = _save_pil_to_enhanced(enhanced_pil, stem)

    # 4. Detect (original + enhanced)
    detector = _get_detector()
    orig_detections: list = []
    enh_detections: list = []
    orig_annotated_url: Optional[str] = None
    enh_annotated_url: Optional[str] = None

    try:
        orig_detections = detector.detect(orig_path)
        ann_orig = detector.annotate(orig_path, orig_detections, f"orig_{upload_filename}")
        orig_annotated_url = f"/images/annotated/{ann_orig.name}"
    except Exception as e:
        logger.warning(f"Detection/annotation on original failed: {e}")

    try:
        enh_detections = detector.detect(enhanced_path)
        ann_enh = detector.annotate(enhanced_path, enh_detections, f"enh_{Path(upload_filename).stem}.png")
        enh_annotated_url = f"/images/annotated/{ann_enh.name}"
    except Exception as e:
        logger.warning(f"Detection/annotation on enhanced failed: {e}")

    # 5. Evaluate
    orig_array = _load_array(orig_path)
    enh_array = _load_array(enhanced_path)
    psnr = compute_psnr(orig_array, enh_array)
    ssim = compute_ssim(orig_array, enh_array)
    lpips = compute_lpips(orig_array, enh_array)
    brightness_orig = compute_brightness(orig_array)
    brightness_enh = compute_brightness(enh_array)

    t_end = time.perf_counter()
    total_time_ms = round((t_end - t_start) * 1000, 2)

    # 6. Save to history
    session_data = {
        "filename": upload_filename,
        "degradation_type": analysis.get("degradation_type"),
        "severity": analysis.get("severity_level"),
        "strategy": enhancement_result.get("strategy_chosen"),
        "psnr": psnr,
        "ssim": ssim,
        "brightness_orig": brightness_orig,
        "brightness_enh": brightness_enh,
        "detection_orig": len(orig_detections),
        "detection_enh": len(enh_detections),
        "processing_time_ms": total_time_ms,
        # extra for report
        "severity_score": analysis.get("severity_score"),
        "confidence": analysis.get("confidence"),
        "model_used": enhancement_result.get("model_used"),
        "lpips": lpips,
        "detections_original": orig_detections,
        "detections_enhanced": enh_detections,
        "details": analysis.get("details", {}),
    }
    session_id = None
    try:
        session_id = _get_history().save_session(session_data)
    except Exception as e:
        logger.warning(f"Failed to save session to history: {e}")

    # 7. Generate report
    report_path: Optional[str] = None
    try:
        rp = generate_report(session_data)
        report_path = f"/images/reports/{rp.name}"
    except Exception as e:
        logger.warning(f"Report generation failed: {e}")

    return {
        "session_id": session_id,
        "upload": {
            "filename": upload_filename,
            "original_filename": upload_result.get("original_filename"),
            "dimensions": upload_result["dimensions"],
            "file_size": upload_result["file_size"],
            "preview_url": upload_result["preview_url"],
        },
        "analysis": analysis,
        "enhancement": {
            "enhanced_url": enhanced_url,
            "strategy_chosen": enhancement_result["strategy_chosen"],
            "model_used": enhancement_result["model_used"],
            "processing_time_ms": enhancement_result["processing_time_ms"],
        },
        "detection": {
            "original": {
                "detections": orig_detections,
                "count": len(orig_detections),
                "annotated_url": orig_annotated_url,
            },
            "enhanced": {
                "detections": enh_detections,
                "count": len(enh_detections),
                "annotated_url": enh_annotated_url,
            },
        },
        "metrics": {
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips,
            "brightness_original": brightness_orig,
            "brightness_enhanced": brightness_enh,
        },
        "report_url": report_path,
        "total_processing_time_ms": total_time_ms,
    }


# ---------------------------------------------------------------------------
# POST /api/batch-process
# ---------------------------------------------------------------------------

@router.post("/batch-process")
async def batch_process(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    """
    Process multiple images through the full pipeline.
    Returns list of per-image results.
    """
    from backend.services.image_service import validate_image, save_upload, ImageServiceError
    from backend.degradation.analyzer import analyze_image as _analyze
    from backend.enhancement.adaptive import get_controller
    from backend.evaluation.metrics import (
        compute_psnr, compute_ssim, compute_brightness
    )

    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Batch limit is 20 images.")

    results = []
    controller = get_controller(_get_zero_dce())
    detector = _get_detector()

    for file in files:
        file_result: dict[str, Any] = {"original_filename": file.filename}
        try:
            await validate_image(file)
            upload_result = await save_upload(file)
            orig_path = Path(upload_result["filepath"])
            upload_filename = upload_result["filename"]
            stem = Path(upload_filename).stem

            analysis = _analyze(orig_path)

            enhancement_result = controller.enhance(orig_path, analysis)
            enhanced_pil = enhancement_result["enhanced_image"]
            enhanced_path, enhanced_url = _save_pil_to_enhanced(enhanced_pil, stem)

            orig_array = _load_array(orig_path)
            enh_array = _load_array(enhanced_path)
            psnr = compute_psnr(orig_array, enh_array)
            ssim = compute_ssim(orig_array, enh_array)
            brightness_orig = compute_brightness(orig_array)
            brightness_enh = compute_brightness(enh_array)

            orig_detections = detector.detect(orig_path)
            enh_detections = detector.detect(enhanced_path)

            session_data = {
                "filename": upload_filename,
                "degradation_type": analysis.get("degradation_type"),
                "severity": analysis.get("severity_level"),
                "strategy": enhancement_result.get("strategy_chosen"),
                "psnr": psnr,
                "ssim": ssim,
                "brightness_orig": brightness_orig,
                "brightness_enh": brightness_enh,
                "detection_orig": len(orig_detections),
                "detection_enh": len(enh_detections),
                "processing_time_ms": enhancement_result.get("processing_time_ms"),
            }
            try:
                _get_history().save_session(session_data)
            except Exception:
                pass

            file_result.update({
                "status": "success",
                "filename": upload_filename,
                "preview_url": upload_result["preview_url"],
                "enhanced_url": enhanced_url,
                "analysis": analysis,
                "strategy": enhancement_result.get("strategy_chosen"),
                "metrics": {
                    "psnr": psnr,
                    "ssim": ssim,
                    "brightness_original": brightness_orig,
                    "brightness_enhanced": brightness_enh,
                },
                "detection_count_original": len(orig_detections),
                "detection_count_enhanced": len(enh_detections),
                "processing_time_ms": enhancement_result.get("processing_time_ms"),
            })

        except ImageServiceError as e:
            file_result["status"] = "error"
            file_result["error"] = str(e)
        except Exception as e:
            logger.error(f"Batch process error for {file.filename}: {e}")
            file_result["status"] = "error"
            file_result["error"] = str(e)

        results.append(file_result)

    success_count = sum(1 for r in results if r.get("status") == "success")
    return {
        "total": len(results),
        "success": success_count,
        "failed": len(results) - success_count,
        "results": results,
    }
