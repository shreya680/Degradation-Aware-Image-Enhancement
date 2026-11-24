"""
Report Service
===============
Generates TXT reports for processing sessions.
Saves to outputs/reports/.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"


def generate_report(session_data: dict[str, Any]) -> Path:
    """
    Generate a human-readable TXT report for a processing session.

    Args:
        session_data: Dict containing processing results.

    Returns:
        Path to the saved report file.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc)
    ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
    filename = session_data.get("filename", "unknown")
    stem = Path(filename).stem if filename else "unknown"
    report_filename = f"report_{stem}_{ts_str}.txt"
    report_path = REPORTS_DIR / report_filename

    lines = [
        "=" * 60,
        "DEGRADATION-AWARE IMAGE ENHANCEMENT REPORT",
        "=" * 60,
        f"Generated At  : {timestamp.isoformat()}",
        f"Image File    : {filename}",
        "",
        "--- DEGRADATION ANALYSIS ---",
        f"Degradation Type  : {session_data.get('degradation_type', 'N/A')}",
        f"Severity Level    : {session_data.get('severity', 'N/A')}",
        f"Severity Score    : {session_data.get('severity_score', 'N/A')}",
        f"Confidence        : {session_data.get('confidence', 'N/A')}",
        "",
        "--- ENHANCEMENT STRATEGY ---",
        f"Strategy Chosen   : {session_data.get('strategy', 'N/A')}",
        f"Model Used        : {session_data.get('model_used', 'N/A')}",
        f"Processing Time   : {session_data.get('processing_time_ms', 'N/A')} ms",
        "",
        "--- IMAGE QUALITY METRICS ---",
        f"PSNR (dB)         : {session_data.get('psnr', 'N/A')}",
        f"SSIM              : {session_data.get('ssim', 'N/A')}",
        f"LPIPS             : {session_data.get('lpips', 'N/A')}",
        f"Brightness (orig) : {session_data.get('brightness_orig', 'N/A')}",
        f"Brightness (enh)  : {session_data.get('brightness_enh', 'N/A')}",
        "",
        "--- OBJECT DETECTION ---",
        f"Objects (original): {session_data.get('detection_orig', 'N/A')}",
        f"Objects (enhanced): {session_data.get('detection_enh', 'N/A')}",
    ]

    # Append detection details if present
    orig_dets = session_data.get("detections_original", [])
    if orig_dets:
        lines.append("")
        lines.append("Original detections:")
        for d in orig_dets:
            lines.append(f"  - {d.get('class_name')} ({d.get('confidence', 0):.2f}) @ {d.get('bbox')}")

    enh_dets = session_data.get("detections_enhanced", [])
    if enh_dets:
        lines.append("")
        lines.append("Enhanced detections:")
        for d in enh_dets:
            lines.append(f"  - {d.get('class_name')} ({d.get('confidence', 0):.2f}) @ {d.get('bbox')}")

    # Additional raw details if present
    details = session_data.get("details", {})
    if details:
        lines.append("")
        lines.append("--- RAW ANALYSIS DETAILS ---")
        for k, v in details.items():
            lines.append(f"  {k}: {v}")

    lines.append("")
    lines.append("=" * 60)
    lines.append("END OF REPORT")
    lines.append("=" * 60)

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Report saved to {report_path}")
    return report_path
