"""
History Service
================
SQLite-backed session history for image processing runs.
DB: outputs/history.db
Table: processing_sessions
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "outputs" / "history.db"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS processing_sessions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp         TEXT NOT NULL,
    filename          TEXT,
    degradation_type  TEXT,
    severity          TEXT,
    strategy          TEXT,
    psnr              REAL,
    ssim              REAL,
    brightness_orig   REAL,
    brightness_enh    REAL,
    detection_orig    INTEGER,
    detection_enh     INTEGER,
    processing_time_ms REAL
);
"""


class HistoryService:
    """Thread-safe (single-connection) SQLite history manager."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = Path(db_path) if db_path else DB_PATH
        self._ensure_db()

    def _ensure_db(self) -> None:
        """Create the database and table if they don't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(CREATE_TABLE_SQL)
            conn.commit()
        logger.debug(f"History DB ready at {self.db_path}")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def save_session(self, data: dict[str, Any]) -> int:
        """
        Persist a processing session.

        Args:
            data: Dict with any subset of:
                  filename, degradation_type, severity, strategy,
                  psnr, ssim, brightness_orig, brightness_enh,
                  detection_orig, detection_enh, processing_time_ms

        Returns:
            The new row's id.
        """
        timestamp = data.get("timestamp") or datetime.now(timezone.utc).isoformat()
        sql = """
        INSERT INTO processing_sessions
            (timestamp, filename, degradation_type, severity, strategy,
             psnr, ssim, brightness_orig, brightness_enh,
             detection_orig, detection_enh, processing_time_ms)
        VALUES
            (:timestamp, :filename, :degradation_type, :severity, :strategy,
             :psnr, :ssim, :brightness_orig, :brightness_enh,
             :detection_orig, :detection_enh, :processing_time_ms)
        """
        params = {
            "timestamp": timestamp,
            "filename": data.get("filename"),
            "degradation_type": data.get("degradation_type"),
            "severity": data.get("severity"),
            "strategy": data.get("strategy"),
            "psnr": data.get("psnr"),
            "ssim": data.get("ssim"),
            "brightness_orig": data.get("brightness_orig"),
            "brightness_enh": data.get("brightness_enh"),
            "detection_orig": data.get("detection_orig"),
            "detection_enh": data.get("detection_enh"),
            "processing_time_ms": data.get("processing_time_ms"),
        }
        with self._connect() as conn:
            cursor = conn.execute(sql, params)
            conn.commit()
            row_id = cursor.lastrowid
        logger.debug(f"Saved session id={row_id} for {data.get('filename')}")
        return row_id

    def get_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """
        Retrieve the most recent processing sessions.

        Args:
            limit: Maximum number of rows to return.

        Returns:
            List of dicts ordered by timestamp descending.
        """
        sql = """
        SELECT * FROM processing_sessions
        ORDER BY id DESC
        LIMIT ?
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (limit,)).fetchall()
        return [dict(row) for row in rows]

    def clear_history(self) -> None:
        """Delete all history records."""
        with self._connect() as conn:
            conn.execute("DELETE FROM processing_sessions")
            conn.commit()


# Module-level singleton
_history_service: Optional[HistoryService] = None


def get_history_service() -> HistoryService:
    global _history_service
    if _history_service is None:
        _history_service = HistoryService()
    return _history_service
