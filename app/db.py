from __future__ import annotations

import sqlite3
from pathlib import Path

from app.config import settings

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS lectures (
    lecture_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    audio_filename TEXT NOT NULL,
    audio_path TEXT NOT NULL,
    language TEXT NOT NULL,
    status TEXT NOT NULL,
    error_message TEXT,
    transcript_path TEXT,
    notes_json_path TEXT,
    notes_md_path TEXT,
    notes_html_path TEXT,
    notes_pdf_path TEXT,
    notes_docx_path TEXT,
    notes_review_path TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS processing_jobs (
    job_id TEXT PRIMARY KEY,
    lecture_id TEXT NOT NULL,
    status TEXT NOT NULL,
    error_message TEXT,
    queued_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (lecture_id) REFERENCES lectures (lecture_id)
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    lecture_id TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    path TEXT NOT NULL,
    content_type TEXT NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (lecture_id) REFERENCES lectures (lecture_id)
);

CREATE TABLE IF NOT EXISTS performance_runs (
    run_id TEXT PRIMARY KEY,
    lecture_id TEXT NOT NULL,
    job_id TEXT,
    queue_backend TEXT NOT NULL,
    worker_backend TEXT NOT NULL,
    llm_enabled INTEGER NOT NULL DEFAULT 0,
    llm_model TEXT NOT NULL DEFAULT '',
    llm_chunk_count INTEGER NOT NULL DEFAULT 0,
    llm_total_ms REAL NOT NULL DEFAULT 0,
    asr_provider TEXT NOT NULL DEFAULT '',
    asr_model_size TEXT NOT NULL DEFAULT '',
    requested_device TEXT NOT NULL DEFAULT '',
    requested_compute_type TEXT NOT NULL DEFAULT '',
    actual_device TEXT NOT NULL DEFAULT '',
    actual_compute_type TEXT NOT NULL DEFAULT '',
    model_backend TEXT NOT NULL DEFAULT '',
    detected_language TEXT NOT NULL DEFAULT '',
    total_ms REAL NOT NULL,
    preprocess_ms REAL NOT NULL,
    transcribe_ms REAL NOT NULL,
    marker_detection_ms REAL NOT NULL,
    note_compose_ms REAL NOT NULL,
    review_apply_ms REAL NOT NULL,
    export_ms REAL NOT NULL,
    segment_count INTEGER NOT NULL,
    marker_count INTEGER NOT NULL,
    note_block_count INTEGER NOT NULL,
    audio_duration_ms INTEGER NOT NULL,
    transcript_char_count INTEGER NOT NULL,
    throughput_audio_x REAL NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (lecture_id) REFERENCES lectures (lecture_id),
    FOREIGN KEY (job_id) REFERENCES processing_jobs (job_id)
);
"""


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or settings.database_path
    connection = sqlite3.connect(path, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database() -> None:
    connection = get_connection()
    try:
        connection.executescript(SCHEMA_SQL)
        _ensure_column(connection, "lectures", "notes_pdf_path", "TEXT")
        _ensure_column(connection, "lectures", "notes_docx_path", "TEXT")
        _ensure_column(connection, "lectures", "notes_review_path", "TEXT")
        _ensure_column(connection, "performance_runs", "llm_enabled", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(connection, "performance_runs", "llm_model", "TEXT NOT NULL DEFAULT ''")
        _ensure_column(connection, "performance_runs", "llm_chunk_count", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(connection, "performance_runs", "llm_total_ms", "REAL NOT NULL DEFAULT 0")
        _ensure_column(connection, "performance_runs", "asr_provider", "TEXT NOT NULL DEFAULT ''")
        _ensure_column(connection, "performance_runs", "asr_model_size", "TEXT NOT NULL DEFAULT ''")
        _ensure_column(connection, "performance_runs", "requested_device", "TEXT NOT NULL DEFAULT ''")
        _ensure_column(connection, "performance_runs", "requested_compute_type", "TEXT NOT NULL DEFAULT ''")
        _ensure_column(connection, "performance_runs", "actual_device", "TEXT NOT NULL DEFAULT ''")
        _ensure_column(connection, "performance_runs", "actual_compute_type", "TEXT NOT NULL DEFAULT ''")
        _ensure_column(connection, "performance_runs", "model_backend", "TEXT NOT NULL DEFAULT ''")
        _ensure_column(connection, "performance_runs", "detected_language", "TEXT NOT NULL DEFAULT ''")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_processing_jobs_lecture_status ON processing_jobs (lecture_id, status)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_lecture_type ON artifacts (lecture_id, artifact_type)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_performance_runs_lecture_created ON performance_runs (lecture_id, created_at DESC)")
        connection.commit()
    finally:
        connection.close()


def _ensure_column(connection: sqlite3.Connection, table_name: str, column_name: str, column_type: str) -> None:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    existing = {row["name"] for row in rows}
    if column_name in existing:
        return
    connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")


if __name__ == "__main__":  # pragma: no cover
    initialize_database()
    print(f"Initialized database at {settings.database_path}")
