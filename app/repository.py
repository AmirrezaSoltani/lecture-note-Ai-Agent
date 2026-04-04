from __future__ import annotations

import uuid
from pathlib import Path

from app.db import get_connection
from app.models import ArtifactRecord, JobRecord, JobStatus, LectureRecord, LectureStatus, PerformanceRunRecord, utc_now_iso


class LectureRepository:
    def create(self, record: LectureRecord) -> None:
        connection = get_connection()
        try:
            connection.execute(
                """
                INSERT INTO lectures (
                    lecture_id, title, audio_filename, audio_path, language, status,
                    error_message, transcript_path, notes_json_path, notes_md_path, notes_html_path,
                    notes_pdf_path, notes_docx_path, notes_review_path, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.lecture_id,
                    record.title,
                    record.audio_filename,
                    record.audio_path,
                    record.language,
                    record.status,
                    record.error_message,
                    record.transcript_path,
                    record.notes_json_path,
                    record.notes_md_path,
                    record.notes_html_path,
                    record.notes_pdf_path,
                    record.notes_docx_path,
                    record.notes_review_path,
                    record.created_at,
                    record.updated_at,
                ),
            )
            connection.commit()
        finally:
            connection.close()

    def update_status(self, lecture_id: str, status: str, error_message: str | None = None) -> None:
        connection = get_connection()
        try:
            connection.execute(
                """
                UPDATE lectures
                SET status = ?, error_message = ?, updated_at = ?
                WHERE lecture_id = ?
                """,
                (status, error_message, utc_now_iso(), lecture_id),
            )
            connection.commit()
        finally:
            connection.close()

    def update_artifacts(self, lecture_id: str, artifact_paths: dict[str, str]) -> None:
        connection = get_connection()
        try:
            connection.execute(
                """
                UPDATE lectures
                SET transcript_path = ?, notes_json_path = ?, notes_md_path = ?, notes_html_path = ?,
                    notes_pdf_path = ?, notes_docx_path = ?, notes_review_path = ?, updated_at = ?, status = ?
                WHERE lecture_id = ?
                """,
                (
                    artifact_paths.get("transcript"),
                    artifact_paths.get("notes_json"),
                    artifact_paths.get("notes_md"),
                    artifact_paths.get("notes_html"),
                    artifact_paths.get("notes_pdf"),
                    artifact_paths.get("notes_docx"),
                    artifact_paths.get("review_json"),
                    utc_now_iso(),
                    LectureStatus.COMPLETED,
                    lecture_id,
                ),
            )
            connection.commit()
        finally:
            connection.close()

    def get(self, lecture_id: str) -> LectureRecord | None:
        connection = get_connection()
        try:
            row = connection.execute("SELECT * FROM lectures WHERE lecture_id = ?", (lecture_id,)).fetchone()
            return self._row_to_record(row) if row else None
        finally:
            connection.close()

    def list_all(self) -> list[LectureRecord]:
        connection = get_connection()
        try:
            rows = connection.execute("SELECT * FROM lectures ORDER BY created_at DESC").fetchall()
            return [self._row_to_record(row) for row in rows]
        finally:
            connection.close()

    @staticmethod
    def _row_to_record(row) -> LectureRecord:
        return LectureRecord(
            lecture_id=row["lecture_id"],
            title=row["title"],
            audio_filename=row["audio_filename"],
            audio_path=row["audio_path"],
            language=row["language"],
            status=row["status"],
            error_message=row["error_message"],
            transcript_path=row["transcript_path"],
            notes_json_path=row["notes_json_path"],
            notes_md_path=row["notes_md_path"],
            notes_html_path=row["notes_html_path"],
            notes_pdf_path=row["notes_pdf_path"],
            notes_docx_path=row["notes_docx_path"],
            notes_review_path=row["notes_review_path"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class JobRepository:
    def create(self, lecture_id: str) -> JobRecord:
        job = JobRecord(job_id=str(uuid.uuid4()), lecture_id=lecture_id)
        connection = get_connection()
        try:
            connection.execute(
                """
                INSERT INTO processing_jobs (
                    job_id, lecture_id, status, error_message, queued_at,
                    started_at, finished_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.job_id,
                    job.lecture_id,
                    job.status,
                    job.error_message,
                    job.queued_at,
                    job.started_at,
                    job.finished_at,
                    job.created_at,
                    job.updated_at,
                ),
            )
            connection.commit()
        finally:
            connection.close()
        return job

    def get(self, job_id: str) -> JobRecord | None:
        connection = get_connection()
        try:
            row = connection.execute("SELECT * FROM processing_jobs WHERE job_id = ?", (job_id,)).fetchone()
            return self._row_to_job(row) if row else None
        finally:
            connection.close()

    def list_for_lecture(self, lecture_id: str) -> list[JobRecord]:
        connection = get_connection()
        try:
            rows = connection.execute(
                "SELECT * FROM processing_jobs WHERE lecture_id = ? ORDER BY created_at DESC",
                (lecture_id,),
            ).fetchall()
            return [self._row_to_job(row) for row in rows]
        finally:
            connection.close()

    def latest_for_lecture(self, lecture_id: str) -> JobRecord | None:
        jobs = self.list_for_lecture(lecture_id)
        return jobs[0] if jobs else None

    def has_active_job(self, lecture_id: str) -> bool:
        connection = get_connection()
        try:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM processing_jobs
                WHERE lecture_id = ? AND status IN (?, ?)
                """,
                (lecture_id, JobStatus.QUEUED, JobStatus.RUNNING),
            ).fetchone()
            return bool(row["count"])
        finally:
            connection.close()

    def mark_running(self, job_id: str) -> None:
        self._update_status(job_id, JobStatus.RUNNING, started_at=utc_now_iso())

    def mark_completed(self, job_id: str) -> None:
        self._update_status(job_id, JobStatus.COMPLETED, finished_at=utc_now_iso())

    def mark_failed(self, job_id: str, error_message: str) -> None:
        self._update_status(job_id, JobStatus.FAILED, error_message=error_message, finished_at=utc_now_iso())

    def queued_jobs(self) -> list[JobRecord]:
        connection = get_connection()
        try:
            rows = connection.execute(
                "SELECT * FROM processing_jobs WHERE status = ? ORDER BY queued_at ASC",
                (JobStatus.QUEUED,),
            ).fetchall()
            return [self._row_to_job(row) for row in rows]
        finally:
            connection.close()

    def mark_stale_running_jobs_failed(self, message: str) -> None:
        connection = get_connection()
        try:
            connection.execute(
                """
                UPDATE processing_jobs
                SET status = ?, error_message = ?, finished_at = ?, updated_at = ?
                WHERE status = ?
                """,
                (JobStatus.FAILED, message, utc_now_iso(), utc_now_iso(), JobStatus.RUNNING),
            )
            connection.commit()
        finally:
            connection.close()

    def _update_status(
        self,
        job_id: str,
        status: str,
        error_message: str | None = None,
        started_at: str | None = None,
        finished_at: str | None = None,
    ) -> None:
        connection = get_connection()
        try:
            current = connection.execute("SELECT * FROM processing_jobs WHERE job_id = ?", (job_id,)).fetchone()
            if current is None:
                raise ValueError(f"Unknown job: {job_id}")
            connection.execute(
                """
                UPDATE processing_jobs
                SET status = ?, error_message = ?, started_at = ?, finished_at = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (
                    status,
                    error_message if error_message is not None else current["error_message"],
                    started_at if started_at is not None else current["started_at"],
                    finished_at if finished_at is not None else current["finished_at"],
                    utc_now_iso(),
                    job_id,
                ),
            )
            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def _row_to_job(row) -> JobRecord:
        return JobRecord(
            job_id=row["job_id"],
            lecture_id=row["lecture_id"],
            status=row["status"],
            error_message=row["error_message"],
            queued_at=row["queued_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class ArtifactRepository:
    def replace_for_lecture(self, lecture_id: str, records: list[ArtifactRecord]) -> None:
        connection = get_connection()
        try:
            connection.execute("DELETE FROM artifacts WHERE lecture_id = ?", (lecture_id,))
            connection.executemany(
                """
                INSERT INTO artifacts (
                    artifact_id, lecture_id, artifact_type, path, content_type, file_size_bytes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        record.artifact_id,
                        record.lecture_id,
                        record.artifact_type,
                        record.path,
                        record.content_type,
                        record.file_size_bytes,
                        record.created_at,
                    )
                    for record in records
                ],
            )
            connection.commit()
        finally:
            connection.close()

    def list_for_lecture(self, lecture_id: str) -> list[ArtifactRecord]:
        connection = get_connection()
        try:
            rows = connection.execute(
                "SELECT * FROM artifacts WHERE lecture_id = ? ORDER BY artifact_type ASC",
                (lecture_id,),
            ).fetchall()
            return [self._row_to_artifact(row) for row in rows]
        finally:
            connection.close()

    def find_by_type(self, lecture_id: str, artifact_type: str) -> ArtifactRecord | None:
        connection = get_connection()
        try:
            row = connection.execute(
                "SELECT * FROM artifacts WHERE lecture_id = ? AND artifact_type = ?",
                (lecture_id, artifact_type),
            ).fetchone()
            return self._row_to_artifact(row) if row else None
        finally:
            connection.close()

    @staticmethod
    def _row_to_artifact(row) -> ArtifactRecord:
        return ArtifactRecord(
            artifact_id=row["artifact_id"],
            lecture_id=row["lecture_id"],
            artifact_type=row["artifact_type"],
            path=row["path"],
            content_type=row["content_type"],
            file_size_bytes=row["file_size_bytes"],
            created_at=row["created_at"],
        )


class PerformanceRunRepository:
    def create(self, record: PerformanceRunRecord) -> None:
        connection = get_connection()
        try:
            connection.execute(
                """
                INSERT INTO performance_runs (
                    run_id, lecture_id, job_id, queue_backend, worker_backend,
                    llm_enabled, llm_model, llm_chunk_count, llm_total_ms,
                    asr_provider, asr_model_size, requested_device, requested_compute_type,
                    actual_device, actual_compute_type, model_backend, detected_language,
                    total_ms, preprocess_ms, transcribe_ms, marker_detection_ms,
                    note_compose_ms, review_apply_ms, export_ms, segment_count, marker_count,
                    note_block_count, audio_duration_ms, transcript_char_count, throughput_audio_x, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.run_id, record.lecture_id, record.job_id, record.queue_backend, record.worker_backend,
                    int(record.llm_enabled), record.llm_model, record.llm_chunk_count, record.llm_total_ms,
                    record.asr_provider, record.asr_model_size, record.requested_device, record.requested_compute_type,
                    record.actual_device, record.actual_compute_type, record.model_backend, record.detected_language,
                    record.total_ms, record.preprocess_ms, record.transcribe_ms, record.marker_detection_ms,
                    record.note_compose_ms, record.review_apply_ms, record.export_ms, record.segment_count, record.marker_count,
                    record.note_block_count, record.audio_duration_ms, record.transcript_char_count, record.throughput_audio_x, record.created_at,
                ),
            )
            connection.commit()
        finally:
            connection.close()

    def list_for_lecture(self, lecture_id: str, limit: int | None = None) -> list[PerformanceRunRecord]:
        connection = get_connection()
        try:
            query = "SELECT * FROM performance_runs WHERE lecture_id = ? ORDER BY created_at DESC"
            params: tuple[object, ...] = (lecture_id,)
            if limit is not None:
                query += " LIMIT ?"
                params = (lecture_id, limit)
            rows = connection.execute(query, params).fetchall()
            return [self._row_to_record(row) for row in rows]
        finally:
            connection.close()

    def latest_for_lecture(self, lecture_id: str) -> PerformanceRunRecord | None:
        runs = self.list_for_lecture(lecture_id, limit=1)
        return runs[0] if runs else None

    def recent(self, limit: int = 25) -> list[PerformanceRunRecord]:
        connection = get_connection()
        try:
            rows = connection.execute(
                "SELECT * FROM performance_runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._row_to_record(row) for row in rows]
        finally:
            connection.close()

    @staticmethod
    def _row_to_record(row) -> PerformanceRunRecord:
        return PerformanceRunRecord(
            run_id=row["run_id"],
            lecture_id=row["lecture_id"],
            job_id=row["job_id"],
            queue_backend=row["queue_backend"],
            worker_backend=row["worker_backend"],
            llm_enabled=bool(row["llm_enabled"]),
            llm_model=row["llm_model"],
            llm_chunk_count=int(row["llm_chunk_count"]),
            llm_total_ms=float(row["llm_total_ms"]),
            asr_provider=row["asr_provider"],
            asr_model_size=row["asr_model_size"],
            requested_device=row["requested_device"],
            requested_compute_type=row["requested_compute_type"],
            actual_device=row["actual_device"],
            actual_compute_type=row["actual_compute_type"],
            model_backend=row["model_backend"],
            detected_language=row["detected_language"],
            total_ms=float(row["total_ms"]),
            preprocess_ms=float(row["preprocess_ms"]),
            transcribe_ms=float(row["transcribe_ms"]),
            marker_detection_ms=float(row["marker_detection_ms"]),
            note_compose_ms=float(row["note_compose_ms"]),
            review_apply_ms=float(row["review_apply_ms"]),
            export_ms=float(row["export_ms"]),
            segment_count=int(row["segment_count"]),
            marker_count=int(row["marker_count"]),
            note_block_count=int(row["note_block_count"]),
            audio_duration_ms=int(row["audio_duration_ms"]),
            transcript_char_count=int(row["transcript_char_count"]),
            throughput_audio_x=float(row["throughput_audio_x"]),
            created_at=row["created_at"],
        )
