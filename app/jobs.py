from __future__ import annotations

import logging
import uuid
from pathlib import Path

from app.config import settings
from app.models import PerformanceRunRecord
from app.repository import ArtifactRepository, JobRepository, LectureRepository, PerformanceRunRepository
from app.services.pipeline import LectureProcessingPipeline

logger = logging.getLogger(__name__)

_PIPELINE_SINGLETON: LectureProcessingPipeline | None = None


def get_pipeline() -> LectureProcessingPipeline:
    global _PIPELINE_SINGLETON
    if _PIPELINE_SINGLETON is None:
        _PIPELINE_SINGLETON = LectureProcessingPipeline()
    return _PIPELINE_SINGLETON


def process_lecture_job(job_id: str, lecture_id: str, worker_backend: str = "local") -> None:
    lecture_repository = LectureRepository()
    job_repository = JobRepository()
    artifact_repository = ArtifactRepository()
    performance_repository = PerformanceRunRepository()
    pipeline = get_pipeline()

    record = lecture_repository.get(lecture_id)
    if record is None:
        raise ValueError(f"Lecture not found: {lecture_id}")

    job_repository.mark_running(job_id)
    try:
        artifacts_root = Path(settings.artifacts_dir) / lecture_id
        result = pipeline.run(
            lecture_id=record.lecture_id,
            lecture_title=record.title,
            input_audio_path=Path(record.audio_path),
            working_dir=artifacts_root,
            language=record.language,
        )
        artifact_path_strings = {
            key: str(path)
            for key, path in result.artifact_paths.items()
            if key != "normalized_audio" and path.exists()
        }
        lecture_repository.update_artifacts(lecture_id=lecture_id, artifact_paths=artifact_path_strings)
        artifact_repository.replace_for_lecture(lecture_id, result.artifact_records)
        performance_repository.create(
            PerformanceRunRecord(
                run_id=str(uuid.uuid4()),
                lecture_id=lecture_id,
                job_id=job_id,
                queue_backend=settings.queue_backend,
                worker_backend=worker_backend,
                **result.performance.to_dict(),
            )
        )
        job_repository.mark_completed(job_id)
        logger.info(
            "Completed lecture job job_id=%s lecture_id=%s total_ms=%s throughput_audio_x=%s",
            job_id,
            lecture_id,
            result.performance.total_ms,
            result.performance.throughput_audio_x,
        )
    except Exception as exc:
        logger.exception("Lecture processing failed for lecture_id=%s", lecture_id)
        lecture_repository.update_status(lecture_id, "failed", str(exc))
        job_repository.mark_failed(job_id, str(exc))
        raise
