from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class LectureStatus(StrEnum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class MarkerType(StrEnum):
    EXAM_HIGH = "exam_high"
    IMPORTANT_STAR = "important_star"
    MEMORIZE_EXACT = "memorize_exact"
    KEYWORD = "keyword"
    QUESTION = "question"
    CASE = "case"
    EXAMPLE = "example"


class NoteBlockType(StrEnum):
    OVERVIEW = "overview"
    MAIN = "main"
    EXAM = "exam"
    IMPORTANT = "important"
    QUESTION = "question"
    CASE = "case"
    EXAMPLE = "example"
    KEYWORD = "keyword"


class ReviewState(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(slots=True)
class LectureRecord:
    lecture_id: str
    title: str
    audio_filename: str
    audio_path: str
    language: str
    status: str = LectureStatus.UPLOADED
    error_message: str | None = None
    transcript_path: str | None = None
    notes_json_path: str | None = None
    notes_md_path: str | None = None
    notes_html_path: str | None = None
    notes_pdf_path: str | None = None
    notes_docx_path: str | None = None
    notes_review_path: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TranscriptSegment:
    index: int
    start_ms: int
    end_ms: int
    text: str
    confidence: float | None = None
    speaker_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Marker:
    marker_id: str
    marker_type: str
    label: str
    score: float
    matched_phrase: str
    text: str
    start_ms: int
    end_ms: int
    source_segment_index: int
    source: str = "text"
    review_state: str = ReviewState.PENDING
    reviewer_note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class NoteBlock:
    block_id: str
    block_type: str
    title: str
    content: str
    marker_types: list[str] = field(default_factory=list)
    source_segment_indexes: list[int] = field(default_factory=list)
    edited: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LectureNoteBundle:
    lecture_id: str
    lecture_title: str
    created_at: str
    segments: list[TranscriptSegment]
    markers: list[Marker]
    note_blocks: list[NoteBlock]

    def to_dict(self) -> dict[str, Any]:
        return {
            "lecture_id": self.lecture_id,
            "lecture_title": self.lecture_title,
            "created_at": self.created_at,
            "segments": [segment.to_dict() for segment in self.segments],
            "markers": [marker.to_dict() for marker in self.markers],
            "note_blocks": [block.to_dict() for block in self.note_blocks],
        }


@dataclass(slots=True)
class JobRecord:
    job_id: str
    lecture_id: str
    status: str = JobStatus.QUEUED
    error_message: str | None = None
    queued_at: str = field(default_factory=utc_now_iso)
    started_at: str | None = None
    finished_at: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)




@dataclass(slots=True)
class PerformanceRunRecord:
    run_id: str
    lecture_id: str
    job_id: str | None
    queue_backend: str
    worker_backend: str
    asr_provider: str
    asr_model_size: str
    requested_device: str
    requested_compute_type: str
    actual_device: str
    actual_compute_type: str
    model_backend: str
    detected_language: str
    total_ms: float
    preprocess_ms: float
    transcribe_ms: float
    marker_detection_ms: float
    note_compose_ms: float
    review_apply_ms: float
    export_ms: float
    segment_count: int
    marker_count: int
    note_block_count: int
    audio_duration_ms: int
    transcript_char_count: int
    throughput_audio_x: float
    llm_enabled: bool = False
    llm_model: str = ""
    llm_chunk_count: int = 0
    llm_total_ms: float = 0.0
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(slots=True)
class ArtifactRecord:
    artifact_id: str
    lecture_id: str
    artifact_type: str
    path: str
    content_type: str
    file_size_bytes: int
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
