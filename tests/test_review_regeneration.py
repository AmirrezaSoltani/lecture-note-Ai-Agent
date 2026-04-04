from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from app.config import settings
from app.main import _regenerate_reviewed_exports, repository, review_service
from app.models import LectureNoteBundle, LectureRecord, Marker, NoteBlock, TranscriptSegment, utc_now_iso


def _sample_bundle(lecture_id: str) -> LectureNoteBundle:
    return LectureNoteBundle(
        lecture_id=lecture_id,
        lecture_title="Bulk Review Demo",
        created_at=utc_now_iso(),
        segments=[TranscriptSegment(index=0, start_ms=0, end_ms=1000, text="Kesin çıkacak konu")],
        markers=[
            Marker(
                marker_id="m-1",
                marker_type="exam_high",
                label="Exam Focus",
                score=8.0,
                matched_phrase="kesin çıkacak",
                text="Kesin çıkacak konu",
                start_ms=0,
                end_ms=1000,
                source_segment_index=0,
            )
        ],
        note_blocks=[
            NoteBlock(
                block_id="main-notes",
                block_type="main",
                title="Main Notes",
                content="- Kesin çıkacak konu",
                source_segment_indexes=[0],
            )
        ],
    )


def test_regenerate_reviewed_exports_persists_without_pipeline_method_error() -> None:
    lecture_id = f"test-review-{uuid.uuid4()}"
    working_dir = settings.artifacts_dir / lecture_id
    working_dir.mkdir(parents=True, exist_ok=True)
    bundle = _sample_bundle(lecture_id)

    notes_json_path = working_dir / "notes.json"
    notes_json_path.write_text(__import__('json').dumps(bundle.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    review_path = working_dir / "review.json"
    review_service.initialize_overlay(review_path)
    review_service.review_marker(review_path, marker_id="m-1", review_state="approved", reviewer_note="bulk ok")

    record = LectureRecord(
        lecture_id=lecture_id,
        title="Bulk Review Demo",
        audio_filename="demo.mp3",
        audio_path=str(working_dir / "demo.mp3"),
        language="tr",
        status="completed",
        notes_json_path=str(notes_json_path),
        notes_md_path=str(working_dir / "notes.md"),
        notes_html_path=str(working_dir / "notes.html"),
        notes_pdf_path=str(working_dir / "notes.pdf"),
        notes_docx_path=str(working_dir / "notes.docx"),
        notes_review_path=str(review_path),
        transcript_path=str(working_dir / "transcript.json"),
    )
    repository.create(record)
    try:
        _regenerate_reviewed_exports(record)
        assert (working_dir / "notes.json").exists()
        assert (working_dir / "notes.md").exists()
        assert (working_dir / "notes.html").exists()
    finally:
        shutil.rmtree(working_dir, ignore_errors=True)
