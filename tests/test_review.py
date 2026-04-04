from pathlib import Path

from app.models import LectureNoteBundle, Marker, NoteBlock, TranscriptSegment, utc_now_iso
from app.services.review import ReviewService


def _bundle() -> LectureNoteBundle:
    return LectureNoteBundle(
        lecture_id="lecture-1",
        lecture_title="Demo",
        created_at=utc_now_iso(),
        segments=[TranscriptSegment(index=0, start_ms=0, end_ms=1000, text="Kesin çıkacak konu.")],
        markers=[
            Marker(
                marker_id="marker-1",
                marker_type="exam_high",
                label="Exam Focus",
                score=8.0,
                matched_phrase="kesin çıkacak",
                text="Kesin çıkacak konu.",
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
                content="- Kesin çıkacak konu.",
                source_segment_indexes=[0],
            )
        ],
    )


def test_review_service_applies_overrides(tmp_path: Path) -> None:
    review_path = tmp_path / "review.json"
    service = ReviewService()
    service.initialize_overlay(review_path)
    service.update_block(review_path, "main-notes", "Edited Notes", "- reviewed")
    service.review_marker(review_path, "marker-1", "approved", "teacher confirmed")

    bundle = service.apply_overlay(_bundle(), service.load_overlay(review_path))

    main_block = next(block for block in bundle.note_blocks if block.block_id == "main-notes")
    assert main_block.title == "Edited Notes"
    assert main_block.edited is True
    assert bundle.markers[0].review_state == "approved"
    assert bundle.markers[0].reviewer_note == "teacher confirmed"
