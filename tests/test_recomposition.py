from pathlib import Path

from app.models import LectureNoteBundle, Marker, NoteBlock, TranscriptSegment, utc_now_iso
from app.services.notes import NoteComposer
from app.services.review import ReviewService


def _bundle() -> LectureNoteBundle:
    return LectureNoteBundle(
        lecture_id="lecture-1",
        lecture_title="Demo",
        created_at=utc_now_iso(),
        segments=[
            TranscriptSegment(index=0, start_ms=0, end_ms=1000, text="Kesin çıkacak bu bilgi."),
            TranscriptSegment(index=1, start_ms=1000, end_ms=2000, text="Bu açıklama tali detaydır."),
            TranscriptSegment(index=2, start_ms=2000, end_ms=3000, text="Anahtar kelime orantısız ağrı."),
        ],
        markers=[
            Marker(
                marker_id="m1",
                marker_type="exam_high",
                label="Exam Focus",
                score=10.0,
                matched_phrase="kesin çıkacak",
                text="Kesin çıkacak bu bilgi.",
                start_ms=0,
                end_ms=1000,
                source_segment_index=0,
                review_state="approved",
            ),
            Marker(
                marker_id="m2",
                marker_type="keyword",
                label="Keyword",
                score=7.0,
                matched_phrase="anahtar kelime",
                text="Anahtar kelime orantısız ağrı.",
                start_ms=2000,
                end_ms=3000,
                source_segment_index=2,
                review_state="rejected",
            ),
        ],
        note_blocks=[
            NoteBlock(
                block_id="main-notes",
                block_type="main",
                title="Main Notes",
                content="- old content",
                source_segment_indexes=[0, 1, 2],
            )
        ],
    )


def test_note_composer_recompose_prefers_approved_markers() -> None:
    bundle = NoteComposer().recompose_from_reviews(_bundle())

    study_notes = next(block for block in bundle.note_blocks if block.block_id == "study-notes")
    rationale = next(block for block in bundle.note_blocks if block.block_id == "marker-rationale")

    assert "exam priority" in study_notes.content
    assert "keywords:" not in study_notes.content
    assert "approved-only" in rationale.content
    assert "keyword" not in rationale.content


def test_review_service_preserves_manual_block_override_after_recomposition(tmp_path: Path) -> None:
    review_path = tmp_path / "review.json"
    service = ReviewService()
    service.initialize_overlay(review_path)
    service.update_block(review_path, "study-notes", "Reviewed Study Notes", "- reviewer summary")

    bundle = service.apply_overlay(_bundle(), service.load_overlay(review_path))

    study_notes = next(block for block in bundle.note_blocks if block.block_id == "study-notes")
    assert study_notes.title == "Reviewed Study Notes"
    assert study_notes.content == "- reviewer summary"
    assert study_notes.edited is True
