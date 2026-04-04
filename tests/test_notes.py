from app.models import Marker, TranscriptSegment
from app.services.notes import NoteComposer


def test_note_composer_builds_exam_block() -> None:
    segments = [
        TranscriptSegment(index=0, start_ms=0, end_ms=1000, text="Kesin çıkacak bu bilgi."),
        TranscriptSegment(index=1, start_ms=1000, end_ms=2000, text="Bir örnek vereyim."),
    ]
    markers = [
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
        ),
        Marker(
            marker_id="m2",
            marker_type="example",
            label="Example",
            score=4.5,
            matched_phrase="örnek",
            text="Bir örnek vereyim.",
            start_ms=1000,
            end_ms=2000,
            source_segment_index=1,
        ),
    ]

    bundle = NoteComposer().compose(
        lecture_id="lecture-1",
        lecture_title="Demo Lecture",
        segments=segments,
        markers=markers,
    )

    titles = [block.title for block in bundle.note_blocks]
    assert "Exam Focus" in titles
    assert "Examples" in titles
    main_block = next(block for block in bundle.note_blocks if block.title == "Main Notes")
    assert "⚠️" in main_block.content
    assert main_block.block_id == "main-notes"
