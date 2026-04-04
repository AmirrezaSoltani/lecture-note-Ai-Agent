from pathlib import Path

from app.models import LectureNoteBundle, Marker, NoteBlock, TranscriptSegment, utc_now_iso
from app.services.exports import DocxExporter, HtmlExporter, PdfExporter


def _bundle() -> LectureNoteBundle:
    return LectureNoteBundle(
        lecture_id="lecture-1",
        lecture_title="Demo Lecture",
        created_at=utc_now_iso(),
        segments=[TranscriptSegment(index=0, start_ms=0, end_ms=1000, text="Kesin çıkacak konu.")],
        markers=[
            Marker(
                marker_id="m1",
                marker_type="exam_high",
                label="Exam Focus",
                score=9.0,
                matched_phrase="kesin çıkacak",
                text="Kesin çıkacak konu.",
                start_ms=0,
                end_ms=1000,
                source_segment_index=0,
                review_state="approved",
            )
        ],
        note_blocks=[
            NoteBlock(
                block_id="exam-focus",
                block_type="exam",
                title="Exam Focus",
                content="- Kesin çıkacak konu.",
                marker_types=["exam_high"],
                source_segment_indexes=[0],
            )
        ],
    )


def test_pdf_and_docx_exporters_create_files(tmp_path: Path) -> None:
    bundle = _bundle()
    html_content = HtmlExporter().render(bundle)

    pdf_path = PdfExporter().render_to_file(html_content, tmp_path / "notes.pdf")
    docx_path = DocxExporter().save(bundle, tmp_path / "notes.docx")

    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 100
    assert docx_path.exists()
    assert docx_path.stat().st_size > 100
    assert "Marker Review Appendix" in html_content


def test_pdf_export_sanitizes_emoji_glyphs() -> None:
    html_content = "<html><body><p>❓ Question</p><p>📝 Example</p><p>⭐ Star</p><p>🧪 Case</p></body></html>"
    sanitized = PdfExporter._sanitize_for_pdf(html_content)
    assert "❓" not in sanitized
    assert "📝" not in sanitized
    assert "⭐" not in sanitized
    assert "🧪" not in sanitized
    assert "[Question]" in sanitized
    assert "[Example]" in sanitized
