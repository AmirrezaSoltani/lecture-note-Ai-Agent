from fastapi.testclient import TestClient

from app.main import app


def test_lecture_detail_shows_final_note_links_when_artifact_exists(tmp_path, monkeypatch) -> None:
    import app.main as main_module
    from app.models import ArtifactRecord, LectureRecord

    lecture = LectureRecord(
        lecture_id="lecture-ui-final",
        title="Demo Lecture",
        audio_filename="demo.wav",
        audio_path=str(tmp_path / "demo.wav"),
        transcript_path=str(tmp_path / "transcript.json"),
        notes_json_path=str(tmp_path / "notes.json"),
        notes_md_path=str(tmp_path / "notes.md"),
        notes_html_path=str(tmp_path / "notes.html"),
        notes_pdf_path=str(tmp_path / "notes.pdf"),
        notes_docx_path=None,
        notes_review_path=str(tmp_path / "review.json"),
        language="tr",
        status="completed",
        created_at="2026-04-02T00:00:00+00:00",
        updated_at="2026-04-02T00:00:00+00:00",
        error_message=None,
    )

    final_note_path = tmp_path / "final_note.pdf"
    final_note_path.write_text("pdf", encoding="utf-8")

    monkeypatch.setattr(main_module.repository, "get", lambda lecture_id: lecture if lecture_id == lecture.lecture_id else None)
    monkeypatch.setattr(main_module.artifact_repository, "list_for_lecture", lambda lecture_id: [
        ArtifactRecord(
            artifact_id="artifact-1",
            lecture_id=lecture.lecture_id,
            artifact_type="final_note_pdf",
            path=str(final_note_path),
            content_type="application/pdf",
            file_size_bytes=final_note_path.stat().st_size,
        )
    ])
    monkeypatch.setattr(main_module.job_repository, "list_for_lecture", lambda lecture_id: [])
    monkeypatch.setattr(main_module.performance_repository, "latest_for_lecture", lambda lecture_id: None)
    monkeypatch.setattr(main_module, "_load_reviewed_bundle", lambda record: None)

    client = TestClient(app)
    response = client.get(f"/lectures/{lecture.lecture_id}")
    assert response.status_code == 200
    assert "Download Final Note PDF" in response.text
    assert "Final Note PDF" in response.text
