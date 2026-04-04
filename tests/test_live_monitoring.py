from __future__ import annotations

import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import _llm_runtime_path, _resolve_llm_runtime, app
from app.models import LectureRecord, LectureStatus, PerformanceRunRecord
from app.repository import JobRepository, LectureRepository, PerformanceRunRepository

client = TestClient(app)


def test_live_endpoint_exposes_runtime_snapshot(tmp_path: Path) -> None:
    lecture_id = f"live-{uuid.uuid4()}"
    audio_path = tmp_path / "sample.mp3"
    audio_path.write_bytes(b"stub")
    LectureRepository().create(
        LectureRecord(
            lecture_id=lecture_id,
            title="Live Lecture",
            audio_filename=audio_path.name,
            audio_path=str(audio_path),
            language="tr",
            status=LectureStatus.COMPLETED,
        )
    )
    job_repository = JobRepository()
    job = job_repository.create(lecture_id)
    job_repository._update_status(
        job.job_id,
        status="completed",
        started_at="2026-04-01T13:21:46.599176+00:00",
        finished_at="2026-04-01T13:36:34.335074+00:00",
    )
    PerformanceRunRepository().create(
        PerformanceRunRecord(
            run_id=str(uuid.uuid4()),
            lecture_id=lecture_id,
            job_id=None,
            queue_backend="rq",
            worker_backend="rq",
            asr_provider="faster_whisper",
            asr_model_size="small",
            requested_device="cuda",
            requested_compute_type="float16",
            actual_device="cuda",
            actual_compute_type="float16",
            model_backend="ctranslate2",
            detected_language="tr",
            total_ms=100.0,
            preprocess_ms=10.0,
            transcribe_ms=40.0,
            marker_detection_ms=8.0,
            note_compose_ms=12.0,
            review_apply_ms=5.0,
            export_ms=25.0,
            segment_count=6,
            marker_count=5,
            note_block_count=4,
            audio_duration_ms=20000,
            transcript_char_count=500,
            throughput_audio_x=200.0,
        )
    )

    response = client.get(f"/api/lectures/{lecture_id}/live")
    assert response.status_code == 200
    payload = response.json()
    assert payload["lecture"]["lecture_id"] == lecture_id
    assert payload["latest_job"]["status"] == "completed"
    assert payload["performance"]["transcribe_ms"] == 40.0
    assert payload["asr_runtime"]["actual_device"] == "cuda"
    assert payload["timeline"]["progress_pct"] == 100
    assert payload["timeline"]["completed_ms"] is not None


def test_detail_page_contains_live_monitoring_ui(tmp_path: Path) -> None:
    lecture_id = f"live-ui-{uuid.uuid4()}"
    audio_path = tmp_path / "sample.mp3"
    audio_path.write_bytes(b"stub")
    LectureRepository().create(
        LectureRecord(
            lecture_id=lecture_id,
            title="Live UI Lecture",
            audio_filename=audio_path.name,
            audio_path=str(audio_path),
            language="tr",
        )
    )
    response = client.get(f"/lectures/{lecture_id}")
    assert response.status_code == 200
    text = response.text
    assert "Live Monitoring" in text
    assert "ASR provider" in text
    assert f'data-live-endpoint="/api/lectures/{lecture_id}/live"' in text
    assert 'id="live-progress-bar"' in text


def test_detail_page_contains_manual_llm_assistant_button(tmp_path: Path) -> None:
    lecture_id = f"live-llm-ui-{uuid.uuid4()}"
    work_dir = tmp_path / lecture_id
    work_dir.mkdir(parents=True, exist_ok=True)
    audio_path = work_dir / "sample.mp3"
    audio_path.write_bytes(b"stub")
    notes_path = work_dir / "notes.json"
    notes_path.write_text(
        '{"lecture_id":"%s","lecture_title":"Demo","created_at":"2026-04-01T00:00:00+00:00","segments":[],"markers":[],"note_blocks":[]}' % lecture_id,
        encoding="utf-8",
    )
    (work_dir / "review.json").write_text('{"marker_reviews":{},"block_overrides":{}}', encoding="utf-8")
    LectureRepository().create(
        LectureRecord(
            lecture_id=lecture_id,
            title="Live UI LLM Lecture",
            audio_filename=audio_path.name,
            audio_path=str(audio_path),
            language="tr",
            status=LectureStatus.COMPLETED,
            notes_json_path=str(notes_path),
            notes_review_path=str(work_dir / "review.json"),
        )
    )
    response = client.get(f"/lectures/{lecture_id}")
    assert response.status_code == 200
    assert 'id="live-llm-trigger"' in response.text
    assert '/api/lectures/' + lecture_id + '/llm-assistant/run' in response.text


def test_manual_llm_trigger_endpoint_marks_runtime_running(tmp_path: Path, monkeypatch) -> None:
    lecture_id = f"live-llm-run-{uuid.uuid4()}"
    work_dir = tmp_path / lecture_id
    work_dir.mkdir(parents=True, exist_ok=True)
    audio_path = work_dir / "sample.mp3"
    audio_path.write_bytes(b"stub")
    review_path = work_dir / "review.json"
    notes_path = work_dir / "notes.json"
    notes_path.write_text(
        '{"lecture_id":"%s","lecture_title":"Demo","created_at":"2026-04-01T00:00:00+00:00","segments":[],"markers":[],"note_blocks":[]}' % lecture_id,
        encoding="utf-8",
    )
    review_path.write_text('{"marker_reviews":{},"block_overrides":{}}', encoding="utf-8")
    LectureRepository().create(
        LectureRecord(
            lecture_id=lecture_id,
            title="LLM Trigger Lecture",
            audio_filename=audio_path.name,
            audio_path=str(audio_path),
            language="tr",
            status=LectureStatus.COMPLETED,
            notes_json_path=str(notes_path),
            notes_review_path=str(review_path),
        )
    )

    class DummyThread:
        def __init__(self, target=None, args=(), daemon=None):
            self.target = target
            self.args = args
        def start(self):
            return None

    monkeypatch.setattr("app.main.Thread", DummyThread)

    response = client.post(f"/api/lectures/{lecture_id}/llm-assistant/run")
    assert response.status_code == 202
    payload = response.json()
    assert payload["started"] is True
    runtime = _resolve_llm_runtime(LectureRepository().get(lecture_id), None)
    assert runtime is not None
    assert runtime["status"] == "running"


def test_resolve_llm_runtime_prefers_latest_runtime_file_over_performance_record(tmp_path: Path) -> None:
    lecture_id = f"live-llm-pref-{uuid.uuid4()}"
    work_dir = tmp_path / lecture_id
    work_dir.mkdir(parents=True, exist_ok=True)
    audio_path = work_dir / "sample.mp3"
    audio_path.write_bytes(b"stub")
    record = LectureRecord(
        lecture_id=lecture_id,
        title="Runtime Priority Lecture",
        audio_filename=audio_path.name,
        audio_path=str(audio_path),
        language="tr",
        status=LectureStatus.COMPLETED,
        notes_review_path=str(work_dir / "review.json"),
    )
    runtime_path = _llm_runtime_path(record)
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_path.write_text('{"status":"applied","model":"gpt-4o-mini","chunk_count":3,"total_ms":321.0}', encoding="utf-8")

    perf = PerformanceRunRecord(
        run_id=str(uuid.uuid4()),
        lecture_id=lecture_id,
        job_id=None,
        queue_backend="local",
        worker_backend="local",
        asr_provider="faster_whisper",
        asr_model_size="small",
        requested_device="cpu",
        requested_compute_type="default",
        actual_device="cpu",
        actual_compute_type="float32",
        model_backend="ctranslate2",
        detected_language="tr",
        total_ms=100.0,
        preprocess_ms=10.0,
        transcribe_ms=40.0,
        marker_detection_ms=8.0,
        note_compose_ms=12.0,
        review_apply_ms=5.0,
        export_ms=25.0,
        segment_count=6,
        marker_count=5,
        note_block_count=4,
        audio_duration_ms=20000,
        transcript_char_count=500,
        throughput_audio_x=200.0,
        llm_enabled=False,
        llm_model="old-model",
        llm_chunk_count=0,
        llm_total_ms=0.0,
    )
    runtime = _resolve_llm_runtime(record, perf)
    assert runtime is not None
    assert runtime["status"] == "applied"
    assert runtime["model"] == "gpt-4o-mini"


def test_openai_settings_diagnostic_endpoint(monkeypatch) -> None:
    monkeypatch.setattr("app.services.openai_notes.build_openai_diagnostics", lambda **_: {
        "status": "dns-failed",
        "detail": "Could not resolve host.",
        "dns": {"ok": False, "detail": "temporary failure"},
    })
    response = client.post("/api/settings/test-openai")
    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "dns-failed"
    assert payload["dns"]["ok"] is False
