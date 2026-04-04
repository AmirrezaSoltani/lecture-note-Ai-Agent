from __future__ import annotations

import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.models import LectureRecord, PerformanceRunRecord
from app.repository import LectureRepository, PerformanceRunRepository
from scripts.benchmark import run_benchmark


client = TestClient(app)


def test_benchmark_script_writes_summary(tmp_path: Path) -> None:
    output_path = tmp_path / "benchmark.json"
    summary = run_benchmark(iterations=1, output_path=output_path)
    assert summary["iterations"] == 1
    assert summary["avg_total_ms"] >= 0
    assert summary["avg_throughput_audio_x"] >= 0
    assert output_path.exists()


def test_performance_endpoint_exposes_latest_run(tmp_path: Path) -> None:
    lecture_id = f"perf-{uuid.uuid4()}"
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"stub")
    LectureRepository().create(
        LectureRecord(
            lecture_id=lecture_id,
            title="Perf Lecture",
            audio_filename=audio_path.name,
            audio_path=str(audio_path),
            language="tr",
        )
    )
    PerformanceRunRepository().create(
        PerformanceRunRecord(
            run_id=str(uuid.uuid4()),
            lecture_id=lecture_id,
            job_id=None,
            queue_backend="local",
            worker_backend="local",
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

    response = client.get(f"/api/lectures/{lecture_id}/performance")
    assert response.status_code == 200
    payload = response.json()
    assert payload["lecture_id"] == lecture_id
    assert payload["latest"]["transcribe_ms"] == 40.0
    assert payload["latest"]["actual_device"] == "cuda"
    assert payload["runs"][0]["throughput_audio_x"] == 200.0
