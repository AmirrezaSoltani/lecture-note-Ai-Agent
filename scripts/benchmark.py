#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
import uuid
import wave
from pathlib import Path

from app.config import settings
from app.db import initialize_database
from app.jobs import get_pipeline
from app.models import PerformanceRunRecord
from app.repository import PerformanceRunRepository


SAMPLE_TRANSCRIPT = (
    "Bugün nekrotizan fasiitten bahsediyoruz. "
    "Burası 4 sefer soruldu, kesin çıkacak, bu kısmı yıldızlı. "
    "Anahtar kelime orantısız ağrı. "
    "Şimdi bir vaka anlatıyorum. "
    "Bir önceki sınavda soruldu. "
    "Örnek olarak diyabetik hastayı düşünün."
)


def run_benchmark(iterations: int, output_path: Path | None = None) -> dict[str, object]:
    initialize_database()
    pipeline = get_pipeline()
    performance_repository = PerformanceRunRepository()
    results: list[dict[str, float | int | str]] = []

    with tempfile.TemporaryDirectory(prefix="lecture-benchmark-") as tmp_dir:
        tmp_root = Path(tmp_dir)
        audio_path = tmp_root / "sample.wav"
        sidecar_path = audio_path.with_suffix(".txt")
        with wave.open(str(audio_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x00\x00" * 16000)
        sidecar_path.write_text(SAMPLE_TRANSCRIPT, encoding="utf-8")

        for _ in range(iterations):
            lecture_id = f"bench-{uuid.uuid4()}"
            result = pipeline.run(
                lecture_id=lecture_id,
                lecture_title="Benchmark Lecture",
                input_audio_path=audio_path,
                working_dir=tmp_root / lecture_id,
                language="tr",
            )
            results.append(result.performance.to_dict())
            performance_repository.create(
                PerformanceRunRecord(
                    run_id=str(uuid.uuid4()),
                    lecture_id=lecture_id,
                    job_id=None,
                    queue_backend="benchmark",
                    worker_backend="benchmark",
                    **result.performance.to_dict(),
                )
            )

    summary = {
        "iterations": iterations,
        "avg_total_ms": round(sum(item["total_ms"] for item in results) / len(results), 3),
        "avg_transcribe_ms": round(sum(item["transcribe_ms"] for item in results) / len(results), 3),
        "avg_throughput_audio_x": round(sum(item["throughput_audio_x"] for item in results) / len(results), 3),
        "max_total_ms": round(max(item["total_ms"] for item in results), 3),
        "min_total_ms": round(min(item["total_ms"] for item in results), 3),
        "results": results,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeatable pipeline benchmarks.")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.data_dir / "benchmarks" / "latest.json",
        help="Path to write the benchmark JSON summary.",
    )
    args = parser.parse_args()
    summary = run_benchmark(iterations=max(1, args.iterations), output_path=args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
