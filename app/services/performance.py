from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter


@dataclass(slots=True)
class StageTiming:
    name: str
    duration_ms: float


@dataclass(slots=True)
class PipelinePerformanceReport:
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

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


class StageTimer:
    def __init__(self) -> None:
        self._started_at = perf_counter()
        self._last_checkpoint = self._started_at

    def checkpoint(self, name: str) -> StageTiming:
        now = perf_counter()
        duration_ms = (now - self._last_checkpoint) * 1000
        self._last_checkpoint = now
        return StageTiming(name=name, duration_ms=round(duration_ms, 3))

    def total_ms(self) -> float:
        return round((perf_counter() - self._started_at) * 1000, 3)


def compute_throughput_audio_x(audio_duration_ms: int, total_ms: float) -> float:
    if audio_duration_ms <= 0 or total_ms <= 0:
        return 0.0
    return round(audio_duration_ms / total_ms, 3)
