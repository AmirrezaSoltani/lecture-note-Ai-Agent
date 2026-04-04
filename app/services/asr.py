from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from app.config import settings
from app.models import TranscriptSegment

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ASRRuntimeInfo:
    provider_name: str
    model_size: str
    requested_device: str
    requested_compute_type: str
    actual_device: str
    actual_compute_type: str
    model_backend: str
    detected_language: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class ASRProvider(Protocol):
    def transcribe(self, audio_path: Path, language: str) -> list[TranscriptSegment]:
        ...

    def runtime_info(self) -> ASRRuntimeInfo:
        ...


class MockASRProvider:
    def __init__(self) -> None:
        self._runtime_info = ASRRuntimeInfo(
            provider_name="mock",
            model_size="mock",
            requested_device="cpu",
            requested_compute_type="mock",
            actual_device="cpu",
            actual_compute_type="mock",
            model_backend="mock",
            detected_language="",
        )

    def transcribe(self, audio_path: Path, language: str) -> list[TranscriptSegment]:
        sidecar = audio_path.with_suffix(".txt")
        if sidecar.exists():
            text = sidecar.read_text(encoding="utf-8")
            logger.info("Using sidecar transcript file for mock ASR: %s", sidecar)
        else:
            text = (
                "Bugün nekrotizan fasiitten bahsediyoruz. "
                "Burası 4 sefer soruldu, kesin çıkacak, bu kısmı yıldızlı. "
                "Anahtar kelime orantısız ağrı. "
                "Şimdi bir vaka anlatıyorum. "
                "Bir önceki sınavda soruldu. "
                "Örnek olarak diyabetik hastayı düşünün."
            )
            logger.info("Using built-in mock transcript sample.")
        self._runtime_info = ASRRuntimeInfo(
            provider_name="mock",
            model_size="mock",
            requested_device="cpu",
            requested_compute_type="mock",
            actual_device="cpu",
            actual_compute_type="mock",
            model_backend="mock",
            detected_language=(language if language not in {"", "auto", "detect"} else "mock"),
        )
        logger.info(
            "ASR runtime provider=%s backend=%s requested_device=%s actual_device=%s requested_compute_type=%s actual_compute_type=%s detected_language=%s",
            self._runtime_info.provider_name,
            self._runtime_info.model_backend,
            self._runtime_info.requested_device,
            self._runtime_info.actual_device,
            self._runtime_info.requested_compute_type,
            self._runtime_info.actual_compute_type,
            self._runtime_info.detected_language,
        )
        return self._segments_from_text(text)

    def runtime_info(self) -> ASRRuntimeInfo:
        return self._runtime_info

    @staticmethod
    def _segments_from_text(text: str) -> list[TranscriptSegment]:
        normalized = text.replace("\n", " ").strip()
        sentences = [part.strip() for part in normalized.split(".") if part.strip()]
        segments: list[TranscriptSegment] = []
        cursor = 0
        for index, sentence in enumerate(sentences):
            duration_ms = max(1800, len(sentence.split()) * 450)
            segments.append(
                TranscriptSegment(
                    index=index,
                    start_ms=cursor,
                    end_ms=cursor + duration_ms,
                    text=sentence + ".",
                    confidence=0.99,
                )
            )
            cursor += duration_ms
        return segments


class FasterWhisperASRProvider:
    _model_cache: dict[tuple[str, str, str], object] = {}

    def __init__(self, model_size: str | None = None) -> None:
        self.model_size = model_size or settings.whisper_model_size
        self.device = settings.whisper_device
        self.compute_type = settings.whisper_compute_type
        self._runtime_info = ASRRuntimeInfo(
            provider_name="faster_whisper",
            model_size=self.model_size,
            requested_device=self.device,
            requested_compute_type=self.compute_type,
            actual_device=self.device,
            actual_compute_type=self.compute_type,
            model_backend="ctranslate2",
            detected_language="",
        )

    def _load_model(self):
        cache_key = (self.model_size, self.device, self.compute_type)
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Install the dependency or use ASR_PROVIDER=mock."
            ) from exc

        model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        self._model_cache[cache_key] = model
        effective_device = str(getattr(model, "device", self.device) or self.device)
        effective_compute = _extract_compute_type(model, self.compute_type)
        self._runtime_info = ASRRuntimeInfo(
            provider_name="faster_whisper",
            model_size=self.model_size,
            requested_device=self.device,
            requested_compute_type=self.compute_type,
            actual_device=effective_device,
            actual_compute_type=effective_compute,
            model_backend="ctranslate2",
            detected_language="",
        )
        logger.info(
            "Loaded faster-whisper model size=%s requested_device=%s actual_device=%s requested_compute_type=%s actual_compute_type=%s backend=%s cuda_visible=%s hf_token_present=%s",
            self.model_size,
            self.device,
            effective_device,
            self.compute_type,
            effective_compute,
            self._runtime_info.model_backend,
            bool(os.getenv("CUDA_VISIBLE_DEVICES", "").strip()),
            bool(os.getenv("HF_TOKEN", "").strip()),
        )
        return model

    def transcribe(self, audio_path: Path, language: str) -> list[TranscriptSegment]:
        model = self._load_model()
        wanted_language = None if language in {"", "auto", "detect"} else language
        logger.info(
            "Starting ASR transcription provider=%s model=%s requested_device=%s actual_device=%s requested_compute_type=%s actual_compute_type=%s language=%s audio_path=%s",
            self._runtime_info.provider_name,
            self._runtime_info.model_size,
            self._runtime_info.requested_device,
            self._runtime_info.actual_device,
            self._runtime_info.requested_compute_type,
            self._runtime_info.actual_compute_type,
            wanted_language or "auto",
            audio_path,
        )
        raw_segments, info = model.transcribe(
            str(audio_path),
            language=wanted_language,
            vad_filter=True,
            condition_on_previous_text=False,
            word_timestamps=False,
        )

        materialized_segments = list(raw_segments)
        detected_language = str(getattr(info, "language", None) or wanted_language or "unknown")
        effective_compute = _extract_compute_type(model, self._runtime_info.actual_compute_type)
        effective_device = str(getattr(model, "device", self._runtime_info.actual_device) or self._runtime_info.actual_device)
        self._runtime_info = ASRRuntimeInfo(
            provider_name="faster_whisper",
            model_size=self.model_size,
            requested_device=self.device,
            requested_compute_type=self.compute_type,
            actual_device=effective_device,
            actual_compute_type=effective_compute,
            model_backend="ctranslate2",
            detected_language=detected_language,
        )
        logger.info(
            "Completed ASR transcription provider=%s segments=%s detected_language=%s requested_device=%s actual_device=%s requested_compute_type=%s actual_compute_type=%s backend=%s",
            self._runtime_info.provider_name,
            len(materialized_segments),
            self._runtime_info.detected_language,
            self._runtime_info.requested_device,
            self._runtime_info.actual_device,
            self._runtime_info.requested_compute_type,
            self._runtime_info.actual_compute_type,
            self._runtime_info.model_backend,
        )

        segments: list[TranscriptSegment] = []
        for index, segment in enumerate(materialized_segments):
            confidence = getattr(segment, "no_speech_prob", None)
            if confidence is not None:
                confidence = max(0.0, 1.0 - float(confidence))
            segments.append(
                TranscriptSegment(
                    index=index,
                    start_ms=int(segment.start * 1000),
                    end_ms=int(segment.end * 1000),
                    text=segment.text.strip(),
                    confidence=confidence,
                )
            )
        return segments

    def runtime_info(self) -> ASRRuntimeInfo:
        return self._runtime_info


class AutoASRProvider:
    def __init__(self) -> None:
        self.fast_provider = FasterWhisperASRProvider()
        self.mock_provider = MockASRProvider()
        self._runtime_info = self.fast_provider.runtime_info()

    def transcribe(self, audio_path: Path, language: str) -> list[TranscriptSegment]:
        try:
            segments = self.fast_provider.transcribe(audio_path, language)
            self._runtime_info = self.fast_provider.runtime_info()
            return segments
        except Exception as exc:  # pragma: no cover - fallback path depends on env
            logger.warning("Falling back to mock ASR provider: %s", exc)
            segments = self.mock_provider.transcribe(audio_path, language)
            self._runtime_info = self.mock_provider.runtime_info()
            logger.info(
                "ASR fallback active provider=%s backend=%s requested_device=%s actual_device=%s requested_compute_type=%s actual_compute_type=%s detected_language=%s",
                self._runtime_info.provider_name,
                self._runtime_info.model_backend,
                self._runtime_info.requested_device,
                self._runtime_info.actual_device,
                self._runtime_info.requested_compute_type,
                self._runtime_info.actual_compute_type,
                self._runtime_info.detected_language,
            )
            return segments

    def runtime_info(self) -> ASRRuntimeInfo:
        return self._runtime_info


def _extract_compute_type(model: object, fallback: str) -> str:
    candidates = [
        getattr(model, "compute_type", None),
        getattr(getattr(model, "model", None), "compute_type", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        text = str(candidate)
        if text:
            return text
    return fallback


def build_asr_provider() -> ASRProvider:
    provider_name = settings.asr_provider
    if provider_name == "faster_whisper":
        return FasterWhisperASRProvider()
    if provider_name == "mock":
        return MockASRProvider()
    return AutoASRProvider()


def write_transcript_json(segments: list[TranscriptSegment], output_path: Path) -> None:
    payload = [segment.to_dict() for segment in segments]
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
