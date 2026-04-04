from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from app.config import settings
from app.models import ArtifactRecord, LectureNoteBundle
from app.services.asr import ASRProvider, build_asr_provider, write_transcript_json
from app.services.audio import AudioPreprocessor
from app.services.exports import DocxExporter, FinalNoteHtmlExporter, HtmlExporter, JsonExporter, MarkdownExporter, PdfExporter
from app.services.markers import MarkerDetector
from app.services.notes import NoteComposer
from app.services.openai_notes import OpenAINotesPolisher, merge_llm_runtimes
from app.services.performance import PipelinePerformanceReport, StageTimer, compute_throughput_audio_x
from app.services.review import ReviewService
from app.services.storage import ArtifactStorage


@dataclass(slots=True)
class PipelineResult:
    bundle: LectureNoteBundle
    artifact_paths: dict[str, Path]
    artifact_records: list[ArtifactRecord]
    performance: PipelinePerformanceReport


class LectureProcessingPipeline:
    def __init__(self, asr_provider: ASRProvider | None = None) -> None:
        self.preprocessor = AudioPreprocessor()
        self._configured_asr_provider = asr_provider
        self.marker_detector = MarkerDetector()
        self.note_composer = NoteComposer()
        self.review_service = ReviewService()
        self.json_exporter = JsonExporter()
        self.markdown_exporter = MarkdownExporter()
        self.html_exporter = HtmlExporter()
        self.final_note_html_exporter = FinalNoteHtmlExporter()
        self.pdf_exporter = PdfExporter()
        self.docx_exporter = DocxExporter()

    def _resolve_asr_provider(self) -> ASRProvider:
        return self._configured_asr_provider or build_asr_provider()

    @staticmethod
    def _resolve_notes_polisher(*, force_enabled: bool = False) -> OpenAINotesPolisher:
        return OpenAINotesPolisher(force_enabled=force_enabled)

    def run(
        self,
        lecture_id: str,
        lecture_title: str,
        input_audio_path: Path,
        working_dir: Path,
        language: str,
    ) -> PipelineResult:
        timer = StageTimer()
        storage = ArtifactStorage(lecture_id=lecture_id, working_dir=working_dir)
        artifact_paths = storage.build_paths()

        self.preprocessor.normalize_to_wav(input_audio_path, artifact_paths["normalized_audio"])
        preprocess_timing = timer.checkpoint("preprocess")

        asr_provider = self._resolve_asr_provider()
        segments = asr_provider.transcribe(artifact_paths["normalized_audio"], language)
        asr_runtime = asr_provider.runtime_info()
        artifact_paths["runtime_info"].write_text(json.dumps(asr_runtime.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        transcribe_timing = timer.checkpoint("transcribe")

        markers = self.marker_detector.detect(segments)
        marker_timing = timer.checkpoint("marker-detection")

        source_bundle = self.note_composer.compose(
            lecture_id=lecture_id,
            lecture_title=lecture_title,
            segments=segments,
            markers=markers,
        )
        note_timing = timer.checkpoint("note-compose")

        write_transcript_json(segments, artifact_paths["transcript"])
        self.json_exporter.save(source_bundle, artifact_paths["notes_source"])
        self.review_service.initialize_overlay(artifact_paths["review_json"])
        reviewed_bundle = self.review_service.apply_overlay(
            source_bundle,
            self.review_service.load_overlay(artifact_paths["review_json"]),
        )
        review_timing = timer.checkpoint("review-apply")

        notes_polisher = self._resolve_notes_polisher()
        polished_bundle, polish_runtime = notes_polisher.polish_bundle(reviewed_bundle)
        final_note_bundle, final_note_runtime = notes_polisher.generate_final_note(reviewed_bundle, polished_bundle)
        llm_runtime = merge_llm_runtimes(polish_runtime, final_note_runtime)
        artifact_paths["llm_runtime"].write_text(json.dumps(llm_runtime.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        self._render_exports(polished_bundle, artifact_paths)
        self._render_final_note_exports(final_note_bundle, artifact_paths)
        export_timing = timer.checkpoint("export")

        artifact_records = storage.build_artifact_records(artifact_paths)
        performance = self._build_performance_report(
            segments=segments,
            markers=markers,
            note_block_count=len(final_note_bundle.note_blocks),
            preprocess_ms=preprocess_timing.duration_ms,
            transcribe_ms=transcribe_timing.duration_ms,
            marker_detection_ms=marker_timing.duration_ms,
            note_compose_ms=note_timing.duration_ms,
            review_apply_ms=review_timing.duration_ms,
            export_ms=export_timing.duration_ms,
            total_ms=timer.total_ms(),
            asr_provider=asr_runtime.provider_name,
            asr_model_size=asr_runtime.model_size,
            requested_device=asr_runtime.requested_device,
            requested_compute_type=asr_runtime.requested_compute_type,
            actual_device=asr_runtime.actual_device,
            actual_compute_type=asr_runtime.actual_compute_type,
            model_backend=asr_runtime.model_backend,
            detected_language=asr_runtime.detected_language,
            llm_enabled=llm_runtime.enabled and llm_runtime.applied,
            llm_model=llm_runtime.model,
            llm_chunk_count=llm_runtime.chunk_count,
            llm_total_ms=llm_runtime.total_ms,
        )
        return PipelineResult(
            bundle=polished_bundle,
            artifact_paths=artifact_paths,
            artifact_records=artifact_records,
            performance=performance,
        )

    def regenerate_reviewed_exports(
        self,
        bundle: LectureNoteBundle,
        artifact_paths: dict[str, Path],
        *,
        force_llm: bool = False,
    ) -> list[ArtifactRecord]:
        notes_polisher = self._resolve_notes_polisher(force_enabled=force_llm)
        polished_bundle, polish_runtime = notes_polisher.polish_bundle(bundle)
        final_note_bundle, final_note_runtime = notes_polisher.generate_final_note(bundle, polished_bundle)
        combined_runtime = merge_llm_runtimes(polish_runtime, final_note_runtime)
        artifact_paths["llm_runtime"].write_text(json.dumps(combined_runtime.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        self._render_exports(polished_bundle, artifact_paths)
        self._render_final_note_exports(final_note_bundle, artifact_paths)
        storage = ArtifactStorage(lecture_id=bundle.lecture_id, working_dir=artifact_paths["notes_json"].parent)
        return storage.build_artifact_records(artifact_paths)

    def _render_exports(self, bundle: LectureNoteBundle, artifact_paths: dict[str, Path]) -> None:
        self.json_exporter.save(bundle, artifact_paths["notes_json"])
        self.markdown_exporter.save(bundle, artifact_paths["notes_md"])
        html_content = self.html_exporter.render(bundle)
        artifact_paths["notes_html"].write_text(html_content, encoding="utf-8")
        if settings.enable_pdf_export:
            self.pdf_exporter.render_to_file(html_content, artifact_paths["notes_pdf"])
        if settings.enable_docx_export:
            self.docx_exporter.save(bundle, artifact_paths["notes_docx"])

    def _render_final_note_exports(self, bundle: LectureNoteBundle, artifact_paths: dict[str, Path]) -> None:
        self.json_exporter.save(bundle, artifact_paths["final_note_json"])
        self.markdown_exporter.save(bundle, artifact_paths["final_note_md"])
        html_content = self.final_note_html_exporter.render(bundle)
        artifact_paths["final_note_html"].write_text(html_content, encoding="utf-8")
        if settings.enable_pdf_export:
            self.pdf_exporter.render_to_file(html_content, artifact_paths["final_note_pdf"])

    @staticmethod
    def _build_performance_report(
        *,
        segments,
        markers,
        note_block_count: int,
        preprocess_ms: float,
        transcribe_ms: float,
        marker_detection_ms: float,
        note_compose_ms: float,
        review_apply_ms: float,
        export_ms: float,
        total_ms: float,
        asr_provider: str,
        asr_model_size: str,
        requested_device: str,
        requested_compute_type: str,
        actual_device: str,
        actual_compute_type: str,
        model_backend: str,
        detected_language: str,
        llm_enabled: bool,
        llm_model: str,
        llm_chunk_count: int,
        llm_total_ms: float,
    ) -> PipelinePerformanceReport:
        audio_duration_ms = max((segment.end_ms for segment in segments), default=0)
        transcript_char_count = sum(len(segment.text) for segment in segments)
        return PipelinePerformanceReport(
            asr_provider=asr_provider,
            asr_model_size=asr_model_size,
            requested_device=requested_device,
            requested_compute_type=requested_compute_type,
            actual_device=actual_device,
            actual_compute_type=actual_compute_type,
            model_backend=model_backend,
            detected_language=detected_language,
            total_ms=round(total_ms, 3),
            preprocess_ms=round(preprocess_ms, 3),
            transcribe_ms=round(transcribe_ms, 3),
            marker_detection_ms=round(marker_detection_ms, 3),
            note_compose_ms=round(note_compose_ms, 3),
            review_apply_ms=round(review_apply_ms, 3),
            export_ms=round(export_ms, 3),
            segment_count=len(segments),
            marker_count=len(markers),
            note_block_count=note_block_count,
            audio_duration_ms=audio_duration_ms,
            transcript_char_count=transcript_char_count,
            throughput_audio_x=compute_throughput_audio_x(audio_duration_ms, total_ms),
            llm_enabled=llm_enabled,
            llm_model=llm_model,
            llm_chunk_count=llm_chunk_count,
            llm_total_ms=round(llm_total_ms, 3),
        )
