from __future__ import annotations

import json
import logging
import re
import socket
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable
from urllib import error, request
from urllib.parse import urlparse

from app.config import settings
from app.models import LectureNoteBundle, NoteBlock, NoteBlockType, TranscriptSegment

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LLMRuntimeInfo:
    enabled: bool
    applied: bool
    provider: str
    model: str
    chunk_count: int
    total_ms: float
    status: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OpenAITimeoutError(RuntimeError):
    """Raised when the OpenAI request times out while waiting for a response."""


class OpenAINotesPolisher:
    def __init__(self, *, force_enabled: bool = False) -> None:
        runtime_settings = settings.snapshot() if hasattr(settings, "snapshot") else settings
        self.force_enabled = force_enabled
        self.api_key = runtime_settings.openai_api_key
        self.enabled = (force_enabled or runtime_settings.llm_notes_enabled) and bool(self.api_key)
        self.model = runtime_settings.openai_model
        self.base_url = runtime_settings.openai_base_url
        self.timeout_seconds = runtime_settings.openai_timeout_seconds
        self.max_input_chars = runtime_settings.openai_max_input_chars
        self._request_count = 0
        self._fallback_split_count = 0

    def diagnostics(self) -> dict[str, Any]:
        return build_openai_diagnostics(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            timeout_seconds=self.timeout_seconds,
        )

    def polish_bundle(
        self,
        bundle: LectureNoteBundle,
        output_path: Path | None = None,
    ) -> tuple[LectureNoteBundle, LLMRuntimeInfo]:
        if not self.enabled:
            runtime = LLMRuntimeInfo(
                enabled=False,
                applied=False,
                provider="openai",
                model=self.model,
                chunk_count=0,
                total_ms=0.0,
                status="disabled",
                detail="LLM polishing disabled or API key missing.",
            )
            self._persist_runtime(output_path, runtime)
            return bundle, runtime

        source_blocks = [block for block in bundle.note_blocks if block.content.strip()]
        if not source_blocks:
            runtime = LLMRuntimeInfo(True, False, "openai", self.model, 0, 0.0, "skipped", "No note blocks to polish.")
            self._persist_runtime(output_path, runtime)
            return bundle, runtime

        block_groups = self._chunk_blocks(source_blocks)
        started = perf_counter()
        polished_blocks: list[NoteBlock] = []
        self._request_count = 0
        self._fallback_split_count = 0
        try:
            for group_index, group in enumerate(block_groups):
                polished_blocks.extend(self._polish_block_group_with_fallback(bundle, group, group_index, len(block_groups)))
            detail_parts: list[str] = []
            if self._fallback_split_count:
                detail_parts.append(
                    f"Recovered from {self._fallback_split_count} timeout split(s) by retrying smaller LLM chunks."
                )
            detail_parts.append("Stage: bundle cleanup.")
            runtime = LLMRuntimeInfo(
                enabled=True,
                applied=True,
                provider="openai",
                model=self.model,
                chunk_count=max(self._request_count, len(block_groups)),
                total_ms=round((perf_counter() - started) * 1000, 3),
                status="applied",
                detail=" ".join(part for part in detail_parts if part),
            )
            polished_bundle = LectureNoteBundle(
                lecture_id=bundle.lecture_id,
                lecture_title=bundle.lecture_title,
                created_at=bundle.created_at,
                segments=bundle.segments,
                markers=bundle.markers,
                note_blocks=polished_blocks or bundle.note_blocks,
            )
            self._persist_runtime(output_path, runtime)
            return polished_bundle, runtime
        except Exception as exc:
            logger.exception("OpenAI note polishing failed: %s", exc)
            runtime = LLMRuntimeInfo(
                enabled=True,
                applied=False,
                provider="openai",
                model=self.model,
                chunk_count=max(self._request_count, len(block_groups)),
                total_ms=round((perf_counter() - started) * 1000, 3),
                status="failed",
                detail=str(exc),
            )
            self._persist_runtime(output_path, runtime)
            return bundle, runtime

    def generate_final_note(
        self,
        source_bundle: LectureNoteBundle,
        polished_bundle: LectureNoteBundle | None = None,
        output_path: Path | None = None,
    ) -> tuple[LectureNoteBundle, LLMRuntimeInfo]:
        reference_bundle = polished_bundle or source_bundle
        fallback_bundle = self._fallback_final_note_bundle(reference_bundle)
        if not self.enabled:
            runtime = LLMRuntimeInfo(
                enabled=False,
                applied=False,
                provider="openai",
                model=self.model,
                chunk_count=0,
                total_ms=0.0,
                status="disabled",
                detail="LLM final-note generation disabled or API key missing. Deterministic final note generated.",
            )
            self._persist_runtime(output_path, runtime)
            return fallback_bundle, runtime

        started = perf_counter()
        self._request_count = 0
        self._fallback_split_count = 0
        try:
            transcript_context = self._build_transcript_context(source_bundle)
            payload = self._build_final_note_payload(source_bundle, reference_bundle, transcript_context)
            parsed = self._request_structured_json(
                developer_message=(
                    "You turn lecture-processing artifacts into a student-facing final handout. "
                    "Use only the provided transcript-derived evidence, reviewed notes, and markers. "
                    "Correct obvious ASR noise, merge duplicates, and produce a clean Turkish final note."
                ),
                user_payload=payload,
                schema=self._final_note_schema(),
            )
            note_blocks = self._coerce_final_note_blocks(parsed, fallback_bundle.note_blocks)
            final_bundle = LectureNoteBundle(
                lecture_id=reference_bundle.lecture_id,
                lecture_title=reference_bundle.lecture_title,
                created_at=reference_bundle.created_at,
                segments=source_bundle.segments,
                markers=source_bundle.markers,
                note_blocks=note_blocks or fallback_bundle.note_blocks,
            )
            detail_parts = [
                f"Stage: final-note generation ({transcript_context['mode']}).",
                "Used reviewed note blocks, marker metadata, and transcript context.",
            ]
            if self._fallback_split_count:
                detail_parts.append(
                    f"Recovered from {self._fallback_split_count} timeout split(s) while digesting transcript chunks."
                )
            runtime = LLMRuntimeInfo(
                enabled=True,
                applied=True,
                provider="openai",
                model=self.model,
                chunk_count=max(1, self._request_count),
                total_ms=round((perf_counter() - started) * 1000, 3),
                status="applied",
                detail=" ".join(detail_parts),
            )
            self._persist_runtime(output_path, runtime)
            return final_bundle, runtime
        except Exception as exc:
            logger.exception("OpenAI final-note generation failed: %s", exc)
            runtime = LLMRuntimeInfo(
                enabled=True,
                applied=False,
                provider="openai",
                model=self.model,
                chunk_count=max(1, self._request_count),
                total_ms=round((perf_counter() - started) * 1000, 3),
                status="failed",
                detail=f"{exc} Falling back to deterministic final note.",
            )
            self._persist_runtime(output_path, runtime)
            return fallback_bundle, runtime

    def _build_transcript_context(self, bundle: LectureNoteBundle) -> dict[str, Any]:
        rendered_segments = [self._segment_payload(segment) for segment in bundle.segments if segment.text.strip()]
        transcript_json = json.dumps(rendered_segments, ensure_ascii=False)
        inline_budget = max(4_000, int(self.max_input_chars * 0.45))
        if len(transcript_json) <= inline_budget:
            return {
                "mode": "raw-transcript",
                "segment_count": len(rendered_segments),
                "segments": rendered_segments,
                "digests": [],
            }

        segment_groups = self._chunk_segments(bundle.segments)
        digests: list[dict[str, Any]] = []
        for group_index, group in enumerate(segment_groups):
            digests.append(self._digest_segment_group_with_fallback(group, group_index, len(segment_groups)))
        return {
            "mode": "chunk-digests",
            "segment_count": len(rendered_segments),
            "segments": [],
            "digests": digests,
        }

    def _chunk_blocks(self, blocks: list[NoteBlock]) -> list[list[NoteBlock]]:
        groups: list[list[NoteBlock]] = []
        current: list[NoteBlock] = []
        current_chars = 0
        for block in blocks:
            block_chars = len(block.title) + len(block.content) + 256
            if current and current_chars + block_chars > self.max_input_chars:
                groups.append(current)
                current = []
                current_chars = 0
            current.append(block)
            current_chars += block_chars
        if current:
            groups.append(current)
        return groups

    def _chunk_segments(self, segments: Iterable[TranscriptSegment]) -> list[list[TranscriptSegment]]:
        groups: list[list[TranscriptSegment]] = []
        current: list[TranscriptSegment] = []
        current_chars = 0
        max_chars = max(2_500, int(self.max_input_chars * 0.28))
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            segment_chars = len(text) + 96
            if current and current_chars + segment_chars > max_chars:
                groups.append(current)
                current = []
                current_chars = 0
            current.append(segment)
            current_chars += segment_chars
        if current:
            groups.append(current)
        return groups or [[]]

    def _digest_segment_group_with_fallback(
        self,
        segments: list[TranscriptSegment],
        group_index: int,
        group_count: int,
    ) -> dict[str, Any]:
        try:
            return self._digest_segment_group(segments, group_index, group_count)
        except OpenAITimeoutError:
            if len(segments) <= 4:
                raise
            mid = max(1, len(segments) // 2)
            self._fallback_split_count += 1
            left = self._digest_segment_group_with_fallback(segments[:mid], group_index, group_count)
            right = self._digest_segment_group_with_fallback(segments[mid:], group_index, group_count)
            return self._merge_digest_payloads(left, right)

    def _digest_segment_group(
        self,
        segments: list[TranscriptSegment],
        group_index: int,
        group_count: int,
    ) -> dict[str, Any]:
        payload = {
            "language": "tr",
            "group_index": group_index + 1,
            "group_count": group_count,
            "instructions": {
                "goal": "Extract only the high-yield facts from this raw transcript chunk.",
                "constraints": [
                    "Do not invent facts.",
                    "Ignore filler, repetitions, jokes, and classroom chatter unless clinically relevant.",
                    "Keep Turkish wording clean and exam-ready.",
                ],
            },
            "segments": [self._segment_payload(segment) for segment in segments if segment.text.strip()],
        }
        parsed = self._request_structured_json(
            developer_message=(
                "You compress raw lecture transcript chunks into faithful evidence digests for a later final-note step."
            ),
            user_payload=payload,
            schema=self._transcript_digest_schema(),
        )
        return {
            "chunk_index": group_index + 1,
            "segment_indexes": sorted({item["index"] for item in payload["segments"]}),
            "high_yield_facts": [self._clean_sentence(item) for item in parsed.get("high_yield_facts", []) if str(item).strip()],
            "exam_cues": [self._clean_sentence(item) for item in parsed.get("exam_cues", []) if str(item).strip()],
            "cases": [self._clean_sentence(item) for item in parsed.get("cases", []) if str(item).strip()],
            "examples": [self._clean_sentence(item) for item in parsed.get("examples", []) if str(item).strip()],
            "keywords": [self._clean_keyword(item) for item in parsed.get("keywords", []) if str(item).strip()],
        }

    def _merge_digest_payloads(self, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
        return {
            "chunk_index": min(left.get("chunk_index", 1), right.get("chunk_index", 1)),
            "segment_indexes": sorted(set(left.get("segment_indexes", [])) | set(right.get("segment_indexes", []))),
            "high_yield_facts": self._unique_strings([*left.get("high_yield_facts", []), *right.get("high_yield_facts", [])]),
            "exam_cues": self._unique_strings([*left.get("exam_cues", []), *right.get("exam_cues", [])]),
            "cases": self._unique_strings([*left.get("cases", []), *right.get("cases", [])]),
            "examples": self._unique_strings([*left.get("examples", []), *right.get("examples", [])]),
            "keywords": self._unique_strings([*left.get("keywords", []), *right.get("keywords", [])]),
        }

    def _polish_block_group_with_fallback(
        self,
        bundle: LectureNoteBundle,
        blocks: list[NoteBlock],
        group_index: int,
        group_count: int,
    ) -> list[NoteBlock]:
        try:
            return self._polish_block_group(bundle, blocks, group_index, group_count)
        except OpenAITimeoutError:
            if len(blocks) <= 1:
                raise
            mid = max(1, len(blocks) // 2)
            self._fallback_split_count += 1
            logger.warning(
                "OpenAI timeout for lecture group %s/%s with %s block(s); retrying with two smaller chunks.",
                group_index + 1,
                group_count,
                len(blocks),
            )
            left = self._polish_block_group_with_fallback(bundle, blocks[:mid], group_index, group_count)
            right = self._polish_block_group_with_fallback(bundle, blocks[mid:], group_index, group_count)
            return left + right

    def _polish_block_group(
        self,
        bundle: LectureNoteBundle,
        blocks: list[NoteBlock],
        group_index: int,
        group_count: int,
    ) -> list[NoteBlock]:
        payload = {
            "lecture_title": bundle.lecture_title,
            "language": "tr",
            "group_index": group_index + 1,
            "group_count": group_count,
            "instructions": {
                "style": "Produce clean Turkish study notes. Keep facts from source only. Remove filler, noise, obvious ASR repetition, and conversational clutter.",
                "constraints": [
                    "Do not invent new facts.",
                    "Preserve exam cues, cases, examples, and keywords when present.",
                    "Keep block_id and block_type unchanged.",
                    "Return concise but complete content.",
                ],
            },
            "note_blocks": [
                {
                    "block_id": block.block_id,
                    "block_type": block.block_type,
                    "title": block.title,
                    "content": block.content,
                    "marker_types": block.marker_types,
                    "source_segment_indexes": block.source_segment_indexes,
                }
                for block in blocks
            ],
        }
        parsed = self._request_structured_json(
            developer_message=(
                "You are cleaning lecture notes into final study notes. Use only provided source material. "
                "Keep the output faithful, concise, and structured."
            ),
            user_payload=payload,
            schema=self._polished_notes_schema(),
        )
        result: list[NoteBlock] = []
        original_lookup = {block.block_id: block for block in blocks}
        for item in parsed.get("note_blocks", []):
            original = original_lookup.get(item.get("block_id"))
            if original is None:
                continue
            result.append(
                NoteBlock(
                    block_id=item["block_id"],
                    block_type=item["block_type"],
                    title=item["title"].strip() or original.title,
                    content=item["content"].strip() or original.content,
                    marker_types=list(item.get("marker_types", original.marker_types)),
                    source_segment_indexes=list(item.get("source_segment_indexes", original.source_segment_indexes)),
                    edited=getattr(original, "edited", False),
                )
            )
        return result or blocks

    def _build_final_note_payload(
        self,
        source_bundle: LectureNoteBundle,
        reference_bundle: LectureNoteBundle,
        transcript_context: dict[str, Any],
    ) -> dict[str, Any]:
        marker_payload = [
            {
                "marker_type": marker.marker_type,
                "label": marker.label,
                "review_state": marker.review_state,
                "score": marker.score,
                "text": marker.text,
                "source_segment_index": marker.source_segment_index,
            }
            for marker in source_bundle.markers
        ]
        note_payload = [
            {
                "block_id": block.block_id,
                "block_type": block.block_type,
                "title": block.title,
                "content": block.content,
                "marker_types": block.marker_types,
                "source_segment_indexes": block.source_segment_indexes,
            }
            for block in reference_bundle.note_blocks
            if block.content.strip()
        ]
        return {
            "lecture_id": source_bundle.lecture_id,
            "lecture_title": source_bundle.lecture_title,
            "language": "tr",
            "instructions": {
                "goal": "Produce one clean final lecture handout for students.",
                "output_rules": [
                    "Use only the provided material.",
                    "Do not include transcript appendix, marker appendix, source references, or system metadata in the student-facing note.",
                    "Merge duplicates and correct obvious ASR artefacts when the intended medical term is clear from context.",
                    "Prefer short paragraphs and bullet lists over long walls of text.",
                    "Make exam-relevant takeaways explicit when they are supported by the source.",
                    "Do not create a standalone Keywords section in the final note.",
                ],
                "target_sections": [
                    "Overview",
                    "Core Notes",
                    "Diagnostic / Classification Points",
                    "Treatment / Management",
                    "Cases / Examples",
                    "Exam Focus",
                ],
            },
            "reviewed_note_blocks": note_payload,
            "markers": marker_payload,
            "transcript_context": transcript_context,
        }

    def _coerce_final_note_blocks(self, parsed: dict[str, Any], fallback_blocks: list[NoteBlock]) -> list[NoteBlock]:
        sections = parsed.get("sections", [])
        if not isinstance(sections, list):
            return fallback_blocks
        result: list[NoteBlock] = []
        for index, item in enumerate(sections, start=1):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            content = self._normalize_note_text(str(item.get("content", "")))
            if not title or not content:
                continue
            block_type = self._normalize_block_type(str(item.get("block_type", "")).strip())
            normalized_title = title.casefold()
            if block_type == NoteBlockType.KEYWORD or normalized_title in {"keywords", "keyword"}:
                continue
            block_id = str(item.get("block_id", "")).strip() or self._slugify(title, prefix=f"section-{index}")
            source_segment_indexes = [int(value) for value in item.get("source_segment_indexes", []) if isinstance(value, int)]
            marker_types = [str(value) for value in item.get("marker_types", []) if str(value).strip()]
            result.append(
                NoteBlock(
                    block_id=block_id,
                    block_type=block_type,
                    title=title,
                    content=content,
                    marker_types=marker_types,
                    source_segment_indexes=source_segment_indexes,
                )
            )
        return result or fallback_blocks

    def _fallback_final_note_bundle(self, bundle: LectureNoteBundle) -> LectureNoteBundle:
        grouped: dict[str, list[NoteBlock]] = {}
        for block in bundle.note_blocks:
            if block.content.strip():
                grouped.setdefault(block.block_type, []).append(block)

        sections: list[NoteBlock] = []
        overview_lines = self._collect_lines(grouped.get(NoteBlockType.OVERVIEW, []), limit=10)
        if not overview_lines:
            overview_lines = self._collect_lines(grouped.get(NoteBlockType.MAIN, []), limit=10)
        if overview_lines:
            sections.append(
                NoteBlock(
                    block_id="final-overview",
                    block_type=NoteBlockType.OVERVIEW,
                    title="Overview",
                    content=self._render_bullets(overview_lines),
                )
            )

        core_lines = self._collect_lines(grouped.get(NoteBlockType.MAIN, []), limit=24)
        if core_lines:
            sections.append(
                NoteBlock(
                    block_id="final-core-notes",
                    block_type=NoteBlockType.MAIN,
                    title="Core Notes",
                    content=self._render_bullets(core_lines),
                )
            )

        exam_lines = self._collect_lines(
            [
                *grouped.get(NoteBlockType.EXAM, []),
                *grouped.get(NoteBlockType.IMPORTANT, []),
                *grouped.get(NoteBlockType.QUESTION, []),
            ],
            limit=18,
        )
        if exam_lines:
            sections.append(
                NoteBlock(
                    block_id="final-exam-focus",
                    block_type=NoteBlockType.EXAM,
                    title="Exam Focus",
                    content=self._render_bullets(exam_lines),
                )
            )

        example_lines = self._collect_lines(
            [*grouped.get(NoteBlockType.CASE, []), *grouped.get(NoteBlockType.EXAMPLE, [])],
            limit=12,
        )
        if example_lines:
            sections.append(
                NoteBlock(
                    block_id="final-cases-examples",
                    block_type=NoteBlockType.CASE,
                    title="Cases and Examples",
                    content=self._render_bullets(example_lines),
                )
            )

        if not sections:
            sections = [
                NoteBlock(
                    block_id="final-note",
                    block_type=NoteBlockType.MAIN,
                    title="Final Note",
                    content=self._render_bullets(self._collect_lines(bundle.note_blocks, limit=24) or [bundle.lecture_title]),
                )
            ]

        return LectureNoteBundle(
            lecture_id=bundle.lecture_id,
            lecture_title=bundle.lecture_title,
            created_at=bundle.created_at,
            segments=bundle.segments,
            markers=bundle.markers,
            note_blocks=sections,
        )

    def _request_structured_json(
        self,
        *,
        developer_message: str,
        user_payload: dict[str, Any],
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        body = {
            "model": self.model,
            "messages": [
                {"role": "developer", "content": developer_message},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_schema", "json_schema": schema},
            "temperature": 0.1,
        }
        self._request_count += 1
        raw = self._request_json(body)
        message = raw["choices"][0]["message"]
        content = message.get("content", "")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return json.loads(content)

    def _request_json(self, body: dict[str, Any]) -> dict[str, Any]:
        endpoint = f"{self.base_url}/chat/completions"
        req = request.Request(
            endpoint,
            method="POST",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:  # pragma: no cover - network/runtime path
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"OpenAI HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:  # pragma: no cover - network/runtime path
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, (TimeoutError, socket.timeout)):
                raise OpenAITimeoutError(_format_openai_timeout_error(self.timeout_seconds, self.max_input_chars)) from exc
            raise RuntimeError(_format_openai_connection_error(exc, self.base_url)) from exc
        except TimeoutError as exc:  # pragma: no cover - network/runtime path
            raise OpenAITimeoutError(_format_openai_timeout_error(self.timeout_seconds, self.max_input_chars)) from exc
        except socket.timeout as exc:  # pragma: no cover - network/runtime path
            raise OpenAITimeoutError(_format_openai_timeout_error(self.timeout_seconds, self.max_input_chars)) from exc

    @staticmethod
    def _segment_payload(segment: TranscriptSegment) -> dict[str, Any]:
        return {
            "index": segment.index,
            "start_ms": segment.start_ms,
            "end_ms": segment.end_ms,
            "text": segment.text.strip(),
        }

    @staticmethod
    def _persist_runtime(path: Path | None, runtime: LLMRuntimeInfo) -> None:
        if path is None:
            return
        path.write_text(json.dumps(runtime.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _clean_sentence(value: str) -> str:
        text = re.sub(r"\s+", " ", str(value or "").strip())
        text = text.lstrip("-•* ")
        if not text:
            return ""
        if text[-1] not in ".!?":
            text += "."
        return text

    @staticmethod
    def _clean_keyword(value: str) -> str:
        text = re.sub(r"\s+", " ", str(value or "").strip())
        return text.strip("-•* ")

    def _normalize_note_text(self, value: str) -> str:
        lines = [re.sub(r"\s+", " ", line.strip()) for line in value.splitlines() if line.strip()]
        if not lines:
            return ""
        normalized: list[str] = []
        for line in lines:
            if line.startswith(("- ", "• ")):
                normalized.append(f"- {line[2:].strip()}")
            elif re.match(r"^\d+[\.)]\s+", line):
                normalized.append(line)
            else:
                normalized.append(line)
        return "\n".join(self._unique_strings(normalized))

    @staticmethod
    def _normalize_block_type(value: str) -> str:
        allowed = {
            NoteBlockType.OVERVIEW,
            NoteBlockType.MAIN,
            NoteBlockType.EXAM,
            NoteBlockType.IMPORTANT,
            NoteBlockType.QUESTION,
            NoteBlockType.CASE,
            NoteBlockType.EXAMPLE,
            NoteBlockType.KEYWORD,
        }
        return value if value in allowed else NoteBlockType.MAIN

    @staticmethod
    def _slugify(value: str, *, prefix: str) -> str:
        candidate = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
        return candidate or prefix

    def _collect_lines(self, blocks: list[NoteBlock], *, limit: int) -> list[str]:
        lines: list[str] = []
        for block in blocks:
            for raw_line in block.content.splitlines():
                cleaned = re.sub(r"\s+", " ", raw_line.strip())
                cleaned = cleaned.lstrip("-•* ")
                if not cleaned:
                    continue
                lines.append(cleaned)
                if len(lines) >= limit:
                    return self._unique_strings(lines)
        return self._unique_strings(lines)

    def _render_bullets(self, lines: list[str]) -> str:
        cleaned = self._unique_strings([self._clean_sentence(line).rstrip(".") for line in lines if str(line).strip()])
        return "\n".join(f"- {line}" for line in cleaned)

    @staticmethod
    def _unique_strings(values: Iterable[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            stripped = str(value).strip()
            if not stripped:
                continue
            key = stripped.casefold()
            if key in seen:
                continue
            seen.add(key)
            result.append(stripped)
        return result

    @staticmethod
    def _polished_notes_schema() -> dict[str, Any]:
        return {
            "name": "polished_notes",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "note_blocks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "block_id": {"type": "string"},
                                "block_type": {"type": "string"},
                                "title": {"type": "string"},
                                "content": {"type": "string"},
                                "marker_types": {"type": "array", "items": {"type": "string"}},
                                "source_segment_indexes": {"type": "array", "items": {"type": "integer"}},
                            },
                            "required": ["block_id", "block_type", "title", "content", "marker_types", "source_segment_indexes"],
                        },
                    }
                },
                "required": ["note_blocks"],
            },
        }

    @staticmethod
    def _transcript_digest_schema() -> dict[str, Any]:
        return {
            "name": "transcript_digest",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "high_yield_facts": {"type": "array", "items": {"type": "string"}},
                    "exam_cues": {"type": "array", "items": {"type": "string"}},
                    "cases": {"type": "array", "items": {"type": "string"}},
                    "examples": {"type": "array", "items": {"type": "string"}},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["high_yield_facts", "exam_cues", "cases", "examples", "keywords"],
            },
        }

    @staticmethod
    def _final_note_schema() -> dict[str, Any]:
        return {
            "name": "final_note",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "sections": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "block_id": {"type": "string"},
                                "block_type": {"type": "string"},
                                "title": {"type": "string"},
                                "content": {"type": "string"},
                                "marker_types": {"type": "array", "items": {"type": "string"}},
                                "source_segment_indexes": {"type": "array", "items": {"type": "integer"}},
                            },
                            "required": ["block_id", "block_type", "title", "content", "marker_types", "source_segment_indexes"],
                        },
                    }
                },
                "required": ["sections"],
            },
        }


def merge_llm_runtimes(*infos: LLMRuntimeInfo) -> LLMRuntimeInfo:
    valid = [info for info in infos if info is not None]
    if not valid:
        return LLMRuntimeInfo(False, False, "openai", settings.openai_model, 0, 0.0, "disabled", "")
    enabled = any(info.enabled for info in valid)
    applied = any(info.applied for info in valid)
    statuses = {info.status for info in valid}
    if applied:
        status = "applied"
    elif "failed" in statuses:
        status = "failed"
    elif enabled:
        status = "skipped"
    else:
        status = "disabled"
    detail = " ".join(info.detail for info in valid if info.detail).strip()
    provider = next((info.provider for info in valid if info.provider), "openai")
    model = next((info.model for info in valid if info.model), settings.openai_model)
    return LLMRuntimeInfo(
        enabled=enabled,
        applied=applied,
        provider=provider,
        model=model,
        chunk_count=sum(info.chunk_count for info in valid),
        total_ms=round(sum(info.total_ms for info in valid), 3),
        status=status,
        detail=detail,
    )


def build_openai_diagnostics(*, base_url: str, api_key: str, model: str, timeout_seconds: float) -> dict[str, Any]:
    parsed = urlparse(base_url)
    host = parsed.hostname or ""
    diagnostics: dict[str, Any] = {
        "provider": "openai",
        "base_url": base_url,
        "host": host,
        "model": model,
        "timeout_seconds": timeout_seconds,
        "api_key_present": bool(api_key),
        "dns": {
            "ok": False,
            "addresses": [],
            "detail": "No hostname configured.",
        },
    }
    if not host:
        diagnostics["status"] = "invalid-base-url"
        diagnostics["detail"] = "OPENAI_BASE_URL is missing a valid hostname."
        return diagnostics
    try:
        infos = socket.getaddrinfo(host, parsed.port or 443, type=socket.SOCK_STREAM)
        addresses = sorted({info[4][0] for info in infos if info[4]})
        diagnostics["dns"] = {
            "ok": True,
            "addresses": addresses,
            "detail": f"Resolved {host} successfully.",
        }
        diagnostics["status"] = "ready" if api_key else "missing-api-key"
        diagnostics["detail"] = (
            "OpenAI hostname resolves and an API key is configured."
            if api_key
            else "OpenAI hostname resolves, but OPENAI_API_KEY is not set."
        )
        return diagnostics
    except socket.gaierror as exc:
        diagnostics["status"] = "dns-failed"
        diagnostics["detail"] = _format_openai_connection_error(exc, base_url)
        diagnostics["dns"] = {
            "ok": False,
            "addresses": [],
            "detail": str(exc),
        }
        return diagnostics


def _format_openai_timeout_error(timeout_seconds: float, max_input_chars: int) -> str:
    return (
        f"OpenAI request timed out after {timeout_seconds:g}s while waiting for a response. "
        f"Try increasing OPENAI_TIMEOUT_SECONDS or lowering OPENAI_MAX_INPUT_CHARS below the current {max_input_chars}."
    )


def _format_openai_connection_error(exc: Exception, base_url: str) -> str:
    parsed = urlparse(base_url)
    host = parsed.hostname or "<missing-host>"
    reason = getattr(exc, "reason", exc)
    if isinstance(reason, socket.gaierror):
        return (
            f"Could not resolve OpenAI host '{host}' from the app container. "
            f"Check OPENAI_BASE_URL, Docker DNS, or outbound internet access. Original error: {reason}"
        )
    return f"OpenAI connection failed for host '{host}': {reason}"
