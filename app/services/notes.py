from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from app.models import LectureNoteBundle, Marker, MarkerType, NoteBlock, NoteBlockType, ReviewState, TranscriptSegment
from app.services.ai_assist import ControlledNoteAssistant
from app.services.markers import MarkerDetector


class NoteComposer:
    def __init__(self) -> None:
        self._assistant = ControlledNoteAssistant()

    def compose(
        self,
        lecture_id: str,
        lecture_title: str,
        segments: list[TranscriptSegment],
        markers: list[Marker],
    ) -> LectureNoteBundle:
        marker_map = self._marker_map(markers)
        blocks = self._compose_blocks(segments=segments, markers=markers, marker_map=marker_map, approved_only=False)
        return LectureNoteBundle(
            lecture_id=lecture_id,
            lecture_title=lecture_title,
            created_at=datetime.now(timezone.utc).isoformat(),
            segments=segments,
            markers=markers,
            note_blocks=blocks,
        )

    def recompose_from_reviews(self, bundle: LectureNoteBundle) -> LectureNoteBundle:
        approved_markers = [marker for marker in bundle.markers if marker.review_state == ReviewState.APPROVED]
        visible_markers = [marker for marker in bundle.markers if marker.review_state != ReviewState.REJECTED]
        reference_markers = approved_markers or visible_markers
        marker_map = self._marker_map(reference_markers)
        recomposed_blocks = self._compose_blocks(
            segments=bundle.segments,
            markers=reference_markers,
            marker_map=marker_map,
            approved_only=bool(approved_markers),
        )
        return LectureNoteBundle(
            lecture_id=bundle.lecture_id,
            lecture_title=bundle.lecture_title,
            created_at=bundle.created_at,
            segments=bundle.segments,
            markers=bundle.markers,
            note_blocks=recomposed_blocks,
        )

    def _compose_blocks(
        self,
        segments: list[TranscriptSegment],
        markers: list[Marker],
        marker_map: dict[int, list[str]],
        approved_only: bool,
    ) -> list[NoteBlock]:
        focus_segments = self._focus_segment_indexes(segments=segments, markers=markers)
        blocks: list[NoteBlock] = []
        blocks.append(
            NoteBlock(
                block_id="overview",
                block_type=NoteBlockType.OVERVIEW,
                title="Overview",
                content=self._build_overview(segments=segments, markers=markers, approved_only=approved_only),
                source_segment_indexes=[segment.index for segment in segments[:3]],
            )
        )
        blocks.append(
            NoteBlock(
                block_id="study-notes",
                block_type=NoteBlockType.MAIN,
                title=self._assistant.generate_section_title([segments[index] for index in focus_segments if index < len(segments)], sorted({marker.marker_type for marker in markers}), "Study Notes"),
                content=self._build_study_notes(segments=segments, markers=markers, focus_segment_indexes=focus_segments),
                source_segment_indexes=focus_segments,
            )
        )
        blocks.append(
            NoteBlock(
                block_id="main-notes",
                block_type=NoteBlockType.MAIN,
                title="Main Notes",
                content=self._build_main_notes(segments, marker_map),
                source_segment_indexes=[segment.index for segment in segments],
            )
        )
        blocks.append(
            NoteBlock(
                block_id="marker-rationale",
                block_type=NoteBlockType.IMPORTANT,
                title="Marker-Guided Rationale",
                content=self._build_marker_rationale(markers=markers, approved_only=approved_only),
                marker_types=sorted({marker.marker_type for marker in markers}),
                source_segment_indexes=sorted({marker.source_segment_index for marker in markers}),
            )
        )

        category_specs = [
            (MarkerType.EXAM_HIGH, NoteBlockType.EXAM, "Exam Focus", "exam-focus"),
            (MarkerType.IMPORTANT_STAR, NoteBlockType.IMPORTANT, "Important / Starred", "important-starred"),
            (MarkerType.QUESTION, NoteBlockType.QUESTION, "Questions", "questions"),
            (MarkerType.CASE, NoteBlockType.CASE, "Cases / Vaka", "cases-vaka"),
            (MarkerType.EXAMPLE, NoteBlockType.EXAMPLE, "Examples", "examples"),
            (MarkerType.KEYWORD, NoteBlockType.KEYWORD, "Keywords", "keywords"),
            (MarkerType.MEMORIZE_EXACT, NoteBlockType.IMPORTANT, "Memorize Exactly", "memorize-exactly"),
        ]
        for marker_type, block_type, title, block_id in category_specs:
            content = self._block_from_markers(markers, marker_type)
            if not content:
                continue
            blocks.append(
                NoteBlock(
                    block_id=block_id,
                    block_type=block_type,
                    title=title,
                    content=content,
                    marker_types=[str(marker_type)],
                    source_segment_indexes=[
                        marker.source_segment_index for marker in MarkerDetector.markers_by_type(markers, marker_type)
                    ],
                )
            )
        return blocks

    @staticmethod
    def _marker_map(markers: list[Marker]) -> dict[int, list[str]]:
        by_segment: dict[int, list[str]] = defaultdict(list)
        for marker in markers:
            by_segment[marker.source_segment_index].append(marker.marker_type)
        return by_segment

    @staticmethod
    def _focus_segment_indexes(segments: list[TranscriptSegment], markers: list[Marker]) -> list[int]:
        if not segments:
            return []
        focus_indexes = {marker.source_segment_index for marker in markers}
        if not focus_indexes:
            return [segment.index for segment in segments[: min(6, len(segments))]]
        expanded_indexes = set(focus_indexes)
        for segment_index in focus_indexes:
            if segment_index > 0:
                expanded_indexes.add(segment_index - 1)
            if segment_index + 1 < len(segments):
                expanded_indexes.add(segment_index + 1)
        return sorted(index for index in expanded_indexes if 0 <= index < len(segments))

    def _build_overview(self, segments: list[TranscriptSegment], markers: list[Marker], approved_only: bool) -> str:
        preview = " ".join(segment.text.strip() for segment in segments[:3]).strip()
        top_markers = MarkerDetector.top_markers(markers, limit=6)
        highlights = "\n".join(f"- [{marker.label}] {marker.text.strip()}" for marker in top_markers)
        average_segment_seconds = 0.0
        if segments:
            average_segment_seconds = sum((segment.end_ms - segment.start_ms) for segment in segments) / len(segments) / 1000
        marker_scope = "approved markers" if approved_only else "visible markers"
        if highlights:
            return (
                f"{preview}\n\n"
                f"Transcript segments: {len(segments)}\n"
                f"Detected markers used for recomposition: {len(markers)} ({marker_scope})\n"
                f"Average segment length: {average_segment_seconds:.1f}s\n\n"
                f"Top detected highlights:\n{highlights}"
            )
        return preview or "No transcript preview available."

    def _build_study_notes(
        self,
        segments: list[TranscriptSegment],
        markers: list[Marker],
        focus_segment_indexes: list[int],
    ) -> str:
        if not segments:
            return "No transcript segments available."
        marker_map = self._marker_map(markers)
        lines: list[str] = []
        for segment_index in focus_segment_indexes:
            segment = segments[segment_index]
            summary_line = self._segment_summary_line(segment=segment, marker_types=marker_map.get(segment.index, []))
            if summary_line:
                lines.append(summary_line)
        if not lines:
            lines = [self._segment_summary_line(segment=segment, marker_types=marker_map.get(segment.index, [])) for segment in segments[:5]]
            lines = [line for line in lines if line]
        return "\n".join(lines) if lines else "No study-note lines could be composed."

    def _segment_summary_line(self, segment: TranscriptSegment, marker_types: list[str]) -> str:
        prefix = self._prefix_for_types(marker_types)
        clean_text = self._assistant.cleanup_text(segment.text)
        if not clean_text:
            return ""
        action_hint = self._action_hint(marker_types)
        keyword_hint = self._keyword_hint(clean_text, marker_types)
        suffix_parts = [part for part in [action_hint, keyword_hint] if part]
        suffix = f" — {' | '.join(suffix_parts)}" if suffix_parts else ""
        return f"- {prefix}{clean_text}{suffix}"

    @staticmethod
    def _action_hint(marker_types: list[str]) -> str:
        ordered_hints = [
            (MarkerType.EXAM_HIGH, "exam priority"),
            (MarkerType.IMPORTANT_STAR, "star this"),
            (MarkerType.MEMORIZE_EXACT, "memorize wording"),
            (MarkerType.KEYWORD, "key term"),
            (MarkerType.CASE, "case framing"),
            (MarkerType.EXAMPLE, "example"),
            (MarkerType.QUESTION, "possible question"),
        ]
        hints = [label for marker_type, label in ordered_hints if marker_type in marker_types]
        return ", ".join(hints)

    @staticmethod
    def _keyword_hint(text: str, marker_types: list[str]) -> str:
        if MarkerType.KEYWORD not in marker_types:
            return ""
        words = [word.strip(".,:;!?()[]{}\"'") for word in text.split() if len(word.strip(".,:;!?()[]{}\"'")) > 3]
        unique_words: list[str] = []
        for word in words:
            lowered = word.lower()
            if lowered in unique_words:
                continue
            unique_words.append(lowered)
            if len(unique_words) == 3:
                break
        return f"keywords: {', '.join(unique_words)}" if unique_words else ""

    def _build_main_notes(self, segments: list[TranscriptSegment], marker_map: dict[int, list[str]]) -> str:
        lines: list[str] = []
        for segment in segments:
            prefix = self._prefix_for_types(marker_map.get(segment.index, []))
            timestamp = f"[{segment.start_ms / 1000:.1f}s]"
            lines.append(f"- {timestamp} {prefix}{segment.text.strip()}")
        return "\n".join(lines)

    @staticmethod
    def _prefix_for_types(types: list[str]) -> str:
        if MarkerType.EXAM_HIGH in types:
            return "⚠️ "
        if MarkerType.IMPORTANT_STAR in types:
            return "⭐ "
        if MarkerType.MEMORIZE_EXACT in types:
            return "🧠 "
        if MarkerType.KEYWORD in types:
            return "🔑 "
        if MarkerType.CASE in types:
            return "🧪 "
        if MarkerType.EXAMPLE in types:
            return "📝 "
        if MarkerType.QUESTION in types:
            return "❓ "
        return ""

    @staticmethod
    def _build_marker_rationale(markers: list[Marker], approved_only: bool) -> str:
        if not markers:
            return "No reviewed markers available yet."
        state_label = "approved-only" if approved_only else "non-rejected"
        grouped: dict[str, list[Marker]] = defaultdict(list)
        for marker in markers:
            grouped[marker.marker_type].append(marker)
        lines = [f"Markers used for recomposition: {len(markers)} ({state_label}).", ""]
        for marker_type, typed_markers in sorted(grouped.items()):
            top_example = sorted(typed_markers, key=MarkerDetector.marker_sort_key)[0]
            lines.append(
                f"- {marker_type}: {len(typed_markers)} segments linked, strongest phrase '{top_example.matched_phrase}' at segment {top_example.source_segment_index}."
            )
        return "\n".join(lines)

    @staticmethod
    def _block_from_markers(markers: list[Marker], marker_type: str | MarkerType) -> str:
        selected = MarkerDetector.markers_by_type(markers, marker_type)
        if not selected:
            return ""
        unique_lines: list[str] = []
        seen_texts: set[tuple[str, str]] = set()
        for marker in sorted(selected, key=MarkerDetector.marker_sort_key):
            key = (marker.text.strip().lower(), marker.source)
            if key in seen_texts:
                continue
            seen_texts.add(key)
            suffix_parts = [
                f"match: {marker.matched_phrase}",
                f"score: {marker.score:.1f}",
                f"segment: {marker.source_segment_index}",
                f"review: {marker.review_state}",
            ]
            if marker.source != "text":
                suffix_parts.append(f"source: {marker.source}")
            unique_lines.append(f"- {marker.text.strip()} ({', '.join(suffix_parts)})")
        return "\n".join(unique_lines)
