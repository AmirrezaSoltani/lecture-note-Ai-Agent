from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from app.models import Marker, MarkerType, TranscriptSegment
from app.services.ai_assist import ControlledNoteAssistant

_TURKISH_TRANSLATION_TABLE = str.maketrans({"İ": "i", "I": "ı", "Ş": "ş", "Ğ": "ğ", "Ü": "ü", "Ö": "ö", "Ç": "ç"})


def normalize_text(value: str) -> str:
    lowered = value.translate(_TURKISH_TRANSLATION_TABLE).lower().strip()
    return re.sub(r"\s+", " ", lowered)


@dataclass(slots=True)
class MarkerDefinition:
    marker_type: str
    label: str
    base_score: float
    patterns: list[re.Pattern[str]]
    context_window: int


class MarkerDetector:
    def __init__(self, lexicon_path: Path | None = None) -> None:
        if lexicon_path is None:
            lexicon_path = Path(__file__).resolve().parent.parent / "lexicons" / "tr_exam_markers.json"
        self._definitions = self._load_definitions(lexicon_path)
        self._assistant = ControlledNoteAssistant()

    def detect(self, segments: list[TranscriptSegment]) -> list[Marker]:
        markers: list[Marker] = []
        normalized_segments = [normalize_text(segment.text) for segment in segments]

        for idx, segment in enumerate(segments):
            normalized = normalized_segments[idx]
            repetition_bonus = self._repetition_bonus(normalized, normalized_segments, idx)
            emphasis_bonus = self._emphasis_bonus(segment.text)
            segment_marker_types: set[str] = set()

            for definition in self._definitions:
                for pattern in definition.patterns:
                    match = pattern.search(normalized)
                    if not match:
                        continue
                    base_marker = self._create_marker(
                        definition=definition,
                        segment=segment,
                        match_text=match.group(0),
                        score=definition.base_score + repetition_bonus + emphasis_bonus,
                        source="text",
                    )
                    markers.append(base_marker)
                    segment_marker_types.add(base_marker.marker_type)
                    if definition.context_window > 0:
                        markers.extend(
                            self._attach_context_markers(
                                segments=segments,
                                center_index=idx,
                                base_marker=base_marker,
                                context_window=definition.context_window,
                            )
                        )
                    break
            markers.extend(
                self._assistant_markers(
                    segment=segment,
                    segment_marker_types=segment_marker_types,
                    emphasis_bonus=emphasis_bonus,
                )
            )
        return self._deduplicate(markers)


    def _assistant_markers(self, segment: TranscriptSegment, segment_marker_types: set[str], emphasis_bonus: float) -> list[Marker]:
        existing_types = set(segment_marker_types)
        attached: list[Marker] = []
        for suggestion in self._assistant.classify_segment(segment.text):
            if suggestion.marker_type in existing_types:
                continue
            if suggestion.score < 3.0:
                continue
            attached.append(
                self._create_marker(
                    definition=MarkerDefinition(
                        marker_type=suggestion.marker_type,
                        label=self._label_for_type(suggestion.marker_type),
                        base_score=suggestion.score + emphasis_bonus,
                        patterns=[],
                        context_window=0,
                    ),
                    segment=segment,
                    match_text=suggestion.matched_phrase,
                    score=suggestion.score + emphasis_bonus,
                    source="ai_heuristic",
                )
            )
        return attached

    @staticmethod
    def _label_for_type(marker_type: str) -> str:
        mapping = {
            str(MarkerType.QUESTION): "Question",
            str(MarkerType.CASE): "Case / Vaka",
            str(MarkerType.EXAMPLE): "Example",
        }
        return mapping.get(marker_type, marker_type.replace("_", " ").title())

    @staticmethod
    def _load_definitions(lexicon_path: Path) -> list[MarkerDefinition]:
        payload = json.loads(lexicon_path.read_text(encoding="utf-8"))
        definitions: list[MarkerDefinition] = []
        for marker_type, config in payload.items():
            definitions.append(
                MarkerDefinition(
                    marker_type=marker_type,
                    label=str(config["label"]),
                    base_score=float(config["base_score"]),
                    patterns=[re.compile(pattern) for pattern in config["patterns"]],
                    context_window=int(config.get("context_window", 0)),
                )
            )
        return definitions

    def _attach_context_markers(
        self,
        segments: list[TranscriptSegment],
        center_index: int,
        base_marker: Marker,
        context_window: int,
    ) -> list[Marker]:
        attached: list[Marker] = []
        for offset in range(1, context_window + 1):
            for neighbor_index in (center_index - offset, center_index + offset):
                if neighbor_index < 0 or neighbor_index >= len(segments):
                    continue
                segment = segments[neighbor_index]
                attached.append(
                    self._create_marker(
                        definition=MarkerDefinition(
                            marker_type=base_marker.marker_type,
                            label=base_marker.label,
                            base_score=max(0.0, base_marker.score - (offset * 1.1)),
                            patterns=[],
                            context_window=0,
                        ),
                        segment=segment,
                        match_text=base_marker.matched_phrase,
                        score=max(0.0, base_marker.score - (offset * 1.1)),
                        source="context",
                    )
                )
        return attached

    @staticmethod
    def _repetition_bonus(normalized_text: str, normalized_segments: list[str], current_index: int) -> float:
        if not normalized_text:
            return 0.0
        repeated_count = 0
        for idx, other in enumerate(normalized_segments):
            if idx == current_index:
                continue
            if normalized_text and normalized_text in other:
                repeated_count += 1
        return min(2.0, repeated_count * 0.4)

    @staticmethod
    def _emphasis_bonus(text: str) -> float:
        bonus = 0.0
        if text.count("!") >= 1:
            bonus += 0.5
        words = text.split()
        if len(words) > 0 and sum(1 for word in words if word.isupper() and len(word) > 2) >= 1:
            bonus += 0.5
        if text.count(":") >= 1:
            bonus += 0.2
        return bonus

    @staticmethod
    def _create_marker(
        definition: MarkerDefinition,
        segment: TranscriptSegment,
        match_text: str,
        score: float,
        source: str,
    ) -> Marker:
        marker_id = hashlib.sha1(
            f"{definition.marker_type}|{segment.index}|{segment.start_ms}|{segment.end_ms}|{match_text}|{source}".encode("utf-8")
        ).hexdigest()[:16]
        return Marker(
            marker_id=marker_id,
            marker_type=definition.marker_type,
            label=definition.label,
            score=score,
            matched_phrase=match_text,
            text=segment.text.strip(),
            start_ms=segment.start_ms,
            end_ms=segment.end_ms,
            source_segment_index=segment.index,
            source=source,
        )

    @staticmethod
    def _deduplicate(markers: list[Marker]) -> list[Marker]:
        best_by_id: dict[str, Marker] = {}
        for marker in markers:
            current = best_by_id.get(marker.marker_id)
            if current is None or marker.score > current.score:
                best_by_id[marker.marker_id] = marker
        return sorted(best_by_id.values(), key=MarkerDetector.marker_sort_key)

    @staticmethod
    def marker_sort_key(marker: Marker) -> tuple[int, int, float, str]:
        return (marker.start_ms, marker.end_ms, -marker.score, marker.marker_type)

    @staticmethod
    def top_markers(markers: list[Marker], limit: int = 5) -> list[Marker]:
        return sorted(markers, key=lambda item: (-item.score, item.start_ms, item.marker_type))[:limit]

    @staticmethod
    def markers_by_type(markers: list[Marker], marker_type: str | MarkerType) -> list[Marker]:
        marker_type_value = str(marker_type)
        return [marker for marker in markers if marker.marker_type == marker_type_value]
