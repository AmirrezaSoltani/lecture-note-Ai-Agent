from __future__ import annotations

import re
from dataclasses import dataclass

from app.models import MarkerType, TranscriptSegment


_TURKISH_TRANSLATION_TABLE = str.maketrans({"İ": "i", "I": "ı", "Ş": "ş", "Ğ": "ğ", "Ü": "ü", "Ö": "ö", "Ç": "ç"})


def normalize_text(value: str) -> str:
    lowered = value.translate(_TURKISH_TRANSLATION_TABLE).lower().strip()
    return re.sub(r"\s+", " ", lowered)


@dataclass(slots=True)
class ClassificationSuggestion:
    marker_type: str
    score: float
    matched_phrase: str
    reason: str


class ControlledNoteAssistant:
    """Low-creativity helper for deterministic cleanup, titles, and light classification.

    This intentionally avoids free-form generation. It uses bounded heuristics so the
    system remains auditable and fast even when the optional AI sprint is enabled.
    """

    _FILLER_PATTERNS = [
        r"\b(yani|şimdi|arkadaşlar|tamam|evet|hani|eee+|ııı+|şey|baktığınızda)\b",
        r"\b(diyeceğim|diyelim)\b",
    ]

    _QUESTION_PATTERNS = [
        r"\?",
        r"\b(neden|niçin|hangi|hangisi|nasıl|kaç|ne zaman|ne olur)\b",
        r"\bşöyle sor(ar|ulur)\b",
        r"\bbunu sorabilir\b",
    ]
    _CASE_PATTERNS = [
        r"\b(vaka|olgu)\b",
        r"\bhasta\b",
        r"\bklinik\s+senaryo\b",
        r"\bbaşvuruyor\b",
    ]
    _EXAMPLE_PATTERNS = [
        r"\b(örnek|örneğin|mesela)\b",
        r"\bdiyelim ki\b",
        r"\bmisal\b",
    ]

    def cleanup_text(self, text: str) -> str:
        normalized = " ".join(text.replace("\n", " ").split()).strip()
        if not normalized:
            return ""
        lowered = normalized
        for pattern in self._FILLER_PATTERNS:
            lowered = re.sub(pattern, " ", lowered, flags=re.IGNORECASE)
        lowered = re.sub(r"\s+", " ", lowered).strip(" ,;:-")
        # Remove immediate word duplication without rewriting semantics.
        lowered = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", lowered, flags=re.IGNORECASE)
        if not lowered:
            return ""
        if lowered[-1] not in ".!?":
            lowered += "."
        return lowered[0].upper() + lowered[1:]

    def classify_segment(self, text: str) -> list[ClassificationSuggestion]:
        normalized = normalize_text(text)
        suggestions: list[ClassificationSuggestion] = []
        for marker_type, base, patterns, phrase in [
            (MarkerType.QUESTION, 3.6, self._QUESTION_PATTERNS, "question cue"),
            (MarkerType.CASE, 3.3, self._CASE_PATTERNS, "case cue"),
            (MarkerType.EXAMPLE, 3.0, self._EXAMPLE_PATTERNS, "example cue"),
        ]:
            matched = next((pattern for pattern in patterns if re.search(pattern, normalized, flags=re.IGNORECASE)), None)
            if matched is None:
                continue
            score = base
            if marker_type == MarkerType.QUESTION and "?" in text:
                score += 0.8
            if marker_type == MarkerType.CASE and re.search(r"\b(hasta|erkek|kadın)\b", normalized):
                score += 0.6
            if marker_type == MarkerType.EXAMPLE and re.search(r"\b(örnek|mesela)\b", normalized):
                score += 0.5
            suggestions.append(
                ClassificationSuggestion(
                    marker_type=str(marker_type),
                    score=score,
                    matched_phrase=phrase,
                    reason=f"matched:{matched}",
                )
            )
        return suggestions

    def generate_section_title(self, segments: list[TranscriptSegment], marker_types: list[str], fallback: str) -> str:
        markers = set(marker_types)
        text = " ".join(segment.text for segment in segments[:3])
        normalized = normalize_text(text)
        if MarkerType.EXAM_HIGH in markers:
            return "Study Notes — Exam Priorities and Recall Triggers"
        if MarkerType.CASE in markers and MarkerType.EXAMPLE in markers:
            return "Study Notes — Cases, Examples, and Teaching Cues"
        if MarkerType.KEYWORD in markers:
            return "Study Notes — Core Terms and Memory Hooks"
        if MarkerType.QUESTION in markers:
            return "Study Notes — Questions and Likely Prompts"
        if re.search(r"\b(tanı|tedavi|sınıflama|yaklaşım)\b", normalized):
            return "Study Notes — Core Definitions and Clinical Framing"
        return fallback
