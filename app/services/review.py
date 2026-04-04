from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.models import LectureNoteBundle, Marker, NoteBlock, ReviewState, utc_now_iso
from app.services.notes import NoteComposer


@dataclass(slots=True)
class ReviewOverlay:
    block_overrides: dict[str, dict[str, object]]
    marker_reviews: dict[str, dict[str, object]]
    updated_at: str

    def to_dict(self) -> dict[str, object]:
        return {
            "block_overrides": self.block_overrides,
            "marker_reviews": self.marker_reviews,
            "updated_at": self.updated_at,
        }


class ReviewService:
    def __init__(self) -> None:
        self.note_composer = NoteComposer()

    def load_overlay(self, path: Path) -> ReviewOverlay:
        if not path.exists():
            return ReviewOverlay(block_overrides={}, marker_reviews={}, updated_at=utc_now_iso())
        payload = json.loads(path.read_text(encoding="utf-8"))
        return ReviewOverlay(
            block_overrides=dict(payload.get("block_overrides", {})),
            marker_reviews=dict(payload.get("marker_reviews", {})),
            updated_at=str(payload.get("updated_at") or utc_now_iso()),
        )

    def initialize_overlay(self, path: Path) -> Path:
        if not path.exists():
            self.save_overlay(path, ReviewOverlay(block_overrides={}, marker_reviews={}, updated_at=utc_now_iso()))
        return path

    def save_overlay(self, path: Path, overlay: ReviewOverlay) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(overlay.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def update_block(self, path: Path, block_id: str, title: str, content: str) -> ReviewOverlay:
        overlay = self.load_overlay(path)
        overlay.block_overrides[block_id] = {
            "title": title.strip(),
            "content": content.strip(),
            "edited": True,
        }
        overlay.updated_at = utc_now_iso()
        self.save_overlay(path, overlay)
        return overlay

    def review_marker(self, path: Path, marker_id: str, review_state: str, reviewer_note: str) -> ReviewOverlay:
        overlay = self.load_overlay(path)
        overlay.marker_reviews[marker_id] = {
            "review_state": review_state,
            "reviewer_note": reviewer_note.strip() or None,
        }
        overlay.updated_at = utc_now_iso()
        self.save_overlay(path, overlay)
        return overlay

    def bulk_review_markers(
        self,
        path: Path,
        bundle: LectureNoteBundle,
        review_state: str,
        marker_ids: list[str] | None = None,
        marker_type: str | None = None,
        current_state: str | None = None,
        text_query: str | None = None,
    ) -> int:
        overlay = self.load_overlay(path)
        selected = self._select_markers(
            bundle,
            marker_ids=marker_ids or [],
            marker_type=marker_type,
            current_state=current_state,
            text_query=text_query,
        )
        for marker in selected:
            overlay.marker_reviews[marker.marker_id] = {
                "review_state": review_state,
                "reviewer_note": overlay.marker_reviews.get(marker.marker_id, {}).get("reviewer_note"),
            }
        overlay.updated_at = utc_now_iso()
        self.save_overlay(path, overlay)
        return len(selected)

    def apply_overlay(self, bundle: LectureNoteBundle, overlay: ReviewOverlay) -> LectureNoteBundle:
        updated_markers: list[Marker] = []
        for marker in bundle.markers:
            review = overlay.marker_reviews.get(marker.marker_id, {})
            updated_markers.append(
                Marker(
                    marker_id=marker.marker_id,
                    marker_type=marker.marker_type,
                    label=marker.label,
                    score=marker.score,
                    matched_phrase=marker.matched_phrase,
                    text=marker.text,
                    start_ms=marker.start_ms,
                    end_ms=marker.end_ms,
                    source_segment_index=marker.source_segment_index,
                    source=marker.source,
                    review_state=str(review.get("review_state") or marker.review_state),
                    reviewer_note=review.get("reviewer_note") or marker.reviewer_note,
                )
            )

        reviewed_bundle = LectureNoteBundle(
            lecture_id=bundle.lecture_id,
            lecture_title=bundle.lecture_title,
            created_at=bundle.created_at,
            segments=bundle.segments,
            markers=updated_markers,
            note_blocks=bundle.note_blocks,
        )
        recomposed_bundle = self.note_composer.recompose_from_reviews(reviewed_bundle)
        updated_blocks: list[NoteBlock] = []
        for block in recomposed_bundle.note_blocks:
            override = overlay.block_overrides.get(block.block_id, {})
            updated_blocks.append(
                NoteBlock(
                    block_id=block.block_id,
                    block_type=block.block_type,
                    title=str(override.get("title") or block.title),
                    content=str(override.get("content") or block.content),
                    marker_types=list(block.marker_types),
                    source_segment_indexes=list(block.source_segment_indexes),
                    edited=bool(override.get("edited", block.edited)),
                )
            )

        return LectureNoteBundle(
            lecture_id=recomposed_bundle.lecture_id,
            lecture_title=recomposed_bundle.lecture_title,
            created_at=recomposed_bundle.created_at,
            segments=recomposed_bundle.segments,
            markers=recomposed_bundle.markers,
            note_blocks=updated_blocks,
        )

    def _select_markers(
        self,
        bundle: LectureNoteBundle,
        marker_ids: list[str],
        marker_type: str | None,
        current_state: str | None,
        text_query: str | None,
    ) -> list[Marker]:
        selected_ids = {item.strip() for item in marker_ids if item.strip()}
        type_norm = (marker_type or "").strip().lower()
        state_norm = (current_state or "").strip().lower()
        query_norm = (text_query or "").strip().lower()
        matched: list[Marker] = []
        for marker in bundle.markers:
            if selected_ids and marker.marker_id not in selected_ids:
                continue
            if type_norm and marker.marker_type.lower() != type_norm:
                continue
            if state_norm and marker.review_state.lower() != state_norm:
                continue
            if query_norm:
                haystack = " ".join(
                    [marker.text, marker.label, marker.matched_phrase, marker.reviewer_note or ""]
                ).lower()
                if query_norm not in haystack:
                    continue
            matched.append(marker)
        return matched

    @staticmethod
    def review_summary(bundle: LectureNoteBundle) -> dict[str, int]:
        counts = {state.value: 0 for state in ReviewState}
        for marker in bundle.markers:
            state = marker.review_state if marker.review_state in counts else ReviewState.PENDING
            counts[str(state)] += 1
        counts["edited_blocks"] = sum(1 for block in bundle.note_blocks if block.edited)
        counts["visible_markers"] = sum(1 for marker in bundle.markers if marker.review_state != ReviewState.REJECTED)
        counts["approved_markers"] = sum(1 for marker in bundle.markers if marker.review_state == ReviewState.APPROVED)
        return counts
