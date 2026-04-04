from app.models import TranscriptSegment
from app.services.ai_assist import ControlledNoteAssistant
from app.services.markers import MarkerDetector
from app.services.notes import NoteComposer


def test_controlled_assistant_cleanup_is_limited_and_deterministic() -> None:
    helper = ControlledNoteAssistant()
    cleaned = helper.cleanup_text("Yani arkadaşlar şey bu konu konu çok önemli")
    assert cleaned == "Bu konu çok önemli."


def test_marker_detector_adds_heuristic_question_marker_when_lexicon_is_weak() -> None:
    detector = MarkerDetector()
    segments = [
        TranscriptSegment(index=0, start_ms=0, end_ms=1000, text="Neden burada nekroz gelişiyor?"),
    ]
    markers = detector.detect(segments)
    question_markers = [marker for marker in markers if marker.marker_type == "question"]
    assert question_markers
    assert any(marker.source == "ai_heuristic" for marker in question_markers)


def test_note_composer_generates_more_specific_study_title_for_exam_markers() -> None:
    segments = [
        TranscriptSegment(index=0, start_ms=0, end_ms=1000, text="Kesin çıkacak tanı yaklaşımı."),
        TranscriptSegment(index=1, start_ms=1000, end_ms=2000, text="Anahtar kelime orantısız ağrı."),
    ]
    markers = MarkerDetector().detect(segments)
    bundle = NoteComposer().compose(
        lecture_id="lecture-ai-1",
        lecture_title="Demo",
        segments=segments,
        markers=markers,
    )
    study_block = next(block for block in bundle.note_blocks if block.block_id == "study-notes")
    assert study_block.title == "Study Notes — Exam Priorities and Recall Triggers"
    assert "exam priority" in study_block.content.lower()


from app.models import LectureNoteBundle, NoteBlock, utc_now_iso
from app.services.openai_notes import OpenAINotesPolisher, OpenAITimeoutError


def test_openai_polisher_retries_timed_out_groups_by_splitting(monkeypatch) -> None:
    bundle = LectureNoteBundle(
        lecture_id="lecture-timeout",
        lecture_title="Timeout Demo",
        created_at=utc_now_iso(),
        segments=[],
        markers=[],
        note_blocks=[
            NoteBlock(block_id="b1", block_type="main", title="One", content="alpha"),
            NoteBlock(block_id="b2", block_type="main", title="Two", content="beta"),
        ],
    )
    polisher = OpenAINotesPolisher(force_enabled=True)
    polisher.api_key = "test-key"
    polisher.enabled = True

    calls: list[list[str]] = []

    def fake_request_json(body):
        payload = __import__("json").loads(body["messages"][1]["content"])
        block_ids = [item["block_id"] for item in payload["note_blocks"]]
        calls.append(block_ids)
        if len(block_ids) > 1:
            raise OpenAITimeoutError("timed out")
        item = payload["note_blocks"][0]
        return {
            "choices": [{
                "message": {
                    "content": __import__("json").dumps({
                        "note_blocks": [{
                            **item,
                            "content": f"polished-{item['block_id']}",
                        }]
                    })
                }
            }]
        }

    monkeypatch.setattr(polisher, "_request_json", fake_request_json)
    polished_bundle, runtime = polisher.polish_bundle(bundle)

    assert [block.content for block in polished_bundle.note_blocks] == ["polished-b1", "polished-b2"]
    assert calls == [["b1", "b2"], ["b1"], ["b2"]]
    assert runtime.applied is True
    assert runtime.chunk_count == 3
    assert "Recovered from 1 timeout split" in runtime.detail
