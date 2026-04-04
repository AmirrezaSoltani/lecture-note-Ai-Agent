from app.main import _filter_markers, _filter_transcript, _markers_by_segment


def test_filter_markers_by_text_type_and_state() -> None:
    markers = [
        {
            "marker_id": "m1",
            "marker_type": "exam_high",
            "review_state": "approved",
            "text": "Kesin çıkacak konu",
            "label": "Exam Focus",
            "matched_phrase": "kesin çıkacak",
            "reviewer_note": "",
            "source_segment_index": 0,
        },
        {
            "marker_id": "m2",
            "marker_type": "keyword",
            "review_state": "pending",
            "text": "Anahtar kelime",
            "label": "Keyword",
            "matched_phrase": "anahtar kelime",
            "reviewer_note": "teacher",
            "source_segment_index": 1,
        },
    ]
    filtered = _filter_markers(markers, q="kesin", marker_type="exam_high", review_state="approved")
    assert [item["marker_id"] for item in filtered] == ["m1"]


def test_filter_transcript_marks_related_segments() -> None:
    transcript = [
        {"index": 0, "text": "Kesin çıkacak konu", "start_ms": 0, "end_ms": 1000},
        {"index": 1, "text": "Diger cumle", "start_ms": 1000, "end_ms": 2000},
    ]
    filtered = _filter_transcript(transcript, transcript_q="", related_segment_indexes={0})
    assert len(filtered) == 1
    assert filtered[0]["index"] == 0
    assert filtered[0]["is_related"] is True


def test_markers_grouped_by_segment() -> None:
    grouped = _markers_by_segment([
        {"marker_id": "m1", "source_segment_index": 2},
        {"marker_id": "m2", "source_segment_index": 2},
        {"marker_id": "m3", "source_segment_index": 4},
    ])
    assert [item["marker_id"] for item in grouped[2]] == ["m1", "m2"]
    assert [item["marker_id"] for item in grouped[4]] == ["m3"]
