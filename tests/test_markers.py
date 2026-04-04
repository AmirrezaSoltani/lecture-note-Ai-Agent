from app.models import TranscriptSegment
from app.services.markers import MarkerDetector


def test_marker_detector_finds_exam_and_keyword_markers() -> None:
    detector = MarkerDetector()
    segments = [
        TranscriptSegment(index=0, start_ms=0, end_ms=1000, text="Kesin çıkacak bu konu."),
        TranscriptSegment(index=1, start_ms=1000, end_ms=2000, text="Anahtar kelime orantısız ağrı."),
        TranscriptSegment(index=2, start_ms=2000, end_ms=3000, text="Şimdi bir vaka anlatıyorum."),
    ]
    markers = detector.detect(segments)

    types = {marker.marker_type for marker in markers}
    assert "exam_high" in types
    assert "keyword" in types
    assert "case" in types
    assert all(marker.marker_id for marker in markers)


def test_marker_detector_supports_numeric_exam_phrase_and_context() -> None:
    detector = MarkerDetector()
    segments = [
        TranscriptSegment(index=0, start_ms=0, end_ms=1000, text="Burası 4x soruldu."),
        TranscriptSegment(index=1, start_ms=1000, end_ms=2000, text="Orantısız ağrı nekrotizan fasiit için önemlidir."),
    ]

    markers = detector.detect(segments)
    exam_markers = [marker for marker in markers if marker.marker_type == "exam_high"]
    assert len(exam_markers) >= 2
    assert any(marker.source == "context" for marker in exam_markers)
