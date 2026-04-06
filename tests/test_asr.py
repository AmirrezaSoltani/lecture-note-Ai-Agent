from app.services.asr import _resolve_whisper_device


def test_resolve_whisper_device_prefers_cuda_when_available(monkeypatch) -> None:
    monkeypatch.setattr("app.services.asr._cuda_available", lambda: True)
    assert _resolve_whisper_device("auto") == "cuda"
    assert _resolve_whisper_device("gpu") == "cuda"


def test_resolve_whisper_device_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr("app.services.asr._cuda_available", lambda: False)
    assert _resolve_whisper_device("auto") == "cpu"
    assert _resolve_whisper_device("gpu") == "cpu"
    assert _resolve_whisper_device("cpu") == "cpu"
