from __future__ import annotations

import base64
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


client = TestClient(app)


def _basic_auth_header(username: str, password: str) -> dict[str, str]:
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}"}


def test_readyz_and_metrics_are_available() -> None:
    ready = client.get("/readyz")
    assert ready.status_code == 200
    assert "database_exists" in ready.json()

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "lecture_app_uptime_seconds" in metrics.text


def test_backup_endpoint_creates_snapshot(tmp_path: Path) -> None:
    original = settings.backup_dir
    settings.backup_dir = tmp_path
    try:
        response = client.post("/api/admin/backup")
        assert response.status_code == 200
        payload = response.json()
        backup_path = Path(payload["backup_path"])
        assert backup_path.exists()
        assert (backup_path / "manifest.json").exists()
        assert (backup_path / "artifacts.tar.gz").exists()
    finally:
        settings.backup_dir = original


def test_basic_auth_protects_private_routes(monkeypatch) -> None:
    monkeypatch.setattr(settings, "basic_auth_enabled", True)
    monkeypatch.setattr(settings, "basic_auth_username", "reviewer")
    monkeypatch.setattr(settings, "basic_auth_password", "secret-pass")
    try:
        unauthorized = client.get("/")
        assert unauthorized.status_code == 401

        public_health = client.get("/healthz")
        assert public_health.status_code == 200

        authorized = client.get("/", headers=_basic_auth_header("reviewer", "secret-pass"))
        assert authorized.status_code == 200
    finally:
        monkeypatch.setattr(settings, "basic_auth_enabled", False)
        monkeypatch.setattr(settings, "basic_auth_username", "admin")
        monkeypatch.setattr(settings, "basic_auth_password", "change-me")
