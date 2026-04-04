from __future__ import annotations

import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings


def create_backup() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_root = settings.backup_dir / f"backup_{timestamp}"
    backup_root.mkdir(parents=True, exist_ok=True)

    database_copy = backup_root / settings.database_path.name
    shutil.copy2(settings.database_path, database_copy)

    artifacts_archive = backup_root / "artifacts.tar.gz"
    with tarfile.open(artifacts_archive, "w:gz") as tar:
        tar.add(settings.artifacts_dir, arcname="artifacts")
        tar.add(settings.uploads_dir, arcname="uploads")

    manifest = {
        "created_at": timestamp,
        "database": str(database_copy.name),
        "artifacts_archive": str(artifacts_archive.name),
        "data_dir": str(settings.data_dir),
    }
    (backup_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return backup_root


if __name__ == "__main__":
    path = create_backup()
    print(path)
