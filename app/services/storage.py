from __future__ import annotations

import mimetypes
import uuid
from pathlib import Path

from app.models import ArtifactRecord


class ArtifactStorage:
    def __init__(self, lecture_id: str, working_dir: Path) -> None:
        self.lecture_id = lecture_id
        self.working_dir = working_dir
        self.working_dir.mkdir(parents=True, exist_ok=True)

    def build_paths(self) -> dict[str, Path]:
        return {
            "normalized_audio": self.working_dir / "normalized.wav",
            "transcript": self.working_dir / "transcript.json",
            "notes_source": self.working_dir / "notes_source.json",
            "notes_json": self.working_dir / "notes.json",
            "review_json": self.working_dir / "review.json",
            "notes_md": self.working_dir / "notes.md",
            "notes_html": self.working_dir / "notes.html",
            "notes_pdf": self.working_dir / "notes.pdf",
            "notes_docx": self.working_dir / "notes.docx",
            "final_note_prep_json": self.working_dir / "final_note_prep.json",
            "final_note_json": self.working_dir / "final_note.json",
            "final_note_md": self.working_dir / "final_note.md",
            "final_note_html": self.working_dir / "final_note.html",
            "final_note_pdf": self.working_dir / "final_note.pdf",
            "runtime_info": self.working_dir / "runtime_info.json",
            "llm_runtime": self.working_dir / "llm_runtime.json",
        }

    def build_artifact_records(self, artifact_paths: dict[str, Path]) -> list[ArtifactRecord]:
        records: list[ArtifactRecord] = []
        for artifact_type, path in artifact_paths.items():
            if artifact_type == "normalized_audio":
                continue
            if not path.exists():
                continue
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            records.append(
                ArtifactRecord(
                    artifact_id=str(uuid.uuid4()),
                    lecture_id=self.lecture_id,
                    artifact_type=artifact_type,
                    path=str(path),
                    content_type=content_type,
                    file_size_bytes=path.stat().st_size,
                )
            )
        return records
