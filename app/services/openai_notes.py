from __future__ import annotations

import json
import logging
import socket
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib import error, request
from urllib.parse import urlparse

from app.config import settings
from app.models import LectureNoteBundle, NoteBlock

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LLMRuntimeInfo:
    enabled: bool
    applied: bool
    provider: str
    model: str
    chunk_count: int
    total_ms: float
    status: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OpenAITimeoutError(RuntimeError):
    """Raised when the OpenAI request times out while waiting for a response."""


class OpenAINotesPolisher:
    def __init__(self, *, force_enabled: bool = False) -> None:
        runtime_settings = settings.snapshot() if hasattr(settings, "snapshot") else settings
        self.force_enabled = force_enabled
        self.api_key = runtime_settings.openai_api_key
        self.enabled = (force_enabled or runtime_settings.llm_notes_enabled) and bool(self.api_key)
        self.model = runtime_settings.openai_model
        self.base_url = runtime_settings.openai_base_url
        self.timeout_seconds = runtime_settings.openai_timeout_seconds
        self.max_input_chars = runtime_settings.openai_max_input_chars
        self._request_count = 0
        self._fallback_split_count = 0

    def diagnostics(self) -> dict[str, Any]:
        return build_openai_diagnostics(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            timeout_seconds=self.timeout_seconds,
        )

    def polish_bundle(self, bundle: LectureNoteBundle, output_path: Path | None = None) -> tuple[LectureNoteBundle, LLMRuntimeInfo]:
        if not self.enabled:
            runtime = LLMRuntimeInfo(
                enabled=False,
                applied=False,
                provider="openai",
                model=self.model,
                chunk_count=0,
                total_ms=0.0,
                status="disabled",
                detail="LLM polishing disabled or API key missing.",
            )
            self._persist_runtime(output_path, runtime)
            return bundle, runtime

        source_blocks = [block for block in bundle.note_blocks if block.content.strip()]
        if not source_blocks:
            runtime = LLMRuntimeInfo(True, False, "openai", self.model, 0, 0.0, "skipped", "No note blocks to polish.")
            self._persist_runtime(output_path, runtime)
            return bundle, runtime

        block_groups = self._chunk_blocks(source_blocks)
        started = perf_counter()
        polished_blocks: list[NoteBlock] = []
        self._request_count = 0
        self._fallback_split_count = 0
        try:
            for group_index, group in enumerate(block_groups):
                polished_blocks.extend(self._polish_block_group_with_fallback(bundle, group, group_index, len(block_groups)))
            detail = ""
            if self._fallback_split_count:
                detail = (
                    f"Recovered from {self._fallback_split_count} timeout split(s) by retrying smaller LLM chunks. "
                    f"Total LLM requests: {self._request_count}."
                )
            runtime = LLMRuntimeInfo(
                enabled=True,
                applied=True,
                provider="openai",
                model=self.model,
                chunk_count=max(self._request_count, len(block_groups)),
                total_ms=round((perf_counter() - started) * 1000, 3),
                status="applied",
                detail=detail,
            )
            polished_bundle = LectureNoteBundle(
                lecture_id=bundle.lecture_id,
                lecture_title=bundle.lecture_title,
                created_at=bundle.created_at,
                segments=bundle.segments,
                markers=bundle.markers,
                note_blocks=polished_blocks or bundle.note_blocks,
            )
            self._persist_runtime(output_path, runtime)
            return polished_bundle, runtime
        except Exception as exc:
            logger.exception("OpenAI note polishing failed: %s", exc)
            runtime = LLMRuntimeInfo(
                enabled=True,
                applied=False,
                provider="openai",
                model=self.model,
                chunk_count=max(self._request_count, len(block_groups)),
                total_ms=round((perf_counter() - started) * 1000, 3),
                status="failed",
                detail=str(exc),
            )
            self._persist_runtime(output_path, runtime)
            return bundle, runtime

    def _chunk_blocks(self, blocks: list[NoteBlock]) -> list[list[NoteBlock]]:
        groups: list[list[NoteBlock]] = []
        current: list[NoteBlock] = []
        current_chars = 0
        for block in blocks:
            block_chars = len(block.title) + len(block.content) + 256
            if current and current_chars + block_chars > self.max_input_chars:
                groups.append(current)
                current = []
                current_chars = 0
            current.append(block)
            current_chars += block_chars
        if current:
            groups.append(current)
        return groups

    def _polish_block_group_with_fallback(
        self,
        bundle: LectureNoteBundle,
        blocks: list[NoteBlock],
        group_index: int,
        group_count: int,
    ) -> list[NoteBlock]:
        try:
            return self._polish_block_group(bundle, blocks, group_index, group_count)
        except OpenAITimeoutError as exc:
            if len(blocks) <= 1:
                raise
            mid = max(1, len(blocks) // 2)
            self._fallback_split_count += 1
            logger.warning(
                "OpenAI timeout for lecture group %s/%s with %s block(s); retrying with two smaller chunks.",
                group_index + 1,
                group_count,
                len(blocks),
            )
            left = self._polish_block_group_with_fallback(bundle, blocks[:mid], group_index, group_count)
            right = self._polish_block_group_with_fallback(bundle, blocks[mid:], group_index, group_count)
            return left + right

    def _polish_block_group(
        self,
        bundle: LectureNoteBundle,
        blocks: list[NoteBlock],
        group_index: int,
        group_count: int,
    ) -> list[NoteBlock]:
        payload = {
            "lecture_title": bundle.lecture_title,
            "language": "tr",
            "group_index": group_index + 1,
            "group_count": group_count,
            "instructions": {
                "style": "Produce clean Turkish study notes. Keep facts from source only. Remove filler, noise, obvious ASR repetition, and conversational clutter.",
                "constraints": [
                    "Do not invent new facts.",
                    "Preserve exam cues, cases, examples, and keywords when present.",
                    "Keep block_id and block_type unchanged.",
                    "Return concise but complete content.",
                ],
            },
            "note_blocks": [
                {
                    "block_id": block.block_id,
                    "block_type": block.block_type,
                    "title": block.title,
                    "content": block.content,
                    "marker_types": block.marker_types,
                    "source_segment_indexes": block.source_segment_indexes,
                }
                for block in blocks
            ],
        }
        schema = {
            "name": "polished_notes",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "note_blocks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "block_id": {"type": "string"},
                                "block_type": {"type": "string"},
                                "title": {"type": "string"},
                                "content": {"type": "string"},
                                "marker_types": {"type": "array", "items": {"type": "string"}},
                                "source_segment_indexes": {"type": "array", "items": {"type": "integer"}},
                            },
                            "required": ["block_id", "block_type", "title", "content", "marker_types", "source_segment_indexes"],
                        },
                    }
                },
                "required": ["note_blocks"],
            },
        }
        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "developer",
                    "content": (
                        "You are cleaning lecture notes into final study notes. Use only provided source material. "
                        "Keep the output faithful, concise, and structured."
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_schema", "json_schema": schema},
            "temperature": 0.1,
        }
        self._request_count += 1
        raw = self._request_json(body)
        message = raw["choices"][0]["message"]
        content = message.get("content", "")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        parsed = json.loads(content)
        result: list[NoteBlock] = []
        for item in parsed.get("note_blocks", []):
            original = next((b for b in blocks if b.block_id == item.get("block_id")), None)
            result.append(
                NoteBlock(
                    block_id=item["block_id"],
                    block_type=item["block_type"],
                    title=item["title"].strip() or (original.title if original else ""),
                    content=item["content"].strip() or (original.content if original else ""),
                    marker_types=list(item.get("marker_types", original.marker_types if original else [])),
                    source_segment_indexes=list(item.get("source_segment_indexes", original.source_segment_indexes if original else [])),
                    edited=getattr(original, "edited", False),
                )
            )
        return result

    def _request_json(self, body: dict[str, Any]) -> dict[str, Any]:
        endpoint = f"{self.base_url}/chat/completions"
        req = request.Request(
            endpoint,
            method="POST",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:  # pragma: no cover - network/runtime path
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"OpenAI HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:  # pragma: no cover - network/runtime path
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, (TimeoutError, socket.timeout)):
                raise OpenAITimeoutError(_format_openai_timeout_error(self.timeout_seconds, self.max_input_chars)) from exc
            raise RuntimeError(_format_openai_connection_error(exc, self.base_url)) from exc
        except TimeoutError as exc:  # pragma: no cover - network/runtime path
            raise OpenAITimeoutError(_format_openai_timeout_error(self.timeout_seconds, self.max_input_chars)) from exc
        except socket.timeout as exc:  # pragma: no cover - network/runtime path
            raise OpenAITimeoutError(_format_openai_timeout_error(self.timeout_seconds, self.max_input_chars)) from exc

    @staticmethod
    def _persist_runtime(path: Path | None, runtime: LLMRuntimeInfo) -> None:
        if path is None:
            return
        path.write_text(json.dumps(runtime.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def build_openai_diagnostics(*, base_url: str, api_key: str, model: str, timeout_seconds: float) -> dict[str, Any]:
    parsed = urlparse(base_url)
    host = parsed.hostname or ""
    diagnostics: dict[str, Any] = {
        "provider": "openai",
        "base_url": base_url,
        "host": host,
        "model": model,
        "timeout_seconds": timeout_seconds,
        "api_key_present": bool(api_key),
        "dns": {
            "ok": False,
            "addresses": [],
            "detail": "No hostname configured.",
        },
    }
    if not host:
        diagnostics["status"] = "invalid-base-url"
        diagnostics["detail"] = "OPENAI_BASE_URL is missing a valid hostname."
        return diagnostics
    try:
        infos = socket.getaddrinfo(host, parsed.port or 443, type=socket.SOCK_STREAM)
        addresses = sorted({info[4][0] for info in infos if info[4]})
        diagnostics["dns"] = {
            "ok": True,
            "addresses": addresses,
            "detail": f"Resolved {host} successfully.",
        }
        diagnostics["status"] = "ready" if api_key else "missing-api-key"
        diagnostics["detail"] = (
            "OpenAI hostname resolves and an API key is configured."
            if api_key
            else "OpenAI hostname resolves, but OPENAI_API_KEY is not set."
        )
        return diagnostics
    except socket.gaierror as exc:
        diagnostics["status"] = "dns-failed"
        diagnostics["detail"] = _format_openai_connection_error(exc, base_url)
        diagnostics["dns"] = {
            "ok": False,
            "addresses": [],
            "detail": str(exc),
        }
        return diagnostics


def _format_openai_timeout_error(timeout_seconds: float, max_input_chars: int) -> str:
    return (
        f"OpenAI request timed out after {timeout_seconds:g}s while waiting for a response. "
        f"Try increasing OPENAI_TIMEOUT_SECONDS or lowering OPENAI_MAX_INPUT_CHARS below the current {max_input_chars}."
    )


def _format_openai_connection_error(exc: Exception, base_url: str) -> str:
    parsed = urlparse(base_url)
    host = parsed.hostname or "<missing-host>"
    reason = getattr(exc, "reason", exc)
    if isinstance(reason, socket.gaierror):
        return (
            f"Could not resolve OpenAI host '{host}' from the app container. "
            f"Check OPENAI_BASE_URL, Docker DNS, or outbound internet access. Original error: {reason}"
        )
    return f"OpenAI connection failed for host '{host}': {reason}"
