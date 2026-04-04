from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class SettingField:
    attr_name: str
    env_name: str
    label: str
    kind: str
    group: str
    description: str = ""
    secret: bool = False
    restart_required: bool = False
    options: tuple[str, ...] = ()
    placeholder: str = ""


SETTINGS_REGISTRY: tuple[SettingField, ...] = (
    SettingField("app_name", "APP_NAME", "App name", "text", "App", restart_required=True),
    SettingField("app_host", "APP_HOST", "Host", "text", "App", description="Bind address for the web server.", restart_required=True, placeholder="127.0.0.1"),
    SettingField("app_port", "APP_PORT", "Port", "int", "App", description="Port used by the web server.", restart_required=True, placeholder="8000"),
    SettingField("data_dir", "DATA_DIR", "Data directory", "path", "Storage", description="Base directory for uploads, artifacts, backups, and the database.", restart_required=True),
    SettingField("database_path", "DATABASE_PATH", "Database path", "path", "Storage", description="SQLite database path. Relative values are resolved from the data directory.", restart_required=True),
    SettingField("backup_dir", "BACKUP_DIR", "Backup directory", "path", "Storage", description="Backup output folder. Relative values are resolved from the data directory."),
    SettingField("default_language", "DEFAULT_LANGUAGE", "Default language", "text", "Processing", placeholder="tr"),
    SettingField("asr_provider", "ASR_PROVIDER", "ASR provider", "select", "Processing", options=("auto", "faster_whisper", "mock")),
    SettingField("whisper_model_size", "WHISPER_MODEL_SIZE", "Whisper model size", "text", "Processing", placeholder="small"),
    SettingField("whisper_device", "WHISPER_DEVICE", "Whisper device", "text", "Processing", placeholder="cpu"),
    SettingField("whisper_compute_type", "WHISPER_COMPUTE_TYPE", "Whisper compute type", "text", "Processing", placeholder="default"),
    SettingField("max_workers", "MAX_WORKERS", "Max workers", "int", "Processing", description="Local worker pool size.", restart_required=True, placeholder="2"),
    SettingField("poll_interval_seconds", "POLL_INTERVAL_SECONDS", "Poll interval seconds", "float", "Processing", placeholder="1.0"),
    SettingField("queue_backend", "QUEUE_BACKEND", "Queue backend", "select", "Queue", description="auto tries RQ first then falls back to local.", options=("auto", "local", "rq"), restart_required=True),
    SettingField("redis_url", "REDIS_URL", "Redis URL", "text", "Queue", restart_required=True, placeholder="redis://redis:6379/0"),
    SettingField("rq_queue_name", "RQ_QUEUE_NAME", "RQ queue name", "text", "Queue", restart_required=True, placeholder="lecture-processing"),
    SettingField("enable_pdf_export", "ENABLE_PDF_EXPORT", "Enable PDF export", "bool", "Exports"),
    SettingField("enable_docx_export", "ENABLE_DOCX_EXPORT", "Enable DOCX export", "bool", "Exports"),
    SettingField("ai_assist_mode", "AI_ASSIST_MODE", "AI assist mode", "text", "AI / LLM", placeholder="heuristic"),
    SettingField("llm_notes_enabled", "LLM_NOTES_ENABLED", "Enable OpenAI note polishing", "bool", "AI / LLM"),
    SettingField("openai_api_key", "OPENAI_API_KEY", "OpenAI API key", "text", "AI / LLM", secret=True, placeholder="sk-..."),
    SettingField("openai_model", "OPENAI_MODEL", "OpenAI model", "text", "AI / LLM", placeholder="gpt-4o-mini"),
    SettingField("openai_base_url", "OPENAI_BASE_URL", "OpenAI base URL", "text", "AI / LLM", placeholder="https://api.openai.com/v1"),
    SettingField("openai_timeout_seconds", "OPENAI_TIMEOUT_SECONDS", "OpenAI timeout seconds", "float", "AI / LLM", placeholder="120"),
    SettingField("openai_max_input_chars", "OPENAI_MAX_INPUT_CHARS", "OpenAI max input chars", "int", "AI / LLM", placeholder="45000"),
    SettingField("basic_auth_enabled", "BASIC_AUTH_ENABLED", "Enable basic auth", "bool", "Security"),
    SettingField("basic_auth_username", "BASIC_AUTH_USERNAME", "Basic auth username", "text", "Security", placeholder="admin"),
    SettingField("basic_auth_password", "BASIC_AUTH_PASSWORD", "Basic auth password", "text", "Security", secret=True, placeholder="change-me"),
    SettingField("log_level", "LOG_LEVEL", "Log level", "select", "Observability", options=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"), restart_required=True),
)

SETTINGS_BY_ATTR = {field.attr_name: field for field in SETTINGS_REGISTRY}
SETTINGS_BY_ENV = {field.env_name: field for field in SETTINGS_REGISTRY}


@dataclass(slots=True)
class Settings:
    app_name: str
    app_host: str
    app_port: int
    data_dir: Path
    database_path: Path
    uploads_dir: Path
    artifacts_dir: Path
    asr_provider: str
    whisper_model_size: str
    whisper_device: str
    whisper_compute_type: str
    default_language: str
    max_workers: int
    log_level: str
    poll_interval_seconds: float
    enable_pdf_export: bool
    enable_docx_export: bool
    ai_assist_mode: str
    queue_backend: str
    redis_url: str
    rq_queue_name: str
    basic_auth_enabled: bool
    basic_auth_username: str
    basic_auth_password: str
    backup_dir: Path
    llm_notes_enabled: bool
    openai_api_key: str
    openai_model: str
    openai_base_url: str
    openai_timeout_seconds: float
    openai_max_input_chars: int

    @classmethod
    def from_env(cls) -> "Settings":
        env_mapping: dict[str, Any] = {field.env_name: os.getenv(field.env_name) for field in SETTINGS_REGISTRY}
        return cls.from_mapping(env_mapping)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "Settings":
        data_dir = _coerce_path(_mapping_value(mapping, "DATA_DIR", "./data")).resolve()
        database_value = _mapping_value(mapping, "DATABASE_PATH", "app.db")
        backup_value = _mapping_value(mapping, "BACKUP_DIR", "backups")
        database_path = _resolve_relative_path(database_value, data_dir)
        backup_dir = _resolve_relative_path(backup_value, data_dir)
        return cls(
            app_name=_coerce_text(mapping.get("APP_NAME", "Lecture Note Builder"), "Lecture Note Builder"),
            app_host=_coerce_text(mapping.get("APP_HOST", "127.0.0.1"), "127.0.0.1"),
            app_port=int(_coerce_int(mapping.get("APP_PORT", 8000), 8000)),
            data_dir=data_dir,
            database_path=database_path,
            uploads_dir=(data_dir / "uploads").resolve(),
            artifacts_dir=(data_dir / "artifacts").resolve(),
            asr_provider=_coerce_text(mapping.get("ASR_PROVIDER", "auto"), "auto").strip().lower(),
            whisper_model_size=_coerce_text(mapping.get("WHISPER_MODEL_SIZE", "small"), "small").strip(),
            whisper_device=_coerce_text(mapping.get("WHISPER_DEVICE", "cpu"), "cpu").strip(),
            whisper_compute_type=_coerce_text(mapping.get("WHISPER_COMPUTE_TYPE", "default"), "default").strip(),
            default_language=_coerce_text(mapping.get("DEFAULT_LANGUAGE", "tr"), "tr").strip(),
            max_workers=max(1, _coerce_int(mapping.get("MAX_WORKERS", 2), 2)),
            log_level=_coerce_text(mapping.get("LOG_LEVEL", "INFO"), "INFO").strip().upper(),
            poll_interval_seconds=max(0.2, _coerce_float(mapping.get("POLL_INTERVAL_SECONDS", 1.0), 1.0)),
            enable_pdf_export=_coerce_bool(mapping.get("ENABLE_PDF_EXPORT", True), True),
            enable_docx_export=_coerce_bool(mapping.get("ENABLE_DOCX_EXPORT", True), True),
            ai_assist_mode=_coerce_text(mapping.get("AI_ASSIST_MODE", "heuristic"), "heuristic").strip().lower(),
            queue_backend=_coerce_text(mapping.get("QUEUE_BACKEND", "auto"), "auto").strip().lower(),
            redis_url=_coerce_text(mapping.get("REDIS_URL", "redis://localhost:6379/0"), "redis://localhost:6379/0").strip(),
            rq_queue_name=_coerce_text(mapping.get("RQ_QUEUE_NAME", "lecture-processing"), "lecture-processing").strip(),
            basic_auth_enabled=_coerce_bool(mapping.get("BASIC_AUTH_ENABLED", False), False),
            basic_auth_username=_coerce_text(mapping.get("BASIC_AUTH_USERNAME", "admin"), "admin").strip(),
            basic_auth_password=_coerce_text(mapping.get("BASIC_AUTH_PASSWORD", "change-me"), "change-me").strip(),
            backup_dir=backup_dir,
            llm_notes_enabled=_coerce_bool(mapping.get("LLM_NOTES_ENABLED", False), False),
            openai_api_key=_coerce_text(mapping.get("OPENAI_API_KEY", ""), "").strip(),
            openai_model=_coerce_text(mapping.get("OPENAI_MODEL", "gpt-4o-mini"), "gpt-4o-mini").strip(),
            openai_base_url=_coerce_text(mapping.get("OPENAI_BASE_URL", "https://api.openai.com/v1"), "https://api.openai.com/v1").strip().rstrip("/"),
            openai_timeout_seconds=max(5.0, _coerce_float(mapping.get("OPENAI_TIMEOUT_SECONDS", 120.0), 120.0)),
            openai_max_input_chars=max(4000, _coerce_int(mapping.get("OPENAI_MAX_INPUT_CHARS", 45000), 45000)),
        )

    def to_env_mapping(self) -> dict[str, Any]:
        return {
            "APP_NAME": self.app_name,
            "APP_HOST": self.app_host,
            "APP_PORT": self.app_port,
            "DATA_DIR": str(self.data_dir),
            "DATABASE_PATH": str(self.database_path),
            "ASR_PROVIDER": self.asr_provider,
            "WHISPER_MODEL_SIZE": self.whisper_model_size,
            "WHISPER_DEVICE": self.whisper_device,
            "WHISPER_COMPUTE_TYPE": self.whisper_compute_type,
            "DEFAULT_LANGUAGE": self.default_language,
            "MAX_WORKERS": self.max_workers,
            "POLL_INTERVAL_SECONDS": self.poll_interval_seconds,
            "ENABLE_PDF_EXPORT": self.enable_pdf_export,
            "ENABLE_DOCX_EXPORT": self.enable_docx_export,
            "LOG_LEVEL": self.log_level,
            "QUEUE_BACKEND": self.queue_backend,
            "REDIS_URL": self.redis_url,
            "RQ_QUEUE_NAME": self.rq_queue_name,
            "AI_ASSIST_MODE": self.ai_assist_mode,
            "BASIC_AUTH_ENABLED": self.basic_auth_enabled,
            "BASIC_AUTH_USERNAME": self.basic_auth_username,
            "BASIC_AUTH_PASSWORD": self.basic_auth_password,
            "BACKUP_DIR": str(self.backup_dir),
            "LLM_NOTES_ENABLED": self.llm_notes_enabled,
            "OPENAI_API_KEY": self.openai_api_key,
            "OPENAI_MODEL": self.openai_model,
            "OPENAI_BASE_URL": self.openai_base_url,
            "OPENAI_TIMEOUT_SECONDS": self.openai_timeout_seconds,
            "OPENAI_MAX_INPUT_CHARS": self.openai_max_input_chars,
        }

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)


class RuntimeSettings:
    def __init__(self, base_settings: Settings) -> None:
        self._base_settings = base_settings
        self._overrides_path = (base_settings.data_dir / "ui_settings.json").resolve()
        self._base_settings.ensure_directories()

    @property
    def overrides_path(self) -> Path:
        return self._overrides_path

    def snapshot(self) -> Settings:
        merged = {**self._base_settings.to_env_mapping(), **self._load_overrides()}
        runtime = Settings.from_mapping(merged)
        runtime.ensure_directories()
        return runtime

    def __getattr__(self, item: str) -> Any:
        return getattr(self.snapshot(), item)

    def ensure_directories(self) -> None:
        self.snapshot().ensure_directories()

    def effective_values(self, *, include_secrets: bool = False) -> dict[str, Any]:
        values = self.snapshot().to_env_mapping()
        if include_secrets:
            return values
        redacted: dict[str, Any] = {}
        for field in SETTINGS_REGISTRY:
            value = values.get(field.env_name)
            if field.secret:
                redacted[field.env_name] = "********" if str(value or "").strip() else ""
            else:
                redacted[field.env_name] = value
        return redacted

    def grouped_entries(self) -> list[dict[str, Any]]:
        current = self.snapshot()
        overrides = self._load_overrides()
        groups: dict[str, list[dict[str, Any]]] = {}
        for field in SETTINGS_REGISTRY:
            value = getattr(current, field.attr_name)
            groups.setdefault(field.group, []).append(
                {
                    "attr_name": field.attr_name,
                    "env_name": field.env_name,
                    "label": field.label,
                    "kind": field.kind,
                    "description": field.description,
                    "secret": field.secret,
                    "restart_required": field.restart_required,
                    "options": field.options,
                    "placeholder": field.placeholder,
                    "is_overridden": field.env_name in overrides,
                    "current_value": _display_value(value, field),
                    "raw_value": "" if field.secret else _raw_form_value(value, field),
                    "is_set": bool(str(value).strip()) if field.secret else True,
                }
            )
        return [{"name": group_name, "entries": entries} for group_name, entries in groups.items()]

    def update_from_form(self, form_data: Mapping[str, Any]) -> tuple[bool, list[str]]:
        overrides = self._load_overrides()
        current = self.snapshot()
        errors: list[str] = []
        for field in SETTINGS_REGISTRY:
            clear_key = f"clear__{field.attr_name}"
            if field.secret and _coerce_bool(form_data.get(clear_key, False), False):
                overrides.pop(field.env_name, None)
                continue

            if field.kind == "bool":
                raw_value = form_data.get(field.attr_name, "false")
            else:
                raw_value = form_data.get(field.attr_name)
                if raw_value is None:
                    continue
                raw_value = str(raw_value).strip()
                if field.secret and raw_value == "":
                    continue

            try:
                coerced = _coerce_form_value(field, raw_value)
            except ValueError as exc:
                errors.append(f"{field.label}: {exc}")
                continue

            current_value = getattr(current, field.attr_name)
            if _equivalent_setting_value(current_value, coerced):
                if field.env_name not in overrides:
                    continue
                if _equivalent_setting_value(getattr(self._base_settings, field.attr_name), coerced):
                    overrides.pop(field.env_name, None)
                else:
                    overrides[field.env_name] = _serialize_override(coerced, field)
                continue

            if _equivalent_setting_value(getattr(self._base_settings, field.attr_name), coerced):
                overrides.pop(field.env_name, None)
            else:
                overrides[field.env_name] = _serialize_override(coerced, field)

        if errors:
            return False, errors
        self._save_overrides(overrides)
        self.snapshot().ensure_directories()
        return True, []

    def clear_overrides(self) -> None:
        if self._overrides_path.exists():
            self._overrides_path.unlink()

    def export_env_text(self) -> str:
        values = self.snapshot().to_env_mapping()
        lines = [f"{field.env_name}={_to_env_string(values[field.env_name], field)}" for field in SETTINGS_REGISTRY]
        return "\n".join(lines) + "\n"

    def _load_overrides(self) -> dict[str, Any]:
        if not self._overrides_path.exists():
            return {}
        try:
            payload = json.loads(self._overrides_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        if not isinstance(payload, dict):
            return {}
        return {str(key): value for key, value in payload.items() if str(key) in SETTINGS_BY_ENV}

    def _save_overrides(self, overrides: dict[str, Any]) -> None:
        self._base_settings.data_dir.mkdir(parents=True, exist_ok=True)
        if not overrides:
            self.clear_overrides()
            return
        with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=self._base_settings.data_dir) as handle:
            json.dump(overrides, handle, ensure_ascii=False, indent=2)
            temp_path = Path(handle.name)
        temp_path.replace(self._overrides_path)



def _mapping_value(mapping: Mapping[str, Any], key: str, default: Any) -> Any:
    value = mapping.get(key, default)
    return default if value in {None, ""} else value



def _coerce_text(value: Any, default: str) -> str:
    if value is None:
        return default
    text = str(value)
    return text if text != "" else default



def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")



def _coerce_int(value: Any, default: int) -> int:
    if value in {None, ""}:
        return default
    return int(str(value).strip())



def _coerce_float(value: Any, default: float) -> float:
    if value in {None, ""}:
        return default
    return float(str(value).strip())



def _coerce_path(value: Any) -> Path:
    return Path(str(value or "./data").strip()).expanduser()



def _resolve_relative_path(value: Any, base_dir: Path) -> Path:
    path = _coerce_path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return path



def _coerce_form_value(field: SettingField, raw_value: Any) -> Any:
    if field.kind in {"text", "select"}:
        return str(raw_value).strip()
    if field.kind == "int":
        return int(str(raw_value).strip())
    if field.kind == "float":
        return float(str(raw_value).strip())
    if field.kind == "bool":
        return _coerce_bool(raw_value, False)
    if field.kind == "path":
        return str(_coerce_path(raw_value))
    raise ValueError(f"Unsupported setting kind: {field.kind}")



def _serialize_override(value: Any, field: SettingField) -> Any:
    if field.kind == "path":
        return str(value)
    return value



def _equivalent_setting_value(left: Any, right: Any) -> bool:
    if isinstance(left, Path):
        left = str(left)
    if isinstance(right, Path):
        right = str(right)
    return left == right



def _display_value(value: Any, field: SettingField) -> str:
    if field.secret:
        return "Configured" if str(value).strip() else "Not set"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)



def _raw_form_value(value: Any, field: SettingField) -> str:
    if field.kind == "bool":
        return "true" if bool(value) else "false"
    return str(value)



def _to_env_string(value: Any, field: SettingField) -> str:
    if field.kind == "bool":
        return "true" if bool(value) else "false"
    return str(value)


_base_settings = Settings.from_env()
settings = RuntimeSettings(_base_settings)
settings.ensure_directories()
