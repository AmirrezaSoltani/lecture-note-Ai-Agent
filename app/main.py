from __future__ import annotations

import json
import shutil
import uuid
from threading import Lock, Thread
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import SETTINGS_REGISTRY, settings
from app.db import initialize_database
from app.logging_utils import configure_logging
from app.models import LectureRecord, ReviewState
from app.services.openai_notes import LLMRuntimeInfo, OpenAINotesPolisher
from app.repository import ArtifactRepository, JobRepository, LectureRepository, PerformanceRunRepository
from app.services.auth import enforce_basic_auth
from app.services.monitoring import build_operational_snapshot, render_prometheus_text, runtime_metrics
from app.services.pipeline import LectureProcessingPipeline
from app.services.review import ReviewService
from app.services.storage import ArtifactStorage
from app.worker import build_job_dispatcher

configure_logging()
initialize_database()

repository = LectureRepository()
job_repository = JobRepository()
artifact_repository = ArtifactRepository()
performance_repository = PerformanceRunRepository()
review_service = ReviewService()
pipeline = LectureProcessingPipeline()
worker = build_job_dispatcher()
_manual_llm_lock = Lock()
_manual_llm_active: set[str] = set()


@asynccontextmanager
async def lifespan(_: FastAPI):
    worker.recover_pending_jobs()
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "web" / "templates"))
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "web" / "static")), name="static")


@app.middleware("http")
async def metrics_and_auth_middleware(request: Request, call_next):
    import time

    started = time.perf_counter()
    try:
        enforce_basic_auth(request)
        response = await call_next(request)
    except HTTPException as exc:
        response = JSONResponse(
            {"detail": exc.detail},
            status_code=exc.status_code,
            headers=exc.headers,
        )
    duration_ms = (time.perf_counter() - started) * 1000
    runtime_metrics.record_request(request.method, request.url.path, response.status_code, duration_ms)
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.3f}"
    return response


@app.get("/healthz")
def healthcheck() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "app": settings.app_name,
            "asr_provider": settings.asr_provider,
            "default_language": settings.default_language,
            "queue_depth": worker.queue_depth(),
            "queue_backend_requested": worker.requested_backend,
            "queue_backend_resolved": worker.backend_name,
            "active_workers": worker.active_count(),
            "max_workers": settings.max_workers,
            "pdf_export": settings.enable_pdf_export,
            "docx_export": settings.enable_docx_export,
            "ai_assist_mode": settings.ai_assist_mode,
        }
    )




@app.get("/readyz")
def readiness() -> JSONResponse:
    db_exists = settings.database_path.exists()
    uploads_exists = settings.uploads_dir.exists()
    artifacts_exists = settings.artifacts_dir.exists()
    return JSONResponse(
        {
            "status": "ready" if db_exists and uploads_exists and artifacts_exists else "not-ready",
            "database_path": str(settings.database_path),
            "database_exists": db_exists,
            "uploads_exists": uploads_exists,
            "artifacts_exists": artifacts_exists,
        }
    )


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    snapshot = build_operational_snapshot()
    snapshot["queue_depth"] = worker.queue_depth()
    snapshot["active_workers"] = worker.active_count()
    return PlainTextResponse(render_prometheus_text(snapshot), media_type="text/plain; version=0.0.4")


@app.post("/api/admin/backup")
def create_backup() -> JSONResponse:
    from scripts.backup import create_backup as run_backup

    backup_path = run_backup()
    return JSONResponse({"status": "ok", "backup_path": str(backup_path)})


@app.get("/settings", response_class=HTMLResponse)
def settings_page(
    request: Request,
    saved: str = Query(default=""),
    error: str = Query(default=""),
    detail: str = Query(default=""),
) -> HTMLResponse:
    save_result = "saved" if saved == "1" else "error" if error == "1" else ""
    return templates.TemplateResponse(
        request,
        "settings.html",
        {
            "request": request,
            "settings_groups": settings.grouped_entries(),
            "overrides_path": str(settings.overrides_path),
            "save_result": save_result,
            "error_detail": detail,
            "restart_required_count": sum(1 for field in SETTINGS_REGISTRY if field.restart_required),
        },
    )


@app.post("/settings")
async def save_settings(request: Request) -> RedirectResponse:
    form = await request.form()
    success, errors = settings.update_from_form(form)
    if success:
        return RedirectResponse(url="/settings?saved=1", status_code=303)
    encoded = urlencode({"error": 1, "detail": " | ".join(errors[:6])})
    return RedirectResponse(url=f"/settings?{encoded}", status_code=303)


@app.post("/settings/reset")
def reset_settings() -> RedirectResponse:
    settings.clear_overrides()
    return RedirectResponse(url="/settings?saved=1", status_code=303)


@app.get("/settings/export.env")
def export_settings_env() -> Response:
    return Response(
        content=settings.export_env_text(),
        media_type="text/plain",
        headers={"Content-Disposition": 'attachment; filename="lecture-note-builder.env"'},
    )




@app.post("/api/settings/test-openai")
def api_test_openai_settings() -> JSONResponse:
    polisher = OpenAINotesPolisher(force_enabled=True)
    diagnostics = polisher.diagnostics()
    status_code = 200 if diagnostics.get("status") in {"ready", "missing-api-key"} else 503
    return JSONResponse(diagnostics, status_code=status_code)

@app.get("/api/settings")
def api_settings() -> JSONResponse:
    return JSONResponse(
        {
            "effective": settings.effective_values(),
            "overrides_path": str(settings.overrides_path),
            "groups": settings.grouped_entries(),
        }
    )


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    lectures = repository.list_all()
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "request": request,
            "lectures": lectures,
            "default_language": settings.default_language,
            "queue_backend": worker.backend_name,
        },
    )


@app.post("/lectures")
async def create_lecture(
    title: str = Form(...),
    language: str = Form(...),
    audio_file: UploadFile = File(...),
) -> RedirectResponse:
    lecture_id = str(uuid.uuid4())
    sanitized_name = Path(audio_file.filename or "lecture_audio.bin").name
    upload_dir = settings.uploads_dir / lecture_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_audio_path = upload_dir / sanitized_name

    with saved_audio_path.open("wb") as destination:
        shutil.copyfileobj(audio_file.file, destination)

    artifacts = ArtifactStorage(lecture_id=lecture_id, working_dir=settings.artifacts_dir / lecture_id).build_paths()
    record = LectureRecord(
        lecture_id=lecture_id,
        title=title.strip() or "Untitled Lecture",
        audio_filename=sanitized_name,
        audio_path=str(saved_audio_path),
        language=(language or settings.default_language).strip(),
        notes_review_path=str(artifacts["review_json"]),
    )
    repository.create(record)
    review_service.initialize_overlay(artifacts["review_json"])
    return RedirectResponse(url=f"/lectures/{lecture_id}", status_code=303)


@app.post("/lectures/{lecture_id}/process")
def process_lecture(lecture_id: str) -> RedirectResponse:
    record = repository.get(lecture_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Lecture not found")
    worker.submit(lecture_id)
    return RedirectResponse(url=f"/lectures/{lecture_id}", status_code=303)


@app.post("/lectures/{lecture_id}/review/block/{block_id}")
def update_block_review(
    lecture_id: str,
    block_id: str,
    title: str = Form(...),
    content: str = Form(...),
) -> RedirectResponse:
    record = _get_lecture_or_404(lecture_id)
    review_path = _review_path(record)
    review_service.update_block(review_path, block_id=block_id, title=title, content=content)
    _regenerate_reviewed_exports(record)
    return RedirectResponse(url=f"/lectures/{lecture_id}#block-{block_id}", status_code=303)


@app.post("/lectures/{lecture_id}/review/marker/{marker_id}")
def update_marker_review(
    lecture_id: str,
    marker_id: str,
    review_state: str = Form(...),
    reviewer_note: str = Form(""),
    q: str = Form(""),
    marker_type: str = Form(""),
    current_state: str = Form(""),
    transcript_q: str = Form(""),
) -> RedirectResponse:
    record = _get_lecture_or_404(lecture_id)
    if review_state not in {state.value for state in ReviewState}:
        raise HTTPException(status_code=400, detail="Invalid review state")
    review_service.review_marker(_review_path(record), marker_id=marker_id, review_state=review_state, reviewer_note=reviewer_note)
    _regenerate_reviewed_exports(record)
    return RedirectResponse(
        url=_lecture_detail_url(
            lecture_id,
            marker_type=marker_type,
            review_state=current_state,
            q=q,
            transcript_q=transcript_q,
            anchor=f"marker-{marker_id}",
        ),
        status_code=303,
    )


@app.post("/lectures/{lecture_id}/review/markers/bulk")
def bulk_update_marker_review(
    lecture_id: str,
    review_state: str = Form(...),
    marker_ids: str = Form(""),
    marker_type: str = Form(""),
    current_state: str = Form(""),
    text_query: str = Form(""),
) -> RedirectResponse:
    record = _get_lecture_or_404(lecture_id)
    if review_state not in {state.value for state in ReviewState}:
        raise HTTPException(status_code=400, detail="Invalid review state")
    bundle = _load_reviewed_bundle(record)
    if bundle is None:
        raise HTTPException(status_code=400, detail="Lecture has not been processed yet")
    selected_ids = [item.strip() for item in marker_ids.split(",") if item.strip()]
    changed = review_service.bulk_review_markers(
        _review_path(record),
        bundle=bundle,
        review_state=review_state,
        marker_ids=selected_ids,
        marker_type=marker_type.strip() or None,
        current_state=current_state.strip() or None,
        text_query=text_query.strip() or None,
    )
    if changed:
        _regenerate_reviewed_exports(record)
    return RedirectResponse(
        url=_lecture_detail_url(
            lecture_id,
            marker_type=marker_type,
            review_state=current_state,
            q=text_query,
            transcript_q=text_query,
            anchor="marker-preview",
        ),
        status_code=303,
    )


@app.get("/lectures/{lecture_id}/review/marker/{marker_id}/{target_state}")
def quick_marker_review(
    lecture_id: str,
    marker_id: str,
    target_state: str,
    q: str = Query(default=""),
    marker_type: str = Query(default=""),
    review_state: str = Query(default=""),
    transcript_q: str = Query(default=""),
) -> RedirectResponse:
    record = _get_lecture_or_404(lecture_id)
    if target_state not in {state.value for state in ReviewState}:
        raise HTTPException(status_code=400, detail="Invalid review state")
    review_service.review_marker(_review_path(record), marker_id=marker_id, review_state=target_state, reviewer_note="")
    _regenerate_reviewed_exports(record)
    return RedirectResponse(
        url=_lecture_detail_url(
            lecture_id,
            marker_type=marker_type,
            review_state=review_state,
            q=q,
            transcript_q=transcript_q,
            anchor=f"marker-{marker_id}",
        ),
        status_code=303,
    )


@app.post("/lectures/{lecture_id}/review/regenerate-exports")
def regenerate_exports(lecture_id: str) -> RedirectResponse:
    record = _get_lecture_or_404(lecture_id)
    _regenerate_reviewed_exports(record)
    return RedirectResponse(url=f"/lectures/{lecture_id}", status_code=303)


@app.post("/api/lectures/{lecture_id}/llm-assistant/run")
def trigger_manual_llm_assistant(lecture_id: str) -> JSONResponse:
    record = _get_lecture_or_404(lecture_id)
    bundle = _load_reviewed_bundle(record)
    if bundle is None:
        raise HTTPException(status_code=400, detail="Lecture has not been processed yet")

    with _manual_llm_lock:
        already_running = lecture_id in _manual_llm_active
        if not already_running:
            _manual_llm_active.add(lecture_id)
            _persist_llm_runtime(record, LLMRuntimeInfo(
                enabled=True,
                applied=False,
                provider="openai",
                model=settings.openai_model,
                chunk_count=0,
                total_ms=0.0,
                status="running",
                detail="Manual LLM assistant requested from the UI.",
            ))
            Thread(target=_run_manual_llm_assistant, args=(lecture_id,), daemon=True).start()

    runtime = _resolve_llm_runtime(record, performance_repository.latest_for_lecture(lecture_id))
    return JSONResponse({
        "lecture_id": lecture_id,
        "started": not already_running,
        "status": "running" if not already_running else "already-running",
        "llm_runtime": runtime,
    }, status_code=202 if not already_running else 200)


@app.get("/api/lectures/{lecture_id}/status")
def lecture_status(lecture_id: str) -> JSONResponse:
    record = _get_lecture_or_404(lecture_id)
    latest_job = job_repository.latest_for_lecture(lecture_id)
    artifacts = artifact_repository.list_for_lecture(lecture_id)
    performance_run = performance_repository.latest_for_lecture(lecture_id)
    bundle = _load_reviewed_bundle(record)
    asr_runtime = _resolve_asr_runtime(record, performance_run)
    llm_runtime = _resolve_llm_runtime(record, performance_run)
    payload = record.to_dict()
    payload["latest_job"] = latest_job.to_dict() if latest_job is not None else None
    payload["artifacts"] = [artifact.to_dict() for artifact in artifacts]
    payload["review_summary"] = review_service.review_summary(bundle) if bundle else None
    payload["queue_backend"] = worker.backend_name
    payload["active_workers"] = worker.active_count()
    payload["performance"] = performance_run.to_dict() if performance_run is not None else None
    payload["asr_runtime"] = asr_runtime
    payload["llm_runtime"] = llm_runtime
    payload["note_block_count"] = len(bundle.note_blocks) if bundle else 0
    payload["recomposition_mode"] = "approved-first" if bundle else None
    payload["search_capabilities"] = {
        "marker_type_filter": True,
        "review_state_filter": True,
        "text_query": True,
        "transcript_query": True,
        "bulk_review": True,
    }
    return JSONResponse(payload)


@app.get("/api/lectures/{lecture_id}/performance")
def lecture_performance(lecture_id: str) -> JSONResponse:
    _get_lecture_or_404(lecture_id)
    runs = performance_repository.list_for_lecture(lecture_id, limit=20)
    latest = runs[0].to_dict() if runs else None
    return JSONResponse({
        "lecture_id": lecture_id,
        "latest": latest,
        "runs": [run.to_dict() for run in runs],
        "active_workers": worker.active_count(),
        "queue_depth": worker.queue_depth(),
    })

@app.get("/api/lectures/{lecture_id}/live")
def lecture_live_snapshot(lecture_id: str) -> JSONResponse:
    record = _get_lecture_or_404(lecture_id)
    latest_job = job_repository.latest_for_lecture(lecture_id)
    artifacts = artifact_repository.list_for_lecture(lecture_id)
    performance_run = performance_repository.latest_for_lecture(lecture_id)
    bundle = _load_reviewed_bundle(record)
    review_summary = review_service.review_summary(bundle) if bundle else None
    asr_runtime = _resolve_asr_runtime(record, performance_run)
    llm_runtime = _resolve_llm_runtime(record, performance_run)
    payload = {
        "lecture": record.to_dict(),
        "latest_job": latest_job.to_dict() if latest_job is not None else None,
        "performance": performance_run.to_dict() if performance_run is not None else None,
        "review_summary": review_summary,
        "artifact_count": len(artifacts),
        "artifact_types": [artifact.artifact_type for artifact in artifacts],
        "active_workers": worker.active_count(),
        "queue_depth": worker.queue_depth(),
        "note_block_count": len(bundle.note_blocks) if bundle else 0,
        "marker_count": len(bundle.markers) if bundle else 0,
        "segment_count": len(bundle.segments) if bundle else 0,
        "timeline": _build_live_timeline(record, latest_job, performance_run),
        "asr_runtime": asr_runtime,
        "llm_runtime": llm_runtime,
        "server_time": datetime.now(timezone.utc).isoformat(),
    }
    return JSONResponse(payload)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> JSONResponse:
    job = job_repository.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(job.to_dict())


@app.get("/lectures/{lecture_id}", response_class=HTMLResponse)
def lecture_detail(
    request: Request,
    lecture_id: str,
    q: str = Query(default=""),
    marker_type: str = Query(default=""),
    review_state: str = Query(default=""),
    transcript_q: str = Query(default=""),
) -> HTMLResponse:
    record = _get_lecture_or_404(lecture_id)
    bundle = _load_reviewed_bundle(record)
    note_payload = bundle.to_dict() if bundle else None
    markers = note_payload.get("markers", []) if note_payload else []
    transcript_payload = note_payload.get("segments", []) if note_payload else []
    filtered_markers = _filter_markers(markers, q=q, marker_type=marker_type, review_state=review_state)
    grouped_markers = _group_markers(filtered_markers) if note_payload else {}
    markers_by_segment = _markers_by_segment(filtered_markers)
    filtered_transcript = _filter_transcript(
        transcript_payload,
        transcript_q=transcript_q,
        related_segment_indexes={item["source_segment_index"] for item in filtered_markers},
    )
    review_summary = review_service.review_summary(bundle) if bundle else None

    jobs = job_repository.list_for_lecture(lecture_id)
    artifacts = artifact_repository.list_for_lecture(lecture_id)
    performance_run = performance_repository.latest_for_lecture(lecture_id)
    asr_runtime = _resolve_asr_runtime(record, performance_run)
    llm_runtime = _resolve_llm_runtime(record, performance_run)
    filters = {
        "q": q,
        "marker_type": marker_type,
        "review_state": review_state,
        "transcript_q": transcript_q,
    }
    filter_summary = {
        "total_markers": len(markers),
        "visible_markers": len(filtered_markers),
        "total_segments": len(transcript_payload),
        "visible_segments": len(filtered_transcript),
        "related_segment_indexes": sorted({item["source_segment_index"] for item in filtered_markers}),
    }
    marker_ids_csv = ",".join(item["marker_id"] for item in filtered_markers)
    return templates.TemplateResponse(
        request,
        "lecture_detail.html",
        {
            "request": request,
            "lecture": record,
            "note_payload": note_payload,
            "transcript_payload": filtered_transcript,
            "grouped_markers": grouped_markers,
            "jobs": jobs,
            "artifacts": artifacts,
            "performance_run": performance_run,
            "queue_backend": worker.backend_name,
            "active_workers": worker.active_count(),
            "review_summary": review_summary,
            "asr_runtime": asr_runtime,
            "llm_runtime": llm_runtime,
            "filters": filters,
            "filter_summary": filter_summary,
            "marker_type_options": _marker_type_options(markers),
            "review_state_options": [state.value for state in ReviewState],
            "marker_ids_csv": marker_ids_csv,
            "markers_by_segment": markers_by_segment,
            "selected_segment_indexes": sorted({item["source_segment_index"] for item in filtered_markers}),
        },
    )


@app.get("/lectures/{lecture_id}/download/{artifact_name}")
def download_artifact(lecture_id: str, artifact_name: str) -> FileResponse:
    record = _get_lecture_or_404(lecture_id)
    mapping = {
        "transcript": record.transcript_path,
        "notes-json": record.notes_json_path,
        "notes-review": record.notes_review_path,
        "notes-md": record.notes_md_path,
        "notes-html": record.notes_html_path,
        "notes-pdf": record.notes_pdf_path,
        "notes-docx": record.notes_docx_path,
    }
    target = mapping.get(artifact_name)
    if not target:
        artifact = artifact_repository.find_by_type(lecture_id, artifact_name)
        target = artifact.path if artifact else None
    if not target:
        raise HTTPException(status_code=404, detail="Artifact not found")
    path = Path(target)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact file missing")
    return FileResponse(path=path, filename=path.name)


def _get_lecture_or_404(lecture_id: str) -> LectureRecord:
    record = repository.get(lecture_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Lecture not found")
    return record


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None



def _resolve_asr_runtime(record: LectureRecord, performance_run) -> dict[str, str] | None:
    if performance_run is not None:
        return {
            "provider": getattr(performance_run, "asr_provider", ""),
            "model_size": getattr(performance_run, "asr_model_size", ""),
            "requested_device": getattr(performance_run, "requested_device", ""),
            "requested_compute_type": getattr(performance_run, "requested_compute_type", ""),
            "actual_device": getattr(performance_run, "actual_device", ""),
            "actual_compute_type": getattr(performance_run, "actual_compute_type", ""),
            "model_backend": getattr(performance_run, "model_backend", ""),
            "detected_language": getattr(performance_run, "detected_language", ""),
        }
    review_path = Path(record.notes_review_path).parent / "runtime_info.json" if record.notes_review_path else None
    if review_path and review_path.exists():
        try:
            import json
            payload = json.loads(review_path.read_text(encoding="utf-8"))
            return {
                "provider": payload.get("provider_name", ""),
                "model_size": payload.get("model_size", ""),
                "requested_device": payload.get("requested_device", ""),
                "requested_compute_type": payload.get("requested_compute_type", ""),
                "actual_device": payload.get("actual_device", ""),
                "actual_compute_type": payload.get("actual_compute_type", ""),
                "model_backend": payload.get("model_backend", ""),
                "detected_language": payload.get("detected_language", ""),
            }
        except Exception:
            return None
    return None




def _resolve_llm_runtime(record: LectureRecord, performance_run) -> dict[str, Any] | None:
    runtime_path = _llm_runtime_path(record)
    if runtime_path.exists():
        try:
            payload = json.loads(runtime_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
    if performance_run is not None:
        return {
            "enabled": bool(getattr(performance_run, "llm_enabled", False)),
            "applied": bool(getattr(performance_run, "llm_enabled", False)),
            "provider": "openai",
            "model": getattr(performance_run, "llm_model", ""),
            "chunk_count": int(getattr(performance_run, "llm_chunk_count", 0) or 0),
            "total_ms": float(getattr(performance_run, "llm_total_ms", 0.0) or 0.0),
            "status": "applied" if bool(getattr(performance_run, "llm_enabled", False)) else "disabled",
        }
    return None

def _llm_runtime_path(record: LectureRecord) -> Path:
    if record.notes_review_path:
        return Path(record.notes_review_path).parent / "llm_runtime.json"
    return settings.artifacts_dir / record.lecture_id / "llm_runtime.json"


def _persist_llm_runtime(record: LectureRecord, runtime: LLMRuntimeInfo | dict[str, Any]) -> None:
    runtime_path = _llm_runtime_path(record)
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    payload = runtime.to_dict() if isinstance(runtime, LLMRuntimeInfo) else runtime
    runtime_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_manual_llm_assistant(lecture_id: str) -> None:
    try:
        record = repository.get(lecture_id)
        if record is None:
            return
        bundle = _load_reviewed_bundle(record)
        if bundle is None:
            _persist_llm_runtime(record, {
                "enabled": True,
                "applied": False,
                "provider": "openai",
                "model": settings.openai_model,
                "chunk_count": 0,
                "total_ms": 0.0,
                "status": "failed",
                "detail": "Lecture has not been processed yet.",
            })
            return
        artifacts = ArtifactStorage(lecture_id=record.lecture_id, working_dir=settings.artifacts_dir / record.lecture_id).build_paths()
        artifact_records = pipeline.regenerate_reviewed_exports(bundle, artifacts, force_llm=True)
        artifact_repository.replace_for_lecture(record.lecture_id, artifact_records)
        repository.update_artifacts(record.lecture_id, {name: str(path) for name, path in artifacts.items()})
    except Exception as exc:
        record = repository.get(lecture_id)
        if record is not None:
            _persist_llm_runtime(record, {
                "enabled": True,
                "applied": False,
                "provider": "openai",
                "model": settings.openai_model,
                "chunk_count": 0,
                "total_ms": 0.0,
                "status": "failed",
                "detail": str(exc),
            })
    finally:
        with _manual_llm_lock:
            _manual_llm_active.discard(lecture_id)


def _build_live_timeline(
    record: LectureRecord,
    latest_job,
    performance_run,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    started_at = _parse_iso_datetime(getattr(latest_job, "started_at", None))
    queued_at = _parse_iso_datetime(getattr(latest_job, "queued_at", None))
    finished_at = _parse_iso_datetime(getattr(latest_job, "finished_at", None))

    elapsed_ms = None
    queue_wait_ms = None
    completed_ms = None
    if started_at is not None:
        if finished_at is not None:
            elapsed_ms = max((finished_at - started_at).total_seconds() * 1000, 0.0)
        else:
            elapsed_ms = max((now - started_at).total_seconds() * 1000, 0.0)
    if queued_at is not None and started_at is not None:
        queue_wait_ms = max((started_at - queued_at).total_seconds() * 1000, 0.0)
    if queued_at is not None and finished_at is not None:
        completed_ms = max((finished_at - queued_at).total_seconds() * 1000, 0.0)

    performance_total_ms = getattr(performance_run, "total_ms", 0.0) if performance_run is not None else 0.0
    progress_pct = 0
    stage = "idle"
    if record.status == "uploaded":
        progress_pct = 5
        stage = "uploaded"
    elif record.status == "processing":
        progress_pct = 55
        stage = "processing"
        if elapsed_ms and elapsed_ms > 60_000:
            progress_pct = 72
    elif record.status == "completed":
        progress_pct = 100
        stage = "completed"
    elif record.status == "failed":
        progress_pct = 100
        stage = "failed"
    if performance_total_ms:
        progress_pct = 100 if record.status == "completed" else max(progress_pct, 85)
        if getattr(performance_run, "transcribe_ms", 0.0) > 0:
            stage = "transcribe-finished"

    return {
        "stage": stage,
        "progress_pct": progress_pct,
        "elapsed_ms": round(elapsed_ms, 3) if elapsed_ms is not None else None,
        "queue_wait_ms": round(queue_wait_ms, 3) if queue_wait_ms is not None else None,
        "completed_ms": round(completed_ms, 3) if completed_ms is not None else None,
        "started_at": getattr(latest_job, "started_at", None) if latest_job is not None else None,
        "queued_at": getattr(latest_job, "queued_at", None) if latest_job is not None else None,
        "finished_at": getattr(latest_job, "finished_at", None) if latest_job is not None else None,
    }


def _load_reviewed_bundle(record: LectureRecord):
    source_path = _notes_source_path(record)
    if source_path is None and not record.notes_json_path:
        return None
    base_path = source_path or Path(record.notes_json_path)
    if not base_path.exists():
        return None
    base_bundle = _bundle_from_path(base_path)
    review_path = _review_path(record)
    overlay = review_service.load_overlay(review_path)
    return review_service.apply_overlay(base_bundle, overlay)




def _notes_source_path(record: LectureRecord) -> Path | None:
    if record.notes_review_path:
        candidate = Path(record.notes_review_path).parent / "notes_source.json"
        if candidate.exists():
            return candidate
    return None

def _review_path(record: LectureRecord) -> Path:
    if record.notes_review_path:
        return Path(record.notes_review_path)
    return ArtifactStorage(lecture_id=record.lecture_id, working_dir=settings.artifacts_dir / record.lecture_id).build_paths()["review_json"]


def _bundle_from_path(path: Path):
    from app.models import LectureNoteBundle, Marker, NoteBlock, TranscriptSegment

    payload = json.loads(path.read_text(encoding="utf-8"))
    return LectureNoteBundle(
        lecture_id=payload["lecture_id"],
        lecture_title=payload["lecture_title"],
        created_at=payload["created_at"],
        segments=[TranscriptSegment(**segment) for segment in payload.get("segments", [])],
        markers=[Marker(**marker) for marker in payload.get("markers", [])],
        note_blocks=[NoteBlock(**block) for block in payload.get("note_blocks", [])],
    )


def _regenerate_reviewed_exports(record: LectureRecord) -> None:
    note_path = _notes_source_path(record) or Path(record.notes_json_path or "")
    if not note_path.exists():
        return
    base_bundle = _bundle_from_path(note_path)
    review_path = _review_path(record)
    overlay = review_service.load_overlay(review_path)
    reviewed_bundle = review_service.apply_overlay(base_bundle, overlay)
    artifacts = ArtifactStorage(lecture_id=record.lecture_id, working_dir=settings.artifacts_dir / record.lecture_id).build_paths()
    artifact_records = pipeline.regenerate_reviewed_exports(reviewed_bundle, artifacts)
    artifact_repository.replace_for_lecture(record.lecture_id, artifact_records)
    repository.update_artifacts(record.lecture_id, {name: str(path) for name, path in artifacts.items()})


def _group_markers(markers: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for marker in markers:
        grouped.setdefault(marker["marker_type"], []).append(marker)
    return grouped


def _filter_markers(
    markers: list[dict[str, Any]],
    q: str,
    marker_type: str,
    review_state: str,
) -> list[dict[str, Any]]:
    q_norm = q.strip().lower()
    type_norm = marker_type.strip().lower()
    state_norm = review_state.strip().lower()
    filtered: list[dict[str, Any]] = []
    for marker in markers:
        if type_norm and str(marker.get("marker_type", "")).lower() != type_norm:
            continue
        if state_norm and str(marker.get("review_state", "")).lower() != state_norm:
            continue
        if q_norm:
            haystack = " ".join(
                [
                    str(marker.get("text", "")),
                    str(marker.get("label", "")),
                    str(marker.get("matched_phrase", "")),
                    str(marker.get("reviewer_note", "")),
                ]
            ).lower()
            if q_norm not in haystack:
                continue
        filtered.append(marker)
    return filtered


def _filter_transcript(
    transcript_payload: list[dict[str, Any]],
    transcript_q: str,
    related_segment_indexes: set[int],
) -> list[dict[str, Any]]:
    transcript_norm = transcript_q.strip().lower()
    if not transcript_norm and not related_segment_indexes:
        return transcript_payload
    filtered: list[dict[str, Any]] = []
    for segment in transcript_payload:
        text = str(segment.get("text", ""))
        matches_query = transcript_norm in text.lower() if transcript_norm else False
        related = int(segment.get("index", -1)) in related_segment_indexes if related_segment_indexes else False
        if transcript_norm and not matches_query and not related:
            continue
        if not transcript_norm and related_segment_indexes and not related:
            continue
        segment_copy = dict(segment)
        segment_copy["is_related"] = related
        filtered.append(segment_copy)
    return filtered


def _marker_type_options(markers: list[dict[str, Any]]) -> list[str]:
    return sorted({str(marker.get("marker_type", "")) for marker in markers if marker.get("marker_type")})


def _markers_by_segment(markers: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for marker in markers:
        grouped.setdefault(int(marker.get("source_segment_index", -1)), []).append(marker)
    return grouped


def _lecture_detail_url(
    lecture_id: str,
    marker_type: str = "",
    review_state: str = "",
    q: str = "",
    transcript_q: str = "",
    anchor: str = "marker-preview",
) -> str:
    params: dict[str, str] = {}
    if q.strip():
        params["q"] = q.strip()
    if marker_type.strip():
        params["marker_type"] = marker_type.strip()
    if review_state.strip():
        params["review_state"] = review_state.strip()
    if transcript_q.strip():
        params["transcript_q"] = transcript_q.strip()
    suffix = f"?{urlencode(params)}" if params else ""
    anchor_part = f"#{anchor}" if anchor else ""
    return f"/lectures/{lecture_id}{suffix}{anchor_part}"
