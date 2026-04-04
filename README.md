# Lecture Note Builder

A performance-first lecture-to-notes web app for turning recorded classroom audio into structured study notes, reviewable exam markers, and exportable study materials.

## What this sprint adds
- Optional queue backend abstraction with `local` and `rq` modes.
- Persistent manual review layer for note blocks and marker approval/rejection.
- Reviewed export regeneration to Markdown, HTML, PDF, and DOCX.
- More professional export template with review appendix and transcript appendix.
- Review JSON persisted beside artifacts.

## Core flow
1. Upload an audio file from the web panel.
2. Run processing.
3. The pipeline normalizes audio, transcribes it, detects markers, and composes notes.
4. Review note blocks and markers in the web panel.
5. Regenerate exports from the reviewed version.

## Queue backends
- `QUEUE_BACKEND=local`: uses in-process thread workers.
- `QUEUE_BACKEND=rq`: enqueues processing jobs to Redis/RQ.
- `QUEUE_BACKEND=auto`: tries RQ first, then falls back to local.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m app.db
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`.

## Docker Compose
```bash
docker compose up
```
This starts:
- the web app
- Redis
- an RQ worker

## Testing and linting
```bash
bash scripts/lint.sh
```

## Main files
- `app/main.py` — web app, routes, review actions
- `app/worker.py` — queue backend abstraction and dispatcher
- `app/jobs.py` — job entrypoint for local and RQ execution
- `app/services/pipeline.py` — end-to-end processing pipeline
- `app/services/review.py` — review overlay persistence and merging
- `app/services/exports.py` — Markdown/HTML/PDF/DOCX exporters

## Current limitations
- Marker review does not yet recompose note blocks automatically from approved markers.
- RQ mode is optional and needs Redis plus `rq`/`redis` installed.
- Review uses a JSON overlay instead of a full database-backed editorial workflow.

## Current sprint highlights

- Review-aware recomposition: note blocks are rebuilt from approved markers first, then from non-rejected markers when nothing is approved yet.
- New `Study Notes` and `Marker-Guided Rationale` blocks improve note quality without adding creative AI generation.
- Regenerated exports now also refresh `notes.json`, so the downloadable structured output matches the reviewed export set.


## Sprint UX additions

- Marker and transcript search/filter on the lecture detail page
- Bulk approve/reject/pending actions for the currently visible markers
- Side-by-side review workspace with transcript, markers, and note blocks
- Quick review links that preserve the current filters
- Transcript highlighting for segments related to the currently visible markers


## Controlled note assistant

This sprint adds a bounded helper layer for:
- deterministic title generation for note sections
- light transcript cleanup without inventing content
- heuristic disambiguation for question / case / example cues

The default mode is `heuristic`, which keeps the system auditable and fast.


## Production hardening added in this sprint
- Dockerfile for repeatable app and worker images
- stronger Docker Compose with health checks, restart policies, and persistent volumes
- optional HTTP Basic Auth for non-public routes
- `/readyz` and `/metrics` endpoints for operational checks
- on-demand backup endpoint plus `scripts/backup.py` for database and artifact snapshots

### Optional Basic Auth
Set these in `.env` when deploying behind a shared URL:
```bash
BASIC_AUTH_ENABLED=true
BASIC_AUTH_USERNAME=admin
BASIC_AUTH_PASSWORD=change-me
```
Public endpoints that stay open for monitoring are:
- `/healthz`
- `/readyz`
- `/metrics`
- `/static/*`

### Backup
Create an offline snapshot manually:
```bash
python scripts/backup.py
```
Or trigger it through the protected admin route:
```bash
curl -u admin:change-me -X POST http://127.0.0.1:8000/api/admin/backup
```

### Monitoring
- `/healthz` for lightweight health
- `/readyz` for local dependency readiness
- `/metrics` for Prometheus-style text metrics


## OpenAI polishing for final notes

This build can optionally polish the reviewed note blocks with the OpenAI API and keep the output JSON-structured.

Set these environment variables:

- `LLM_NOTES_ENABLED=true`
- `OPENAI_API_KEY=...`
- `OPENAI_MODEL=gpt-4o-mini`
- `OPENAI_BASE_URL=https://api.openai.com/v1`

Behavior:

- `notes_source.json` keeps the review-aware source bundle used for recomposition.
- `notes.json`, `notes.md`, `notes.html`, `notes.pdf`, and `notes.docx` are rendered from the polished bundle.
- `llm_runtime.json` records whether polishing was applied, which model was used, how many chunks were sent, and the total time.
