from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from statistics import mean
from threading import Lock
from time import monotonic

from app.repository import ArtifactRepository, JobRepository, LectureRepository, PerformanceRunRepository


@dataclass(slots=True)
class RequestMetric:
    method: str
    path: str
    status_code: int
    duration_ms: float


@dataclass(slots=True)
class RuntimeMetrics:
    window_size: int = 200
    _recent_requests: deque[RequestMetric] = field(default_factory=lambda: deque(maxlen=200))
    _started_at: float = field(default_factory=monotonic)
    _lock: Lock = field(default_factory=Lock)

    def record_request(self, method: str, path: str, status_code: int, duration_ms: float) -> None:
        with self._lock:
            if self._recent_requests.maxlen != self.window_size:
                self._recent_requests = deque(self._recent_requests, maxlen=self.window_size)
            self._recent_requests.append(RequestMetric(method, path, status_code, duration_ms))

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            requests = list(self._recent_requests)
        durations = [item.duration_ms for item in requests]
        errors = [item for item in requests if item.status_code >= 500]
        return {
            "uptime_seconds": round(monotonic() - self._started_at, 3),
            "recent_request_count": len(requests),
            "recent_error_count": len(errors),
            "recent_avg_duration_ms": round(mean(durations), 3) if durations else 0.0,
            "recent_max_duration_ms": round(max(durations), 3) if durations else 0.0,
        }


runtime_metrics = RuntimeMetrics()


def build_operational_snapshot() -> dict[str, object]:
    lecture_repository = LectureRepository()
    job_repository = JobRepository()
    artifact_repository = ArtifactRepository()
    performance_repository = PerformanceRunRepository()
    lectures = lecture_repository.list_all()
    jobs = sum((job_repository.list_for_lecture(item.lecture_id) for item in lectures), start=[])
    artifacts = sum((artifact_repository.list_for_lecture(item.lecture_id) for item in lectures), start=[])
    performance_runs = performance_repository.recent(limit=50)
    lecture_status_counts: dict[str, int] = {}
    job_status_counts: dict[str, int] = {}
    for lecture in lectures:
        lecture_status_counts[lecture.status] = lecture_status_counts.get(lecture.status, 0) + 1
    for job in jobs:
        job_status_counts[job.status] = job_status_counts.get(job.status, 0) + 1
    throughput_values = [item.throughput_audio_x for item in performance_runs if item.throughput_audio_x > 0]
    total_values = [item.total_ms for item in performance_runs if item.total_ms > 0]
    transcribe_values = [item.transcribe_ms for item in performance_runs if item.transcribe_ms > 0]
    return {
        "lectures_total": len(lectures),
        "jobs_total": len(jobs),
        "artifacts_total": len(artifacts),
        "performance_runs_total": len(performance_runs),
        "lecture_status_counts": lecture_status_counts,
        "job_status_counts": job_status_counts,
        "avg_pipeline_total_ms": round(mean(total_values), 3) if total_values else 0.0,
        "avg_transcribe_ms": round(mean(transcribe_values), 3) if transcribe_values else 0.0,
        "avg_throughput_audio_x": round(mean(throughput_values), 3) if throughput_values else 0.0,
        "max_throughput_audio_x": round(max(throughput_values), 3) if throughput_values else 0.0,
        **runtime_metrics.snapshot(),
    }


def render_prometheus_text(snapshot: dict[str, object]) -> str:
    lecture_counts = snapshot.get("lecture_status_counts", {})
    job_counts = snapshot.get("job_status_counts", {})
    lines = [
        "# HELP lecture_app_uptime_seconds Process uptime in seconds",
        "# TYPE lecture_app_uptime_seconds gauge",
        f"lecture_app_uptime_seconds {snapshot['uptime_seconds']}",
        "# HELP lecture_recent_requests Total requests tracked in the rolling window",
        "# TYPE lecture_recent_requests gauge",
        f"lecture_recent_requests {snapshot['recent_request_count']}",
        "# HELP lecture_recent_errors Total 5xx responses tracked in the rolling window",
        "# TYPE lecture_recent_errors gauge",
        f"lecture_recent_errors {snapshot['recent_error_count']}",
        "# HELP lecture_recent_avg_duration_ms Average request duration in milliseconds",
        "# TYPE lecture_recent_avg_duration_ms gauge",
        f"lecture_recent_avg_duration_ms {snapshot['recent_avg_duration_ms']}",
        "# HELP lecture_store_lectures Total lecture records",
        "# TYPE lecture_store_lectures gauge",
        f"lecture_store_lectures {snapshot['lectures_total']}",
        "# HELP lecture_store_jobs Total processing jobs",
        "# TYPE lecture_store_jobs gauge",
        f"lecture_store_jobs {snapshot['jobs_total']}",
        "# HELP lecture_store_artifacts Total stored artifacts",
        "# TYPE lecture_store_artifacts gauge",
        f"lecture_store_artifacts {snapshot['artifacts_total']}",
        "# HELP lecture_performance_runs_total Recent performance runs stored",
        "# TYPE lecture_performance_runs_total gauge",
        f"lecture_performance_runs_total {snapshot['performance_runs_total']}",
        "# HELP lecture_pipeline_avg_total_ms Average pipeline runtime in milliseconds",
        "# TYPE lecture_pipeline_avg_total_ms gauge",
        f"lecture_pipeline_avg_total_ms {snapshot['avg_pipeline_total_ms']}",
        "# HELP lecture_pipeline_avg_transcribe_ms Average ASR runtime in milliseconds",
        "# TYPE lecture_pipeline_avg_transcribe_ms gauge",
        f"lecture_pipeline_avg_transcribe_ms {snapshot['avg_transcribe_ms']}",
        "# HELP lecture_pipeline_avg_throughput_audio_x Average audio throughput multiplier",
        "# TYPE lecture_pipeline_avg_throughput_audio_x gauge",
        f"lecture_pipeline_avg_throughput_audio_x {snapshot['avg_throughput_audio_x']}",
        "# HELP lecture_pipeline_max_throughput_audio_x Max audio throughput multiplier",
        "# TYPE lecture_pipeline_max_throughput_audio_x gauge",
        f"lecture_pipeline_max_throughput_audio_x {snapshot['max_throughput_audio_x']}",
    ]
    for status_name, count in sorted(lecture_counts.items()):
        lines.append(f'lecture_status_total{{status="{status_name}"}} {count}')
    for status_name, count in sorted(job_counts.items()):
        lines.append(f'lecture_job_status_total{{status="{status_name}"}} {count}')
    return "\n".join(lines) + "\n"
