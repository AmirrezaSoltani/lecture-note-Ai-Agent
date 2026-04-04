from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import Protocol

from app.config import settings
from app.jobs import process_lecture_job
from app.models import LectureStatus
from app.repository import JobRepository, LectureRepository

logger = logging.getLogger(__name__)


class JobDispatcher(Protocol):
    backend_name: str

    def submit(self, lecture_id: str) -> str:
        ...

    def recover_pending_jobs(self) -> None:
        ...

    def queue_depth(self) -> int:
        ...


class LocalJobDispatcher:
    backend_name = "local"

    def __init__(self) -> None:
        self.executor = ThreadPoolExecutor(max_workers=settings.max_workers, thread_name_prefix="lecture-worker")
        self.lecture_repository = LectureRepository()
        self.job_repository = JobRepository()
        self.futures: dict[str, Future] = {}
        self._lock = Lock()

    def submit(self, lecture_id: str) -> str:
        self._cleanup_finished_futures()
        record = self.lecture_repository.get(lecture_id)
        if record is None:
            raise ValueError(f"Lecture not found: {lecture_id}")
        if self.job_repository.has_active_job(lecture_id):
            existing = self.job_repository.latest_for_lecture(lecture_id)
            return existing.job_id if existing is not None else ""

        job = self.job_repository.create(lecture_id)
        self.lecture_repository.update_status(lecture_id, LectureStatus.PROCESSING)
        future = self.executor.submit(process_lecture_job, job.job_id, lecture_id, self.backend_name)
        with self._lock:
            self.futures[job.job_id] = future
        return job.job_id

    def recover_pending_jobs(self) -> None:
        self._cleanup_finished_futures()
        self.job_repository.mark_stale_running_jobs_failed("Worker restarted before the previous run finished.")
        for job in self.job_repository.queued_jobs():
            self.lecture_repository.update_status(job.lecture_id, LectureStatus.PROCESSING)
            future = self.executor.submit(process_lecture_job, job.job_id, job.lecture_id, self.backend_name)
            with self._lock:
                self.futures[job.job_id] = future

    def queue_depth(self) -> int:
        self._cleanup_finished_futures()
        return len(self.job_repository.queued_jobs())

    def active_count(self) -> int:
        self._cleanup_finished_futures()
        with self._lock:
            return sum(1 for future in self.futures.values() if not future.done())

    def _cleanup_finished_futures(self) -> None:
        with self._lock:
            finished = [job_id for job_id, future in self.futures.items() if future.done()]
            for job_id in finished:
                self.futures.pop(job_id, None)


class RQJobDispatcher:
    backend_name = "rq"

    def __init__(self) -> None:
        try:
            from redis import Redis
            from rq import Queue
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError("RQ backend requested but rq/redis are not installed.") from exc

        self.lecture_repository = LectureRepository()
        self.job_repository = JobRepository()
        self.connection = Redis.from_url(settings.redis_url)
        self.connection.ping()
        self.queue = Queue(settings.rq_queue_name, connection=self.connection)

    def submit(self, lecture_id: str) -> str:
        record = self.lecture_repository.get(lecture_id)
        if record is None:
            raise ValueError(f"Lecture not found: {lecture_id}")
        if self.job_repository.has_active_job(lecture_id):
            existing = self.job_repository.latest_for_lecture(lecture_id)
            return existing.job_id if existing is not None else ""

        job = self.job_repository.create(lecture_id)
        self.lecture_repository.update_status(lecture_id, LectureStatus.PROCESSING)
        self.queue.enqueue("app.jobs.process_lecture_job", job.job_id, lecture_id, self.backend_name, job_timeout="2h")
        return job.job_id

    def recover_pending_jobs(self) -> None:
        self.job_repository.mark_stale_running_jobs_failed("RQ worker restarted before the previous run finished.")

    def queue_depth(self) -> int:
        return len(self.job_repository.queued_jobs())

    def active_count(self) -> int:
        try:
            return self.queue.count
        except Exception:  # pragma: no cover - runtime dependent
            return 0


class DispatcherFacade:
    def __init__(self, dispatcher: JobDispatcher, requested_backend: str) -> None:
        self._dispatcher = dispatcher
        self.requested_backend = requested_backend
        self.backend_name = dispatcher.backend_name

    def submit(self, lecture_id: str) -> str:
        return self._dispatcher.submit(lecture_id)

    def recover_pending_jobs(self) -> None:
        self._dispatcher.recover_pending_jobs()

    def queue_depth(self) -> int:
        return self._dispatcher.queue_depth()

    def active_count(self) -> int:
        if hasattr(self._dispatcher, "active_count"):
            return int(self._dispatcher.active_count())
        return 0


def build_job_dispatcher() -> DispatcherFacade:
    requested = settings.queue_backend
    if requested == "local":
        return DispatcherFacade(LocalJobDispatcher(), requested)
    if requested == "rq":
        return DispatcherFacade(RQJobDispatcher(), requested)
    try:
        return DispatcherFacade(RQJobDispatcher(), requested)
    except Exception as exc:  # pragma: no cover - depends on runtime
        logger.info("Falling back to local queue backend: %s", exc)
        return DispatcherFacade(LocalJobDispatcher(), requested)
