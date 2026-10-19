"""The routing service: a FastAPI application (Chapter 15).

    uvicorn routing_service.app:app --host 0.0.0.0 --port 8000

POST /plans/solve   solve now and return the plan (short time limits)
POST /plans         start a background job, return its id at once
GET  /plans/{id}    the job's status, and its plan when done
GET  /health        liveness check
GET  /metrics       counters for monitoring, in Prometheus text format
"""
from __future__ import annotations

import logging
import multiprocessing
import threading
import time
import uuid
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

from fastapi import Depends, FastAPI, Header, HTTPException

from . import config
from .models import Job, PlanRequest, PlanResult
from .solver import RETRY_MIN_S, solve

log = logging.getLogger("routing_service")
app = FastAPI(title="Routing service", version="1.1")
# solves run in separate processes: the server keeps answering while they work
pool = ProcessPoolExecutor(max_workers=config.MAX_WORKERS,
                           mp_context=multiprocessing.get_context("spawn"))
# in memory, for one server: use a database or Redis with several servers
jobs: dict[str, Job] = {}
futures: dict[str, Future] = {}
pending = threading.BoundedSemaphore(config.MAX_PENDING)
metrics = {"plans_total": 0, "plans_failed_total": 0, "plans_fallback_total": 0,
           "plans_rejected_busy_total": 0, "solve_seconds_sum": 0.0,
           "stops_total": 0, "unassigned_total": 0, "cost_per_stop_last": 0.0}
if not config.API_KEY:
    log.warning("ROUTING_API_KEY is not set: anyone can call this service "
                "(development mode, never run it like this in production)")


def check_key(x_api_key: str | None = Header(default=None)):
    if config.API_KEY and x_api_key != config.API_KEY:
        raise HTTPException(status_code=401, detail="missing or wrong API key")


def check_limits(req: PlanRequest):
    if req.time_limit_s > config.MAX_TIME_LIMIT_S:
        raise HTTPException(status_code=422,
                            detail=f"time_limit_s above {config.MAX_TIME_LIMIT_S}")


# <<record>>
def record(req: PlanRequest, result: PlanResult) -> None:
    """One log line and a few counters per plan: enough to watch quality."""
    served = len(req.stops) - len(result.unassigned)
    metrics["plans_total"] += 1
    metrics["plans_fallback_total"] += result.fallback
    metrics["solve_seconds_sum"] += result.solve_seconds
    metrics["stops_total"] += len(req.stops)
    metrics["unassigned_total"] += len(result.unassigned)
    metrics["cost_per_stop_last"] = result.cost / max(1, served)
    log.info("plan stops=%d vehicles=%d unassigned=%d miles=%.1f cost=%.2f "
             "cost_per_stop=%.2f seconds=%.2f fallback=%s", len(req.stops),
             len(result.routes), len(result.unassigned), result.miles, result.cost,
             metrics["cost_per_stop_last"], result.solve_seconds, result.fallback)
# <<end_record>>


# <<endpoints>>
def submit(req: PlanRequest) -> Future:
    """Admission control: at most MAX_PENDING solves queued or running."""
    if not pending.acquire(blocking=False):
        metrics["plans_rejected_busy_total"] += 1
        raise HTTPException(status_code=429, headers={"Retry-After": "10"},
                            detail="busy: too many plans in progress")
    try:
        future = pool.submit(solve, req)
    except Exception:
        pending.release()
        raise
    future.add_done_callback(lambda f: pending.release())
    return future


@app.post("/plans/solve", response_model=PlanResult,
          dependencies=[Depends(check_key)])
def solve_now(req: PlanRequest) -> PlanResult:
    check_limits(req)
    if req.time_limit_s > config.SYNC_LIMIT_S:
        raise HTTPException(status_code=422, detail=(
            f"time_limit_s above {config.SYNC_LIMIT_S}: use POST /plans"))
    future = submit(req)
    retry = max(RETRY_MIN_S, req.time_limit_s)        # the solver's second try
    deadline = req.time_limit_s + retry + config.SYNC_GRACE_S
    try:
        result = future.result(timeout=deadline)       # an end-to-end deadline
    except FutureTimeout:
        future.cancel()                                # if it has not started yet
        raise HTTPException(status_code=503, detail="no plan within the deadline: "
                            "the service is busy. Retry, or use POST /plans")
    record(req, result)
    return result


@app.post("/plans", response_model=Job, status_code=202,
          dependencies=[Depends(check_key)])
def start_job(req: PlanRequest) -> Job:
    check_limits(req)
    expire_jobs()
    future = submit(req)
    job = Job(id=uuid.uuid4().hex, status="queued", created=time.time())
    jobs[job.id], futures[job.id] = job, future

    def done(f):
        try:
            job.result = f.result()
            job.status = "done"
            record(req, job.result)
        except Exception as exc:                     # report, never crash the server
            job.status, job.error = "failed", str(exc)
            metrics["plans_failed_total"] += 1
        job.finished = time.time()
        futures.pop(job.id, None)
    future.add_done_callback(done)
    return job


@app.get("/plans/{job_id}", response_model=Job, dependencies=[Depends(check_key)])
def get_job(job_id: str) -> Job:
    expire_jobs()
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="no such job, or expired")
    job = jobs[job_id]
    f = futures.get(job_id)
    if job.status == "queued" and f is not None and f.running():
        job.status = "running"
    return job


def expire_jobs() -> None:
    """Forget finished jobs after JOB_TTL_S, so memory stays bounded."""
    now = time.time()
    for job_id in [j.id for j in jobs.values()
                   if j.finished and now - j.finished > config.JOB_TTL_S]:
        del jobs[job_id]
# <<end_endpoints>>


@app.get("/health")
def health():
    return {"status": "ok", "time": time.time(), "auth": bool(config.API_KEY)}


@app.get("/metrics")
def get_metrics():
    from fastapi.responses import PlainTextResponse
    lines = [f"routing_{k} {v}" for k, v in metrics.items()]
    return PlainTextResponse("\n".join(lines) + "\n")
