"""
FastAPI server for controlling NeMo Curator curation pipeline jobs.
Runs on port 8094, manages job lifecycle and progress monitoring.
"""

import asyncio
import json
import os
import subprocess
import threading
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ─── Constants ──────────────────────────────────────────────────────────

WORKSPACE = Path("/workspace/curator")
CONFIGS_DIR = WORKSPACE / "configs"
DATA_DIR = WORKSPACE / "data"
LOGS_DIR = WORKSPACE / "logs"
JOBS_DIR = WORKSPACE / "jobs"
JOBS_STATE_FILE = JOBS_DIR / "jobs.json"
VENV_BIN = Path("/opt/venv/bin")
PYTHON = VENV_BIN / "python"
RUN_PIPELINE_SCRIPT = Path("/app/api/run_pipeline.py")

API_VERSION = "1.0.0"

# ─── Enums and Models ───────────────────────────────────────────────────


class JobStatus(str, Enum):
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StageConfig(BaseModel):
    type: str
    params: Dict[str, Any] = {}


class CurationJobCreate(BaseModel):
    name: str = "curation-job"
    input_path: str
    output_path: str
    text_field: str = "text"
    stages: List[StageConfig]
    output_format: Optional[str] = None
    scheduled_for: Optional[datetime] = None

    class Config:
        json_schema_extra = {
            "example": {
                "name": "quality-filter",
                "input_path": "/workspace/curator/data/input.jsonl",
                "output_path": "/workspace/curator/data/output.jsonl",
                "text_field": "text",
                "stages": [
                    {
                        "type": "ScoreFilter",
                        "params": {
                            "filter": "WordCountFilter",
                            "min_words": 50,
                            "max_words": 100000,
                        },
                    }
                ],
            }
        }


class ScheduleJobRequest(BaseModel):
    scheduled_for: float  # Unix timestamp


class CurationJob(BaseModel):
    job_id: str
    name: str
    status: JobStatus
    input_path: str
    output_path: str
    stages_count: int
    pid: Optional[int] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    scheduled_for: Optional[datetime] = None
    exit_code: Optional[int] = None
    log_file: str
    config_file: str
    error_message: Optional[str] = None
    output_format: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    running_jobs: int
    jobs_total: int
    api_version: str


# ─── State ──────────────────────────────────────────────────────────────

_jobs: Dict[str, CurationJob] = {}
_processes: Dict[str, subprocess.Popen] = {}

app = FastAPI(title="Curator API", version=API_VERSION)


# ─── Persistence ────────────────────────────────────────────────────────


def _ensure_dirs():
    """Create all required directories."""
    for d in [CONFIGS_DIR, DATA_DIR, LOGS_DIR, JOBS_DIR, WORKSPACE / "custom_stages"]:
        d.mkdir(parents=True, exist_ok=True)


def _cleanup_orphan_tmp_files():
    """Remove leftover jobs.*.tmp files from aborted _save_jobs() writes.

    _save_jobs() uses per-call unique tmp filenames to avoid the race
    between concurrent writers. On SIGKILL / OOM / container crash
    between write_text and replace, the .tmp file survives. This
    sweep runs at startup so the jobs directory doesn't accumulate.
    """
    if not JOBS_DIR.exists():
        return
    stem = JOBS_STATE_FILE.stem  # "jobs"
    for p in JOBS_DIR.glob(f"{stem}.*.tmp"):
        try:
            p.unlink()
        except OSError:
            pass


def _load_jobs():
    """Load job state from JSON file."""
    global _jobs
    _cleanup_orphan_tmp_files()
    if JOBS_STATE_FILE.exists():
        data = json.loads(JOBS_STATE_FILE.read_text())
        for job_id, job_data in data.items():
            try:
                job_data["created_at"] = datetime.fromisoformat(
                    job_data["created_at"]
                )
                if job_data.get("started_at"):
                    job_data["started_at"] = datetime.fromisoformat(
                        job_data["started_at"]
                    )
                if job_data.get("finished_at"):
                    job_data["finished_at"] = datetime.fromisoformat(
                        job_data["finished_at"]
                    )
                job = CurationJob(**job_data)
                # Mark any RUNNING jobs as FAILED (process lost on restart)
                if job.status == JobStatus.RUNNING:
                    job.status = JobStatus.FAILED
                    job.error_message = "Process lost on container restart"
                    job.finished_at = datetime.now(timezone.utc)
                _jobs[job_id] = job
            except Exception as e:
                print(f"Failed to load job {job_id}: {e}")


_jobs_lock = threading.Lock()


def _save_jobs():
    """Persist job state to JSON file (atomic write, thread-safe).

    Uses a per-call unique tmp filename (pid + thread id + uuid) plus
    a module-level lock so concurrent callers do not collide on the
    shared .tmp path or produce inconsistent state snapshots.
    """
    with _jobs_lock:
        data = {jid: j.model_dump(mode="json") for jid, j in _jobs.items()}
        unique = f"{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex[:8]}"
        tmp = JOBS_STATE_FILE.with_suffix(f".{unique}.tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str))
        tmp.replace(JOBS_STATE_FILE)


# ─── Job Execution ──────────────────────────────────────────────────────


def _start_job(job: CurationJob) -> None:
    """Launch a curation pipeline subprocess."""
    config_path = Path(job.config_file)
    if not config_path.exists():
        job.status = JobStatus.FAILED
        job.error_message = f"Config file not found: {config_path}"
        job.finished_at = datetime.now(timezone.utc)
        _save_jobs()
        return

    log_path = Path(job.log_file)
    try:
        log_fh = open(log_path, "w")
    except Exception as e:
        job.status = JobStatus.FAILED
        job.error_message = f"Failed to open log: {e}"
        job.finished_at = datetime.now(timezone.utc)
        _save_jobs()
        return

    env = os.environ.copy()
    env["PATH"] = f"{VENV_BIN}:{env.get('PATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    # Ensure PYTHONPATH includes the Curator package
    env["PYTHONPATH"] = f"/opt/Curator:{env.get('PYTHONPATH', '')}"

    cmd = [str(PYTHON), str(RUN_PIPELINE_SCRIPT), str(config_path)]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(WORKSPACE),
        )
    except Exception as e:
        job.status = JobStatus.FAILED
        job.error_message = f"Failed to start pipeline: {e}"
        job.finished_at = datetime.now(timezone.utc)
        _save_jobs()
        return

    job.status = JobStatus.RUNNING
    job.pid = proc.pid
    job.started_at = datetime.now(timezone.utc)
    _processes[job.job_id] = proc
    _save_jobs()


def _poll_jobs_once():
    """Run a single iteration of the poll loop.

    Checks every RUNNING job for subprocess completion and transitions
    it to COMPLETED / FAILED. Scheduled jobs are intentionally NOT
    touched here — the self.UI daemon is responsible for approving
    scheduled jobs when their time arrives.

    Extracted from `_poll_jobs()` so tests can exercise the poll
    behavior without the infinite `await asyncio.sleep(5)` loop.
    """
    for job_id, job in list(_jobs.items()):
        if job.status != JobStatus.RUNNING:
            continue

        proc = _processes.get(job_id)
        if not proc:
            continue

        rc = proc.poll()
        if rc is not None:
            job.exit_code = rc
            job.finished_at = datetime.now(timezone.utc)
            job.status = (
                JobStatus.COMPLETED if rc == 0 else JobStatus.FAILED
            )
            if rc != 0:
                try:
                    log_lines = Path(job.log_file).read_text().splitlines()
                    tail = "\n".join(log_lines[-10:])
                    job.error_message = f"Process exited with code {rc}. Tail:\n{tail}"
                except Exception:
                    job.error_message = f"Process exited with code {rc}"
            del _processes[job_id]
            _save_jobs()


async def _poll_jobs():
    """Background task: poll job subprocess status every 5 seconds."""
    while True:
        await asyncio.sleep(5)
        _poll_jobs_once()


# ─── Startup ────────────────────────────────────────────────────────────


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    _ensure_dirs()
    _load_jobs()
    asyncio.create_task(_poll_jobs())


# ─── Health ─────────────────────────────────────────────────────────────


@app.get("/health")
def health_check() -> HealthResponse:
    running = sum(1 for j in _jobs.values() if j.status == JobStatus.RUNNING)
    return HealthResponse(
        status="ok",
        running_jobs=running,
        jobs_total=len(_jobs),
        api_version=API_VERSION,
    )


# ─── Stages Endpoints ──────────────────────────────────────────────────
# Custom stage routes MUST be registered before the parameterized
# {category} routes, otherwise FastAPI matches "/api/text/custom/stages"
# as category="custom" on the "/{category}/stages" route.


@app.get("/api/text")
def list_text_categories():
    """List all text stage categories and their stages."""
    from stage_registry import get_text_stages_by_category

    return get_text_stages_by_category()


# ─── Custom Stages (before parameterized routes) ───────────────────────


class CustomStageCreate(BaseModel):
    name: str
    category: str
    code: str


@app.post("/api/text/custom/stages", status_code=201)
def create_custom_stage(req: CustomStageCreate):
    """Save a user-defined custom stage.

    The code must define exactly one concrete ProcessingStage subclass.
    Name must be unique across builtins and existing custom stages.
    """
    from stage_registry import save_custom_stage

    try:
        result = save_custom_stage(req.name, req.category, req.code)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@app.get("/api/text/custom/stages")
def list_custom_stages():
    """List all user-defined custom stages."""
    from stage_registry import _load_custom_index

    index = _load_custom_index()
    return [
        {
            "id": uid,
            "name": entry["name"],
            "source": "custom",
            "category": entry["category"],
            "created_at": entry.get("created_at"),
        }
        for uid, entry in index.items()
    ]


@app.get("/api/text/custom/stages/{stage_uuid}")
def get_custom_stage(stage_uuid: str):
    """Get full details for a custom stage, including source code."""
    from stage_registry import get_custom_stage_detail

    detail = get_custom_stage_detail(stage_uuid)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Custom stage not found: {stage_uuid}")
    return detail


@app.delete("/api/text/custom/stages/{stage_uuid}")
def remove_custom_stage(stage_uuid: str):
    """Delete a custom stage."""
    from stage_registry import delete_custom_stage

    if not delete_custom_stage(stage_uuid):
        raise HTTPException(status_code=404, detail=f"Custom stage not found: {stage_uuid}")
    return {"status": "deleted", "id": stage_uuid}


@app.post("/api/text/custom/stages/validate-name")
def validate_stage_name(name: str):
    """Check if a stage name is available (no conflicts)."""
    from stage_registry import validate_custom_stage_name

    conflict = validate_custom_stage_name(name)
    if conflict:
        return {"available": False, "reason": conflict}
    return {"available": True}


# ─── Builtin Stages (parameterized routes last) ────────────────────────


@app.get("/api/text/{category}/stages")
def list_category_stages(category: str):
    """List stages in a specific text category."""
    from stage_registry import get_category_stages

    stages = get_category_stages(category)
    if stages is None:
        raise HTTPException(status_code=404, detail=f"Category not found: {category}")
    return stages


@app.get("/api/text/{category}/stages/{stage_id}")
def get_text_stage(category: str, stage_id: str):
    """Get full details and parameters for a specific text stage."""
    from stage_registry import get_text_stage_detail

    detail = get_text_stage_detail(stage_id)
    if detail is None or detail["category"] != category:
        raise HTTPException(status_code=404, detail=f"Stage not found: {stage_id} in {category}")
    return detail


# ─── Jobs Endpoints ─────────────────────────────────────────────────────


@app.post("/api/jobs", status_code=201)
def create_job(req: CurationJobCreate) -> CurationJob:
    """Create and start a new curation pipeline job."""
    # Validate input path exists
    input_path = Path(req.input_path)
    if not input_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Input path not found: {req.input_path}",
        )

    # Validate stages reference known types
    from stage_registry import get_text_stage_detail as _get_detail
    from run_pipeline import _FILTER_CLASS_REGISTRY, _MODIFIER_CLASS_REGISTRY, _CLASSIFIER_CLASS_REGISTRY, _DEDUP_TYPES

    for stage in req.stages:
        if _get_detail(stage.type) is None and stage.type not in _FILTER_CLASS_REGISTRY and stage.type not in _MODIFIER_CLASS_REGISTRY and stage.type not in _CLASSIFIER_CLASS_REGISTRY and stage.type not in _DEDUP_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown stage type: {stage.type}",
            )

    job_id = str(uuid.uuid4())[:8]

    # Write pipeline config to disk
    config = {
        "name": req.name,
        "input_path": req.input_path,
        "output_path": req.output_path,
        "text_field": req.text_field,
        "stages": [s.model_dump() for s in req.stages],
        "output_format": req.output_format,
    }
    config_path = CONFIGS_DIR / f"{job_id}.json"
    config_path.write_text(json.dumps(config, indent=2))

    log_file = LOGS_DIR / f"{job_id}.log"

    scheduled_for = req.scheduled_for
    if scheduled_for and scheduled_for.timestamp() <= datetime.now(timezone.utc).timestamp():
        scheduled_for = None  # past time → treat as immediate

    initial_status = JobStatus.SCHEDULED if scheduled_for else JobStatus.PENDING

    job = CurationJob(
        job_id=job_id,
        name=req.name,
        status=initial_status,
        input_path=req.input_path,
        output_path=req.output_path,
        stages_count=len(req.stages),
        created_at=datetime.now(timezone.utc),
        scheduled_for=scheduled_for,
        log_file=str(log_file),
        config_file=str(config_path),
        output_format=req.output_format,
    )

    _jobs[job_id] = job
    _save_jobs()

    return _jobs[job_id]


@app.get("/api/jobs")
def list_jobs() -> List[CurationJob]:
    """List all curation jobs."""
    return sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> CurationJob:
    """Get curation job details."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/jobs/{job_id}/logs")
async def get_job_logs(
    job_id: str, tail: int = Query(100), stream: bool = Query(False)
):
    """Get job logs. Use stream=true for live tailing."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    log_path = Path(job.log_file)
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Log file not found")

    if not stream:
        try:
            lines = log_path.read_text().splitlines()
            return {"lines": lines[-tail:]}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def generate():
        async with aiofiles.open(log_path, "r") as f:
            await f.seek(0, 2)
            while True:
                line = await f.readline()
                if line:
                    yield line
                else:
                    await asyncio.sleep(0.3)

    return StreamingResponse(generate(), media_type="text/plain")


def _do_cancel(job_id: str) -> dict:
    """Shared cancel logic for DELETE and POST cancel endpoints."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status not in (JobStatus.RUNNING, JobStatus.PENDING, JobStatus.SCHEDULED):
        raise HTTPException(status_code=400, detail="Job is not cancellable")

    proc = _processes.get(job_id)
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        del _processes[job_id]

    job.status = JobStatus.CANCELLED
    job.finished_at = datetime.now(timezone.utc)
    _save_jobs()
    return {"status": "cancelled", "job_id": job_id}


@app.delete("/api/jobs/{job_id}")
def cancel_job(job_id: str):
    """Cancel a running, pending, or scheduled curation job."""
    return _do_cancel(job_id)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job_post(job_id: str):
    """Cancel a job (POST alias for schedule-page compatibility)."""
    return _do_cancel(job_id)


@app.post("/api/jobs/{job_id}/schedule")
def schedule_job(job_id: str, body: ScheduleJobRequest):
    """Set a future run time for a pending job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status not in (JobStatus.PENDING, JobStatus.SCHEDULED):
        raise HTTPException(status_code=400, detail="Job cannot be scheduled in its current state")

    scheduled_dt = datetime.fromtimestamp(body.scheduled_for, tz=timezone.utc)
    job.scheduled_for = scheduled_dt
    job.status = JobStatus.SCHEDULED
    _save_jobs()
    return _jobs[job_id]


@app.post("/api/jobs/{job_id}/unschedule")
def unschedule_job(job_id: str):
    """Remove the scheduled time from a job (returns it to pending)."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.SCHEDULED:
        raise HTTPException(status_code=400, detail="Job is not scheduled")

    job.scheduled_for = None
    job.status = JobStatus.PENDING
    _save_jobs()
    return _jobs[job_id]


@app.post("/api/jobs/{job_id}/approve")
def approve_job(job_id: str):
    """Start a pending or scheduled job immediately."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status not in (JobStatus.PENDING, JobStatus.SCHEDULED):
        raise HTTPException(status_code=400, detail="Job is not pending or scheduled")

    job.scheduled_for = None
    _start_job(job)
    return _jobs[job_id]


# ─── Data Endpoint ──────────────────────────────────────────────────────


@app.get("/api/data")
def list_data():
    """List files in the data volume."""
    files = []
    if DATA_DIR.exists():
        for p in sorted(DATA_DIR.rglob("*")):
            if p.is_file():
                files.append({
                    "path": str(p),
                    "name": p.name,
                    "size_bytes": p.stat().st_size,
                    "relative_path": str(p.relative_to(DATA_DIR)),
                })
    return {"data_dir": str(DATA_DIR), "files": files}
