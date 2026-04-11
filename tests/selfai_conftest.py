"""
Shared fixtures for self.ai curator API tests.

Usage in sub-package conftest.py files:
    pytest_plugins = ["selfai_conftest"]
"""

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Make api/ importable from the repo layout (not the Docker /app/api path)
_API_DIR = str(Path(__file__).resolve().parent.parent / "api")
if _API_DIR not in sys.path:
    sys.path.insert(0, _API_DIR)

from fastapi.testclient import TestClient  # noqa: E402

from main import (  # noqa: E402
    CurationJob,
    JobStatus,
    app,
)
import main as main_module  # noqa: E402


# ── TestClient ───────────────────────────��──────────────────────


@pytest.fixture(scope="session")
def client():
    """Session-scoped FastAPI TestClient. No real server is started."""
    return TestClient(app)


# ── Temporary workspace ──────���──────────────────────────────────


@pytest.fixture()
def temp_workspace(tmp_path, monkeypatch):
    """Function-scoped isolated workspace matching the production layout.

    Creates the full directory tree under *tmp_path*, monkeypatches every
    path constant in ``api.main``, and resets the in-memory job / process
    dicts.  Cleanup is automatic (pytest's ``tmp_path``).
    """
    workspace = tmp_path / "workspace" / "curator"
    subdirs = ["configs", "data", "logs", "jobs", "custom_stages"]
    for name in subdirs:
        (workspace / name).mkdir(parents=True)

    monkeypatch.setattr(main_module, "WORKSPACE", workspace)
    monkeypatch.setattr(main_module, "CONFIGS_DIR", workspace / "configs")
    monkeypatch.setattr(main_module, "DATA_DIR", workspace / "data")
    monkeypatch.setattr(main_module, "LOGS_DIR", workspace / "logs")
    monkeypatch.setattr(main_module, "JOBS_DIR", workspace / "jobs")
    monkeypatch.setattr(
        main_module, "JOBS_STATE_FILE", workspace / "jobs" / "jobs.json"
    )

    if hasattr(main_module, "CUSTOM_STAGES_DIR"):
        monkeypatch.setattr(
            main_module, "CUSTOM_STAGES_DIR", workspace / "custom_stages"
        )

    # Reset in-memory state
    main_module._jobs.clear()
    main_module._processes.clear()

    yield workspace

    # Belt-and-suspenders cleanup
    main_module._jobs.clear()
    main_module._processes.clear()


# ── Job state factory ──────────���────────────────────────────────


@pytest.fixture()
def job_factory(temp_workspace):
    """Factory that creates CurationJob instances and injects them
    into ``api.main._jobs``.

    Usage::

        job = job_factory(status=JobStatus.RUNNING, name="my-job")
    """

    def _create(
        *,
        job_id: str | None = None,
        name: str | None = None,
        status: JobStatus = JobStatus.PENDING,
        input_path: str | None = None,
        output_path: str | None = None,
        stages_count: int = 1,
        created_at: datetime | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
        scheduled_for: datetime | None = None,
        exit_code: int | None = None,
        error_message: str | None = None,
        output_format: str | None = None,
        pid: int | None = None,
    ) -> CurationJob:
        _id = job_id or uuid.uuid4().hex[:8]
        now = datetime.now(timezone.utc)

        job = CurationJob(
            job_id=_id,
            name=name or f"test-job-{_id}",
            status=status,
            input_path=input_path or str(temp_workspace / "data" / "input.jsonl"),
            output_path=output_path or str(temp_workspace / "data" / "output"),
            stages_count=stages_count,
            pid=pid,
            created_at=created_at or now,
            started_at=started_at,
            finished_at=finished_at,
            scheduled_for=scheduled_for,
            exit_code=exit_code,
            log_file=str(temp_workspace / "logs" / f"{_id}.log"),
            config_file=str(temp_workspace / "configs" / f"{_id}.json"),
            error_message=error_message,
            output_format=output_format,
        )
        main_module._jobs[_id] = job
        return job

    return _create
