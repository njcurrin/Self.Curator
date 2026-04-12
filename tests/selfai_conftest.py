"""
Shared fixtures for self.ai curator API tests.

Sub-package conftest.py files import fixtures from here directly.
"""

import os
import subprocess
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

    # Patch stage_registry's module-level CUSTOM_STAGES_DIR and index path.
    # stage_registry.py has its own import of the path — must be patched separately.
    try:
        import stage_registry as sr_module
        monkeypatch.setattr(sr_module, "CUSTOM_STAGES_DIR", workspace / "custom_stages")
        monkeypatch.setattr(
            sr_module, "CUSTOM_STAGES_INDEX", workspace / "custom_stages" / "index.json"
        )
    except ImportError:
        pass

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


# ── VRAM guard ─────────────────────────────────────────────────


def _get_free_vram_mb() -> float:
    """Query free GPU memory in MB. Returns 0 if no GPU available."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            # Sum free memory across all GPUs
            return sum(float(x) for x in result.stdout.strip().split("\n") if x.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return 0.0


@pytest.fixture(autouse=True)
def _vram_guard(request):
    """Auto-skip gpu-marked tests if free VRAM is below threshold.

    Threshold (in GB) is set via VRAM_THRESHOLD env var (default: 4).
    Only applies to tests with the ``gpu`` marker.
    """
    marker = request.node.get_closest_marker("gpu")
    if marker is None:
        return  # Not a GPU test — no check

    threshold_gb = float(os.environ.get("VRAM_THRESHOLD", "4"))
    free_mb = _get_free_vram_mb()
    free_gb = free_mb / 1024

    if free_gb < threshold_gb:
        pytest.skip(
            f"Insufficient VRAM: {free_gb:.1f} GB free, "
            f"{threshold_gb:.1f} GB required (set VRAM_THRESHOLD to override)"
        )
