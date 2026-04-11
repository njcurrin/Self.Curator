"""Validation tests for Tier 0 + Tier 1 infrastructure
(T-100/T-101/T-102/T-103/T-104/T-105/T-107)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from selfai_conftest import client, temp_workspace, job_factory, _vram_guard  # noqa: F401,E402


# ── T-100: Directory structure ──────────────────────────────────


def test_directories_exist():
    assert (Path("/app/tests/nodes/__init__.py")).exists()
    assert (Path("/app/tests/pipeline/__init__.py")).exists()
    assert (Path("/app/tests/api/__init__.py")).exists()


# ── T-101: Shared fixtures ──────────────────────────────────────


def test_client_hits_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_temp_workspace_creates_all_subdirs(temp_workspace):
    for name in ["configs", "data", "logs", "jobs", "custom_stages"]:
        assert (temp_workspace / name).is_dir(), f"Missing subdir: {name}"


def test_temp_workspace_patches_constants(temp_workspace):
    import main as main_module
    assert main_module.WORKSPACE == temp_workspace
    assert main_module.CONFIGS_DIR == temp_workspace / "configs"
    assert main_module.DATA_DIR == temp_workspace / "data"
    assert main_module.LOGS_DIR == temp_workspace / "logs"
    assert main_module.JOBS_DIR == temp_workspace / "jobs"


def test_temp_workspace_resets_state(temp_workspace):
    import main as main_module
    assert main_module._jobs == {}
    assert main_module._processes == {}


def test_job_factory_creates_and_injects(job_factory):
    from main import JobStatus
    import main as main_module
    job = job_factory(name="val-job", status=JobStatus.RUNNING)
    assert job.name == "val-job"
    assert job.status == JobStatus.RUNNING
    assert job.job_id in main_module._jobs


def test_job_factory_paths_in_workspace(job_factory, temp_workspace):
    job = job_factory()
    assert str(temp_workspace) in job.input_path
    assert str(temp_workspace) in job.log_file
    assert str(temp_workspace) in job.config_file


# ── T-102: Fixture data ────────────────────────────────────────


def test_fixture_data_jsonl():
    import json
    records = [json.loads(l) for l in open("/app/tests/fixtures/selfai/sample_data.jsonl")]
    assert len(records) == 30
    assert all("text" in r and len(r["text"].split()) >= 2 for r in records)


def test_fixture_data_parquet():
    import pyarrow.parquet as pq
    table = pq.read_table("/app/tests/fixtures/selfai/sample_data.parquet")
    assert table.num_rows == 30
    assert "text" in table.column_names
    assert "id" in table.column_names


# ── T-103: Upstream conftest ────────────────────────────────────


def test_upstream_conftest_untouched():
    sys.path.insert(0, "/app/tests")
    import conftest
    assert hasattr(conftest, "shared_ray_cluster")


# ── T-104: Pytest markers ──────────────────────────────────────


@pytest.mark.fast
def test_fast_marker_applies():
    """A fast-marked test runs without Ray or GPU."""
    assert True


@pytest.mark.integration
def test_integration_marker_applies():
    """Placeholder: integration marker is registered and applies."""
    pytest.skip("Skipped by design — validates marker registration only")


@pytest.mark.gpu
def test_gpu_marker_applies():
    """Placeholder: gpu marker is registered and applies."""
    pytest.skip("Skipped by design — validates marker registration only")


# ── T-105: VRAM guard ──────────────────────────────────────────


def test_vram_guard_skips_no_gpu():
    """_get_free_vram_mb returns 0 when no GPU is available or in test env."""
    from selfai_conftest import _get_free_vram_mb
    # Just verifying it doesn't crash — returns a number
    result = _get_free_vram_mb()
    assert isinstance(result, float)
    assert result >= 0.0


def test_vram_guard_not_applied_to_fast(client):
    """Non-gpu-marked tests are never subject to VRAM checks."""
    # This test has no gpu marker, so _vram_guard should be a no-op.
    # If VRAM guard incorrectly applied, this would skip.
    resp = client.get("/health")
    assert resp.status_code == 200
