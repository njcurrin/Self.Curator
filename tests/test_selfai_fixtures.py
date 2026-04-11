"""Validation tests for selfai_conftest fixtures (T-100/T-101/T-102/T-103)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from selfai_conftest import client, temp_workspace, job_factory  # noqa: F401,E402


def test_client_hits_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


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


def test_directories_exist():
    from pathlib import Path
    assert (Path("/app/tests/nodes/__init__.py")).exists()
    assert (Path("/app/tests/pipeline/__init__.py")).exists()
    assert (Path("/app/tests/api/__init__.py")).exists()


def test_upstream_conftest_untouched():
    """shared_ray_cluster fixture available from upstream conftest."""
    import importlib, sys
    sys.path.insert(0, "/app/tests")
    # Just verify conftest is importable and has the fixture
    import conftest
    assert hasattr(conftest, "shared_ray_cluster")
