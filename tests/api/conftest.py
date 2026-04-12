import sys
from pathlib import Path

# Portable paths — tests/ is parents[2] of this file; api/ is sibling of tests/.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_API_DIR = str(_REPO_ROOT / "api")
_TESTS_DIR = str(_REPO_ROOT / "tests")

for p in (_API_DIR, _TESTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from selfai_conftest import client, temp_workspace, job_factory, _vram_guard  # noqa: F401,E402
