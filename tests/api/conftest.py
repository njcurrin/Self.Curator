import sys
from pathlib import Path

# Make api/ and tests/ importable
sys.path.insert(0, "/app/api")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from selfai_conftest import client, temp_workspace, job_factory, _vram_guard  # noqa: F401,E402
