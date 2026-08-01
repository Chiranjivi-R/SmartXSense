"""Regression coverage for launching the backend from its normal directory."""

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = ROOT / "backend"


def test_run_module_creates_app_from_backend_directory_without_pythonpath():
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    command = (
        "import run; "
        "assert any(str(rule) == '/api/v1/datasets/<dataset_id>/preprocess' "
        "for rule in run.app.url_map.iter_rules()); "
        "print('startup-ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", command],
        cwd=BACKEND_DIR,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "startup-ok" in result.stdout
