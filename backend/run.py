"""Development entry point for the SmartXSense backend."""

import sys
from pathlib import Path

# ``python run.py`` is intentionally supported from backend/.  Add the
# repository root so the sibling reusable ML package is importable without
# requiring developers to configure PYTHONPATH themselves.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import create_app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
