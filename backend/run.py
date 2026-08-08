"""Development entry point for the SmartXSense backend."""

import os
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
    # Never enable the Werkzeug interactive debugger (no HTML traceback UI).
    # Optional auto-reload: set SMARTXSENSE_RELOAD=1 for local edit-reload loops.
    use_reloader = os.getenv("SMARTXSENSE_RELOAD", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_debugger=False,
        use_reloader=use_reloader,
    )
