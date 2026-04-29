from __future__ import annotations

# Compatibility wrapper. Prefer `backend.app.main:app` for new deployments.
from backend.app.main import app

__all__ = ["app"]
