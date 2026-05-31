"""
App package.

Convenience re-export so you can run:
  uvicorn app:app --reload
or:
  python -m uvicorn app.app:app --reload
"""

from .app import app

__all__ = ["app"]

