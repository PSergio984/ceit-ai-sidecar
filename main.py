"""FastAPI Cloud entrypoint.

FastAPI Cloud's default ``fastapi run`` command discovers the repository-root
``main.py`` module, so re-export the real ASGI application from the package.
"""

from app.main import app

__all__ = ["app"]
