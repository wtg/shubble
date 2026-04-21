web: uv run alembic -c backend/alembic.ini upgrade head && uv run uvicorn shubble:app --host 0.0.0.0 --port ${PORT:-8080}
worker: uv run python -m backend.worker
