web: alembic -c backend/alembic.ini upgrade head && uvicorn shubble:app --host 0.0.0.0 --port ${PORT:-8080}
worker: python -m backend.worker
