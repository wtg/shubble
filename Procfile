release: flask --app server:create_app db upgrade
web: gunicorn shubble:app --bind 0.0.0.0:$PORT --log-level $LOG_LEVEL
worker: python -m server.worker
