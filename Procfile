release: flask --app server:create_app db upgrade
web: gunicorn shubble:app --bind 0.0.0.0:$PORT
worker: python -m server.worker
