release: flask db upgrade
web: gunicorn shubble:app --bind 0.0.0.0:$PORT
worker: python worker.py
