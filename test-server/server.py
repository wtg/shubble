from flask import Flask, jsonify, request, send_from_directory
from threading import Thread, Lock
import time
import os
from .shuttle import Shuttle, ShuttleState, ShuttleLoop

app = Flask(__name__, static_folder="../test-client/dist", static_url_path="")

shuttles = {}
shuttle_counter = 1
shuttle_lock = Lock()

# --- Background Thread ---
def update_loop():
    while True:
        time.sleep(5)
        with shuttle_lock:
            for shuttle in shuttles.values():
                shuttle.update_state()

# Start background updater
t = Thread(target=update_loop, daemon=True)
t.start()

# --- API Routes ---
@app.route("/api/shuttles", methods=["GET"])
def list_shuttles():
    with shuttle_lock:
        return jsonify([s.to_dict() for s in shuttles.values()])

@app.route("/api/shuttles", methods=["POST"])
def create_shuttle():
    global shuttle_counter
    loop_type = request.json.get("loop", "north").lower()
    try:
        loop_enum = ShuttleLoop(loop_type)
    except ValueError:
        return {"error": "Invalid loop type"}, 400

    with shuttle_lock:
        shuttle_id = str(shuttle_counter).zfill(15)
        shuttle = Shuttle(shuttle_id, loop_enum)
        shuttles[shuttle_id] = shuttle
        shuttle_counter += 1
        return jsonify(shuttle.to_dict()), 201

@app.route("/api/shuttles/<shuttle_id>/action", methods=["POST"])
def trigger_action(shuttle_id):
    action = request.json.get("action")
    with shuttle_lock:
        shuttle = shuttles.get(shuttle_id)
        if not shuttle:
            return {"error": "Shuttle not found"}, 404

        try:
            desired_state = ShuttleState(action)
        except ValueError:
            return {"error": "Invalid action"}, 400

        shuttle.trigger(desired_state)
        return jsonify(shuttle.to_dict())

# --- Frontend Serving ---
@app.route("/")
@app.route("/<path:path>")
def serve_frontend(path=""):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    app.run(debug=True)
