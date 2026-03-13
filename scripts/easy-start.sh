#!/usr/bin/env bash
set -e

# ============================================================
# Shubble Development Launcher
# Usage: ./shubble.sh {start|stop}
# ============================================================

# -------- CONFIG --------
SHUBBLE_DIR="$HOME/shubble" # <--- Replace "" with "$PARENT_DIRECTORY_NAME/shubble_folder_name"
VITE_DIR="$SHUBBLE_DIR/test/client"
SESSION="shubble-dev"

# -------- CLEANUP FUNCTION --------
# This is called automatically if the script receives an exit signal
cleanup() {
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    cd "$SHUBBLE_DIR"
    docker compose --profile frontend down
    docker compose down redis postgres
    echo "✅ Shubble stack stopped."
}
# -------- ARGUMENT HANDLING --------
case "$1" in
    stop)
        echo "🛑 Stopping Shubble services..."
        tmux kill-session -t "$SESSION" 2>/dev/null || true
        docker compose --profile frontend down
        docker compose down redis postgres
        echo "✅ Shubble stack stopped."
        ;;
    start|*)
        # 1. Validation
        if [ -z "$SHUBBLE_DIR" ]; then
            echo "ERROR: SHUBBLE_DIR not set in script."
            exit 1
        fi
        for cmd in docker uv npm tmux; do
            if ! command -v "$cmd" &> /dev/null; then
                echo "ERROR: Required command '$cmd' not found."
                exit 1
            fi
        done

        # 2. Trap signals so cleanup runs on exit
        trap cleanup EXIT SIGINT SIGTERM

        # 3. Clean old session
        if tmux has-session -t "$SESSION" 2>/dev/null; then
            echo "Stopping existing session..."
            tmux kill-session -t "$SESSION"
        fi

        echo "🚀 Starting Shubble dev environment..."

        # 4. Create Session
        tmux new-session -d -s "$SESSION" -n "docker" -c "$SHUBBLE_DIR" "docker compose up -d postgres redis"
        tmux new-window -t "$SESSION" -n "worker" -c "$SHUBBLE_DIR" "uv run python -m backend.worker"
        tmux new-window -t "$SESSION" -n "test-server" -c "$SHUBBLE_DIR" "uv run uvicorn test.server.server:app --port 4000"
        tmux new-window -t "$SESSION" -n "api" -c "$SHUBBLE_DIR" "uv run uvicorn shubble:app --reload --port 8000"
        tmux new-window -t "$SESSION" -n "frontend" -c "$SHUBBLE_DIR" "docker compose --profile frontend up"
        tmux new-window -t "$SESSION" -n "vite-client" -c "$VITE_DIR" "npm run dev -- --port 5174 --host"
        tmux new-window -t "$SESSION" -n "command" -c "../$SHUBBLE_DIR" bash

        # 5. Attach
        echo "✅ Shubble services started."
        tmux attach -t "$SESSION"
        ;;
esac