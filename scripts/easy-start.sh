#!/usr/bin/env bash
set -e

# ************************************************************************* #
# *** NOTE Gnome Terminal needs to be installed for this script to work *** #
# *** Install using "sudo apt install gnome-terminal"                   *** #  
# ************************************************************************* #

# --------------------------------------------------------------------------------------- #
# Copy this file outside of your Shubble work folder and fill in the necessary data below #
# --------------------------------------------------------------------------------------- #

SHUBBLE_DIR="" # <--- Replace "" with "$PARENT_DIRECTORY_NAME/shubble_folder_name"
VITE_DIR="$SHUBBLE_DIR/test/client" 


echo "Starting FULL Shubble stack in separate terminal tabs"

# ------------------ Postgres & Redis ------------------
gnome-terminal --tab --title="Postgres+Redis" -- bash -c "cd $SHUBBLE_DIR && docker compose up -d postgres redis; exec bash"

# ------------------ Backend worker ------------------
gnome-terminal --tab --title="Backend Worker" -- bash -c "cd $SHUBBLE_DIR && uv run python -m backend.worker; exec bash"

# ------------------ Test server ------------------
gnome-terminal --tab --title="Test Server 4000" -- bash -c "cd $SHUBBLE_DIR && uv run uvicorn test.server.server:app --port 4000; exec bash"

# ------------------ Main API ------------------
gnome-terminal --tab --title="Main API 8000" -- bash -c "cd $SHUBBLE_DIR && source .venv/bin/activate && uvicorn shubble:app --reload --port 8000; exec bash"

# ------------------ Docker Frontend ------------------
gnome-terminal --tab --title="Frontend Docker 3000" -- bash -c "cd $SHUBBLE_DIR && docker compose --profile frontend up; exec bash"

# ------------------ Vite Test Client ------------------
gnome-terminal --tab --title="Vite Client 5174" -- bash -c "cd $VITE_DIR && npm run dev -- --port 5174 --host; exec bash"

echo "All services launched in separate tabs."
echo "   • Frontend (Docker): http://localhost:3000"
echo "   • Test Client (Vite): http://localhost:5174"
echo "   • Main API: http://localhost:8000"
echo "   • Test Server: http://localhost:4000"