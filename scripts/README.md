# Shubble Development Launcher

A cross-platform Python script that starts and stops the full Shubble development environment using tmux (Mac/Linux/WSL) or Windows Terminal (native Windows).

---

## Files

- `quick_dev.py` — Main development launcher script

---

## Requirements

Before using this script, make sure the following are installed:

### All Platforms
- Python 3
- Docker + Docker Compose
- uv
- npm

### Mac / Linux / WSL
- tmux

### Native Windows
- Windows Terminal (`wt`) — install from the [Microsoft Store](https://aka.ms/terminal)

You can verify installation with:

```bash
python3 --version
docker --version
docker compose version
uv --version
npm --version

# Mac/Linux/WSL
tmux -V

# Windows
wt --version
```

---

## Setup

Set the `SHUBBLE_DIR` environment variable to the path of your local Shubble folder.

**Mac/Linux/WSL** — add to `~/.bashrc` or `~/.zshrc`:
```bash
export SHUBBLE_DIR=$HOME/shubble
source ~/.bashrc
```

**Windows** — run in Command Prompt:
```cmd
setx SHUBBLE_DIR %USERPROFILE%\shubble
```

---

## Usage

```bash
# Start all services
python quick_dev.py start

# Stop all services
python quick_dev.py stop
```

---

## What It Starts

| Window | Service |
|---|---|
| `docker` | PostgreSQL + Redis containers |
| `worker` | Background worker |
| `test-server` | Test server (auto-restarts on crash) |
| `api` | Shubble API (port 8000) |
| `frontend` | Frontend Docker container (port 3000) |
| `vite-client` | Vite dev server (port 5174) |
| `command` | General-purpose shell in `/scripts` |