#!/usr/bin/env python3
"""
Shubble Development Launcher
Usage: python quick_dev.py start|stop

Requirements:
  - All platforms: docker, uv, npm
  - Mac/Linux/WSL: tmux
  - Native Windows: Windows Terminal (wt)
"""

import os
import sys
import platform
import shutil
import subprocess

# -------- CONFIG --------
SHUBBLE_DIR = os.environ.get("SHUBBLE_DIR", "")
VITE_DIR = os.path.join(SHUBBLE_DIR, "test", "client")
SESSION = "shubble-dev"

# Detect native Windows (not WSL)
IS_NATIVE_WINDOWS = platform.system() == "Windows"

# -------- HELPERS --------
def run(cmd, **kwargs):
    return subprocess.run(cmd, check=True, **kwargs)

def run_silent(cmd):
    return subprocess.run(cmd, capture_output=True)

def check_deps():
    deps = ["docker", "uv", "npm"]
    if IS_NATIVE_WINDOWS:
        deps.append("wt")
    else:
        deps.append("tmux")
    for cmd in deps:
        if not shutil.which(cmd):
            print(f"ERROR: Required command '{cmd}' not found.")
            sys.exit(1)

# -------- TMUX HELPERS (Mac / Linux / WSL) --------
def tmux(args: list):
    run(["tmux"] + args)

def tmux_silent(args: list):
    run_silent(["tmux"] + args)

def tmux_new_session(session, name, cwd, cmd):
    tmux(["new-session", "-d", "-s", session, "-n", name, "-c", cwd, "--", "bash", "-c", cmd])

def tmux_new_window(session, name, cwd, cmd):
    tmux(["new-window", "-t", session, "-n", name, "-c", cwd, "--", "bash", "-c", cmd])

# -------- UNIX: Mac / Linux / WSL --------
POSTGRES_WAIT = (
    "until docker compose exec postgres pg_isready -q 2>/dev/null; "
    "do echo 'Waiting for postgres...'; sleep 1; done"
)
TEST_SERVER_LOOP = (
    "while true; do "
    "uv run uvicorn test.server.server:app --port 4000; "
    'echo "test-server exited ($?), restarting in 2s..."; '
    "sleep 2; done"
)

UNIX_WINDOWS = [
    ("docker",      SHUBBLE_DIR,                            "docker compose up -d postgres redis"),
    ("worker",      SHUBBLE_DIR,                            "uv run python -m backend.worker"),
    ("test-server", SHUBBLE_DIR,                            f"{POSTGRES_WAIT}; {TEST_SERVER_LOOP}"),
    ("api",         SHUBBLE_DIR,                            "uv run uvicorn shubble:app --reload --port 8000"),
    ("frontend",    SHUBBLE_DIR,                            "docker compose --profile frontend up"),
    ("vite-client", VITE_DIR,                               "npm run dev -- --port 5174 --host"),
    ("command",     os.path.join(SHUBBLE_DIR, "scripts"),   "bash"),
]

def start_unix():
    if run_silent(["tmux", "has-session", "-t", SESSION]).returncode == 0:
        print("Stopping existing session...")
        tmux_silent(["kill-session", "-t", SESSION])

    print("Starting Shubble dev environment...")

    first = True
    for name, cwd, cmd in UNIX_WINDOWS:
        if first:
            tmux_new_session(SESSION, name, cwd, cmd)
            first = False
        else:
            tmux_new_window(SESSION, name, cwd, cmd)

    print("✅ Shubble services started.")
    os.execvp("tmux", ["tmux", "attach", "-t", SESSION])

def stop_unix():
    print("🛑 Stopping Shubble services...")
    run(["docker", "compose", "--profile", "frontend", "down"], cwd=SHUBBLE_DIR)
    run(["docker", "compose", "down", "redis", "postgres"], cwd=SHUBBLE_DIR)
    tmux_silent(["kill-session", "-t", SESSION])
    print("✅ Shubble stack stopped.")

# -------- WINDOWS: Native Windows Terminal --------
WIN_SERVICES = [
    ("docker",      SHUBBLE_DIR,                            "docker compose up -d postgres redis"),
    ("worker",      SHUBBLE_DIR,                            "uv run python -m backend.worker"),
    ("test-server", SHUBBLE_DIR,                            (
        "cmd /k \"FOR /L %i IN (0,0,1) DO ("
        "uv run uvicorn test.server.server:app --port 4000 & "
        "timeout /t 2 >nul)\""
    )),
    ("api",         SHUBBLE_DIR,                            "uv run uvicorn shubble:app --reload --port 8000"),
    ("frontend",    SHUBBLE_DIR,                            "docker compose --profile frontend up"),
    ("vite-client", VITE_DIR,                               "npm run dev -- --port 5174 --host"),
    ("command",     os.path.join(SHUBBLE_DIR, "scripts"),   "cmd /k"),
]

def start_windows():
    print("Starting Shubble dev environment (Windows Terminal)...")

    # Build wt command: first tab, then subsequent tabs
    first_name, first_cwd, first_cmd = WIN_SERVICES[0]
    wt_args = [
        "wt", "--title", first_name,
        "--startingDirectory", first_cwd,
        "cmd", "/k", first_cmd
    ]
    for name, cwd, cmd in WIN_SERVICES[1:]:
        wt_args += [
            ";", "new-tab",
            "--title", name,
            "--startingDirectory", cwd,
            "cmd", "/k", cmd
        ]

    subprocess.Popen(wt_args)
    print("✅ Shubble services started in Windows Terminal.")
    print("ℹ️  To stop: run 'python quick_dev.py stop' or close the Windows Terminal window.")

def stop_windows():
    print("🛑 Stopping Shubble services...")
    run(["docker", "compose", "--profile", "frontend", "down"], cwd=SHUBBLE_DIR)
    run(["docker", "compose", "down", "redis", "postgres"], cwd=SHUBBLE_DIR)
    print("✅ Shubble stack stopped.")
    print("ℹ️  Close the Windows Terminal window to finish.")

# -------- MAIN --------
def main():
    if not SHUBBLE_DIR:
        print("ERROR: SHUBBLE_DIR environment variable is not set.\n")
        if IS_NATIVE_WINDOWS:
            print("  Set it permanently:  setx SHUBBLE_DIR %USERPROFILE%\\shubble")
            print("  Set it temporarily:  set SHUBBLE_DIR=%USERPROFILE%\\shubble")
        else:
            print("  Set it permanently:  echo 'export SHUBBLE_DIR=$HOME/shubble' >> ~/.bashrc && source ~/.bashrc")
            print("  Set it temporarily:  export SHUBBLE_DIR=$HOME/shubble")
        sys.exit(1)

    check_deps()

    cmd = sys.argv[1] if len(sys.argv) > 1 else ""

    if cmd == "start":
        start_windows() if IS_NATIVE_WINDOWS else start_unix()
    elif cmd == "stop":
        stop_windows() if IS_NATIVE_WINDOWS else stop_unix()
    else:
        print(f"Usage: python {sys.argv[0]} start|stop")
        sys.exit(1)

if __name__ == "__main__":
    main()