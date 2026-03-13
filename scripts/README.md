# Shubble Development Launcher

This script starts and stops the full Shubble development environment using tmux and Docker.

---

## Files

- `easy_dev.sh` — Main development launcher script

---

## Requirements

Before using this script, make sure the following are installed:

- docker
- docker compose
- uv
- npm
- tmux

You can verify installation with:

```bash
docker --version
docker compose version
uv --version
npm --version
tmux -V