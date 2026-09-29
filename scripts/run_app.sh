#!/usr/bin/env bash
# DepthWizard: start the local app (FastAPI backend + Three.js viewer) with one command.
#
#   bash scripts/run_app.sh              # http://127.0.0.1:8000/app/ , opens your browser
#   DW_PORT=8080 bash scripts/run_app.sh
#   DW_NO_BROWSER=1 bash scripts/run_app.sh
#
# Needs the environment from `bash setup/setup_env.sh` (once). Runs fully offline once the model weights and
# the elevation data for your area are cached (the first run of each needs internet).
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${DW_PORT:-8000}"
HOST="127.0.0.1"

if [ ! -x .venv/bin/python ]; then
  echo "No Python environment yet. Run this once (downloads ~6 GB: PyTorch + CUDA libraries):"
  echo "    bash setup/setup_env.sh"
  exit 1
fi
if ! .venv/bin/python -c "import fastapi, depth_anything_3" 2>/dev/null; then
  echo "The environment is incomplete (fastapi or depth_anything_3 missing). Re-run: bash setup/setup_env.sh"
  exit 1
fi
if (exec 3<>"/dev/tcp/$HOST/$PORT") 2>/dev/null; then
  echo "Port $PORT is already in use. Stop the other program or choose another port: DW_PORT=8080 bash scripts/run_app.sh"
  exit 1
fi

CUDA=$(.venv/bin/python -c "import torch; print('yes' if torch.cuda.is_available() else 'no')" 2>/dev/null || echo no)
[ "$CUDA" = "yes" ] || echo "Note: no CUDA GPU visible to PyTorch. The app still runs, but on the CPU (much slower)."

URL="http://$HOST:$PORT/app/"
echo "Starting DepthWizard on $URL  (Ctrl+C to stop)"
if [ "${DW_NO_BROWSER:-0}" != "1" ] && command -v xdg-open >/dev/null 2>&1; then
  ( for _ in $(seq 1 60); do
      if curl -s -o /dev/null "http://$HOST:$PORT/health"; then xdg-open "$URL" >/dev/null 2>&1; break; fi
      sleep 1
    done ) &
fi
export PYTHONPATH="src:."
export DW_JOBS_DIR="${DW_JOBS_DIR:-runs/jobs}"
exec .venv/bin/python -m uvicorn backend.app:app --host "$HOST" --port "$PORT"
