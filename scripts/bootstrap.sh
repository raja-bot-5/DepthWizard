#!/usr/bin/env bash
# DepthWizard bootstrap — runs automatically when the folder is opened in VS Code
# (see .vscode/tasks.json). Safe to re-run: every step is skipped once done.
#
#   1. git init (so Claude Code can track and review changes)
#   2. Python/CUDA environment      -> setup/setup_env.sh   (first run only, ~10-20 min)
#   3. Hardware/environment check   -> scripts/check_hardware.py
#   4. Claude Code                  -> installed if missing, then started with the next task
#
# Env flags:
#   DW_NO_CLAUDE=1   stop after the checks, don't launch Claude Code
#   DW_FORCE_SETUP=1 rebuild the environment even if it exists

set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
STATE=".dw_state"
mkdir -p "$STATE" runs/env

log()  { printf '\n\033[1;36m━━ DepthWizard ━━\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }

log "Project: $ROOT"

# ------------------------------------------------------------------ 1. git
if [ ! -d .git ]; then
  log "Initialising git repository"
  git init -q && git add -A && git commit -qm "Phase 1a: setup scripts and project brief" \
    || warn "git commit skipped (set git user.name / user.email)"
fi

# ---------------------------------------------------------- 2. environment
if [ "${DW_FORCE_SETUP:-0}" = "1" ] || [ ! -f "$STATE/setup_complete" ] || [ ! -x .venv/bin/python ]; then
  log "Step 1/3 — building Python environment (first run takes a while)"
  rm -f "$STATE/setup_complete"
  # pipefail is set, so this fails if setup_env.sh fails (not just tee)
  if bash setup/setup_env.sh 2>&1 | tee runs/env/setup.log; then
    date -u +%Y-%m-%dT%H:%M:%SZ > "$STATE/setup_complete"
  fi
  if [ ! -f "$STATE/setup_complete" ]; then
    warn "Environment setup failed — see runs/env/setup.log. Claude Code will be asked to fix it."
  fi
else
  log "Step 1/3 — environment already built ($(cat "$STATE/setup_complete")), skipping"
fi

# ---------------------------------------------------------- 3. hardware check
log "Step 2/3 — hardware / environment check"
if [ -x .venv/bin/python ]; then
  .venv/bin/python scripts/check_hardware.py --out runs/env/hardware.json
else
  python3 scripts/check_hardware.py --out runs/env/hardware.json || true
fi

# ---------------------------------------------------------- 4. Claude Code
if [ "${DW_NO_CLAUDE:-0}" = "1" ]; then
  log "DW_NO_CLAUDE=1 — stopping here."
  exit 0
fi

log "Step 3/3 — Claude Code"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v claude >/dev/null 2>&1; then
  log "Claude Code not found — installing (official installer)"
  curl -fsSL https://claude.ai/install.sh | bash
  export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v claude >/dev/null 2>&1; then
  warn "Claude Code still not on PATH. Open a new terminal and run: claude"
  exit 1
fi

if [ -f "$STATE/setup_complete" ]; then
  PROMPT="Read CLAUDE.md. The environment is built; results are in runs/env/setup.log and runs/env/hardware.json. \
Review them, update the Phase status in CLAUDE.md, then continue with the first unchecked phase. \
Explain your plan before editing code."
else
  PROMPT="Read CLAUDE.md. Environment setup FAILED — read runs/env/setup.log and runs/env/hardware.json, \
diagnose the root cause, propose a fix, and ask me before changing system packages."
fi

log "Launching Claude Code (sign in on first use)"
exec claude "$PROMPT"
