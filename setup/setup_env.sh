#!/usr/bin/env bash
# DepthWizard — Phase 1 environment setup (Ubuntu 24.04, NVIDIA GPU)
#
# Creates a uv-managed virtualenv in ./.venv with:
#   - PyTorch (CUDA build)            -> model inference / training
#   - transformers + huggingface_hub  -> Depth Anything V2 Small
#   - Depth Anything 3 (from source)  -> DA3-Mono-Large
#   - rasterio, pyproj                -> GeoTIFF / CRS handling
#   - FastAPI, uvicorn                -> Stack B backend (used later)
#
# Every run writes a frozen lockfile + provenance record to setup/, so the
# exact environment can be reproduced on Kaggle/Colab.
#
# Usage (from repo root):
#   bash setup/setup_env.sh                 # default CUDA wheel channel cu126
#   TORCH_CUDA=cu128 bash setup/setup_env.sh
#   SKIP_DA3=1 bash setup/setup_env.sh      # skip DA3 source install
#   SKIP_GPU_CHECK=1 bash setup/setup_env.sh  # driver broken: build anyway, CUDA unverified

set -euo pipefail

PY_VERSION="${PY_VERSION:-3.11}"
TORCH_CUDA="${TORCH_CUDA:-cu126}"
SKIP_GPU_CHECK="${SKIP_GPU_CHECK:-0}"
CUDA_VERIFIED=false
DA3_REPO="https://github.com/ByteDance-Seed/depth-anything-3"
DA3_DIR="third_party/depth-anything-3"

log()  { printf '\n\033[1;34m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\n\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\n\033[1;31m[error]\033[0m %s\n' "$*"; exit 1; }

# ---------------------------------------------------------------- preflight
log "Checking NVIDIA driver"
if ! nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; then
  if [ "${SKIP_GPU_CHECK}" = "1" ]; then
    warn "nvidia-smi failed — SKIP_GPU_CHECK=1, continuing. CUDA will be marked unverified."
  else
    die "nvidia-smi failed (driver missing or no module for kernel $(uname -r)).
    Fix the driver, or rerun with SKIP_GPU_CHECK=1 to build without CUDA verification."
  fi
fi

log "Checking free disk space in $(pwd)"
FREE_GB=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
echo "Free: ${FREE_GB} GB"
if [ "${FREE_GB}" -lt 25 ]; then
  warn "Less than 25 GB free. PyTorch+CUDA (~6 GB), models (~2 GB), GAMUS and DEM tiles will not all fit."
fi

log "Checking uv"
if ! command -v uv >/dev/null 2>&1; then
  log "Installing uv (user-local)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

# ------------------------------------------------------------------- venv
if [ -x .venv/bin/python ]; then
  log "Reusing existing .venv ($(.venv/bin/python --version))"
else
  log "Creating .venv with Python ${PY_VERSION}"
  uv venv --python "${PY_VERSION}" .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# ------------------------------------------------------------------ torch
log "Installing PyTorch (${TORCH_CUDA} wheels)"
uv pip install torch torchvision --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

if python - <<'PY'
import torch, sys
ok = torch.cuda.is_available()
print(f"torch {torch.__version__} | CUDA build {torch.version.cuda} | cuda available: {ok}")
if not ok:
    sys.exit("CUDA not available to PyTorch. Check driver version vs wheel channel (try TORCH_CUDA=cu124).")
print("GPU:", torch.cuda.get_device_name(0))
PY
then
  CUDA_VERIFIED=true
elif [ "${SKIP_GPU_CHECK}" = "1" ]; then
  warn "PyTorch CUDA check failed — SKIP_GPU_CHECK=1, continuing (cuda_verified: false)."
else
  die "PyTorch CUDA check failed."
fi
TORCH_BEFORE=$(python -c "import torch; print(torch.__version__)")

# ------------------------------------------------------------- core stack
log "Installing core requirements"
uv pip install -r setup/requirements.txt

# --------------------------------------------------------------------- DA3
if [ "${SKIP_DA3:-0}" != "1" ]; then
  log "Installing Depth Anything 3 from source"
  mkdir -p third_party
  if [ ! -d "${DA3_DIR}/.git" ]; then
    git clone --depth 1 "${DA3_REPO}" "${DA3_DIR}"
  fi
  uv pip install -e "${DA3_DIR}"

  TORCH_AFTER=$(python -c "import torch; print(torch.__version__)")
  if [ "${TORCH_BEFORE}" != "${TORCH_AFTER}" ]; then
    die "DA3 install changed torch ${TORCH_BEFORE} -> ${TORCH_AFTER}. Stopping; not forcing it.
    Reinstall torch from the ${TORCH_CUDA} index or pin DA3's torch requirement, then rerun."
  fi
else
  warn "SKIP_DA3=1 — DA3 not installed; smoke test will skip it."
fi

# ------------------------------------------------------------- provenance
log "Writing lockfile and provenance"
uv pip freeze > setup/requirements.lock.txt
# nvidia-smi prints its failure message on stdout, so test the exit code, not stderr
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null) || DRIVER="UNAVAILABLE"
{
  echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "python: $(python --version 2>&1)"
  echo "torch_channel: ${TORCH_CUDA}"
  echo "torch: $(python -c 'import torch; print(torch.__version__)')"
  echo "driver: ${DRIVER}"
  echo "kernel: $(uname -r)"
  echo "cuda_verified: ${CUDA_VERIFIED}"
  if [ -d "${DA3_DIR}/.git" ]; then
    echo "da3_commit: $(git -C "${DA3_DIR}" rev-parse HEAD)"
  fi
} > setup/provenance.txt
cat setup/provenance.txt

log "Done. Next:
  source .venv/bin/activate
  python scripts/check_hardware.py
  python experiments/00_smoke/smoke_test_models.py --image <overhead_rgb.tif|png>"
