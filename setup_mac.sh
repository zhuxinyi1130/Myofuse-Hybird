#!/bin/zsh
set -euo pipefail

script_dir="${0:A:h}"
cd "$script_dir"

bootstrap_python="${BOOTSTRAP_PYTHON:-python3}"
if ! "$bootstrap_python" -c 'import sys; assert sys.version_info[:2] == (3, 9)' 2>/dev/null; then
  print -u2 "Python 3.9 is required. Selected interpreter: $bootstrap_python"
  exit 1
fi

"$bootstrap_python" -m venv .venv
export PIP_NO_CACHE_DIR=1
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-mac.txt

# Svetlana declares a CUDA-only CuPy dependency. The saved MyoFuse classifier
# needs its CNN2D class but does not need the Napari interface or CUDA on macOS.
.venv/bin/python -m pip install --no-deps napari-svetlana==1.6.1

.venv/bin/python - <<'PY'
import torch
from aicsimageio import AICSImage
from aicspylibczi import CziFile
from cellpose import models
from napari_svetlana.CNN2D import CNN2D

print("MyoFuse Hybrid Actin-structure runtime ready; PyTorch", torch.__version__)
PY
