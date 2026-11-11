#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if command -v docker >/dev/null 2>&1; then
    docker build -t isic-stylegan-pilot:2026-11-11 .
    docker run --rm --gpus 'device=0' --shm-size=4g -v "$PWD:/pilot" isic-stylegan-pilot:2026-11-11 --execute
else
    echo 'Using the current Python CUDA environment (PyTorch 2.11.0 + CUDA 13 and nvcc required).'
    python3 -m pip install -r requirements.txt
    python3 -u run_pilot.py --execute
fi
