#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/environment.sh"
exec 9>"$STUDY_ROOT/.setup.lock"
flock 9
setup_one gan
setup_one cnn
script=/study/bootstrap.py
[[ "$BACKEND" == native ]] && script="$STUDY_ROOT/bootstrap.py"
container_command cnn no python "$script"
echo 'Software setup complete. Full training still requires an allocated NVIDIA GPU and original images in data/real/.'
