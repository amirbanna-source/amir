#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/environment.sh"
exec 9>"$STUDY_ROOT/.setup.lock"
flock 9
if [[ "$REFERENCE_DIR" == "$STUDY_ROOT/.cache/reference" ]]; then
    python3 "$STUDY_ROOT/bootstrap.py" "$REFERENCE_DIR" --fetch
fi
reference=/reference
script=/study/bootstrap.py
if [[ "$BACKEND" == native ]]; then
    script="$STUDY_ROOT/bootstrap.py"
    reference="$ISIC_REFERENCE_DIR"
fi
if [[ "${1:-}" == --verify-reference ]]; then
    # Metadata-only check using the host Python and pandas.
    python3 "$STUDY_ROOT/bootstrap.py" "$REFERENCE_DIR"
    exit
fi
setup_one gan
setup_one cnn
container_command cnn no python "$script" "$reference"
echo 'Software setup complete. Run the input preflight before allocating a GPU.'
