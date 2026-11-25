#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-}"
case "$MODE" in gan|gan-conventional|baseline|preflight) ;; *) echo 'Use gan, gan-conventional, baseline or preflight' >&2; exit 2;; esac
source "$(dirname -- "${BASH_SOURCE[0]}")/environment.sh"
exec 8>"$STUDY_ROOT/.workflow.lock"
flock -n 8 || { echo 'Another workflow is already running in this folder. Run the two conditions sequentially.' >&2; exit 2; }
bash "$STUDY_ROOT/SETUP.sh"
run_stage cnn preflight no
if [[ "$MODE" == preflight ]]; then exit; fi
if [[ "$MODE" == baseline ]]; then
    run_stage cnn baseline
    exit
fi
run_stage gan gan
run_stage gan pool
if [[ "$MODE" == gan ]]; then run_stage cnn cnn
else run_stage cnn cnn-conventional
fi
echo "Completed $MODE. Preserve the results folder. No test evaluation was run."
