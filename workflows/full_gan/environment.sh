#!/usr/bin/env bash
# Shared Linux container support. Source this file from the launchers.
set -euo pipefail
STUDY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GAN_IMAGE='pytorch/pytorch@sha256:6e8a7a6dedf900096f90190f66f988e7e658cda4f1e6cbc7c17e3a38980a4f89'
CNN_IMAGE='tensorflow/tensorflow:2.20.0-gpu'
BACKEND="${ISIC_BACKEND:-auto}"
if [[ "$BACKEND" == auto ]]; then
    if command -v apptainer >/dev/null 2>&1; then BACKEND=apptainer
    elif command -v singularity >/dev/null 2>&1; then BACKEND=singularity
    elif command -v docker >/dev/null 2>&1; then BACKEND=docker
    else BACKEND=native
    fi
fi
case "$BACKEND" in apptainer|singularity|docker|native) ;; *) echo 'ISIC_BACKEND must be auto, apptainer, singularity, docker, or native.' >&2; exit 2;; esac
mkdir -p "$STUDY_ROOT/.containers" "$STUDY_ROOT/.deps" "$STUDY_ROOT/results"
mkdir -p "$STUDY_ROOT/data/real"
SOURCE_IMAGES="${ISIC_IMAGE_DIR:-$STUDY_ROOT/data/real}"
[[ -d "$SOURCE_IMAGES" ]] || { echo "Image directory does not exist: $SOURCE_IMAGES" >&2; exit 2; }
SOURCE_IMAGES="$(cd -- "$SOURCE_IMAGES" && pwd)"
REFERENCE_DIR="$(cd -- "$STUDY_ROOT/../.." && pwd)/artifacts"
if [[ ! -f "$REFERENCE_DIR/isic2019_binary_split.csv" ]]; then
    REFERENCE_DIR="$STUDY_ROOT/.cache/reference"
fi
if [[ "$BACKEND" == native ]]; then
    export ISIC_IMAGE_DIR="$SOURCE_IMAGES" ISIC_REFERENCE_DIR="$REFERENCE_DIR"
fi


ensure_image() {
    local kind="$1" image="$2"
    if [[ "$BACKEND" == apptainer || "$BACKEND" == singularity ]]; then
        if [[ ! -f "$STUDY_ROOT/.containers/$kind.sif" ]]; then
            "$BACKEND" pull "$STUDY_ROOT/.containers/$kind.sif" "docker://$image"
        fi
    elif [[ "$BACKEND" == docker ]]; then
        if ! docker image inspect "$image" >/dev/null 2>&1; then docker pull "$image"; fi
    fi
}

container_command() {
    local kind="$1" gpu="$2"; shift 2
    local image="$CNN_IMAGE"
    [[ "$kind" == gan ]] && image="$GAN_IMAGE"
    if [[ "$BACKEND" == apptainer || "$BACKEND" == singularity ]]; then
        local opts=()
        [[ "$gpu" == yes ]] && opts+=(--nv)
        "$BACKEND" exec "${opts[@]}" --cleanenv --bind "$STUDY_ROOT:/study" --bind "$SOURCE_IMAGES:/study/data/real:ro" --bind "$REFERENCE_DIR:/reference:ro" --pwd /study \
            --env "PYTHONPATH=/study/.deps/$kind" --env "ISIC_IMAGE_DIR=/study/data/real" --env "ISIC_REFERENCE_DIR=/reference" \
            --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}" \
            --env "OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}" \
            --env "TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK:-4}" \
            --env 'TF_FORCE_GPU_ALLOW_GROWTH=true' \
            "$STUDY_ROOT/.containers/$kind.sif" "$@"
    elif [[ "$BACKEND" == docker ]]; then
        local opts=()
        local allocated="${CUDA_VISIBLE_DEVICES:-0}"
        allocated="${allocated%%,*}"
        [[ "$gpu" == yes ]] && opts+=(--gpus "device=$allocated")
        docker run --rm "${opts[@]}" --user "$(id -u):$(id -g)" --shm-size=4g \
            --entrypoint '' -v "$STUDY_ROOT:/study" -w /study \
            -v "$SOURCE_IMAGES:/study/data/real:ro" -e "ISIC_IMAGE_DIR=/study/data/real" -v "$REFERENCE_DIR:/reference:ro" -e "ISIC_REFERENCE_DIR=/reference" -e "PYTHONPATH=/study/.deps/$kind" -e 'CUDA_VISIBLE_DEVICES=0' \
            -e "OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}" -e TF_FORCE_GPU_ALLOW_GROWTH=true \
            -e 'PYTHONUSERBASE=/study/.cache/python' -e 'TORCH_HOME=/study/.cache/torch' \
            -e 'XDG_CACHE_HOME=/study/.cache' "$image" "$@"
    else
        local interpreter="${CNN_PYTHON:-python3}"
        [[ "$kind" == gan ]] && interpreter="${GAN_PYTHON:-python3}"
        # Native interpreters must have their respective frameworks installed already.
        ( cd "$STUDY_ROOT"; PYTHONPATH="$STUDY_ROOT/.deps/$kind${PYTHONPATH:+:$PYTHONPATH}" \
            "$interpreter" "${@:2}" )
    fi
}

setup_one() {
    local kind="$1" image="$CNN_IMAGE"
    [[ "$kind" == gan ]] && image="$GAN_IMAGE"
    ensure_image "$kind" "$image"
    local signature
    signature="$(sha256sum "$STUDY_ROOT/requirements-$kind.txt" | cut -d ' ' -f 1)"
    local marker="$STUDY_ROOT/.deps/$kind/.requirements-sha256"
    if [[ ! -f "$marker" ]] || [[ "$(cat "$marker")" != "$signature" ]]; then
        mkdir -p "$STUDY_ROOT/.deps/$kind"
        local destination="/study/.deps/$kind" requirements="/study/requirements-$kind.txt"
        if [[ "$BACKEND" == native ]]; then
            destination="$STUDY_ROOT/.deps/$kind"; requirements="$STUDY_ROOT/requirements-$kind.txt"
        fi
        container_command "$kind" no python -m pip install --no-cache-dir --upgrade \
            --target "$destination" -r "$requirements"
        printf '%s\n' "$signature" > "$marker"
    fi
}

run_stage() {
    local kind="$1" stage="$2" gpu="${3:-yes}"
    local script=/study/full_workflow.py
    [[ "$BACKEND" == native ]] && script="$STUDY_ROOT/full_workflow.py"
    container_command "$kind" "$gpu" python -u "$script" "$stage"
}
