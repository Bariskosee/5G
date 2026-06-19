#!/usr/bin/env bash
# T4 Docker Validation — Remote Linux x86_64 + NVIDIA GPU
#
# Run this script on a Linux x86_64 machine with an NVIDIA GPU after:
#   1. Cloning the repository
#   2. Copying models/model_b_plate/best.pt into place (see docs/T4_REMOTE_RUNBOOK.md §6)
#   3. Uploading a test video to this machine (see docs/T4_REMOTE_RUNBOOK.md §7)
#
# Usage:
#   bash scripts/t4_docker_validation_commands.sh
#
# Override defaults with environment variables:
#   VIDEO_PATH=/abs/path/to/video.mp4 bash scripts/t4_docker_validation_commands.sh
#
# Do NOT run this on Mac ARM64 — the Docker build requires Linux x86_64 CUDA wheels.
#
# See docs/T4_REMOTE_RUNBOOK.md for the full step-by-step guide.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — override via environment variables
# ---------------------------------------------------------------------------
VIDEO_PATH="${VIDEO_PATH:-/tmp/video_1.mp4}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/5g_docker_output}"
IMAGE_NAME="${IMAGE_NAME:-teknofest/5g-road-safety:local}"
EVIDENCE_DIR="${EVIDENCE_DIR:-/tmp/5g_t4_validation_evidence}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================"
echo "T4 Docker Validation"
echo "  REPO:       $REPO_ROOT"
echo "  VIDEO_PATH: $VIDEO_PATH"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  IMAGE_NAME: $IMAGE_NAME"
echo "  EVIDENCE:   $EVIDENCE_DIR"
echo "========================================"

# ---------------------------------------------------------------------------
# Step 1: Platform check
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 1: Platform check ---"
ARCH="$(uname -m)"
echo "Architecture: $ARCH"
if [[ "$ARCH" != "x86_64" ]]; then
    echo "WARNING: This script targets Linux x86_64."
    echo "  Current arch is $ARCH. Docker build will likely fail."
    echo "  Continuing anyway — use a Linux x86_64 GPU machine for official validation."
fi

# ---------------------------------------------------------------------------
# Step 2: GPU check
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 2: GPU check ---"
nvidia-smi
echo "[OK] nvidia-smi passed"

# ---------------------------------------------------------------------------
# Step 3: Guard — model weight must exist
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 3: Model weight guard ---"
MODEL_PT="$REPO_ROOT/models/model_b_plate/best.pt"
if [[ ! -f "$MODEL_PT" ]]; then
    echo "ERROR: $MODEL_PT not found."
    echo "  Copy best.pt before running this script."
    echo "  See docs/T4_REMOTE_RUNBOOK.md §6 for instructions."
    exit 1
fi
ls -lh "$MODEL_PT"
echo "[OK] best.pt exists"

# ---------------------------------------------------------------------------
# Step 4: Guard — video must exist
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 4: Video guard ---"
if [[ ! -f "$VIDEO_PATH" ]]; then
    echo "ERROR: VIDEO_PATH=$VIDEO_PATH not found."
    echo "  Set VIDEO_PATH to an existing video file."
    echo "  Example: VIDEO_PATH=/tmp/video_1.mp4 bash $0"
    exit 1
fi
ls -lh "$VIDEO_PATH"
echo "[OK] Video exists"

# ---------------------------------------------------------------------------
# Step 5: Static packaging check
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 5: Static Docker packaging check ---"
cd "$REPO_ROOT"
python3 scripts/check_docker_packaging.py 2>&1 | tee /tmp/check_docker_packaging.txt
if ! grep -q "PASS: 11/11" /tmp/check_docker_packaging.txt; then
    echo "ERROR: check_docker_packaging.py did not pass 11/11. Fix before building."
    exit 1
fi
echo "[OK] 11/11 checks passed"

# ---------------------------------------------------------------------------
# Step 6: Docker build
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 6: Docker build ---"
docker build -t "$IMAGE_NAME" "$REPO_ROOT" 2>&1 | tee /tmp/docker_build.log
echo "[OK] docker build completed"

# ---------------------------------------------------------------------------
# Step 7: Image size
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 7: Image size ---"
docker images "$IMAGE_NAME" | tee /tmp/docker_images.txt
echo "NOTE: size shown above is uncompressed. Competition limit (<=8 GB) is for compressed export."
echo "  Compressed: docker save $IMAGE_NAME | gzip | wc -c  (run separately if needed)"

# ---------------------------------------------------------------------------
# Step 8: Container debug (best.pt + CUDA check)
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 8: Container debug ---"
docker run --rm --gpus all --entrypoint /bin/bash \
    "$IMAGE_NAME" -lc "
ls -lh /app/models/model_b_plate/best.pt
python3 - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda_available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
    print('vram_gb:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
PY
" 2>&1 | tee /tmp/container_debug.txt

if ! grep -q "cuda_available: True" /tmp/container_debug.txt; then
    echo "WARNING: cuda_available is not True inside container."
    echo "  Check NVIDIA Container Toolkit and --gpus all flag."
    echo "  Continuing to inference step..."
fi
echo "[OK] Container debug done"

# ---------------------------------------------------------------------------
# Step 9: Inference run
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 9: Inference run ---"
mkdir -p "$OUTPUT_DIR"
VIDEO_ABSPATH="$(realpath "$VIDEO_PATH")"
OUTPUT_ABSPATH="$(realpath "$OUTPUT_DIR")"

echo "Mounting: $VIDEO_ABSPATH -> /app/data/input/video.mp4"
echo "Mounting: $OUTPUT_ABSPATH -> /app/data/output"

time docker run --rm --gpus all \
    -v "$VIDEO_ABSPATH":/app/data/input/video.mp4 \
    -v "$OUTPUT_ABSPATH":/app/data/output \
    "$IMAGE_NAME" \
    2>&1 | tee /tmp/docker_run.log

echo "[OK] Inference run completed"
ls -lh "$OUTPUT_DIR/results.json"

# ---------------------------------------------------------------------------
# Step 10: Validate output JSON
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 10: Validate results.json ---"
cd "$REPO_ROOT"
python3 scripts/validate_results_json.py "$OUTPUT_DIR/results.json" \
    2>&1 | tee /tmp/validation.log

if ! grep -q "is valid" /tmp/validation.log; then
    echo "ERROR: results.json did not pass validation."
    echo "  See /tmp/validation.log for details."
    exit 1
fi
echo "[OK] results.json is valid"

# ---------------------------------------------------------------------------
# Step 11: Collect evidence
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 11: Collect evidence ---"
mkdir -p "$EVIDENCE_DIR"

cp /tmp/check_docker_packaging.txt "$EVIDENCE_DIR/"
cp /tmp/docker_build.log           "$EVIDENCE_DIR/"
cp /tmp/docker_images.txt          "$EVIDENCE_DIR/"
cp /tmp/container_debug.txt        "$EVIDENCE_DIR/"
cp /tmp/docker_run.log             "$EVIDENCE_DIR/"
cp /tmp/validation.log             "$EVIDENCE_DIR/"
cp "$OUTPUT_DIR/results.json"      "$EVIDENCE_DIR/"

nvidia-smi > "$EVIDENCE_DIR/nvidia_smi.txt"

echo "Evidence files:"
find "$EVIDENCE_DIR" -maxdepth 1 -type f | sort

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "PASS: All validation steps completed."
echo ""
echo "Evidence saved to: $EVIDENCE_DIR"
echo "Transfer to local machine for FTR report:"
echo "  scp -r USER@REMOTE_IP:$EVIDENCE_DIR /local/path/ftr_evidence/"
echo ""
echo "See docs/T4_REMOTE_RUNBOOK.md §15 for the full pass/fail checklist."
echo "========================================"
