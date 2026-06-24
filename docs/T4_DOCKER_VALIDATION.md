# T4 / Linux x86_64 Docker Validation Checklist

This document describes how to perform the **real** Docker validation on an
environment that matches the competition grader.

> For the full step-by-step operational guide (exact commands, evidence collection,
> troubleshooting), see [T4_REMOTE_RUNBOOK.md](T4_REMOTE_RUNBOOK.md).

> Mac ARM64 Docker builds are expected to fail (CUDA PyTorch wheels not available
> for ARM64). This does NOT mean the Dockerfile is wrong. The official grader
> environment is Linux x86_64 + NVIDIA Tesla T4.

---

## 1. Required Environment

| Requirement | Value |
|---|---|
| OS | Linux x86_64 (e.g. Ubuntu 22.04) |
| NVIDIA driver | ≥ 525 (CUDA 12.1 compatible) |
| Docker | ≥ 24.0 |
| NVIDIA Container Toolkit | installed and configured |
| GPU | NVIDIA Tesla T4 or equivalent CUDA-capable GPU |

Verify GPU visibility before building:

```bash
nvidia-smi
# Expected: Tesla T4 listed, Driver Version, CUDA Version 12.x
```

---

## 2. Pre-Build Checks

Clone the repository and place the model weights:

```bash
git clone https://github.com/Bariskosee/5G.git
cd 5G
```

> `models/model_b_plate/best.pt` is **gitignored** and not included in the clone.
> Copy it into place before building:

```bash
# Replace /path/to/best.pt with the actual location of the trained weights file.
cp /path/to/best.pt models/model_b_plate/best.pt
ls -lh models/model_b_plate/best.pt
```

Run the static packaging check (no Docker daemon needed):

```bash
python scripts/check_docker_packaging.py
# Expected: 11/11 checks passed.
```

---

## 3. Build

```bash
docker build -t teknofest/5g-road-safety:local .
```

Verify image size is within the 8 GB limit:

```bash
docker images teknofest/5g-road-safety
# IMAGE SIZE column must show ≤ 8 GB
```

---

## 4. Run

```bash
mkdir -p /tmp/5g_docker_output

docker run --rm --gpus all \
  -v /absolute/path/to/video.mp4:/app/data/input/video.mp4 \
  -v /tmp/5g_docker_output:/app/data/output \
  teknofest/5g-road-safety:local
```

Replace `/absolute/path/to/video.mp4` with the path to a test video on the host.

---

## 5. Validate Output

```bash
# Schema validation
python scripts/validate_results_json.py /tmp/5g_docker_output/results.json

# Human-readable check
python -m json.tool /tmp/5g_docker_output/results.json
```

Expected: validator prints `OK: ... is valid.`

The JSON must contain all three top-level keys:

```json
{
  "video_id": "video.mp4",
  "arac_bilgisi": { "tip": "...", "plaka": "...", "renk": "...", "confidence_score": 0.0 },
  "tespitler": []
}
```

---

## 6. Container Debug (if needed)

If the container exits with an error, run an interactive debug shell:

```bash
docker run --rm --entrypoint /bin/bash teknofest/5g-road-safety:local -lc \
  "ls -lh /app/models/model_b_plate/best.pt && \
   python3 -c 'import torch; print(\"CUDA available:\", torch.cuda.is_available())'"
```

Expected output:
```
-rw-r--r-- 1 root root 21M ...  /app/models/model_b_plate/best.pt
CUDA available: True
```

If `CUDA available: False` → verify NVIDIA Container Toolkit and `--gpus all` flag.

---

## 7. Expected Results

| Check | Expected |
|---|---|
| `/app/data/output/results.json` exists | ✅ |
| JSON validates against schema | ✅ |
| No crash from missing model file | ✅ |
| No crash if EasyOCR files not packaged | ✅ (fallback: `tespit_edilemedi`) |
| `torch.cuda.is_available()` | `True` |
| Runtime for a ≤5 min video at frame-stride 10 | ≤ 4 min (well within 10 min limit) |
| Image size (`docker images`) | Reported as **uncompressed** virtual size. Competition limit (≤ 8 GB) applies to compressed export. Measure with `docker save IMAGE \| gzip \| wc -c`. |

---

## 8. Known Limitations / Status

| Item | Status |
|---|---|
| Mac ARM64 Docker build | Expected to fail — CUDA PyTorch wheels not available for ARM64 |
| T4 / Linux x86_64 Docker build | **Validated on Lightning AI Tesla T4 on 2026-06-24** |
| EasyOCR OCR model files | ✅ Fixed 2026-06-25 — `/app/easyocr_models` path, `tr`+`en` packs baked in; no double-slash bug |
| Vehicle type, color, driver actions | ✅ Implemented 2026-06-25 — HSV+Lab color, COCO YOLOv8m vehicle type, MediaPipe esneme all wired into `main.py` |
| Final FTR submission deadline | 28 June 2026, 17:00 Turkey time |

For detailed troubleshooting (CUDA not found, best.pt missing in container, etc.)
see [T4_REMOTE_RUNBOOK.md §14](T4_REMOTE_RUNBOOK.md#14-troubleshooting).

---

## 9. Actual T4 Validation Result — 2026-06-24

The Docker image was validated on Lightning AI Studio using a Linux x86_64 environment
with an NVIDIA Tesla T4 GPU.

| Check | Result |
|---|---|
| Host architecture (`uname -m`) | `x86_64` |
| GPU | NVIDIA Tesla T4 (Driver 580.159.03) |
| Docker version | 28.0.1 |
| CUDA base image GPU test (`docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`) | Passed |
| Static packaging check (`scripts/check_docker_packaging.py`) | 11/11 passed |
| Docker build | Passed |
| Docker image virtual size | ~7.51 GB |
| Docker run with `--gpus all` | Passed |
| Output file produced | `/app/data/output/results.json` |
| JSON schema validation (`scripts/validate_results_json.py`) | `OK: ... is valid.` |
| Compressed image size | ~3.4 GB |
| Compressed archive SHA256 | `a7cb1036fa590749e7206f1b06c5154d5ee7e2993556b91ad9a07980afb46a60` |
| `gzip -t` on final archive | Exit code 0 |
| Final image archive on Mac | `/Users/bariskose/Desktop/5g_road_safety.tar.gz` |
| T4 evidence folder on Mac | `/Users/bariskose/Desktop/5g_t4_validation_evidence/` |

Evidence files committed to `docs/ftr_report/milestone4_t4_evidence/`
(except `docker_build.log` which is large and archived on Mac only):

```
check_docker_packaging.txt
container_debug.txt
docker_images.txt
docker_run.log
final_image_size.txt
image_sha256.txt
nvidia_smi.txt
results.json
validation.log
```

`docker_build.log` — archived at `/Users/bariskose/Desktop/5g_t4_validation_evidence/docker_build.log` (not committed — large file).

The JSON produced during this validation run was:

```json
{
  "video_id": "video.mp4",
  "arac_bilgisi": {
    "tip": "sedan",
    "plaka": "tespit_edilemedi",
    "renk": "beyaz",
    "confidence_score": 0.01
  },
  "tespitler": []
}
```

**This validates Docker packaging / runtime / schema compliance. It does NOT represent
final AI model output quality.**

> **Note on large file transfer:** `scp` of a single 3.5 GB archive failed mid-transfer.
> The working approach was to split the archive into 500 MB parts with
> `split -b 500M`, download the parts, then concatenate locally with
> `cat part_* > archive.tar.gz`. Use `/teamspace/studios/this_studio/` (not `/tmp`)
> as the persistent handoff directory on Lightning, since `/tmp` does not survive
> long enough for multi-step downloads.

---

## 10. After Validation

Once the container passes all checks on a real Linux x86_64 + NVIDIA GPU machine:

1. Collect evidence logs into `/tmp/5g_t4_validation_evidence/`:
   - `docker_build.log`, `docker_run.log`, `docker_images.txt`
   - `nvidia_smi.txt` (`nvidia-smi > nvidia_smi.txt`)
   - `container_debug.txt` (best.pt ls + `torch.cuda.is_available()`)
   - `check_docker_packaging.txt`, `validation.log`, `results.json`
2. Record the image size and runtime in the FTR report (§4 Çözümün Sınanması).
3. Export the image: `docker save teknofest/5g-road-safety:local | gzip > 5g_road_safety.tar.gz`
4. Verify the `.tar.gz` is within the 8 GB compressed limit.
5. Upload via KYS before the submission deadline.

See [T4_REMOTE_RUNBOOK.md §13](T4_REMOTE_RUNBOOK.md#13-collect-evidence) for the full evidence collection commands.
