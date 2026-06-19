# T4 / Linux x86_64 Docker Validation Checklist

This document describes how to perform the **real** Docker validation on an
environment that matches the competition grader.

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

---

## 8. Known Limitations at Time of Writing

| Item | Status |
|---|---|
| Mac ARM64 Docker build | Expected to fail — CUDA PyTorch wheels not available for ARM64 |
| T4 / Linux x86_64 Docker build | Not yet validated — this checklist covers it |
| EasyOCR OCR model files | Not yet baked into Docker image — OCR falls back to `tespit_edilemedi` |
| Vehicle type, color, driver actions | Not yet implemented — skeleton outputs hardcoded placeholders |
| Final FTR submission deadline | 28 June 2026, 17:00 Turkey time |

---

## 9. After Validation

Once the container passes all checks on a real Linux x86_64 + NVIDIA GPU machine:

1. Record the image size and runtime in the FTR report (§4 Çözümün Sınanması).
2. Export the image: `docker save teknofest/5g-road-safety:local | gzip > 5g_road_safety.tar.gz`
3. Verify the `.tar.gz` is within the 8 GB compressed limit.
4. Upload via KYS before the submission deadline.
