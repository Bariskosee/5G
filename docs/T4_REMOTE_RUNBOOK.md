# T4 Remote Docker Validation Runbook

Step-by-step operational guide for validating the FTR Docker image on a real
Linux x86_64 + NVIDIA GPU machine (RunPod T4, GCP T4 VM, university GPU server, etc.).

> For the high-level validation checklist, see [T4_DOCKER_VALIDATION.md](T4_DOCKER_VALIDATION.md).
> This runbook is the executable companion: exact commands, expected outputs, and troubleshooting.

---

## 1. Goal

Confirm that `docker build` completes, `docker run --gpus all` executes the plate-detection
inference, and `/app/data/output/results.json` passes `scripts/validate_results_json.py` —
all on a Linux x86_64 machine with an NVIDIA GPU, without any internet access after build time.

---

## 2. Success Criteria

The milestone is **PASS** only when every item below is true:

- [ ] `uname -m` returns `x86_64`
- [ ] `nvidia-smi` shows an NVIDIA GPU
- [ ] `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi` works
- [ ] `models/model_b_plate/best.pt` placed in repo before build
- [ ] `python3 scripts/check_docker_packaging.py` → 11/11 PASS
- [ ] `docker build` completes without error
- [ ] Image size recorded (target ≤ 8 GB compressed)
- [ ] `/app/models/model_b_plate/best.pt` exists inside container
- [ ] `docker run --gpus all` completes without crash
- [ ] `/app/data/output/results.json` is written
- [ ] `validate_results_json.py` exits 0
- [ ] Evidence folder created and saved

If any item fails, do not mark the milestone complete. Diagnose using §14.

---

## 3. Required Local Files

Before touching the remote machine, have these ready on your development machine:

| File | Notes |
|---|---|
| `models/model_b_plate/best.pt` | **Not in Git** (gitignored). Must be copied manually. ~21 MB. |
| A test video (e.g. `video_1.mp4`) | Any H.264/MP4 that OpenCV can open. The 4K/50 FPS videos from local tests work. |

> `models/model_b_plate/best.pt` is gitignored because `.gitignore` matches `models/**/*.pt`.
> **Never commit it.** The `.dockerignore` intentionally does NOT exclude `models/`, so
> `docker build` copies the local `best.pt` into the image automatically — but only if the
> file exists before you run `docker build`.

---

## 4. Remote Setup Check

Run these on the remote machine before cloning anything:

```bash
# Platform must be x86_64
uname -m
# Expected: x86_64

# NVIDIA driver and GPU must be visible
nvidia-smi
# Expected: table showing GPU name (e.g. Tesla T4), Driver Version, CUDA Version 12.x

# Docker must be installed
docker --version
# Expected: Docker version 24.x or higher

# NVIDIA Container Toolkit must be configured
docker info 2>&1 | grep -i runtime
# Expected: Runtimes: nvidia runc  (or similar)

# End-to-end GPU test in Docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
# Expected: same nvidia-smi table as host
```

If any of these fail, fix the environment before proceeding. See §14 for common causes.

---

## 5. Clone Repository

```bash
git clone https://github.com/Bariskosee/5G.git
cd 5G
git log --oneline -5
```

Expected: recent commit history including the FTR skeleton and packaging commits.

---

## 6. Copy Model Weight

`best.pt` is not on GitHub. Transfer it to the remote machine before building.

**Option A — scp from your local machine (run on local machine):**

```bash
scp models/model_b_plate/best.pt USER@REMOTE_IP:/tmp/best.pt
```

Then on the remote machine:

```bash
mkdir -p ~/5G/models/model_b_plate
cp /tmp/best.pt ~/5G/models/model_b_plate/best.pt
ls -lh ~/5G/models/model_b_plate/best.pt
# Expected: -rw-r--r-- ... 21M ... best.pt
```

**Option B — cloud provider file manager / web console:**

Upload `best.pt` through the cloud provider's UI, then move it:

```bash
mv /path/to/uploaded/best.pt ~/5G/models/model_b_plate/best.pt
```

**Verify it is in place:**

```bash
cd ~/5G
python3 scripts/check_docker_packaging.py
# Expected: [PASS] Local model weights exist: models/model_b_plate/best.pt  (21 MB)
```

> **Never commit best.pt.** It is gitignored for size and reproducibility reasons.
> If you need to share it between machines, use scp, cloud storage, or a private artifact store.

---

## 7. Copy Test Video

The Docker CMD expects the input video mounted at `/app/data/input/video.mp4` exactly.
You can mount any video at that path — the filename inside the container is always `video.mp4`.

Transfer a video to the remote machine:

```bash
# From local machine:
scp /path/to/video_1.mp4 USER@REMOTE_IP:/tmp/video_1.mp4

# Verify on remote:
ls -lh /tmp/video_1.mp4
```

You will mount this as `-v /tmp/video_1.mp4:/app/data/input/video.mp4` in the run command.

> Using a 4K/50 FPS video (similar to local smoke tests) is recommended for representative
> timing. Shorter or lower-resolution videos are fine for a quick pass/fail check.

---

## 8. Static Pre-Build Validation

```bash
cd ~/5G

# Static packaging check — no Docker daemon needed
python3 scripts/check_docker_packaging.py 2>&1 | tee /tmp/check_docker_packaging.txt
# Expected: PASS: 11/11 checks passed.
```

If any check fails, fix it before running `docker build`. Common issues:

- `best.pt` not in place → repeat §6
- Dockerfile modified incorrectly → check `git diff Dockerfile`

---

## 9. Build Docker Image

```bash
cd ~/5G

docker build -t teknofest/5g-road-safety:local . 2>&1 | tee /tmp/docker_build.log
```

This takes several minutes on first build (PyTorch CUDA wheels are large).
Subsequent builds reuse cached layers.

**Check image size:**

```bash
docker images teknofest/5g-road-safety:local | tee /tmp/docker_images.txt
```

> **Note on image size:** `docker images` reports the **uncompressed** virtual size.
> The competition limit (≤ 8 GB) applies to the **compressed** export size.
> To measure compressed size: `docker save teknofest/5g-road-safety:local | gzip | wc -c`
> (divide by 1073741824 for GB). This is slower but the accurate number for the grader.

---

## 10. Inspect Container (Model + CUDA Check)

Run a debug shell inside the container to confirm `best.pt` was copied in and
CUDA is accessible. This step does **not** need `--gpus all` for the file check,
but does need it for the CUDA check.

```bash
docker run --rm --gpus all --entrypoint /bin/bash \
  teknofest/5g-road-safety:local -lc "
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
```

Expected output:

```
-rw-r--r-- 1 root root 21M ...  /app/models/model_b_plate/best.pt
torch: 2.3.1+cu121
cuda_available: True
device: Tesla T4
vram_gb: 15.8
```

If `cuda_available: False`:
- Confirm you passed `--gpus all`
- Confirm NVIDIA Container Toolkit is installed and configured (`docker info | grep nvidia`)
- Confirm the host `nvidia-smi` works

---

## 11. Run Inference Container

```bash
mkdir -p /tmp/5g_docker_output

time docker run --rm --gpus all \
  -v /tmp/video_1.mp4:/app/data/input/video.mp4 \
  -v /tmp/5g_docker_output:/app/data/output \
  teknofest/5g-road-safety:local \
  2>&1 | tee /tmp/docker_run.log
```

Replace `/tmp/video_1.mp4` with the absolute path to your test video.

Expected:
- Container prints progress logs to stderr
- `time` output shows wall-clock elapsed (target ≤ 10 min for a ≤5 min video)
- Exit code 0

Check output was written:

```bash
ls -lh /tmp/5g_docker_output/results.json
```

---

## 12. Validate Output

```bash
cd ~/5G

# Schema validation
python3 scripts/validate_results_json.py /tmp/5g_docker_output/results.json \
  2>&1 | tee /tmp/validation.log
# Expected: OK: /tmp/5g_docker_output/results.json is valid.

# Human-readable check
python3 -m json.tool /tmp/5g_docker_output/results.json
```

The JSON must contain all three top-level keys:

```json
{
  "video_id": "video.mp4",
  "arac_bilgisi": { "tip": "...", "plaka": "...", "renk": "...", "confidence_score": ... },
  "tespitler": [ ... ]
}
```

> In the current FTR skeleton, `arac_bilgisi.tip` and `arac_bilgisi.renk` are placeholder
> values and `tespitler` may be empty. The schema is still valid — auto-grader scores by
> field presence and format, not by accuracy of the placeholder values.

---

## 13. Collect Evidence

Save everything needed for the FTR report (§4 Çözümün Sınanması):

```bash
mkdir -p /tmp/5g_t4_validation_evidence

# Logs from this run
cp /tmp/docker_build.log      /tmp/5g_t4_validation_evidence/
cp /tmp/docker_run.log        /tmp/5g_t4_validation_evidence/
cp /tmp/docker_images.txt     /tmp/5g_t4_validation_evidence/
cp /tmp/container_debug.txt   /tmp/5g_t4_validation_evidence/
cp /tmp/check_docker_packaging.txt /tmp/5g_t4_validation_evidence/
cp /tmp/validation.log        /tmp/5g_t4_validation_evidence/
cp /tmp/5g_docker_output/results.json /tmp/5g_t4_validation_evidence/

# GPU snapshot
nvidia-smi > /tmp/5g_t4_validation_evidence/nvidia_smi.txt

# List what was saved
find /tmp/5g_t4_validation_evidence -maxdepth 1 -type f | sort
```

Expected evidence folder:

```
/tmp/5g_t4_validation_evidence/
├── check_docker_packaging.txt  # 11/11 PASS
├── container_debug.txt         # best.pt ls + cuda_available: True
├── docker_build.log            # build output (no error at end)
├── docker_images.txt           # image name + size
├── docker_run.log              # inference log (exit 0)
├── nvidia_smi.txt              # GPU info
├── results.json                # schema-valid output
└── validation.log              # "OK: ... is valid."
```

> Transfer this folder back to your local machine for inclusion in the FTR report.
> **Do not commit these files to the repository.**

---

## 14. Troubleshooting

| Error / Symptom | Likely Cause | Fix |
|---|---|---|
| `torch==2.3.1` wheel not found during build | Not Linux x86_64, or wrong CUDA index | Use Linux x86_64 machine; Dockerfile uses `--index-url https://download.pytorch.org/whl/cu121` which only provides x86_64 wheels |
| `could not select device driver "nvidia" with capabilities: [[gpu]]` | NVIDIA Container Toolkit missing or misconfigured | Install `nvidia-container-toolkit`; run `sudo systemctl restart docker`; re-test with `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi` |
| `/app/models/model_b_plate/best.pt: No such file or directory` inside container | `best.pt` was not in place before `docker build`, or `.dockerignore` excludes `*.pt` | Confirm `ls models/model_b_plate/best.pt` before building; confirm `.dockerignore` has no `*.pt` or `models/` exclusion |
| `results.json` not created | `main.py` crash or output directory not mounted | Check `docker_run.log` for traceback; confirm `-v` volume mount path is absolute and exists |
| `validate_results_json.py` fails | Schema mismatch in output JSON | Check `validation.log` for the specific key/value that failed; fix `src/output/` modules |
| `EasyOCR` attempts internet download at runtime | OCR model files not baked into image | Expected in this milestone — `--disable-ocr` or OCR fallback to `tespit_edilemedi` is acceptable; full OCR packaging is a later milestone |
| `cuda_available: False` in container | `--gpus all` missing from run command, or Container Toolkit not configured | Always pass `--gpus all` for inference; verify with `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi` |

---

## 15. Pass/Fail Checklist

After completing all steps, confirm every box is checked:

- [ ] `uname -m` → `x86_64`
- [ ] `nvidia-smi` shows GPU on host
- [ ] `docker run --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi` works
- [ ] `models/model_b_plate/best.pt` present before build
- [ ] `check_docker_packaging.py` → 11/11 PASS
- [ ] `docker build` completed (check end of `docker_build.log` — no ERROR)
- [ ] Image size recorded in `docker_images.txt`
- [ ] `/app/models/model_b_plate/best.pt` visible in `container_debug.txt`
- [ ] `cuda_available: True` in `container_debug.txt`
- [ ] `docker run` exit code 0 (check end of `docker_run.log`)
- [ ] `results.json` exists in output volume
- [ ] `validation.log` contains "is valid."
- [ ] Evidence folder created with all 8 files

All 13 items checked → **Milestone 4 COMPLETE**.

> Record the pass date and GPU machine spec (GPU model, driver version, Docker version)
> in the FTR report for §4 Çözümün Sınanması.
