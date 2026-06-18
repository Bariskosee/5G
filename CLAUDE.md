# CLAUDE.md

> This file is read automatically by Claude Code at the start of every session
> in this repository. Keep it accurate and concise — when something changes
> in the project (architecture, constraints, deadlines), update this file
> FIRST, then everything else.

---

## 1. Project Identity

**Competition.** TEKNOFEST 2026 — "5G & Yapay Zekâ ile Akıllı Yol Güvenliği
Yarışması," organized by Turkcell. We are team [TEAM_NAME].

**Current phase.** We are in **Phase 2 (Final Tasarım Raporu — FTR).**
- Phase 1 (PDR / Ön Tasarım Raporu) is **complete and accepted**.
- FTR submission deadline: **28 June 2026, 17:00 (Türkiye time)** via KYS.
  (This was extended from the original 14 June deadline.)
- After FTR: Phase 3 = on-site final competition (Aug-Sep 2026, Şanlıurfa).

**Phase 1 task.** Build an AI pipeline that processes offline video files
and outputs a `results.json` matching the competition schema exactly. The
mobile app, 5G API integration, TOGG-specific features, Number Verification
and Quality on Demand APIs all belong to Phase 2/3 of the COMPETITION and
are **out of scope** for the FTR submission. Do not propose adding them.

**Deliverables for FTR.**
1. A Docker image that contains the AI pipeline and all model weights.
2. The Final Design Report (text document).
3. Source code repository (this one).

The Docker image is what the auto-grader runs. The report (50% weight)
explains the architecture and validates the team's engineering quality.

---

## 2. Hard Runtime Constraints

The auto-grader runs your container in this exact environment. Violating any
of these is an automatic failure mode:

| Constraint | Value | Implication |
| --- | --- | --- |
| Base image | `nvidia/cuda:12.1.0-base-ubuntu22.04` | Use this exact tag in Dockerfile |
| GPU | NVIDIA Tesla T4 | 16 GB VRAM, FP16 supported, no FP8 |
| System RAM | 16 GB | Avoid loading entire video into memory |
| vCPU | 4 | Multi-process I/O is fine; CPU inference is not |
| Image size | **≤ 8 GB** | Compressed size of the final image |
| Wall-clock runtime | **≤ 10 minutes per video** | Budget is hard. See §5 for breakdown. |
| Internet at runtime | **NONE** | All weights, models, OCR data must be inside the image |
| Internet at build | Available | All `pip install` happens at build time |
| Output path | `/app/data/output/results.json` | Hardcoded by grader |
| Input path | Provided at runtime under `/app/data/input/` | Single video file per run |

**Therefore:**
- Pre-download all model weights into the image. No `model.download()` at runtime.
- Pre-download all EasyOCR language packs into the image (`tr`, `en`).
- Pre-download MediaPipe model files (`.task` or `.tflite`).
- Pin every Python dependency to an exact version.
- Use `opencv-python-headless`, never `opencv-python` (saves ~150 MB and avoids GUI deps).

---

## 3. Output Schema (CRITICAL — wrong key = 0 points)

The competition auto-grader scores by exact key match. **Memorize this schema.**

```json
{
  "arac_bilgisi": {
    "tip": "sedan",
    "plaka": "34ABC123",
    "renk": "beyaz",
    "confidence_score": 0.87
  },
  "tespitler": [
    {
      "zaman_saniye": 4.2,
      "kategori": "sofor_eylemi",
      "etiket": "telefonla_konusma",
      "confidence_score": 0.81
    }
  ]
}
```

**Allowed values.** All values are ASCII, lowercase, underscore-separated.
**No Turkish characters (ş, ç, ğ, ü, ö, ı) anywhere in any label.**

| Field | Allowed values |
| --- | --- |
| `arac_bilgisi.tip` | `sedan`, `suv`, `hatchback`, `pickup`, `minibus`, `panelvan`, `kamyon` |
| `arac_bilgisi.renk` | `beyaz`, `siyah`, `gri`, `kirmizi`, `mavi`, `sari`, `yesil`, `turuncu`, `kahverengi` |
| `arac_bilgisi.plaka` | Turkish plate format (e.g. `34ABC123`) OR `tespit_edilemedi` if OCR failed |
| `tespitler[].kategori` | `sofor_eylemi`, `nesneler`, `yolcular` |
| `tespitler[].etiket` (kategori=sofor_eylemi) | `arkaya_bakma`, `esneme`, `sigara_icme`, `su_icme`, `telefonla_konusma`, `slalom`, `etrafa_bakinma`, `emniyet_kemeri_ihlali` |
| `tespitler[].etiket` (kategori=nesneler) | `teknocan`, `bilgisayar` |
| `tespitler[].etiket` (kategori=yolcular) | `arka_koltuk_1`, `arka_koltuk_2`, `on_koltuk` |
| `confidence_score` | float in `[0.0, 1.0]` |
| `zaman_saniye` | float, video time in seconds |

**Validation.** `scripts/validate_results_json.py` checks all of this.
Run it before every Docker build:

```bash
python scripts/validate_results_json.py outputs/results.json
```

If the validator fails, the auto-grader will fail. Fix it before building.

---

## 4. Architecture (Hybrid — 2 trained models + 5 helper modules)

We do NOT use a single end-to-end model. The pipeline is intentionally
hybrid because each detection task has a different geometry:

```
INPUT VIDEO
    │
    ├──> frame sampler (every 5 frames)
    │
    ├──> Model A: YOLOv8m unified detector
    │        ├── vehicle types (sedan/suv/hatchback/pickup/minibus/panelvan/kamyon)
    │        ├── driver objects (phone/cigarette/drink/no_seatbelt)
    │        ├── general objects (teknocan/laptop)
    │        └── person (used for passenger ROI)
    │
    ├──> Model B: YOLOv8s plate detector → license_plate bbox
    │        └──> EasyOCR (lazy, only when bbox conf > 0.7) → plate text
    │              └──> regex normalize → arac_bilgisi.plaka
    │
    ├──> MediaPipe FaceLandmarker (on driver crop)
    │        ├── mouth aspect ratio → esneme
    │        ├── head yaw → arkaya_bakma
    │        └── head yaw temporal pattern → etrafa_bakinma
    │
    ├──> Vehicle bbox tracker (Norfair or ByteTrack)
    │        └── lateral oscillation → slalom
    │
    ├──> HSV+Lab color analyzer (on vehicle bbox)
    │        └── arac_bilgisi.renk
    │
    └──> ROI mapper (on Model A person bboxes)
            └── on_koltuk / arka_koltuk_1 / arka_koltuk_2

OUTPUT: aggregated arac_bilgisi (one per video) + tespitler list → results.json
```

### Why this split?

- **Object detection** (Model A, B): tasks where bounding boxes are the right tool.
- **OCR**: tasks where text recognition is the right tool. Don't make YOLO read text.
- **Landmarks**: behaviors defined by face/head geometry (yawning, looking back). Don't make YOLO learn these — data is scarce and accuracy is low.
- **Tracking**: behaviors defined by motion patterns (slalom). Bbox over time, not appearance.
- **Color**: pixel statistics on a crop. Don't make YOLO learn colors.
- **ROI**: rule-based geometric assignment. Don't make YOLO learn seat positions.

### Why YOLOv8 specifically?

Production-ready, well-documented, robust pretrained weights, FP16 on T4
with no extra effort, simple Ultralytics API. Not YOLOv10 (newer but
ecosystem is thinner, marginal accuracy gain not worth the risk this close
to the deadline).

---

## 5. The 10-Minute Budget

Assuming a 5-minute video at 30 fps = ~9000 frames. With `frame_sample_every_n=5`,
we process ~1800 frames.

Per-sampled-frame budget on T4:

| Operation | Per-frame time | 1800 frames total |
| --- | --- | --- |
| Decode + resize | ~3 ms | ~5 sec |
| Model A (YOLOv8m, FP16, 640×640) | ~14 ms | ~25 sec |
| Model B (YOLOv8s, FP16, 640×640) | ~8 ms | ~15 sec |
| MediaPipe (face only, driver crop) | ~12 ms | ~22 sec |
| HSV+Lab color (per vehicle bbox) | ~2 ms | ~4 sec |
| Tracker update | ~3 ms | ~6 sec |
| ROI mapping | ~1 ms | ~2 sec |
| **EasyOCR (lazy, ~30 calls per video)** | ~80 ms × 30 | ~2.4 sec |
| Aggregation + JSON write | — | ~1 sec |
| Total compute | | **~80 sec** |
| Buffer for I/O, warmup, GC | | ~120 sec |
| **Total wall-clock estimate** | | **~3-4 min** |

This leaves comfortable margin. If we add operations, this table is the
place to estimate cost first.

**Optimizations already baked in:**
- Frame sampling (every 5)
- Lazy OCR (only on high-conf plates, cooldown 30 frames)
- FP16 inference for both YOLOs (T4 has no FP8)
- `opencv-python-headless`
- Single CUDA context shared across YOLO + MediaPipe

**Optimizations NOT to use yet:**
- TensorRT export (added complexity, often breaks T4 compatibility)
- INT8 quantization (degrades accuracy, hard to validate in 10 days)
- Multi-process pipelining (adds RAM pressure)

If runtime is ever > 8 minutes in testing, revisit these in this order:
TensorRT → smaller Model A (YOLOv8s instead of m) → coarser frame sampling.

---

## 6. Repository Layout

```
.
├── CLAUDE.md                    # ← you are here
├── README.md                    # human onboarding
├── Dockerfile                   # built last, after pipeline works locally
├── .dockerignore
├── .gitignore
├── requirements.txt             # pinned versions only
├── configs/
│   ├── model_a_classes.yaml     # Model A: 14 classes (internal names)
│   ├── model_b_classes.yaml     # Model B: 1 class (license_plate)
│   ├── final_label_mapping.yaml # Model class → competition JSON label
│   └── thresholds.yaml          # All runtime parameters
├── docs/
│   ├── annotation_guidelines.md # Manual labeling rules
│   ├── data_sources.md          # Per-source tracking table
│   └── architecture.md          # Long-form arch document (FTR-ready)
├── datasets/                    # gitignored
│   ├── model_a_unified/
│   │   ├── images/{train,val,test}/
│   │   ├── labels/{train,val,test}/
│   │   └── data.yaml
│   └── model_b_plate/
│       └── (same structure)
├── models/                      # gitignored, large .pt files
│   ├── model_a_best.pt
│   ├── model_b_best.pt
│   └── mediapipe/face_landmarker.task
├── src/
│   ├── __init__.py
│   ├── predict.py               # Main entrypoint — called by Docker CMD
│   ├── pipeline.py              # Orchestrates one video → results.json
│   ├── detection/
│   │   ├── model_a.py           # YOLOv8m wrapper
│   │   └── model_b.py           # YOLOv8s plate wrapper
│   ├── ocr/
│   │   ├── reader.py            # EasyOCR wrapper + lazy invocation
│   │   └── plate_regex.py       # Turkish plate normalization
│   ├── landmark/
│   │   ├── face.py              # MediaPipe FaceLandmarker
│   │   ├── mouth.py             # Mouth aspect ratio → esneme
│   │   └── head_pose.py         # Yaw → arkaya_bakma / etrafa_bakinma
│   ├── tracking/
│   │   └── slalom.py            # Vehicle bbox temporal oscillation
│   ├── color/
│   │   └── hsv_lab.py           # Color classification
│   ├── roi/
│   │   ├── seat_assignment.py   # Person bbox → seat label
│   │   └── driver_proximity.py  # Object-near-driver check for sofor_eylemi
│   ├── output/
│   │   ├── formatter.py         # Build the final dict
│   │   ├── aggregator.py        # Per-frame → per-video aggregation
│   │   └── confidence.py        # Vehicle confidence formula
│   └── utils/
│       ├── video.py             # Frame iteration with sampling
│       ├── config.py            # YAML loader
│       └── logger.py            # Structured logging
├── scripts/
│   ├── validate_results_json.py # Schema validator — run before every build
│   ├── remap_labels.py          # Source class names → Model A names
│   ├── split_dataset.py         # 80/10/10 split per source
│   ├── train_model_a.py         # YOLOv8m fine-tuning entry
│   ├── train_model_b.py         # YOLOv8s plate fine-tuning entry
│   └── benchmark_runtime.py     # Measure wall-clock on a test video
├── tests/
│   ├── test_validator.py        # Test the schema validator itself
│   ├── test_label_mapping.py    # Test config mapping correctness
│   ├── test_plate_regex.py      # Test Turkish plate normalization
│   └── test_output_formatter.py # Test JSON output building
└── outputs/                     # gitignored
    └── results.json             # Latest pipeline output
```

---

## 7. Conventions

### Code style
- **Python 3.10+**. Use modern syntax: `list[int]`, `dict[str, Any]`, `|` for unions.
- **Type hints everywhere.** No `Any` unless genuinely opaque.
- Format with `black` (line length 100). Lint with `ruff`.
- Imports: stdlib, then third-party, then local. Sorted alphabetically.

### Configuration
- **Never hardcode** thresholds, paths, class names. Read from
  `configs/*.yaml` via `src/utils/config.py`.
- If you need a new parameter, add it to `thresholds.yaml` and document the
  default in a comment.

### Labels and class names
- The 14 Model A class names are the **internal** model vocabulary.
- The competition JSON labels are produced ONLY via `final_label_mapping.yaml`.
- Never write a hardcoded string like `"telefonla_konusma"` outside of the
  mapping module. Always go through the label mapper.

### Imports inside the package
```python
from src.detection.model_a import ModelADetector
from src.output.formatter import build_results_json
```
Not `from detection.model_a import ...`. Always full path from project root.

### Tests
- Every config-driven module gets unit tests with fixed inputs.
- The validator script is itself tested with fixtures in `tests/test_validator.py`.
- Run tests with `pytest tests/` before opening a PR.

### Logging
- Use the project logger (`src/utils/logger.py`), not `print()`.
- One line per detected event. Format:
  `[t=4.20s] sofor_eylemi/telefonla_konusma conf=0.81 bbox=[x,y,x,y]`
- This log is part of the "automated proof" requirement (FR-07).

### Commits and branches
- Branch naming: `feature/<short-name>`, `fix/<short-name>`, `docs/<short-name>`.
- Commit messages: imperative mood, ≤ 72 chars. "Add OCR cooldown" not
  "Added OCR cooldown" or "Adding OCR cooldown".
- Never commit `.pt` weights, video files, or anything in `datasets/`.

---

## 8. Common Tasks — How to Approach Them

When the user asks for one of these, follow the linked checklist.

### "Add a new detection class"
1. Update `configs/model_a_classes.yaml` (add to `names:` and increment `nc`).
2. Update `configs/final_label_mapping.yaml` (add `model_to_json` entry).
3. Decide if it needs ROI/proximity (sofor_eylemi usually does).
4. Add training data to `datasets/model_a_unified/`.
5. Retrain Model A.
6. Update `tests/test_label_mapping.py`.
7. **Confirm** the new label is in the competition's allowed enum (§3).
   If not, do NOT add it.

### "Improve detection accuracy for class X"
1. Check the per-class confusion matrix in the validation report first.
2. Diagnose: data scarcity? bad labels? confusing class boundary?
3. Augmentation (mosaic, mixup, brightness, blur) — try first, free win.
4. Lower confidence threshold in `thresholds.yaml > detection.per_class_conf`.
5. Add training data — last resort, expensive.

### "Reduce Docker image size"
1. Audit with `docker history <image>` — find the largest layer.
2. Common wins:
   - `opencv-python-headless` not `opencv-python`
   - `torch` CPU vs CUDA — we need CUDA, no choice
   - `--no-cache-dir` on every `pip install`
   - Combine RUN layers
   - Multi-stage build (build deps in stage 1, copy only runtime to stage 2)
3. NEVER remove EasyOCR or MediaPipe model files — these are required at runtime.

### "Reduce runtime under 10 minutes"
See §5 for the optimization order. Always measure with
`scripts/benchmark_runtime.py` first; don't optimize blindly.

### "Generate / fix / regenerate results.json"
1. Run the pipeline: `python src/predict.py --input <video> --output outputs/results.json`
2. **Always validate:** `python scripts/validate_results_json.py outputs/results.json`
3. If validation fails, fix the formatter, not the validator.

### "Build the Docker image"
1. Make sure local pipeline runs end-to-end first.
2. Make sure `validate_results_json.py` passes on the local output.
3. Build: `docker build -t teknofest-road-safety:latest .`
4. Check size: `docker images teknofest-road-safety` — must be ≤ 8 GB.
5. Test: `docker run --gpus all -v $(pwd)/data:/app/data teknofest-road-safety:latest`
6. Validate the container's output: `python scripts/validate_results_json.py data/output/results.json`

### "Write a section of the FTR report"
1. The report goes in `docs/ftr_report/` (separate from CLAUDE.md scope).
2. Use English (team preference). Translate to Turkish only at final step.
3. Cite all datasets with license info from `docs/data_sources.md`.
4. Architecture section: use diagrams from `docs/architecture.md` as source.

---

## 9. Common Mistakes — DO NOT do these

These have been considered and rejected. If you find yourself proposing
any of them, stop and consult the team first.

| Mistake | Why it's wrong |
| --- | --- |
| Hardcoding Turkish label strings in `src/` | Bypasses `final_label_mapping.yaml`. One typo = 0 points. |
| Adding Turkish characters anywhere in output | Auto-grader rejects non-ASCII labels. |
| Calling YOLO on every frame | Blows the 10-minute budget. Sample every 5. |
| Calling OCR on every plate detection | Same. Use lazy OCR with cooldown. |
| Training YOLO to detect "esneme" or "arkaya_bakma" | Data is too scarce, MediaPipe is the right tool. |
| Using `opencv-python` instead of `opencv-python-headless` | ~150 MB bloat + GUI deps that fail in headless Docker. |
| Adding TensorRT, ONNX export, or INT8 quantization without measuring | Often breaks T4 compatibility. Only do this if runtime is genuinely too slow. |
| Loading the full video into RAM | 16 GB limit. Use the frame iterator in `src/utils/video.py`. |
| `pip install` at container runtime | No internet at runtime. All installs at build. |
| Downloading model weights with `model.download()` at runtime | Same. Bake weights into the image. |
| Adding new competition labels not in the official allowed list | Auto-grader rejects them. |
| Switching to a fundamentally different model architecture | We have ~10 days. The current architecture is committed. |
| Trying to integrate Number Verification / Quality on Demand APIs | These are Phase 2/3 of the competition, not Phase 1 / FTR. |
| Building the Dockerfile before the local pipeline runs end-to-end | Wastes hours debugging in a slower environment. |

---

## 10. Key Reference Files (read these on session start)

| When you need to know... | Read this file |
| --- | --- |
| What classes Model A detects | `configs/model_a_classes.yaml` |
| How model output becomes JSON labels | `configs/final_label_mapping.yaml` |
| Any threshold / runtime parameter | `configs/thresholds.yaml` |
| Output schema and validation rules | `scripts/validate_results_json.py` |
| What we expect annotators to do | `docs/annotation_guidelines.md` |
| What data we trained on, with licenses | `docs/data_sources.md` |
| Detailed architecture for the FTR report | `docs/architecture.md` |

---

## 11. External Context — Competition Rules That Affect Code

**FR-07 (automated proof requirement).** Every detection in `results.json`
must be provably automated. Unprovable detections are disqualified. Practical
implication: keep the detection log in `src/utils/logger.py`. Each detection
in JSON must have a corresponding log entry with timestamp, frame index,
model confidence, and bbox.

**Scoring weight (final phase).**
- 40% — AI analysis accuracy and precision (detection quality)
- 40% — 5G API integration (Phase 2/3, NOT our concern for FTR)
- 20% — Final design report quality and modern software practices

**For FTR specifically:** the report (text) and the working AI solution
(Docker + code + weights) are weighted ~50/50. So: a working container that
produces a clean JSON is half the battle. Don't under-invest in either.

**Submission medium.** KYS (Kurumsal Yönetim Sistemi). Final artifacts:
- `<team>_FTR.pdf`
- `<team>_docker_image.tar` (or registry link)
- Repository link

---

## 12. Communication Conventions

- **Code, configs, file names, commit messages:** English.
- **The FTR report itself:** Turkish (per competition convention).
- **Chat with the user:** Turkish (user's preference).
- **Documentation in this repo (`docs/`):** English unless explicitly marked
  for the report.

When Claude Code generates text, follow the rule for that text's destination,
not the language of the prompt.

---

## 13. Things That Change Over Time — Keep Updated

The following sections of this file go stale fast. Update them in the same PR
that changes the underlying reality:

- §1 — current phase and deadline
- §3 — output schema (re-check against latest FTR document if Turkcell updates it)
- §5 — runtime budget (after every benchmark run)
- §6 — repository layout (after every reorganization)
- §10 — reference files list (after every config addition)

The team lead is responsible for these updates. If you (Claude) notice
something has drifted, flag it explicitly rather than working around it.

---

## 14. Quick Start for a New Session

If this is your first time in this repo this session, do this:

```bash
# 1. Confirm you're on the right branch
git status

# 2. Read the latest config to understand current class set
cat configs/model_a_classes.yaml
cat configs/final_label_mapping.yaml

# 3. Read the latest data status
cat docs/data_sources.md | head -50

# 4. Check the latest run output
ls -la outputs/

# 5. If there's a results.json, validate it
python scripts/validate_results_json.py outputs/results.json 2>&1 | tail -10
```

That's enough context to start being useful.

---

*Last updated: 18 June 2026 — pre-FTR setup phase. Update this date whenever
the file is edited.*
