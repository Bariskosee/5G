# Annotation Guidelines — TEKNOFEST 2026 5G & AI Smart Road Safety

**Purpose.** These rules apply to ALL manual labeling work for Model A
(unified detector) and Model B (plate detector). Inconsistent labels across
team members will degrade model precision and cost points (precision is 50%
of the AI score weight).

When in doubt, **leave the image unlabeled** rather than guessing.

---

## 1. General Rules

- Always label the **driver's vehicle** as the subject. Surrounding vehicles
  in traffic should NOT be labeled for behavior classes.
- Use tight bounding boxes. Padding > 10% of the object size is incorrect.
- Skip images that are too dark, too blurred, or where the subject is < 5%
  of the frame area.
- One annotator per image. Cross-check 10% of each annotator's work.

---

## 2. Vehicle Type (sedan / suv / hatchback / pickup / minibus / panelvan / kamyon)

- Label only the **full visible body** of a single vehicle per image.
- If two vehicles fully visible and equally prominent, skip the image.
- If body is partially cut off (> 30% missing), skip.

| Class       | Definition                                                       |
| ----------- | ---------------------------------------------------------------- |
| `sedan`     | 3-box car: separate engine + cabin + trunk                       |
| `suv`       | High ride height, monobody, often with rear hatch                |
| `hatchback` | 2-box car: cabin and trunk merged, rear lift gate                |
| `pickup`    | Cabin with open cargo bed                                        |
| `minibus`   | Passenger transport, multiple windows, < 9m length (e.g. Sprinter, Transit Combi) |
| `panelvan`  | Commercial closed cargo van, NO rear side windows (e.g. Transit Van, Doblo Cargo) |
| `kamyon`    | Large truck, > 3.5t, separate cabin + cargo, often box-truck or flatbed |

**Tie-breaker — minibus vs panelvan:** windows on cargo side = minibus,
solid panels = panelvan.

---

## 3. Driver Behavior — phone / cigarette / drink / no_seatbelt

**Critical rule:** only label objects when they are in the **driver's
hand or near the driver's mouth/face**. Phones/bottles/cigarettes that are
on the dashboard, in the passenger seat, or in the back are **not** labeled.

| Class         | Label if...                                                        | Do NOT label if...                          |
| ------------- | ------------------------------------------------------------------ | ------------------------------------------- |
| `phone`       | Phone is in driver's hand AND raised to ear/face/in front of face  | Phone resting on dashboard, in cupholder    |
| `cigarette`   | Cigarette in driver's hand or mouth, smoke visible OR clearly lit  | Unlit cigarette in a pack, ashtray          |
| `drink`       | Bottle/cup in driver's hand AND raised toward mouth                | Bottle in cupholder, on passenger seat      |
| `no_seatbelt` | Driver torso clearly visible AND no seatbelt strap visible across  | Image too dark to see strap; only passenger missing belt |

**Do NOT label** esneme (yawning), arkaya_bakma (looking backward), or
etrafa_bakinma (looking around) — these are produced by MediaPipe at
inference time, not by YOLO. Manual annotations for these would conflict
with the head-pose pipeline.

---

## 4. Objects — teknocan / laptop

### `teknocan`
- **First step before any annotation:** confirm with the team what `teknocan`
  refers to in the competition reference materials. If the team has not seen
  a confirmed reference frame, STOP and escalate. Do not guess.
- Once confirmed, label tightly. Include partial occlusions up to 30%.

### `laptop`
- Any open or closed laptop in the cabin, regardless of who is holding it.
- Do NOT label tablets, e-readers, or large phones.

---

## 5. Person (for passenger ROI assignment)

- Label any visible human in the cabin as `person`.
- One bounding box per person. Do NOT split into "driver" vs "passenger" —
  the ROI module assigns seat position at inference time.
- Partial visibility (face + shoulder) is acceptable.

---

## 6. License Plates (Model B)

- Bounding box must contain the **full plate rectangle** plus 2-3 pixels of
  border padding.
- If the plate is rotated > 30°, label it but flag in metadata for OCR review.
- Skip plates that are completely illegible (no characters readable to the
  human eye) — OCR will not save these.
- Foreign plates: **skip**. The model should only learn Turkish plate shape.

---

## 7. Class Name Discipline

The model class names are fixed in `model_a_classes.yaml` and
`model_b_classes.yaml`. Roboflow / labeling tool class names MUST match
these strings exactly. If a class appears in source data under a different
name (e.g. `mobile_phone`, `cell_phone`, `Phone`), use the remap script
(`scripts/remap_labels.py`) — do not rename manually one-by-one.

---

## 8. Test Set Hygiene

- Test split is sacred. Never look at test images during training, even
  for "spot checking".
- Build separate small test sets for special conditions:
  - `night_or_blur_test/` — 50-100 images for low-light/blurred conditions
  - `rare_class_test/` — 30-50 images each for minibus, panelvan, teknocan
  - `manual_teknocan_test/` — held-out teknocan images (do not use in train)
